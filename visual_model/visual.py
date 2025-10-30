# how to run
# python visual.py --video_folder_id "1Z8ZyToYAb6xG7aRWSiNDkbxBX8HOdUgV" --output_folder_id "15eifVkFWOhUxgOlc4Uf_i-WorhlagTKF"     

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import time

# --- PYDRIVE AUTHENTICATION ---
gauth = GoogleAuth()
gauth.LocalWebserverAuth() 
drive = GoogleDrive(gauth)
# -----------------------------

def get_resnet_model(device: str):
    """ (This function is unchanged) """
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(device)
    preprocess = weights.transforms()
    return model, preprocess

def extract_resnet_embeddings(
    video_path: Path, 
    model, 
    preprocess, 
    device: str, 
    frame_sample_rate: int = 30, 
    batch_size: int = 32
) -> np.ndarray:
    """ (This function is unchanged) """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_features = []
    frame_batch = []
    frame_idx = 0
    pbar = tqdm(total=frame_count, desc=f"Frames for {video_path.name}", leave=False)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret: break
            pbar.update(1)
            
            if frame_idx % frame_sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                frame_batch.append(pil_img)

                if len(frame_batch) == batch_size:
                    image_inputs = torch.stack(
                        [preprocess(img) for img in frame_batch]
                    ).to(device)
                    image_features = model(image_inputs)
                    all_features.append(image_features.squeeze().cpu().numpy())
                    frame_batch = []
            frame_idx += 1
        
        if frame_batch:
            image_inputs = torch.stack(
                [preprocess(img) for img in frame_batch]
            ).to(device)
            image_features = model(image_inputs)
            all_features.append(image_features.squeeze().cpu().numpy())

    cap.release()
    pbar.close()
    if not all_features:
        raise ValueError(f"No frames sampled for {video_path.name}")

    embeddings = np.vstack(all_features)
    mean_embedding = np.mean(embeddings, axis=0)
    return mean_embedding

def main():
    parser = argparse.ArgumentParser(description="Extract ResNet-50 embeddings from Google Drive.")
    parser.add_argument("--video_folder_id", type=str, required=True, help="Google Drive Folder ID for input videos.")
    parser.add_argument("--output_folder_id", type=str, required=True, help="Google Drive Folder ID for output embeddings.")
    parser.add_argument("--frame_sample_rate", type=int, default=30, help="Sample every Nth frame.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing frames.")
    args = parser.parse_args()
    
    print("Setting up model and device...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = get_resnet_model(device)
    print(f"Using device: {device}")
    
    print("Listing files from Google Drive...")
    query = f"'{args.video_folder_id}' in parents and trashed=false"
    video_files = drive.ListFile({'q': query}).GetList()
    
    output_query = f"'{args.output_folder_id}' in parents and trashed=false"
    existing_embeddings = [f['title'] for f in drive.ListFile({'q': output_query}).GetList()]
    
    print(f"Found {len(video_files)} videos in Drive.")
    
    for video_file in tqdm(video_files, desc="Processing Videos"):
        video_title = video_file['title']
        video_stem = Path(video_title).stem
        output_filename = f"{video_stem}_resnet.npy"

        if output_filename in existing_embeddings:
            continue
            
        temp_video_path = Path(".") / video_title
        temp_npy_path = Path(".") / output_filename
        
        try:
            tqdm.write(f"Downloading {video_title}...")
            video_file.GetContentFile(str(temp_video_path))
            
            tqdm.write(f"Processing {video_title}...")
            mean_embedding = extract_resnet_embeddings(
                video_path=temp_video_path,
                model=model,
                preprocess=preprocess,
                device=device,
                frame_sample_rate=args.frame_sample_rate,
                batch_size=args.batch_size
            )
            np.save(temp_npy_path, mean_embedding)

            tqdm.write(f"Uploading {output_filename}...")
            new_file = drive.CreateFile({
                'title': output_filename,
                'parents': [{'id': args.output_folder_id}]
            })
            new_file.SetContentFile(str(temp_npy_path))
            new_file.Upload()

        except Exception as e:
            tqdm.write(f"\n[ERROR] Failed to process {video_title}: {e}")
            
        finally:
            if temp_video_path.exists():
                os.remove(temp_video_path)
            
            if temp_npy_path.exists():
                try:
                    os.remove(temp_npy_path)
                except PermissionError:
                    tqdm.write(f"\n[INFO] {temp_npy_path.name} is locked. Retrying in 5s...")
                    try:
                        time.sleep(5)
                        os.remove(temp_npy_path)
                    except Exception as e:
                        tqdm.write(f"\n[WARNING] Could not delete temp file {temp_npy_path.name}: {e}")
                except Exception as e:
                     tqdm.write(f"\n[WARNING] Could not delete temp file {temp_npy_path.name}: {e}")

    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()