import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from tqdm import tqdm

# --- PYDRIVE AUTHENTICATION ---
# Assumes 'client_secrets.json' is in the parent folder
os.chdir("..") # Go up from 'visual_model' to 'context-aware-video-retrieval'
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)
print(f"Authenticated with Google Drive. Current dir: {os.getcwd()}")
# -----------------------------

def load_embedding(path: Path) -> np.ndarray:
    """Loads a .npy file and returns it as a numpy array."""
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    return np.load(path)

def sync_embeddings_from_drive(folder_id, local_dir: Path):
    """
    Downloads any .npy files from Google Drive that are not
    already present in the local cache directory.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Syncing embeddings from GDrive to local cache: '{local_dir.name}'...")
    
    # Get list of local files
    local_files = {f.name for f in local_dir.glob('*.npy')}
    
    # Get list of remote files
    query = f"'{folder_id}' in parents and trashed=false"
    remote_files = drive.ListFile({'q': query}).GetList()
    
    # Filter for .npy files that need to be downloaded
    files_to_download = [
        f for f in remote_files 
        if f['title'].endswith('.npy') and f['title'] not in local_files
    ]

    if not files_to_download:
        print("Local cache is already up to date.")
        return

    print(f"Downloading {len(files_to_download)} new embedding files...")
    for file in tqdm(files_to_download, desc="Downloading embeddings"):
        local_path = local_dir / file['title']
        try:
            file.GetContentFile(str(local_path))
        except Exception as e:
            print(f"Warning: Failed to download {file['title']}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Rank video similarity against a query video."
    )
    parser.add_argument(
        "--query", 
        type=str, 
        required=True, 
        help="The stem name of the query video (e.g., '1-1', 'vid-abc')."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=['clip', 'resnet'], 
        help="The model type to compare."
    )
    # --- ARGUMENT CHANGED ---
    parser.add_argument(
        "--embeddings_folder_id", 
        type=str, 
        required=True, 
        help="Google Drive Folder ID where embeddings are stored."
    )
    # --- NEW ARGUMENT ---
    parser.add_argument(
        "--local_cache_dir", 
        type=Path, 
        default=Path("./gdrive_embeddings_cache"), 
        help="Local directory to cache/download embeddings."
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.7, 
        help="Similarity threshold to be considered a 'match'. Default is 0.7."
    )
    args = parser.parse_args()

    # --- NEW: Sync files from Google Drive ---
    sync_embeddings_from_drive(args.embeddings_folder_id, args.local_cache_dir)
    
    # --- MODIFIED: Use the local_cache_dir path ---
    suffix = f"_{args.model}.npy"
    all_embedding_files = list(args.local_cache_dir.glob(f"*{suffix}"))

    if not all_embedding_files:
        print(f"[ERROR] No embeddings found with suffix '*{suffix}' in {args.local_cache_dir}")
        sys.exit(1)

    query_file_path = args.local_cache_dir / (args.query + suffix)
    try:
        query_vec = load_embedding(query_file_path)
        query_vec = query_vec.reshape(1, -1) 
    except FileNotFoundError:
        print(f"[ERROR] Query file not found: {query_file_path}")
        sys.exit(1)

    print(f"Loading {len(all_embedding_files) - 1} other embeddings for comparison...")
    candidate_names = []
    candidate_vectors = []

    for file_path in all_embedding_files:
        video_name = file_path.stem.replace(suffix, "")
        
        if video_name == args.query:
            continue
            
        try:
            candidate_vectors.append(load_embedding(file_path))
            candidate_names.append(video_name)
        except Exception as e:
            print(f"Warning: Could not load {file_path.name}: {e}")

    if not candidate_vectors:
        print("[ERROR] No other videos found to compare against.")
        sys.exit(1)

    all_vectors_matrix = np.vstack(candidate_vectors)

    print("Calculating similarities...")
    scores = cosine_similarity(query_vec, all_vectors_matrix)[0]
    
    results = list(zip(candidate_names, scores))
    
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "="*40)
    print(f"Similarity Ranking for: {args.query} (Model: {args.model.upper()})")
    print("="*40)
    
    for i, (name, score) in enumerate(results):
        print(f"  Rank {i+1:02d}: {name:<10} (Score: {score:.4f})")

    print("\n" + "="*40)
    print(f"Videos Passing Threshold (> {args.threshold:.2f})")
    print("="*40)
    
    matches = [res for res in results if res[1] > args.threshold]
    
    if not matches:
        print("  None.")
    else:
        for i, (name, score) in enumerate(matches):
            print(f"  Match {i+1:02d}: {name:<10} (Score: {score:.4f})")

if __name__ == "__main__":
    main()