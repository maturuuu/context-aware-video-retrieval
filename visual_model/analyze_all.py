import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
import pandas as pd

# --- Assumes script is in a subfolder, e.g., 'visual_model' ---
# Go up to the main project folder, e.g., 'context-aware-video-retrieval'
try:
    os.chdir("..")
    print(f"Working directory set to: {os.getcwd()}")
except FileNotFoundError:
    print(f"Could not change directory. Assuming already in: {os.getcwd()}")
# -----------------------------------------------------------------

# --- THIS IS THE CORRECTED FUNCTION ---
def get_trend(video_name: str) -> str:
    """
    Extracts the trend name (e.g., 'trend1') from 'trend1vid1'
    or 'airball' from 'airball_1'.
    """
    if 'vid' in video_name:
        return video_name.split('vid')[0]
    # Fallback for other formats like 'airball_1'
    return video_name.split('_')[0]
# --- ------------------------------ ---

def load_all_embeddings(embeddings_dir: Path, suffix: str) -> tuple[list, np.ndarray]:
    """Loads all embeddings into a sorted list and a 2D numpy matrix."""
    video_names = []
    video_vectors = []
    
    all_files = list(embeddings_dir.glob(f"*{suffix}"))
    if not all_files:
        print(f"[ERROR] No embeddings found with suffix '*{suffix}' in {embeddings_dir.resolve()}")
        sys.exit(1)
        
    print(f"Loading {len(all_files)} embeddings...")
    for file_path in all_files:
        # --- FIX: Get the stem from the *full* filename ---
        # e.g., "trend1vid1_emb-visual2048"
        full_stem = file_path.stem
        
        # This will find the correct file suffix (e.g., "_emb-visual2048")
        # and remove it, leaving just "trend1vid1"
        if "_emb-" in full_stem:
            video_name = full_stem.split("_emb-")[0]
        else:
             # Fallback for your other naming convention
            video_name = file_path.stem.replace(suffix, "")

        try:
            video_names.append(video_name)
            video_vectors.append(np.load(file_path))
        except Exception as e:
            print(f"Warning: Could not load {file_path.name}: {e}")
            
    return video_names, np.vstack(video_vectors)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze all queries to find best/worst examples and trend confusion."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=['clip', 'resnet'], 
        help="The model type to analyze."
    )
    parser.add_argument(
        "--embeddings_dir", 
        type=Path, 
        required=True, 
        help="Local directory where the .npy embedding files are stored."
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=5, 
        help="Top-K value to use for Precision@K calculation. Default is 5."
    )
    args = parser.parse_args()

    # --- 1. Load Data ---
    # This suffix logic is now a fallback, as load_all_embeddings is smarter
    suffix = f"_{args.model}.npy" 
    
    # We must determine the REAL suffix from your pipeline
    if args.model == 'resnet':
        embedding_suffix = "_resnet.npy"
    elif args.model == 'clip':
         embedding_suffix = "_clip.npy" # <-- You'll need to confirm this filename
    else:
        embedding_suffix = suffix # Fallback
        
    print(f"Using model '{args.model}' and suffix '{embedding_suffix}'")

    video_names, all_vectors = load_all_embeddings(args.embeddings_dir, embedding_suffix)
    
    # --- 2. Calculate All-vs-All Similarity Matrix (Fast) ---
    print("Calculating all-pairs similarity matrix...")
    sim_matrix = cosine_similarity(all_vectors)
    
    # --- 3. Calculate Precision@K for Every Video ---
    query_scores = [] 
    all_trends = sorted(list(set(get_trend(name) for name in video_names)))
    confusion_matrix = pd.DataFrame(0, index=all_trends, columns=all_trends)

    print(f"Analyzing all {len(video_names)} queries for Precision@{args.k}...")
    
    for i, query_name in enumerate(video_names):
        query_trend = get_trend(query_name)
        
        scores = sim_matrix[i]
        
        results = []
        for j, score in enumerate(scores):
            if i == j: 
                continue
            results.append((video_names[j], score))
            
        results.sort(key=lambda x: x[1], reverse=True)
        top_k_results = results[:args.k]
        
        correct_matches = 0
        for (video_name, score) in top_k_results:
            result_trend = get_trend(video_name)
            
            # This check is to prevent an error if a new trend appears
            if result_trend in confusion_matrix.index and query_trend in confusion_matrix.columns:
                confusion_matrix.loc[query_trend, result_trend] += 1
            
            if result_trend == query_trend:
                correct_matches += 1
                
        precision_at_k = correct_matches / args.k
        query_scores.append((query_name, precision_at_k))

    # --- 4. Print Results ---
    
    query_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*50)
    print(f"🏆 BEST Examples (Highest Precision@{args.k})")
    print("   These are your 'poster children' for success.")
    print("="*50)
    for i in range(min(10, len(query_scores))):
        name, score = query_scores[i]
        print(f"  Rank {i+1:02d}: {name:<15} (P@{args.k} = {score*100:6.2f}%)")

    query_scores.sort(key=lambda x: x[1], reverse=False)

    print("\n" + "="*50)
    print(f"🔥 WORST Examples (Lowest Precision@{args.k})")
    print("   These are your best discussion points for failure.")
    print("="*50)
    for i in range(min(10, len(query_scores))):
        name, score = query_scores[i]
        print(f"  Rank {i+1:02d}: {name:<15} (P@{args.k} = {score*100:6.2f}%)")
        
    print("\n" + "="*50)
    print(f"📊 Trend Confusion Matrix (Based on Top {args.k} results)")
    print("   (Row = Query Trend, Column = Result Trend)")
    print("="*50)
    print(confusion_matrix)

if __name__ == "__main__":
    main()