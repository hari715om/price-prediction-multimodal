# File: scripts/rebuild_index.py
import pandas as pd
from pathlib import Path
import argparse

def rebuild_index(dataset_name: str):
    """
    Scans for existing .npy embedding files and creates a new index.csv.
    This is useful for resuming from an interrupted process.
    """
    print(f"--- Rebuilding index for '{dataset_name}' dataset ---")

    # Define paths
    raw_csv_path = Path(f'data/raw/{dataset_name}.csv')
    embeddings_dir = Path(f'data/processed/embeddings/{dataset_name}/')
    output_path = Path(f'data/processed/embedding_index_{dataset_name}.csv')

    if not raw_csv_path.exists():
        print(f"ERROR: Raw CSV file not found at {raw_csv_path}")
        return

    # 1. Load all sample IDs from the original CSV
    all_samples_df = pd.read_csv(raw_csv_path)[['sample_id']]
    print(f"Found {len(all_samples_df)} total samples in {raw_csv_path.name}")

    # 2. Find all existing embedding files
    existing_embeddings = list(embeddings_dir.glob('*.npy'))
    print(f"Found {len(existing_embeddings)} existing .npy files in {embeddings_dir}")

    if not existing_embeddings:
        print("Warning: No embedding files found. Creating an empty index.")
        records = []
    else:
        # 3. Create records for the files that exist
        records = [{
            'sample_id': p.stem, # '12345.npy' -> '12345'
            'embedding_path': str(p),
            'has_embedding': True
        } for p in existing_embeddings]

    # 4. Create the final index
    embedding_index = pd.DataFrame(records)

    all_samples_df['sample_id'] = all_samples_df['sample_id'].astype(str)
    embedding_index['sample_id'] = embedding_index['sample_id'].astype(str)
    
    # Merge with all samples to include those with missing embeddings
    full_index = all_samples_df.merge(embedding_index, on='sample_id', how='left')
    full_index['has_embedding'] = full_index['has_embedding'].fillna(False)

    # Convert sample_id to the correct type to avoid merge issues later
    full_index['sample_id'] = full_index['sample_id'].astype(all_samples_df['sample_id'].dtype)

    # 5. Save the new index file
    full_index.to_csv(output_path, index=False)
    print(f"\n✅ Successfully created new index file at: {output_path}")
    print(f"   - Total entries: {len(full_index)}")
    print(f"   - Embeddings found: {full_index['has_embedding'].sum()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild embedding index from existing files.")
    parser.add_argument("--dataset", type=str, required=True, choices=['train', 'test'], help="The dataset to process ('train' or 'test').")
    args = parser.parse_args()
    rebuild_index(args.dataset)