import sys
sys.path.append('.')

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("IMAGE EMBEDDING EXTRACTION")
print("="*80)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load pretrained EfficientNet-B0 (best balance of speed + quality)
print("\nLoading EfficientNet-B0...")
model = models.efficientnet_b0(pretrained=True)
# Remove final classification layer, keep feature extractor
model.classifier = torch.nn.Identity()  # This gives 1280-dim embeddings
model.eval()
model.to(device)
print("✓ Model loaded")

# Image preprocessing (ImageNet normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

def extract_embedding(image_path):
    """Extract 1280-dim embedding from image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = model(img_tensor)
            embedding = embedding.squeeze().cpu().numpy()
        
        return embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_dataset(csv_path, image_dir, output_dir, dataset_name='train'):
    """Extract embeddings for entire dataset."""
    
    print(f"\n{'='*80}")
    print(f"Processing {dataset_name.upper()} set")
    print(f"{'='*80}")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    
    # Create output directory
    embeddings_dir = Path(output_dir) / 'embeddings' / dataset_name
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Load download report to find successful images
    download_report = Path(image_dir) / f'download_report_{dataset_name}.csv'
    if download_report.exists():
        report = pd.read_csv(download_report)
        successful = report[report['status'] == 'success']['sample_id'].values
        print(f"Found {len(successful)} successfully downloaded images")
    else:
        print("⚠️  No download report found, will try all images")
        successful = df['sample_id'].values
    
    # Extract embeddings
    embedding_records = []
    failed_count = 0
    
    for sample_id in tqdm(successful, desc=f"Extracting {dataset_name} embeddings"):
        # Find image path (check subfolders)
        subfolder = str(sample_id)[:2]
        image_path = Path(image_dir) / subfolder / f"{sample_id}.jpg"
        
        if not image_path.exists():
            # Try without subfolder
            image_path = Path(image_dir) / f"{sample_id}.jpg"
        
        if not image_path.exists():
            failed_count += 1
            continue
        
        embedding = extract_embedding(image_path)
        
        if embedding is not None:
            # Save embedding
            emb_path = embeddings_dir / f"{sample_id}.npy"
            np.save(emb_path, embedding)
            
            embedding_records.append({
                'sample_id': sample_id,
                'embedding_path': str(emb_path),
                'embedding_dim': len(embedding),
                'has_embedding': True
            })
        else:
            failed_count += 1
    
    # Create index with ALL samples (mark missing ones)
    all_samples = df[['sample_id']].copy()
    embedding_index = pd.DataFrame(embedding_records)
    
    # Merge to include all samples
    full_index = all_samples.merge(embedding_index, on='sample_id', how='left')
    full_index['has_embedding'] = full_index['has_embedding'].fillna(False)
    
    # Save index
    index_path = Path(output_dir) / f'embedding_index_{dataset_name}.csv'
    full_index.to_csv(index_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ Extracted {len(embedding_records)} embeddings")
    print(f"✓ Failed/Missing: {failed_count}")
    print(f"✓ Success rate: {len(embedding_records)/len(successful)*100:.1f}%")
    print(f"✓ Saved to: {embeddings_dir}")
    print(f"✓ Index saved to: {index_path}")
    print(f"{'='*80}\n")
    
    return full_index

# Process train and test
if __name__ == "__main__":
    output_dir = 'data/processed'
    
    # Train set
    train_index = process_dataset(
        csv_path='data/raw/train.csv',
        image_dir='data/raw/train_images',
        output_dir=output_dir,
        dataset_name='train'
    )
    
    # Test set
    test_index = process_dataset(
        csv_path='data/raw/test.csv',
        image_dir='data/raw/test_images',
        output_dir=output_dir,
        dataset_name='test'
    )
    
    print("\n" + "="*80)
    print("✅ EMBEDDING EXTRACTION COMPLETE!")
    print("="*80)