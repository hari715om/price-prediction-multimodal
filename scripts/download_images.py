import os
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from typing import Tuple

class ImageDownloader:
    """Robust image downloader with retry logic and checkpointing"""
    
    def __init__(self, output_dir: str, max_workers: int = 24, 
                 max_retries: int = 5, timeout: int = 30):
        """
        Args:
            output_dir: Directory to save images
            max_workers: Number of concurrent download threads
            max_retries: Maximum retry attempts per image
            timeout: Request timeout in seconds
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_single(self, sample_id: str, url: str, 
                       shard_dir: str = None) -> Tuple[str, str, str]:
        """
        Download a single image with retry logic
        
        Returns:
            (sample_id, local_path, status)
        """
        if pd.isna(url) or url == '':
            return sample_id, '', 'no_url'
        
        # Determine save path
        if shard_dir:
            save_dir = self.output_dir / shard_dir
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.output_dir
        
        save_path = save_dir / f"{sample_id}.jpg"
        
        # Skip if already exists
        if save_path.exists() and save_path.stat().st_size > 0:
            return sample_id, str(save_path), 'already_exists'
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout, stream=True)
                response.raise_for_status()
                
                # Save image
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Verify file was written
                if save_path.stat().st_size > 0:
                    return sample_id, str(save_path), 'success'
                else:
                    save_path.unlink()  # Remove empty file
                    return sample_id, '', 'empty_file'
            
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return sample_id, '', 'timeout'
            
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return sample_id, '', f'error: {str(e)[:50]}'
            
            except Exception as e:
                return sample_id, '', f'error: {str(e)[:50]}'
        
        return sample_id, '', 'max_retries_exceeded'
    
    def get_shard_name(self, sample_id: str, shard_size: int = 10000) -> str:
        """
        Determine shard directory for sample_id to avoid single-folder bottleneck
        
        Returns:
            Shard name like '000000-009999'
        """
        # Extract numeric part from sample_id if possible
        try:
            # Assuming sample_id is like 'ABC123' or numeric
            num = int(''.join(filter(str.isdigit, sample_id)))
            shard_start = (num // shard_size) * shard_size
            shard_end = shard_start + shard_size - 1
            return f"{shard_start:06d}-{shard_end:06d}"
        except:
            return 'misc'
    
    def download_batch(self, df: pd.DataFrame, 
                      use_sharding: bool = True,
                      checkpoint_path: str = None) -> pd.DataFrame:
        """
        Download all images from dataframe with parallel processing
        
        Args:
            df: DataFrame with columns ['sample_id', 'image_link']
            use_sharding: Whether to use shard directories
            checkpoint_path: Path to save/load checkpoint CSV
        
        Returns:
            DataFrame with download results
        """
        # Load checkpoint if exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint_df = pd.read_csv(checkpoint_path)
            completed_ids = set(checkpoint_df[checkpoint_df['status'].isin(['success', 'already_exists'])]['sample_id'])
            df_remaining = df[~df['sample_id'].isin(completed_ids)].copy()
            print(f"Resuming: {len(completed_ids)} already downloaded, {len(df_remaining)} remaining")
        else:
            df_remaining = df.copy()
            checkpoint_df = pd.DataFrame()
        
        results = []
        
        print(f"\nDownloading {len(df_remaining)} images with {self.max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for _, row in df_remaining.iterrows():
                sample_id = row['sample_id']
                url = row['image_link']
                
                shard_dir = self.get_shard_name(sample_id) if use_sharding else None
                future = executor.submit(self.download_single, sample_id, url, shard_dir)
                futures[future] = sample_id
            
            # Progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
                sample_id, local_path, status = future.result()
                results.append({
                    'sample_id': sample_id,
                    'local_path': local_path,
                    'status': status
                })
                
                # Periodic checkpointing (every 1000 images)
                if checkpoint_path and len(results) % 1000 == 0:
                    temp_df = pd.DataFrame(results)
                    combined_df = pd.concat([checkpoint_df, temp_df], ignore_index=True)
                    combined_df.to_csv(checkpoint_path, index=False)
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Combine with checkpoint data
        if not checkpoint_df.empty:
            results_df = pd.concat([checkpoint_df, results_df], ignore_index=True)
        
        # Merge with original dataframe
        final_df = df.merge(results_df, on='sample_id', how='left')
        
        # Save final checkpoint
        if checkpoint_path:
            final_df[['sample_id', 'image_link', 'local_path', 'status']].to_csv(
                checkpoint_path, index=False
            )
            print(f"\n✓ Saved checkpoint to {checkpoint_path}")
        
        return final_df
    
    def generate_report(self, results_df: pd.DataFrame, output_path: str):
        """Generate download report"""
        
        total = len(results_df)
        status_counts = results_df['status'].value_counts()
        
        success_count = status_counts.get('success', 0) + status_counts.get('already_exists', 0)
        fail_count = total - success_count
        
        report = {
            'total_urls': total,
            'success_count': success_count,
            'fail_count': fail_count,
            'success_rate': round(success_count / total * 100, 2) if total > 0 else 0,
            'status_breakdown': status_counts.to_dict()
        }
        
        # Save report
        report_df = pd.DataFrame([report])
        report_df.to_csv(output_path, index=False)
        
        # Print summary
        print("\n" + "=" * 80)
        print("DOWNLOAD REPORT")
        print("=" * 80)
        print(f"Total URLs: {total}")
        print(f"Successful: {success_count} ({report['success_rate']}%)")
        print(f"Failed: {fail_count}")
        print("\nStatus Breakdown:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
        print("=" * 80)
        
        return report


# ==================== MAIN SCRIPT ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download product images')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file (train.csv or test.csv)')
    parser.add_argument('--output', type=str, required=True, help='Output directory for images')
    parser.add_argument('--workers', type=int, default=24, help='Number of concurrent workers')
    parser.add_argument('--sample', type=int, default=None, help='Download only N samples (for testing)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint file path')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    # Sample if requested
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
        print(f"Sampling {len(df)} images for testing")
    
    # Set checkpoint path
    if args.checkpoint is None:
        checkpoint_name = f"checkpoint_{Path(args.data).stem}.csv"
        args.checkpoint = os.path.join(args.output, checkpoint_name)
    
    # Initialize downloader
    downloader = ImageDownloader(
        output_dir=args.output,
        max_workers=args.workers,
        max_retries=5,
        timeout=30
    )
    
    # Download images
    results_df = downloader.download_batch(
        df=df,
        use_sharding=True,
        checkpoint_path=args.checkpoint
    )
    
    # Generate report
    report_path = os.path.join(args.output, f'download_report_{Path(args.data).stem}.csv')
    downloader.generate_report(results_df, report_path)
    
    print(f"\n✓ Download complete!")
    print(f"✓ Images saved to: {args.output}")
    print(f"✓ Report saved to: {report_path}")
    print(f"✓ Checkpoint saved to: {args.checkpoint}")