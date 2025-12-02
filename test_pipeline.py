"""
Simple test script to verify the pipeline works.
Run this from the project root directory.
"""

from backend.models import run_full_pipeline
from pathlib import Path

if __name__ == "__main__":
    # Get path to CSV
    csv_path = Path("data") / "stocks_raw.csv"
    
    print(f"Running pipeline with CSV: {csv_path}")
    print("This may take a few minutes...")
    
    # Run pipeline
    result = run_full_pipeline(str(csv_path))
    
    # Print summary
    print("\n=== Pipeline Results ===")
    print(f"Best model: {result['models_info']['best_model_name']}")
    print(f"\nModel metrics:")
    print(result['models_info']['metrics'])
    print(f"\nCluster stats:")
    print(result['cluster_stats'])
    print(f"\nTrain shape: {result['splits']['X_train_scaled'].shape}")
    print(f"Val shape: {result['splits']['X_val_scaled'].shape}")
    print(f"Test shape: {result['splits']['X_test_scaled'].shape}")
    print("\nPipeline completed successfully!")

