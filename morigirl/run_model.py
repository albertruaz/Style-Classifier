#!/usr/bin/env python3
# run_model.py

import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import sys
sys.path.append('..')
from database import DatabaseManager
from morigirl.morigirl_model import MoriGirlVectorClassifier
from tqdm import tqdm
import argparse

# Hardcoded product_id list - modify as needed
PRODUCT_IDS = [
    620033, 611959, 609042, 586172, 610956, 610695, 595703, 590458, 414379,
    601246, 542363, 535167, 596296, 596919, 
]

class MorigirlModelRunner:
    """Run Morigirl model predictions on specific product IDs"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.db_manager = DatabaseManager()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get checkpoint path
        self.checkpoint_path = self._get_checkpoint_path()
        
        print(f"🚀 Morigirl Model Runner initialized")
        print(f"  - Device: {self.device}")
        print(f"  - Checkpoint: {self.checkpoint_path}")

    def load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ Configuration loaded: {config_path}")
            return config
        except Exception as e:
            print(f"❌ Configuration load failed: {e}")
            # Return default configuration
            return {
                "data": {"max_products_per_type": 5000},
                "model": {"input_vector_dim": 1024, "hidden_dim": 256, "hidden_dim2": 128, "dropout_rate": 0.1}
            }

    def _get_checkpoint_path(self) -> str:
        """Get checkpoint path from config"""
        data_config = self.config["data"]
        test_paths = data_config.get("test_paths", {})
        
        # 1. Use checkpoint_path if directly set
        if test_paths.get("checkpoint_path"):
            checkpoint_path = test_paths["checkpoint_path"]
            if os.path.exists(checkpoint_path):
                print(f"🔍 User-defined checkpoint: {checkpoint_path}")
                return checkpoint_path
            else:
                print(f"⚠️  Specified checkpoint does not exist: {checkpoint_path}")
        
        # 2. Auto-search if auto_find_best_model is enabled
        if test_paths.get("auto_find_best_model", True):
            target_experiment = test_paths.get("target_experiment")
            
            if target_experiment:
                # Find best_model.pth of specific experiment
                checkpoint_path = f"result/{target_experiment}/checkpoints/best_model.pth"
                if os.path.exists(checkpoint_path):
                    print(f"🔍 Auto-found checkpoint: {checkpoint_path}")
                    return checkpoint_path
                else:
                    print(f"⚠️  Target experiment checkpoint does not exist: {checkpoint_path}")
            
            # Find latest experiment's best_model.pth
            result_dir = "result"
            if os.path.exists(result_dir):
                experiments = [d for d in os.listdir(result_dir) 
                             if os.path.isdir(os.path.join(result_dir, d))]
                if experiments:
                    # Sort by experiment name (date-time based, latest first)
                    experiments.sort(reverse=True)
                    for exp in experiments:
                        checkpoint_path = f"{result_dir}/{exp}/checkpoints/best_model.pth"
                        if os.path.exists(checkpoint_path):
                            print(f"🔍 Latest experiment checkpoint: {checkpoint_path}")
                            return checkpoint_path
        
        # 3. Error if not found
        raise FileNotFoundError(
            "Could not find checkpoint. Please set one of the following:\n"
            "1. Set test_paths.checkpoint_path in config.json\n"
            "2. Set test_paths.target_experiment in config.json\n"
            "3. Check if experiment results exist in result/ folder"
        )

    def load_model(self) -> nn.Module:
        """Load model from checkpoint"""
        print(f"\n📦 Loading model...")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Read model parameters from config
        model_config = self.config["model"]
        model_kwargs = {
            "input_dim": model_config["input_vector_dim"],
            "hidden_dim": model_config["hidden_dim"],
            "hidden_dim2": model_config["hidden_dim2"],
            "dropout_rate": model_config["dropout_rate"]
        }
        
        # Create model
        model = MoriGirlVectorClassifier(**model_kwargs)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Validation accuracy: {checkpoint.get('metrics', {}).get('accuracy', 'N/A')}")
        
        return model

    def get_product_vectors(self, product_ids: List[int]) -> Dict[int, List[float]]:
        """Get product vectors from Vector DB"""
        if not product_ids:
            return {}
            
        vector_session = self.db_manager.vector_db.Session()
        try:
            from sqlalchemy import text
            
            print(f"🔍 Fetching vectors for {len(product_ids)} products...")
            
            # Process in batches
            batch_size = 1000
            all_vectors = {}
            
            for i in range(0, len(product_ids), batch_size):
                batch_ids = product_ids[i:i + batch_size]
                
                # Direct comparison with product_id
                placeholders = ','.join([str(batch_id) for batch_id in batch_ids])
                sql = text(f"""
                    SELECT product_id, vector
                    FROM product_image_vector
                    WHERE product_id IN ({placeholders})
                      AND vector IS NOT NULL
                """)
                
                result = vector_session.execute(sql)
                
                for product_id, vector_str in result.fetchall():
                    if vector_str:
                        # Parse "[1.0,2.0,3.0,...]" format
                        if isinstance(vector_str, str):
                            vector_str = vector_str.strip('[]')
                            vector = [float(x) for x in vector_str.split(',')]
                        else:
                            # Already list or array
                            vector = list(vector_str)
                        all_vectors[int(product_id)] = vector
            
            print(f"✅ Found vectors for {len(all_vectors)} products")
            return all_vectors
            
        finally:
            vector_session.close()

    def predict_products(self, product_ids: List[int], output_file: str = "morigirl_predictions.csv") -> pd.DataFrame:
        """Predict morigirl probabilities for given product IDs"""
        print(f"\n🎯 Starting predictions for {len(product_ids)} products...")
        
        # 1. Load model
        model = self.load_model()
        
        # 2. Get vectors from DB
        product_vectors = self.get_product_vectors(product_ids)
        
        # 3. Prepare results
        results = []
        
        print(f"\n🔮 Running predictions...")
        
        # Process each product
        for product_id in tqdm(product_ids, desc="Predicting"):
            if product_id in product_vectors:
                vector = product_vectors[product_id]
                
                # Convert to tensor
                vector_tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Predict
                with torch.no_grad():
                    prob = model(vector_tensor).item()
                    pred_label = int(prob > 0.5)
                
                results.append({
                    'product_id': product_id,
                    'morigirl_probability': prob,
                    'predicted_label': pred_label,
                    'prediction': 'Morigirl' if pred_label == 1 else 'Non-Morigirl',
                    'confidence': prob if pred_label == 1 else (1 - prob),
                    'vector_found': True
                })
            else:
                # No vector found
                results.append({
                    'product_id': product_id,
                    'morigirl_probability': None,
                    'predicted_label': None,
                    'prediction': 'No vector data',
                    'confidence': None,
                    'vector_found': False
                })
        
        # 4. Create DataFrame and save to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        
        print(f"\n✅ Predictions completed!")
        print(f"📁 Results saved to: {output_file}")
        
        # Print summary
        vector_found_count = results_df['vector_found'].sum()
        morigirl_count = results_df['predicted_label'].sum() if vector_found_count > 0 else 0
        
        print(f"\n📊 Summary:")
        print(f"  - Total products: {len(product_ids)}")
        print(f"  - Vectors found: {vector_found_count}")
        print(f"  - Vectors missing: {len(product_ids) - vector_found_count}")
        if vector_found_count > 0:
            print(f"  - Predicted as Morigirl: {morigirl_count}")
            print(f"  - Predicted as Non-Morigirl: {vector_found_count - morigirl_count}")
            avg_prob = results_df[results_df['vector_found']]['morigirl_probability'].mean()
            print(f"  - Average Morigirl probability: {avg_prob:.4f}")
        
        return results_df

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'db_manager'):
            self.db_manager.dispose_all()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run Morigirl model predictions on specific product IDs')
    parser.add_argument('--config-path', default='config.json', help='Configuration file path')
    parser.add_argument('--output-file', default='morigirl_predictions.csv', help='Output CSV file path')
    parser.add_argument('--product-ids', nargs='+', type=int, help='List of product IDs to predict (overrides hardcoded list)')
    
    args = parser.parse_args()
    
    try:
        # Determine product IDs to use
        if args.product_ids:
            product_ids = args.product_ids
            print(f"✅ Using product IDs from command line: {len(product_ids)} products")
        else:
            product_ids = PRODUCT_IDS
            print(f"✅ Using hardcoded product IDs: {len(product_ids)} products")
        
        print(f"🎯 Product IDs: {product_ids}")
        
        # Initialize runner
        runner = MorigirlModelRunner(config_path=args.config_path)
        
        # Run predictions
        results_df = runner.predict_products(product_ids, args.output_file)
        
        print(f"\n🎉 Prediction completed successfully!")
        print(f"📄 Results saved to: {args.output_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            runner.close()
        except:
            pass

if __name__ == "__main__":
    main() 