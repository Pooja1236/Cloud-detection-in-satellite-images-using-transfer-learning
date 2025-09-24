import numpy as np
import matplotlib.pyplot as plt
import glob

def diagnose_files():
    files = glob.glob("*.npy")
    
    for f in files:
        if f.startswith('._'):
            continue
            
        print(f"\n📁 File: {f}")
        try:
            data = np.load(f)
            print(f"   Shape: {data.shape}")
            print(f"   Type: {data.dtype}")
            print(f"   Range: {data.min()} to {data.max()}")
            print(f"   Mean: {data.mean():.2f}")
            print(f"   Std: {data.std():.2f}")
            
            if 'mask' in f.lower():
                unique_vals = np.unique(data)
                print(f"   Unique values: {unique_vals}")
                if len(unique_vals) == 2:
                    coverage = (data == unique_vals[-1]).sum() / data.size * 100
                    print(f"   Coverage: {coverage:.2f}%")
            
        except Exception as e:
            print(f"   Error: {e}")

if __name__ == "__main__":
    diagnose_files()
