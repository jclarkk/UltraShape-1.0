import argparse
import os
import sys
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from flashpack.serialization import pack_to_file

def convert_to_flashpack(input_path, output_dir):
    print(f"Loading weights from {input_path}...")
    weights = torch.load(input_path, map_location='cpu', weights_only=False)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    components = ['vae', 'dit', 'conditioner']
    
    for component in components:
        if component in weights:
            print(f"Converting {component} to flashpack...")
            state_dict = weights[component]
            output_path = os.path.join(output_dir, f"{component}.flashpack")
            
            # Determine dtype from the first tensor
            first_tensor = next(iter(state_dict.values()))
            dtype = first_tensor.dtype
            
            pack_to_file(
                state_dict,
                output_path,
                target_dtype=dtype,
                silent=False
            )
            print(f"Saved {component} to {output_path}")
        else:
            print(f"Warning: Component {component} not found in weights.")

    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert UltraShape weights to Flashpack format")
    parser.add_argument("--input", type=str, required=True, help="Path to input .pt file")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    
    args = parser.parse_args()
    
    convert_to_flashpack(args.input, args.output)
