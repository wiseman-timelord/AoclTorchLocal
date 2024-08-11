import os
import json
import torch
from flask import Flask, request, jsonify
from pathlib import Path
from functools import wraps
import time
import ctypes

# Load AOCL libraries
try:
    ctypes.CDLL("libamdblascpu.dll")
    print("AOCL BLAS library loaded successfully")
except OSError:
    print("Failed to load AOCL BLAS library. Make sure it's in your PATH")

# Function to load or create config
def load_or_create_config():
    config_path = Path("./settings_persistent.json")
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        default_config = {
            "torch_model_library": "",
            "api_port_hosting": 5000,
            "selected_model": "",
            "rate_limit": 10,  # requests per second
            "num_threads": os.cpu_count()  # default to all CPU threads
        }
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=4)
        return default_config

# Function to update config
def update_config(config):
    with open("./settings_persistent.json", "w") as f:
        json.dump(config, f, indent=4)

# Function to find .ckpt files
def find_ckpt_files(directory):
    ckpt_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".ckpt"):
                ckpt_files.append(os.path.join(root, file))
    return ckpt_files

# Function to set models library folder
def set_models_library():
    global config
    new_library = input("Enter the path to the models library folder: ").strip()
    if os.path.isdir(new_library):
        config["torch_model_library"] = new_library
        config["selected_model"] = ""  # Reset selected model when changing library
        update_config(config)
        print(f"Models library folder set to: {new_library}")
    else:
        print("Invalid directory. Please try again.")

# Function to select model
def select_model():
    global config
    if not config["torch_model_library"] or not os.path.isdir(config["torch_model_library"]):
        print("Models library folder not set or invalid. Please set it first.")
        return

    ckpt_files = find_ckpt_files(config["torch_model_library"])
    if not ckpt_files:
        print("No .ckpt files found in the specified directory.")
        return

    print("\nAvailable models:")
    for i, file_path in enumerate(ckpt_files, 1):
        print(f"{i}. {os.path.relpath(file_path, config['torch_model_library'])}")

    while True:
        try:
            selection = int(input("Select a model number: "))
            if 1 <= selection <= len(ckpt_files):
                config["selected_model"] = ckpt_files[selection - 1]
                update_config(config)
                print(f"Selected model: {os.path.relpath(config['selected_model'], config['torch_model_library'])}")
                return
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

# Function to change API address
def change_api_address():
    global config
    new_address = input("Enter new API address (press Enter to keep current): ").strip()
    if new_address:
        config["api_address_hosting"] = new_address
        update_config(config)
        print(f"API address updated to: {new_address}")

# Function to change API port
def change_api_port():
    global config
    while True:
        new_port = input("Enter new API port (press Enter to keep current): ").strip()
        if not new_port:
            return
        try:
            new_port = int(new_port)
            if 1024 <= new_port <= 65535:
                config["api_port_hosting"] = new_port
                update_config(config)
                print(f"API port updated to: {new_port}")
                return
            else:
                print("Port must be between 1024 and 65535. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

# Function to load model
def load_model(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                # This is likely a checkpoint dict
                model = checkpoint['model'] if 'model' in checkpoint else None
                if model is None:
                    raise ValueError("Model architecture not found in checkpoint")
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # This is likely just a state dict
                # You might need to define your model architecture here
                from torch import nn
                model = nn.Sequential(
                    nn.Linear(10, 50),
                    nn.ReLU(),
                    nn.Linear(50, 1)
                )
                model.load_state_dict(checkpoint)
        else:
            # This is likely the full model
            model = checkpoint
        model.eval()
        
        # Optimize model for inference
        model = torch.jit.script(model)
        model = torch.jit.freeze(model)
        
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Function to start hosting
def start_hosting():
    global config
    if not config["selected_model"]:
        print("No model selected. Please select a model first.")
        return

    if not os.path.exists(config["selected_model"]):
        print(f"Error: Selected model file does not exist: {config['selected_model']}")
        return

    print(f"Loading model from {config['selected_model']}...")
    model = load_model(config["selected_model"])
    if model is None:
        return

    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    @rate_limit(config['rate_limit'])
    def predict():
        try:
            data = request.json['input']
            
            # Input validation and preprocessing
            if not isinstance(data, list):
                return jsonify({'error': 'Input must be a list'}), 400
            
            input_tensor = torch.tensor(data, dtype=torch.float32)
            
            # You might want to add more preprocessing here depending on your model's requirements
            
            with torch.no_grad():
                output = model(input_tensor)
            
            # Post-processing (if needed)
            processed_output = output.tolist()  # Convert to list for JSON serialization
            
            return jsonify({'output': processed_output})
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    print(f"Starting model server at http://localhost:{config['api_port_hosting']}/")
    app.run(host='localhost', port=config['api_port_hosting'])
    
# Function to set number of threads
def set_num_threads():
    global config
    max_threads = os.cpu_count()
    while True:
        new_threads = input(f"Enter number of threads to use (1-{max_threads}, current: {config['num_threads']}): ").strip()
        if not new_threads:
            return
        try:
            new_threads = int(new_threads)
            if 1 <= new_threads <= max_threads:
                config['num_threads'] = new_threads
                update_config(config)
                torch.set_num_threads(new_threads)
                print(f"Number of threads updated to: {new_threads}")
                return
            else:
                print(f"Please enter a number between 1 and {max_threads}.")
        except ValueError:
            print("Please enter a valid number.")

# Main menu function
def main_menu():
    while True:
        print("\n=================")
        print("    AmdTorchLocal")
        print("------------------------")
        print(f"1. Set Models Library Folder")
        print(f"   (Current: {config['torch_model_library'] or 'Not set'})")
        print(f"2. Select Model")
        print(f"   (Current: {get_current_model_path()})")
        print(f"3. Change Port")
        print(f"   (Current: {config['api_port_hosting']})")
        print(f"4. Change Rate Limit")
        print(f"   (Current: {config['rate_limit']} requests/second)")
        print(f"5. Set Number of Threads")
        print(f"   (Current: {config['num_threads']})")
        print("------------------------")
        choice = input("Select: Menu Options (1-5), Begin Hosting (B), Exit Program (X): ").strip().upper()

        if choice == '1':
            set_models_library()
        elif choice == '2':
            select_model()
        elif choice == '3':
            change_api_port()
        elif choice == '4':
            new_limit = input("Enter new rate limit (requests per second): ").strip()
            try:
                config['rate_limit'] = int(new_limit)
                update_config(config)
                print(f"Rate limit updated to: {config['rate_limit']} requests/second")
            except ValueError:
                print("Invalid input. Please enter a number.")
        elif choice == '5':
            set_num_threads()
        elif choice == 'B':
            start_hosting()
            break  # Exit the loop after hosting (you might want to change this behavior)
        elif choice == 'X':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

# Load config
config = load_or_create_config()

# Set up PyTorch optimizations
torch.set_num_threads(config['num_threads'])
torch.set_num_interop_threads(1)  # Set to 1 for best performance with AOCL
torch.backends.mkldnn.enabled = True
torch.backends.openmp.enabled = True

# Enable AVX2 if supported
if torch.backends.cpu.is_avx2_supported():
    torch.backends.cpu.enable_avx2()
    print("AVX2 is enabled")
else:
    print("AVX2 is not supported on this CPU")

print(f"PyTorch is using {config['num_threads']} threads")
print(f"MKL-DNN is enabled: {torch.backends.mkldnn.enabled}")
print(f"OpenMP is enabled: {torch.backends.openmp.enabled}")

# Start the main menu
if __name__ == "__main__":
    main_menu()