import os
import sys
from pathlib import Path
import argparse
from utils import get_model

def setup_directories():
    """Create necessary directories if they don't exist."""
    print("\n=== Creating directories ===")
    directories = [
        'agent/pong',
        'agent/breakout',
        'pretrained_models',
        'models',
        'trajs_block'  # Required for detection and explanation
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def download_pretrained_models():
    """Download pretrained models for both games."""
    print("\n=== Downloading pretrained models ===")
    games = ['PongNoFrameskip-v4', 'BreakoutNoFrameskip-v4']
    for game in games:
        print(f"\nDownloading {game} model...")
        args = argparse.Namespace(
            game=game,
            model_dir='pretrained_models',
            use_pretrained_model=True
        )
        try:
            model, _ = get_model(args)
            print(f"✓ Successfully downloaded {game} model")
        except Exception as e:
            print(f"✗ Error downloading {game} model: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    print("=== SHINE Environment Setup ===")
    setup_directories()
    download_pretrained_models()
    print("\n✓ Setup completed successfully!") 