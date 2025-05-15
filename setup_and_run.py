import os
import sys
import argparse
import subprocess
from pathlib import Path
import torch
from utils import get_model
import time

def setup_directories():
    """Create necessary directories if they don't exist."""
    print("\n=== Setting up directories ===")
    directories = [
        'agent/pong',
        'agent/breakout',
        'pretrained_models',
        'models',
        'trajs_block',
        'trajs_cross',
        'trajs_equal',
        'trajs_rand0.2',
        'trajs_rand0.3',
        'trajs_4x4',
        'trajs_5x5'
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

def run_detection(game, pattern_type):
    """Run the detection phase."""
    print(f"\n=== Running detection for {game} with {pattern_type} pattern ===")
    base_args = f"--name {game.lower()} --subname {pattern_type}"
    try:
        subprocess.run(f"python detection.py {base_args}", shell=True, check=True)
        print("✓ Detection completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Detection failed: {str(e)}")
        sys.exit(1)

def run_explanation(game, pattern_type):
    """Run the explanation phase."""
    print(f"\n=== Generating explanations for {game} with {pattern_type} pattern ===")
    base_args = f"--name {game.lower()} --subname {pattern_type}"
    try:
        subprocess.run(f"python explain.py {base_args}", shell=True, check=True)
        print("✓ Explanation generation completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Explanation generation failed: {str(e)}")
        sys.exit(1)

def run_retraining(game, pattern_type):
    """Run the retraining phase."""
    print(f"\n=== Retraining model for {game} with {pattern_type} pattern ===")
    base_args = f"--name {game.lower()} --subname {pattern_type}"
    try:
        subprocess.run(f"python retrain.py {base_args}", shell=True, check=True)
        print("✓ Retraining completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Retraining failed: {str(e)}")
        sys.exit(1)

def run_evaluation(game, pattern_type, poisoned=True):
    """Run the evaluation phase."""
    env_type = "poisoned" if poisoned else "clean"
    print(f"\n=== Evaluating {game} with {pattern_type} pattern in {env_type} environment ===")
    base_args = f"--name {game.lower()} --subname {pattern_type}"
    if not poisoned:
        base_args += " --no_poison"
    try:
        subprocess.run(f"python eval.py {base_args}", shell=True, check=True)
        print(f"✓ Evaluation in {env_type} environment completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Evaluation in {env_type} environment failed: {str(e)}")
        sys.exit(1)

def test_shielded_agent(game):
    """Test the shielded agent."""
    print(f"\n=== Testing shielded agent for {game} ===")
    model_path = f"models/shielded_{game.lower().split('-')[0]}_model"
    try:
        subprocess.run(
            f"python test_shielded_agent.py --env {game} --model_path {model_path}",
            shell=True,
            check=True
        )
        print("✓ Shielded agent testing completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Shielded agent testing failed: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Setup and run SHINE pipeline')
    parser.add_argument('--games', nargs='+', default=['PongNoFrameskip-v4', 'BreakoutNoFrameskip-v4'],
                      help='Games to run (default: both Pong and Breakout)')
    parser.add_argument('--patterns', nargs='+', 
                      default=['block', 'cross', 'equal', 'rand0.2', 'rand0.3', '4x4', '5x5'],
                      help='Pattern types to use')
    args = parser.parse_args()

    start_time = time.time()

    # Setup directories
    setup_directories()

    # Download pretrained models
    download_pretrained_models()

    # Run pipeline for each game and pattern
    for game in args.games:
        for pattern in args.patterns:
            print(f"\n{'='*50}")
            print(f"Processing {game} with {pattern} pattern")
            print(f"{'='*50}")

            # Run each phase sequentially
            run_detection(game, pattern)
            run_explanation(game, pattern)
            run_retraining(game, pattern)
            run_evaluation(game, pattern, poisoned=True)
            run_evaluation(game, pattern, poisoned=False)
            test_shielded_agent(game)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n{'='*50}")
    print(f"Pipeline completed in {total_time:.2f} seconds")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 