import argparse
import subprocess
import sys
import os
from pathlib import Path

def create_trajectory_dir(pattern_type):
    """Create trajectory directory if it doesn't exist."""
    traj_dir = f'trajs_{pattern_type}'
    Path(traj_dir).mkdir(exist_ok=True)
    return traj_dir

def run_detection(game):
    """Run detection for a specific game."""
    print(f"\n=== Running detection for {game} ===")
    try:
        # Create trajectory directory for block pattern
        create_trajectory_dir('block')
        
        subprocess.run(f"python detection.py --name {game} --subname block", 
                      shell=True, check=True)
        print(f"✓ Detection completed successfully for {game}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Detection failed for {game}: {str(e)}")
        sys.exit(1)

def run_explanation(game):
    """Run explanation for a specific game."""
    print(f"\n=== Generating explanations for {game} ===")
    try:
        subprocess.run(f"python explain.py --name {game} --subname block", 
                      shell=True, check=True)
        print(f"✓ Explanation completed successfully for {game}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Explanation failed for {game}: {str(e)}")
        sys.exit(1)

def run_retraining(game):
    """Run retraining for a specific game."""
    print(f"\n=== Retraining model for {game} ===")
    try:
        # Ensure agent directory exists
        agent_dir = f'agent/{game.lower().split("-")[0]}'
        Path(agent_dir).mkdir(parents=True, exist_ok=True)
        
        subprocess.run(f"python retrain.py --name {game} --subname block --mode ours", 
                      shell=True, check=True)
        print(f"✓ Retraining completed successfully for {game}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Retraining failed for {game}: {str(e)}")
        sys.exit(1)

def run_evaluation(game):
    """Run evaluation for a specific game."""
    print(f"\n=== Evaluating {game} ===")
    try:
        # Evaluate in poisoned environment
        subprocess.run(f"python eval.py --name {game} --subname block", 
                      shell=True, check=True)
        print(f"✓ Poisoned environment evaluation completed for {game}")
        
        # Evaluate in clean environment
        subprocess.run(f"python eval.py --name {game} --subname block --no_poison", 
                      shell=True, check=True)
        print(f"✓ Clean environment evaluation completed for {game}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Evaluation failed for {game}: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Train SHINE models')
    parser.add_argument('--games', nargs='+', 
                      default=['pong', 'breakout'],
                      help='Games to train (default: both pong and breakout)')
    args = parser.parse_args()

    print("=== SHINE Training Pipeline ===")
    
    for game in args.games:
        print(f"\n{'='*50}")
        print(f"Processing {game}")
        print(f"{'='*50}")
        
        run_detection(game)
        run_explanation(game)
        run_retraining(game)
        run_evaluation(game)

    print("\n✓ Training pipeline completed successfully!")

if __name__ == "__main__":
    main() 