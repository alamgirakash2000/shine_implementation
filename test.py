import argparse
import subprocess
import sys
from pathlib import Path

def ensure_models_dir():
    """Ensure models directory exists."""
    Path('models').mkdir(exist_ok=True)

def test_shielded_agent(game):
    """Test a shielded agent for a specific game."""
    print(f"\n=== Testing shielded agent for {game} ===")
    game_name = game.lower().split('-')[0]
    model_path = f"agent/{game_name}/block_{game_name}_retrain_ours.tar"
    
    try:
        subprocess.run(
            f"python test_shielded_agent.py --env {game} --model_path {model_path}",
            shell=True,
            check=True
        )
        print(f"✓ Testing completed successfully for {game}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Testing failed for {game}: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Test SHINE shielded agents')
    parser.add_argument('--games', nargs='+', 
                      default=['PongNoFrameskip-v4', 'BreakoutNoFrameskip-v4'],
                      help='Games to test (default: both Pong and Breakout)')
    args = parser.parse_args()

    print("=== SHINE Testing Pipeline ===")
    
    # Ensure models directory exists
    ensure_models_dir()
    
    for game in args.games:
        test_shielded_agent(game)

    print("\n✓ Testing pipeline completed successfully!")

if __name__ == "__main__":
    main()
