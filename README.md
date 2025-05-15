# SHINE: Shielded Human-in-the-loop Interactive Learning

This project implements a shielded reinforcement learning agent with human-in-the-loop interactive learning for Atari games (Pong and Breakout).

The main repository of the project is [this](https://github.com/eurekayuan/SHINE)

## Setup

1. Create and activate a virtual environment:

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The project workflow is split into three main steps:

### 1. Setup Environment and Models

Run the setup script to create necessary directories and download pretrained models:

```bash
python setup.py
```

This will:
- Create required directories
- Download pretrained models for Pong and Breakout

### 2. Train and Retrain Models

Run the training script to perform detection, explanation, retraining, and evaluation:

```bash
python train.py --games pong breakout
```

This will:
- Run detection for each game using DGP models
- Generate explanations and triggers
- Retrain models with shielding
- Evaluate performance in both poisoned and clean environments

### 3. Test Shielded Agents

Run the testing script to evaluate the shielded agents:

```bash
python test.py --games PongNoFrameskip-v4 BreakoutNoFrameskip-v4
```

This will:
- Test each shielded agent
- Display performance metrics
- Show shield activation statistics

## Expected Output

After running the complete pipeline, you should see:

1. Setup completion message with downloaded models
2. Training progress for each game:
   - Detection results
   - Explanation generation
   - Retraining progress
   - Evaluation metrics
3. Testing results showing:
   - Average rewards
   - Success rates
   - Shield activation statistics
