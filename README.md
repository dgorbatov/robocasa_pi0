# Minimal Pi0 RoboCasa Rollouts

Stripped-down implementation for running Pi0 policy rollouts on RoboCasa tasks.

## Files

```
robocasa_pi0/
├── run_rollout.py         # Main rollout script
├── pi0_model.py           # Model loading and inference
├── robocasa_transforms.py # Observation/action transforms
└── README.md
```

## Setup

Requires the following packages (same as RLinf):
```bash
pip install robosuite robocasa openpi torch safetensors einops jax
```

## Usage

```bash
# Basic usage
python run_rollout.py --checkpoint ../RLinf/RLinf-Pi0-RoboCasa --task CloseDrawer

# Full options
python run_rollout.py \
    --checkpoint ../RLinf/RLinf-Pi0-RoboCasa \
    --task CloseDrawer \
    --num_episodes 10 \
    --max_steps 300 \
    --seed 42 \
    --device cuda
```

## Available Tasks

| Task | Description |
|------|-------------|
| OpenSingleDoor | open cabinet or microwave door |
| CloseSingleDoor | close cabinet or microwave door |
| OpenDoubleDoor | open double cabinet doors |
| CloseDoubleDoor | close double cabinet doors |
| OpenDrawer | open drawer |
| CloseDrawer | close drawer |
| PnPCounterToCab | pick and place from counter to cabinet |
| PnPCabToCounter | pick and place from cabinet to counter |
| PnPCounterToSink | pick and place from counter to sink |
| PnPSinkToCounter | pick and place from sink to counter |
| PnPCounterToStove | pick and place from counter to stove |
| PnPStoveToCounter | pick and place from stove to counter |
| PnPCounterToMicrowave | pick and place from counter to microwave |
| PnPMicrowaveToCounter | pick and place from microwave to counter |
| TurnOnMicrowave | turn on microwave |
| TurnOffMicrowave | turn off microwave |
| TurnOnSinkFaucet | turn on sink faucet |
| TurnOffSinkFaucet | turn off sink faucet |
| TurnSinkSpout | turn sink spout |
| TurnOnStove | turn on stove |
| TurnOffStove | turn off stove |
| CoffeeSetupMug | setup mug for coffee |
| CoffeeServeMug | serve coffee into mug |
| CoffeePressButton | press coffee machine button |

## Programmatic Usage

```python
from pi0_model import Pi0RobocasaModel
from run_rollout import make_env, extract_obs, TASK_DESCRIPTIONS

# Load model
model = Pi0RobocasaModel("../RLinf/RLinf-Pi0-RoboCasa", device="cuda")

# Create environment
env = make_env("CloseDrawer", seed=0)
raw_obs = env.reset()

# Run inference
obs = extract_obs(raw_obs, TASK_DESCRIPTIONS["CloseDrawer"])
actions = model.predict_actions(obs)  # (1, 10, 12)

# Execute first action
raw_obs, reward, done, info = env.step(actions[0, 0])
```
