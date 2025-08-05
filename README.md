# Mining Reward Simulation Tools

This repository contains a collection of Python simulation tools for analyzing mining reward strategies in blockchain networks, with a focus on fee withholding strategies and their impact on miner profitability.

## Overview

The simulations model different mining strategies where pools can choose to either:
- **Regular/Honest**: Always include transaction fees in blocks
- **Strategic/Dishonest**: Withhold fees unless they mined the matured block (after maturity period)

## Project Structure

### Core Simulation Files

- **`2_miners.py`** - Two-pool simulation comparing regular vs strategic strategies
- **`3_miners.py`** - Three-pool simulation with 3D visualization capabilities
- **`6_miners.py`** - Six-pool simulation scenarios
- **`cooperative_strategic.py`** - Analysis of cooperative strategic behavior
- **`strategics.py`** - Multiple strategic pools analysis (2, 3, 7 pools)
- **`fees_random.py`** - Simulation with randomized fee distributions
- **`strategic_vs_strategics.py`** - Comparison between single and multiple strategic pools
- **`regulars_vs_strategics.py`** - Analysis of regular pools vs strategic pools

### Specialized Simulations

#### Reward Sharing Proposals
- **`proposal.py`** - Current block reward strategy analysis
- **`proposal_multi_pool.py`** - Multi-pool current block reward analysis

#### `/results/` - Generated Visualizations
Contains PNG files with simulation results, including:
- Reward share comparisons
- Difference from baseline analyses
- 3D visualizations
- Multi-pool results
- Theoretical vs simulation comparisons

## Key Parameters

All simulations use these common parameters:

```python
blocks = 50000              # Total blocks to simulate
maturity_period = 4000       # Blocks before rewards mature
reward_percentage = 0.10     # Percentage of fees distributed as rewards
base_fee = 0.001            # Base transaction fee
spike_fee = 1.0             # Fee during spike periods
num_spikes = 200            # Number of fee spikes in simulation
```

## Simulation Strategy

### Regular/Honest Strategy
- Pools always include transaction fees in blocks
- Standard mining behavior

### Strategic/Dishonest Strategy
- Pools only include fees if they mined the matured block
- If no matured block exists, fees are included to avoid infinite withholding
- This creates a "withholding" behavior where fees accumulate until strategic conditions are met

## Usage

### Running Basic Simulations

```bash
# Run two-pool comparison
python 2_miners.py

# Run three-pool simulation with 3D plots
python 3_miners.py

# Run cooperative strategic analysis
python cooperative_strategic.py

# Run reward sharing proposal analysis
python proposal.py

# Run multi-pool proposal analysis
python proposal_multi_pool.py

# Run six-pool simulation
python 6_miners.py
```

### Key Functions

Most simulation files include these main functions:

- `run_hashrate_sweep_single_run()` - Single simulation run
- `run_multiple_simulations()` - Multiple runs for statistical analysis
- `plot_*()` - Various plotting functions for results visualization

### Output

Simulations generate:
1. **Console output** with key metrics and statistics
2. **PNG files** in the `/results/` directory showing:
   - Reward share comparisons
   - Difference from baseline
   - 3D visualizations (for 3+ pools)
   - Multi-pool analysis results

## Key Metrics Tracked

- **Reward Balance**: Accumulated fees in the system
- **Miner Rewards**: Individual pool rewards
- **Fee Inclusion Rate**: Percentage of fees included vs withheld
- **Withholding Streaks**: Consecutive blocks with withheld fees
- **Accumulated Fees**: Total fees withheld before inclusion

## Dependencies

```python
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots
```

## Research Applications

These simulations are designed to analyze:

1. **Fee Withholding Strategies**: Impact of strategic fee inclusion on profitability
2. **Multi-Pool Dynamics**: How multiple strategic pools interact
3. **Reward Distribution**: Fairness and efficiency of different reward mechanisms
4. **Network Stability**: Effects of strategic behavior on overall network health

## Results Interpretation

- **Positive differences from baseline**: Strategic pools gain advantage
- **Negative differences**: Strategic pools lose compared to honest behavior
- **3D plots**: Show reward surfaces across different hashrate combinations
- **Multi-pool results**: Demonstrate how strategic behavior scales with pool count

## Contributing

When adding new simulations:
1. Follow the existing parameter structure
2. Include comprehensive metrics tracking
3. Generate appropriate visualizations
4. Document any new strategies or behaviors
5. Add results to the `/results/` directory

## License

This project is part of the RSK blockchain research initiative. 