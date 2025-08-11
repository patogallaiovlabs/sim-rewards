import random
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from scipy.stats import expon

# Parameters
blocks = 100000
maturity_period = 4000
reward_percentage = 0.10
base_fee = 0.001
spike_fee = 1.0
num_spikes = 200  # Number of random spikes to generate
num_runs = 5  # Number of simulation runs for statistical analysis

# Sibling parameters
sibling_mean = 0.6  # Average number of siblings
sibling_min = 0
sibling_max = 4

# Fixed percentage parameters to test
fixed_percentages = [50, 40, 30, 20]  # Percentage that goes to the matured block miner

# Pool configuration - 7 miners with equal hashrates
num_miners = 7
miner_hashrate = 1.0 / num_miners  # Equal hashrate for all miners

# Sibling production probabilities for each miner (must sum to 1.0)
# This determines how much of each miner's blocks become siblings vs main trunk
# Ordered from highest to lowest sibling production probability
sibling_production_probabilities = {
    0: 0.30,  # Miner 0: 30% of siblings (highest sibling producer)
    1: 0.25,  # Miner 1: 25% of siblings (high sibling producer)
    2: 0.15,  # Miner 2: 15% of siblings (medium-high)
    3: 0.10,  # Miner 3: 10% of siblings (medium-low)
    4: 0.10,  # Miner 4: 10% of siblings (medium-low)
    5: 0.05,  # Miner 5: 5% of siblings (lowest sibling producer)
    6: 0.05   # Miner 6: 5% of siblings (lowest sibling producer)
}

# Verify sibling probabilities sum to 1.0
total_sibling_prob = sum(sibling_production_probabilities.values())
if abs(total_sibling_prob - 1.0) > 0.001:
    print(f"Warning: Sibling production probabilities don't sum to 1.0: {total_sibling_prob:.3f}")
    # Normalize to sum to 1.0
    for miner_id in sibling_production_probabilities:
        sibling_production_probabilities[miner_id] /= total_sibling_prob

# Sort miners by sibling production probability (highest to lowest)
sorted_miner_ids = sorted(sibling_production_probabilities.keys(), 
                         key=lambda x: sibling_production_probabilities[x], reverse=True)

# Use time-based seed for different results each run
random.seed(int(time.time()))

# Generate random spike positions (deterministic)
spike_positions = set(random.sample(range(blocks), num_spikes))

def generate_siblings_and_assign_miners():
    """Generate number of siblings and assign miners to them"""
    # Generate exponential random variable with mean 0.6
    sibling_count = expon.rvs(scale=sibling_mean)
    sibling_count = int(round(sibling_count))
    sibling_count = max(sibling_min, min(sibling_max, sibling_count))
    
    # Assign miners to siblings based on sibling production probabilities
    sibling_miners = []
    for _ in range(sibling_count):
        rand = random.random()
        cumulative = 0
        for miner_id, prob in sibling_production_probabilities.items():
            cumulative += prob
            if rand < cumulative:
                sibling_miners.append(miner_id)
                break
    
    return sibling_count, sibling_miners

def setup_miners():
    """Setup miner configuration with equal hashrates"""
    miner_hashrates = {}
    for i in range(num_miners):
        miner_hashrates[i] = miner_hashrate
    
    return miner_hashrates

def simulate_current_reward_logic(miner_sequence, miner_hashrates):
    """Simulate current reward logic with siblings"""
    reward_balance = 0.0
    miner_rewards = {miner_id: 0.0 for miner_id in miner_hashrates.keys()}
    block_owners = []
    block_siblings = []  # Track siblings for each block
    fee = 0

    for block in range(blocks):
        miner = miner_sequence[block]
        block_owners.append(miner)
        
        # Generate siblings for this block and assign miners to them
        num_siblings, sibling_miners = generate_siblings_and_assign_miners()
        block_siblings.append((num_siblings, sibling_miners))

        # Fee amount for current block - check if this block is a spike
        fee += spike_fee if block in spike_positions else base_fee
        matured_block = block - maturity_period

        # Always include fees (regular miners)
        reward_balance += fee
        fee = 0

        # Reward distribution for matured block
        if matured_block >= 0:
            matured_miner = block_owners[matured_block]
            matured_siblings, matured_sibling_miners = block_siblings[matured_block]
            reward = reward_balance * reward_percentage
            reward_balance -= reward
            
            # Current logic: 80% of reward split equally among matured miner and siblings
            total_reward = reward * 0.8  # 80% of total reward
            total_participants = 1 + matured_siblings  # matured miner + siblings
            reward_per_participant = total_reward / total_participants if total_participants > 0 else 0
            
            # Give reward to the matured block miner
            miner_rewards[matured_miner] += reward_per_participant
            
            # Give equal reward to each sibling (if any)
            if matured_siblings > 0:
                # Each sibling has its own miner assigned based on sibling production probabilities
                for sibling_miner in matured_sibling_miners:
                    miner_rewards[sibling_miner] += reward_per_participant

    return miner_rewards

def simulate_new_reward_logic(miner_sequence, miner_hashrates, fixed_percentage):
    """Simulate new reward logic with siblings and fixed percentage"""
    reward_balance = 0.0
    miner_rewards = {miner_id: 0.0 for miner_id in miner_hashrates.keys()}
    block_owners = []
    block_siblings = []  # Track siblings for each block
    fee = 0

    for block in range(blocks):
        miner = miner_sequence[block]
        block_owners.append(miner)
        
        # Generate siblings for this block and assign miners to them
        num_siblings, sibling_miners = generate_siblings_and_assign_miners()
        block_siblings.append((num_siblings, sibling_miners))

        # Fee amount for current block - check if this block is a spike
        fee += spike_fee if block in spike_positions else base_fee
        matured_block = block - maturity_period

        # Always include fees (regular miners)
        reward_balance += fee
        fee = 0

        # Reward distribution for matured block
        if matured_block >= 0:
            matured_miner = block_owners[matured_block]
            matured_siblings, matured_sibling_miners = block_siblings[matured_block]
            reward = reward_balance * reward_percentage
            reward_balance -= reward
            
            # Split reward according to new logic
            total_reward = reward * 0.8  # 80% of total reward
            
            # Fixed percentage goes to the matured block miner
            fixed_reward = total_reward * (fixed_percentage / 100)
            miner_rewards[matured_miner] += fixed_reward
            
            # Remaining reward is split among matured block miner and sibling miners
            remaining_reward = total_reward - fixed_reward
            
            if matured_siblings > 0:
                # If there are siblings, split remaining reward among all participants
                total_participants = 1 + matured_siblings  # matured miner + siblings
                reward_per_participant = remaining_reward / total_participants
                
                # Give remaining reward share to the matured block miner
                miner_rewards[matured_miner] += reward_per_participant
                
                # Give equal reward to each sibling
                for sibling_miner in matured_sibling_miners:
                    miner_rewards[sibling_miner] += reward_per_participant
            else:
                # If no siblings, matured block miner gets 100% of the reward
                miner_rewards[matured_miner] += remaining_reward

    return miner_rewards

def generate_block_sequence_with_sibling_relationship():
    """Generate block sequence where high sibling producers produce fewer main trunk blocks"""
    miner_sequence = []
    
    # Track block production statistics
    miner_main_blocks = {miner_id: 0 for miner_id in range(num_miners)}
    miner_sibling_blocks = {miner_id: 0 for miner_id in range(num_miners)}
    
    for _ in range(blocks):
        # First, determine which miner gets this block based on hashrate
        rand = random.random()
        cumulative = 0
        selected_miner = None
        for miner_id in range(num_miners):
            cumulative += miner_hashrate
            if rand < cumulative:
                selected_miner = miner_id
                break
        
        # Now, based on the miner's sibling production probability,
        # determine if this becomes a main trunk block or a sibling
        sibling_prob = sibling_production_probabilities[selected_miner]
        is_sibling = random.random() < sibling_prob
        
        if is_sibling:
            # This becomes a sibling, so we need to assign a different miner for the main trunk block
            # The main trunk block should be assigned based on adjusted hashrates
            # Miners with high sibling production should have lower probability of getting main trunk blocks
            
            # Track this as a sibling for the selected miner
            miner_sibling_blocks[selected_miner] += 1
            
            # Calculate adjusted hashrates for main trunk blocks
            adjusted_hashrates = {}
            total_adjusted = 0
            for miner_id in range(num_miners):
                # Reduce hashrate based on sibling production probability
                adjusted_hashrate = miner_hashrate * (1 - sibling_production_probabilities[miner_id])
                adjusted_hashrates[miner_id] = adjusted_hashrate
                total_adjusted += adjusted_hashrate
            
            # Normalize to sum to 1.0
            if total_adjusted > 0:
                for miner_id in adjusted_hashrates:
                    adjusted_hashrates[miner_id] /= total_adjusted
                
                # Assign main trunk block based on adjusted hashrates
                rand = random.random()
                cumulative = 0
                for miner_id, adj_hashrate in adjusted_hashrates.items():
                    cumulative += adj_hashrate
                    if rand < cumulative:
                        miner_sequence.append(miner_id)
                        miner_main_blocks[miner_id] += 1
                        break
            else:
                # Fallback: assign randomly
                fallback_miner = random.randint(0, num_miners - 1)
                miner_sequence.append(fallback_miner)
                miner_main_blocks[fallback_miner] += 1
        else:
            # This becomes a main trunk block
            miner_sequence.append(selected_miner)
            miner_main_blocks[selected_miner] += 1
    
    return miner_sequence, miner_main_blocks, miner_sibling_blocks

def run_single_simulation():
    """Run a single simulation comparing both reward logics"""
    # Generate miner sequence with proper sibling relationship
    random.seed(int(time.time() * 1000) % (2**32))
    miner_sequence, miner_main_blocks, miner_sibling_blocks = generate_block_sequence_with_sibling_relationship()
    
    # Setup miners
    miner_hashrates = setup_miners()
    
    # Run current reward logic
    current_rewards = simulate_current_reward_logic(miner_sequence, miner_hashrates)
    
    # Run new reward logic for each fixed percentage
    new_rewards = {}
    for fixed_pct in fixed_percentages:
        new_rewards[fixed_pct] = simulate_new_reward_logic(miner_sequence, miner_hashrates, fixed_pct)
    
    return current_rewards, new_rewards, miner_main_blocks, miner_sibling_blocks

def run_multiple_simulations():
    """Run multiple simulations and collect statistics"""
    print(f"Running {num_runs} simulations with {num_miners} miners...")
    print(f"Testing fixed percentages: {fixed_percentages}%")
    print(f"Sibling production probabilities: {sibling_production_probabilities}")
    print()
    
    # Collect results for each miner
    current_results = {miner_id: [] for miner_id in range(num_miners)}
    new_results = {fixed_pct: {miner_id: [] for miner_id in range(num_miners)} for fixed_pct in fixed_percentages}
    
    # Track block production statistics
    main_trunk_blocks = {miner_id: 0 for miner_id in range(num_miners)}
    sibling_blocks = {miner_id: 0 for miner_id in range(num_miners)}
    
    for run in range(num_runs):
        if run % 10 == 0:
            print(f"Run {run + 1}/{num_runs}...")
        
        current_rewards, new_rewards, miner_main_blocks, miner_sibling_blocks = run_single_simulation()
        
        # Store current reward results
        for miner_id in range(num_miners):
            current_results[miner_id].append(current_rewards[miner_id])
        
        # Store new reward results for each fixed percentage
        for fixed_pct in fixed_percentages:
            for miner_id in range(num_miners):
                new_results[fixed_pct][miner_id].append(new_rewards[fixed_pct][miner_id])
        
        # Accumulate block production statistics
        for miner_id in range(num_miners):
            main_trunk_blocks[miner_id] += miner_main_blocks[miner_id]
            sibling_blocks[miner_id] += miner_sibling_blocks[miner_id]
    
    # Calculate statistics
    current_stats = {}
    for miner_id in range(num_miners):
        rewards = current_results[miner_id]
        current_stats[miner_id] = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': min(rewards),
            'max_reward': max(rewards)
        }
    
    new_stats = {}
    for fixed_pct in fixed_percentages:
        new_stats[fixed_pct] = {}
        for miner_id in range(num_miners):
            rewards = new_results[fixed_pct][miner_id]
            new_stats[fixed_pct][miner_id] = {
                'avg_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'min_reward': min(rewards),
                'max_reward': max(rewards)
            }
    
    # Print block production statistics
    print(f"\n=== BLOCK PRODUCTION STATISTICS ({num_runs} runs) ===")
    print("Miner ID (Siblings%) | Main Trunk Blocks | Sibling Blocks | Total Blocks | Main Trunk % | Sibling %")
    print("-" * 95)
    
    for miner_id in sorted_miner_ids:
        main_count = main_trunk_blocks[miner_id]
        sibling_count = sibling_blocks[miner_id]
        total_count = main_count + sibling_count
        main_percentage = (main_count / total_count * 100) if total_count > 0 else 0
        sibling_percentage = (sibling_count / total_count * 100) if total_count > 0 else 0
        sibling_pct = sibling_production_probabilities[miner_id] * 100
        
        print(f"{miner_id} ({sibling_pct:.0f}%)      | {main_count:16d} | {sibling_count:13d} | {total_count:12d} | {main_percentage:11.1f}% | {sibling_percentage:9.1f}%")
    
    return current_stats, new_stats

# Run simulations
print("=== REWARD DISTRIBUTION COMPARISON SIMULATION ===")
print(f"Number of miners: {num_miners}")
print(f"Equal hashrate per miner: {miner_hashrate:.3f}")
print(f"Fixed percentages to test: {fixed_percentages}%")
print()

current_stats, new_stats = run_multiple_simulations()

# Print results
print(f"\n=== RESULTS ({num_runs} runs) ===")
print("Current Reward Logic:")
print("Miner ID | Average Reward | Std Dev | Min Reward | Max Reward")
print("-" * 65)
for miner_id in range(num_miners):
    stats = current_stats[miner_id]
    print(f"{miner_id:8} | {stats['avg_reward']:13.2f} | {stats['std_reward']:7.2f} | {stats['min_reward']:10.2f} | {stats['max_reward']:10.2f}")

for fixed_pct in fixed_percentages:
    print(f"\nNew Reward Logic (Fixed {fixed_pct}%):")
    print("Miner ID | Average Reward | Std Dev | Min Reward | Max Reward")
    print("-" * 65)
    for miner_id in range(num_miners):
        stats = new_stats[fixed_pct][miner_id]
        print(f"{miner_id:8} | {stats['avg_reward']:13.2f} | {stats['std_reward']:7.2f} | {stats['min_reward']:10.2f} | {stats['max_reward']:10.2f}")

# Print percentage difference from baseline table
print(f"\n=== PERCENTAGE DIFFERENCE FROM BASELINE ({num_runs} runs) ===")
print("Miner ID (Siblings%) | Fixed 50% | Fixed 40% | Fixed 30% | Fixed 20%")
print("-" * 65)
for miner_id in sorted_miner_ids:
    current_avg = current_stats[miner_id]['avg_reward']
    percentage_differences = []
    for fixed_pct in fixed_percentages:
        new_avg = new_stats[fixed_pct][miner_id]['avg_reward']
        if current_avg > 0:
            percentage_diff = (new_avg - current_avg) / current_avg * 100
        else:
            percentage_diff = 0
        percentage_differences.append(f"{percentage_diff:+7.2f}%")
    sibling_pct = sibling_production_probabilities[miner_id] * 100
    print(f"{miner_id} ({sibling_pct:.0f}%)      | {percentage_differences[0]} | {percentage_differences[1]} | {percentage_differences[2]} | {percentage_differences[3]}")

# Print total rewards table
print(f"\n=== TOTAL REWARDS COMPARISON ({num_runs} runs) ===")
print("Miner ID (Siblings%) | Current Logic | Fixed 50% | Fixed 40% | Fixed 30% | Fixed 20%")
print("-" * 100)
for miner_id in sorted_miner_ids:
    current_avg = current_stats[miner_id]['avg_reward']
    total_rewards = [f"{current_avg:13.2f}"]
    for fixed_pct in fixed_percentages:
        new_avg = new_stats[fixed_pct][miner_id]['avg_reward']
        total_rewards.append(f"{new_avg:11.2f}")
    sibling_pct = sibling_production_probabilities[miner_id] * 100
    print(f"{miner_id} ({sibling_pct:.0f}%)      | {total_rewards[0]} | {total_rewards[1]} | {total_rewards[2]} | {total_rewards[3]} | {total_rewards[4]}")

# Create comparison plots
os.makedirs('results', exist_ok=True)

# Plot 1: Average rewards comparison
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(num_miners)
width = 0.15

# Plot current rewards
current_avg_rewards = [current_stats[miner_id]['avg_reward'] for miner_id in sorted_miner_ids]
ax.bar(x - 2*width, current_avg_rewards, width, label='Current Logic', alpha=0.8)

# Plot new rewards for each fixed percentage
colors = ['red', 'blue', 'green', 'orange']
for i, fixed_pct in enumerate(fixed_percentages):
    new_avg_rewards = [new_stats[fixed_pct][miner_id]['avg_reward'] for miner_id in sorted_miner_ids]
    ax.bar(x + i*width, new_avg_rewards, width, label=f'Fixed {fixed_pct}%', alpha=0.8, color=colors[i])

ax.set_xlabel('Miner ID')
ax.set_ylabel('Average Total Reward')
ax.set_title(f'Reward Distribution Comparison ({num_runs} runs)\n{num_miners} miners with equal hashrates')
ax.set_xticks(x)
ax.set_xticklabels([f'Miner {i} ({sibling_production_probabilities[i]*100:.0f}%)' for i in sorted_miner_ids])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/reward_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot saved to: results/reward_distribution_comparison.png")

# Plot 2: Percentage difference from baseline comparison
fig, ax = plt.subplots(figsize=(12, 8))

# Plot standard deviations
current_stds = [current_stats[miner_id]['std_reward'] for miner_id in range(num_miners)]
ax.bar(x - 2*width, current_stds, width, label='Current Logic', alpha=0.8)

for i, fixed_pct in enumerate(fixed_percentages):
    new_stds = [new_stats[fixed_pct][miner_id]['std_reward'] for miner_id in range(num_miners)]
    ax.bar(x + i*width, new_stds, width, label=f'Fixed {fixed_pct}%', alpha=0.8, color=colors[i])

ax.set_xlabel('Miner ID')
ax.set_ylabel('Standard Deviation of Reward')
ax.set_title(f'Reward Variance Comparison ({num_runs} runs)\n{num_miners} miners with equal hashrates')
ax.set_xticks(x)
ax.set_xticklabels([f'Miner {i}' for i in range(num_miners)])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/reward_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot saved to: results/reward_distribution_comparison.png")

# Plot 2: Percentage difference from baseline comparison
fig, ax = plt.subplots(figsize=(12, 8))

# Calculate percentage differences from baseline
current_avg_rewards = [current_stats[miner_id]['avg_reward'] for miner_id in sorted_miner_ids]

# Plot percentage differences for each fixed percentage
colors = ['red', 'blue', 'green', 'orange']
for i, fixed_pct in enumerate(fixed_percentages):
    new_avg_rewards = [new_stats[fixed_pct][miner_id]['avg_reward'] for miner_id in sorted_miner_ids]
    percentage_differences = [(new_avg - current_avg) / current_avg * 100 if current_avg > 0 else 0 
                             for new_avg, current_avg in zip(new_avg_rewards, current_avg_rewards)]
    ax.bar(x + i*width, percentage_differences, width, label=f'Fixed {fixed_pct}%', alpha=0.8, color=colors[i])

# Add horizontal line at zero for reference
ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

ax.set_xlabel('Miner ID')
ax.set_ylabel('Percentage Difference from Current Logic (%)')
ax.set_title(f'Reward Percentage Difference from Baseline ({num_runs} runs)\n{num_miners} miners with equal hashrates')
ax.set_xticks(x)
ax.set_xticklabels([f'Miner {i} ({sibling_production_probabilities[i]*100:.0f}%)' for i in sorted_miner_ids])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/reward_percentage_difference_from_baseline.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot saved to: results/reward_percentage_difference_from_baseline.png")

# Print summary statistics
print(f"\n=== SUMMARY STATISTICS ===")
print(f"Total average reward per miner (Current Logic): {np.mean([current_stats[i]['avg_reward'] for i in range(num_miners)]):.2f}")
for fixed_pct in fixed_percentages:
    total_avg = np.mean([new_stats[fixed_pct][i]['avg_reward'] for i in range(num_miners)])
    print(f"Total average reward per miner (Fixed {fixed_pct}%): {total_avg:.2f}")

print(f"\nTested configurations:")
print(f"- {num_miners} miners with equal hashrates ({miner_hashrate:.3f} each)")
print(f"- Current logic: 80% of reward goes to matured block miner")
print(f"- New logic: Fixed percentage to matured block miner + remaining split among all participants")
print(f"- Fixed percentages tested: {fixed_percentages}%")
print(f"- Siblings: exponential distribution with mean={sibling_mean}, min={sibling_min}, max={sibling_max}")
print(f"- Sibling production probabilities: {sibling_production_probabilities}")
print(f"\nKey insight: Hashrate determines total blocks (main trunk + siblings), while sibling production")
print(f"probabilities determine how much of each miner's blocks become siblings vs main trunk blocks.") 