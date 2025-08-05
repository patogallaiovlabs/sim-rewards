import random
import matplotlib.pyplot as plt
import numpy as np
import time

# Parameters
blocks = 50000
maturity_period = 4000
reward_percentage = 0.10
base_fee = 0.001
spike_fee = 1.0
num_spikes = 200  # Number of random spikes to generate

# Pool hashrates (must sum to 1.0)
pool_hashrates = {
    0: 0.1,    # Pool 0: 10%
    1: 0.9,   # Pool 1: 90%
}

# Configure which pools are strategic (set to empty set for all regular)
strategic_pools = {0}  # Pool 0 will be strategic

# Use time-based seed for different results each run
random.seed(int(time.time()))

# Fixed miner sequence for deterministic simulation
fixed_miner_sequence = []
for _ in range(blocks):
    rand = random.random()
    cumulative = 0
    for pool_id, hashrate in pool_hashrates.items():
        cumulative += hashrate
        if rand < cumulative:
            fixed_miner_sequence.append(pool_id)
            break

# Generate random spike positions (deterministic)
spike_positions = set(random.sample(range(blocks), num_spikes))

def deterministic_simulation(strategy: str, miner_sequence):
    reward_balance = 0.0
    miner_rewards = {pool_id: 0.0 for pool_id in pool_hashrates.keys()}
    block_owners = []
    block_fees = []
    fee = 0
    
    # Metrics tracking
    total_blocks = 0
    fees_not_included = 0
    pool_blocks = {pool_id: 0 for pool_id in pool_hashrates.keys()}
    pool_fees_not_included = {pool_id: 0 for pool_id in pool_hashrates.keys()}
    
    # Accumulation tracking
    accumulated_fees = 0
    blocks_with_accumulated_fees = 0
    strategic_wins_with_accumulated_fees = 0
    total_accumulated_fees_won = 0
    
    # Consecutive withholding tracking
    current_streak = 0
    withholding_streaks = []
    max_streak = 0
    min_streak = float('inf')

    for block in range(blocks):
        miner = miner_sequence[block]
        block_owners.append(miner)
        total_blocks += 1
        
        # Track pool blocks
        pool_blocks[miner] += 1

        # Fee amount for current block - check if this block is a spike
        fee += spike_fee if block in spike_positions else base_fee
        matured_block = block - maturity_period

        # Fee inclusion decision
        include_fee = False
        if strategy == "regular":
            include_fee = True
        elif strategy == "strategic":
            if miner in strategic_pools:
                # Strategic pool only includes fees if matured block was also strategic
                # If no matured block yet (matured_block < 0), include fees to avoid infinite withholding
                if matured_block < 0 or (matured_block >= 0 and block_owners[matured_block] in strategic_pools):
                    include_fee = True
            else:
                # regular pools always include fees
                include_fee = True

        if include_fee:
            reward_balance += fee
            block_fees.append(fee)
            fee = 0
            # End current streak if there was one
            if current_streak > 0:
                withholding_streaks.append(current_streak)
                max_streak = max(max_streak, current_streak)
                min_streak = min(min_streak, current_streak)
                current_streak = 0
        else:
            block_fees.append(0)
            fees_not_included += 1
            pool_fees_not_included[miner] += 1
            # Track accumulation
            accumulated_fees += fee
            blocks_with_accumulated_fees += 1
            # Track consecutive withholding
            current_streak += 1

        # Reward distribution for matured block
        if matured_block >= 0:
            matured_miner = block_owners[matured_block]
            reward = reward_balance * reward_percentage
            reward_balance -= reward
            miner_rewards[matured_miner] += reward * 0.8
            # Track if strategic miner won accumulated fees
            if accumulated_fees > 0 and matured_miner in strategic_pools:
                strategic_wins_with_accumulated_fees += 1
                total_accumulated_fees_won += accumulated_fees
                accumulated_fees = 0  # Reset after winning

    # Handle final streak if simulation ends with withholding
    if current_streak > 0:
        withholding_streaks.append(current_streak)
        max_streak = max(max_streak, current_streak)
        min_streak = min(min_streak, current_streak)

    # Calculate metrics
    avg_streak = sum(withholding_streaks) / len(withholding_streaks) if withholding_streaks else 0
    min_streak = min_streak if min_streak != float('inf') else 0
    
    metrics = {
        'total_blocks': total_blocks,
        'fees_not_included': fees_not_included,
        'fees_not_included_percentage': (fees_not_included / total_blocks) * 100,
        'pool_blocks': pool_blocks,
        'pool_fees_not_included': pool_fees_not_included,
        'pool_fees_not_included_percentage': {pool_id: (pool_fees_not_included[pool_id] / pool_blocks[pool_id] * 100) if pool_blocks[pool_id] > 0 else 0 for pool_id in pool_hashrates.keys()},
        'blocks_with_accumulated_fees': blocks_with_accumulated_fees,
        'strategic_wins_with_accumulated_fees': strategic_wins_with_accumulated_fees,
        'total_accumulated_fees_won': total_accumulated_fees_won,
        'accumulation_success_rate': (strategic_wins_with_accumulated_fees / blocks_with_accumulated_fees * 100) if blocks_with_accumulated_fees > 0 else 0,
        'withholding_streaks': withholding_streaks,
        'avg_withholding_streak': avg_streak,
        'min_withholding_streak': min_streak,
        'max_withholding_streak': max_streak,
        'total_streaks': len(withholding_streaks)
    }
    
    return miner_rewards, metrics

def run_hashrate_sweep_single_run(strategic_hashrate):
    """Run a single simulation for a given hashrate"""
    # Update pool hashrates
    pool_hashrates = {
        0: strategic_hashrate,    # Strategic pool
        1: 1.0 - strategic_hashrate,   # regular pool
    }
    
    # Generate miner sequence for this hashrate
    random.seed(int(time.time() * 1000) % (2**32))  # Use time-based seed
    fixed_miner_sequence = []
    for _ in range(blocks):
        rand = random.random()
        cumulative = 0
        for pool_id, hashrate in pool_hashrates.items():
            cumulative += hashrate
            if rand < cumulative:
                fixed_miner_sequence.append(pool_id)
                break
    
    # Run simulations
    regular_rewards, regular_metrics = deterministic_simulation("regular", fixed_miner_sequence)
    strategic_rewards, strategic_metrics = deterministic_simulation("strategic", fixed_miner_sequence)
    
    # Calculate reward improvement for strategic pool
    regular_reward_strategic = regular_rewards[0]  # Pool 0 is strategic
    strategic_reward = strategic_rewards[0]
    strategic_improvement = (strategic_reward - regular_reward_strategic) / regular_reward_strategic * 100 if regular_reward_strategic > 0 else 0
    
    # Calculate reward change for regular pool
    regular_reward_regular = regular_rewards[1]  # Pool 1 is regular
    strategic_reward_regular = strategic_rewards[1]
    regular_improvement = (strategic_reward_regular - regular_reward_regular) / regular_reward_regular * 100 if regular_reward_regular > 0 else 0
    
    # Calculate total rewards and percentages
    total_regular_reward = sum(regular_rewards.values())
    total_strategic_reward = sum(strategic_rewards.values())
    
    regular_percentages = {pool_id: (reward / total_regular_reward * 100) if total_regular_reward > 0 else 0 
                         for pool_id, reward in regular_rewards.items()}
    strategic_percentages = {pool_id: (reward / total_strategic_reward * 100) if total_strategic_reward > 0 else 0 
                           for pool_id, reward in strategic_rewards.items()}
    
    return strategic_hashrate, strategic_improvement, regular_improvement, regular_percentages, strategic_percentages

def run_multiple_simulations(num_runs=100):
    """Run multiple simulations and calculate averages"""
    hashrates = np.arange(0.0, 1.1, 0.1)  # 0.0 to 1.0
    all_strategic_results = {h: [] for h in hashrates}
    all_regular_results = {h: [] for h in hashrates}
    all_regular_percentages = {h: {0: [], 1: []} for h in hashrates}
    all_strategic_percentages = {h: {0: [], 1: []} for h in hashrates}
    
    print(f"Running {num_runs} simulations for each hashrate...")
    
    for run in range(num_runs):
        if run % 10 == 0:
            print(f"Run {run + 1}/{num_runs}...")
        
        for strategic_hashrate in hashrates:
            _, strategic_improvement, regular_improvement, regular_percentages, strategic_percentages = run_hashrate_sweep_single_run(strategic_hashrate)
            all_strategic_results[strategic_hashrate].append(strategic_improvement)
            all_regular_results[strategic_hashrate].append(regular_improvement)
            
            # Store percentages for each pool
            for pool_id in [0, 1]:
                all_regular_percentages[strategic_hashrate][pool_id].append(regular_percentages[pool_id])
                all_strategic_percentages[strategic_hashrate][pool_id].append(strategic_percentages[pool_id])
    
    # Calculate averages and statistics
    avg_results = []
    for hashrate in hashrates:
        strategic_improvements = all_strategic_results[hashrate]
        regular_improvements = all_regular_results[hashrate]
        
        avg_strategic_improvement = np.mean(strategic_improvements)
        std_strategic_improvement = np.std(strategic_improvements)
        avg_regular_improvement = np.mean(regular_improvements)
        std_regular_improvement = np.std(regular_improvements)
        
        # Calculate average percentages
        avg_regular_pct = {
            pool_id: np.mean(all_regular_percentages[hashrate][pool_id]) 
            for pool_id in [0, 1]
        }
        avg_strategic_pct = {
            pool_id: np.mean(all_strategic_percentages[hashrate][pool_id]) 
            for pool_id in [0, 1]
        }
        
        avg_results.append({
            'hashrate': hashrate,
            'avg_strategic_improvement': avg_strategic_improvement,
            'std_strategic_improvement': std_strategic_improvement,
            'min_strategic_improvement': min(strategic_improvements),
            'max_strategic_improvement': max(strategic_improvements),
            'avg_regular_improvement': avg_regular_improvement,
            'std_regular_improvement': std_regular_improvement,
            'min_regular_improvement': min(regular_improvements),
            'max_regular_improvement': max(regular_improvements),
            'avg_regular_percentages': avg_regular_pct,
            'avg_strategic_percentages': avg_strategic_pct
        })
        
        print(f"Hashrate {hashrate:.1f}: Strategic avg {avg_strategic_improvement:+.2f}% ± {std_strategic_improvement:.2f}%, regular avg {avg_regular_improvement:+.2f}% ± {std_regular_improvement:.2f}%")
    
    return avg_results

# Run simulations
regular_rewards, regular_metrics = deterministic_simulation("regular", fixed_miner_sequence)
strategic_rewards, strategic_metrics = deterministic_simulation("strategic", fixed_miner_sequence)

# Print results
print("=== SCENARIO COMPARISON ===")
print(f"Scenario 1: All pools regular")
print(f"Scenario 2: Pool {strategic_pools} strategic, others regular")
print()

print("=== REWARD COMPARISON ===")
for pool_id in pool_hashrates.keys():
    regular_reward = regular_rewards[pool_id]
    strategic_reward = strategic_rewards[pool_id]
    improvement = (strategic_reward - regular_reward) / regular_reward * 100 if regular_reward > 0 else 0
    pool_type = "STRATEGIC" if pool_id in strategic_pools else "regular"
    print(f"Pool {pool_id} ({pool_hashrates[pool_id]*100:.0f}% hashrate, {pool_type}):")
    print(f"  All-regular reward: {regular_reward:.2f}")
    print(f"  Strategic scenario reward: {strategic_reward:.2f}")
    print(f"  Change: {improvement:+.2f}%")
    print()

print(f"Generated {num_spikes} random spikes across {blocks} blocks")
print()

# Print fee inclusion metrics
print("=== FEE INCLUSION METRICS ===")
print(f"All-regular Scenario:")
print(f"  Total blocks: {regular_metrics['total_blocks']}")
print(f"  Fees not included: {regular_metrics['fees_not_included']} ({regular_metrics['fees_not_included_percentage']:.2f}%)")
print(f"  Pool blocks: {regular_metrics['pool_blocks']}")
print()

print(f"Strategic Scenario (Pool {strategic_pools} strategic):")
print(f"  Total blocks: {strategic_metrics['total_blocks']}")
print(f"  Fees not included: {strategic_metrics['fees_not_included']} ({strategic_metrics['fees_not_included_percentage']:.2f}%)")
print(f"  Pool blocks: {strategic_metrics['pool_blocks']}")
print(f"  Pool fees not included: {strategic_metrics['pool_fees_not_included']}")
print()

# Print accumulation metrics
print("=== ACCUMULATION STRATEGY METRICS ===")
print(f"All-regular Scenario:")
print(f"  Blocks with accumulated fees: {regular_metrics['blocks_with_accumulated_fees']}")
print(f"  Strategic wins with accumulated fees: {regular_metrics['strategic_wins_with_accumulated_fees']}")
print(f"  Total accumulated fees won: {regular_metrics['total_accumulated_fees_won']:.2f}")
print()

print(f"Strategic Scenario (Pool {strategic_pools} strategic):")
print(f"  Blocks with accumulated fees: {strategic_metrics['blocks_with_accumulated_fees']}")
print(f"  Strategic wins with accumulated fees: {strategic_metrics['strategic_wins_with_accumulated_fees']}")
print(f"  Total accumulated fees won: {strategic_metrics['total_accumulated_fees_won']:.2f}")
print(f"  Accumulation success rate: {strategic_metrics['accumulation_success_rate']:.2f}%")
print()

# Print withholding streak metrics
print("=== WITHHOLDING STREAK METRICS ===")
print(f"All-regular Scenario:")
print(f"  Total withholding streaks: {regular_metrics['total_streaks']}")
print(f"  Average streak length: {regular_metrics['avg_withholding_streak']:.2f} blocks")
print(f"  Min streak length: {regular_metrics['min_withholding_streak']} blocks")
print(f"  Max streak length: {regular_metrics['max_withholding_streak']} blocks")
print()

print(f"Strategic Scenario (Pool {strategic_pools} strategic):")
print(f"  Total withholding streaks: {strategic_metrics['total_streaks']}")
print(f"  Average streak length: {strategic_metrics['avg_withholding_streak']:.2f} blocks")
print(f"  Min streak length: {strategic_metrics['min_withholding_streak']} blocks")
print(f"  Max streak length: {strategic_metrics['max_withholding_streak']} blocks")

# Run multiple simulations
print("Running multiple simulations to get average results...")
avg_results = run_multiple_simulations(100)

# Create the plot
hashrates = [r['hashrate'] for r in avg_results]
avg_strategic_improvements = [r['avg_strategic_improvement'] for r in avg_results]
std_strategic_improvements = [r['std_strategic_improvement'] for r in avg_results]
avg_regular_improvements = [r['avg_regular_improvement'] for r in avg_results]
std_regular_improvements = [r['std_regular_improvement'] for r in avg_results]

plt.figure(figsize=(12, 8))

# Plot strategic pool line
plt.plot(hashrates, avg_strategic_improvements, 'bo-', linewidth=2, markersize=8, label='Strategic Pool Improvement')

# Plot regular pool line
plt.plot(hashrates, avg_regular_improvements, 'ro-', linewidth=2, markersize=8, label='regular Pool Impact')

# Plot error bars (standard deviation)
plt.errorbar(hashrates, avg_strategic_improvements, yerr=std_strategic_improvements, fmt='none', capsize=5, alpha=0.7, color='blue')
plt.errorbar(hashrates, avg_regular_improvements, yerr=std_regular_improvements, fmt='none', capsize=5, alpha=0.7, color='red')

plt.xlabel('Strategic Pool Hashrate')
plt.ylabel('Average Reward Change (%)')
plt.title('Strategic vs regular Pool Reward Impact (100 runs)')
plt.grid(True, alpha=0.3)
plt.xticks(hashrates)

# Add zero line for reference
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='No Change')

plt.legend()
plt.tight_layout()
plt.show()

# Create a comparison plot showing the difference
plt.figure(figsize=(12, 8))

# Calculate the difference (strategic gain - regular loss)
differences = [s - h for s, h in zip(avg_strategic_improvements, avg_regular_improvements)]
std_differences = [np.sqrt(s**2 + h**2) for s, h in zip(std_strategic_improvements, std_regular_improvements)]

plt.plot(hashrates, differences, 'go-', linewidth=2, markersize=8, label='Net Impact (Strategic - regular)')
plt.errorbar(hashrates, differences, yerr=std_differences, fmt='none', capsize=5, alpha=0.7, color='green')

plt.xlabel('Strategic Pool Hashrate')
plt.ylabel('Net Reward Impact (%)')
plt.title('Net Impact of Strategic Behavior (100 runs)')
plt.grid(True, alpha=0.3)
plt.xticks(hashrates)

# Add zero line for reference
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='No Net Impact')

plt.legend()
plt.tight_layout()
plt.show()

# Create streak distribution plot
plt.figure(figsize=(12, 8))

# Get streak data from the strategic scenario
strategic_streaks = strategic_metrics['withholding_streaks']
regular_streaks = regular_metrics['withholding_streaks']

# Create histogram bins
max_streak = max(max(strategic_streaks) if strategic_streaks else 0, 
                 max(regular_streaks) if regular_streaks else 0)
bins = range(1, max_streak + 2)

# Plot histograms
plt.hist(strategic_streaks, bins=bins, alpha=0.7, label=f'Strategic Scenario (Pool {strategic_pools})', 
         color='red', edgecolor='black', density=True)
plt.hist(regular_streaks, bins=bins, alpha=0.7, label='All-regular Scenario', 
         color='blue', edgecolor='black', density=True)

plt.xlabel('Withholding Streak Length (blocks)')
plt.ylabel('Density')
plt.title('Distribution of Withholding Streaks')
plt.legend()
plt.grid(True, alpha=0.3)

# Add statistics as text
strategic_stats = f"Strategic: {len(strategic_streaks)} streaks, avg={strategic_metrics['avg_withholding_streak']:.1f}"
regular_stats = f"regular: {len(regular_streaks)} streaks, avg={regular_metrics['avg_withholding_streak']:.1f}"
plt.text(0.02, 0.98, f"{strategic_stats}\n{regular_stats}", 
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

# Print summary
print("\n=== SUMMARY (100 runs) ===")
print("Hashrate | Strategic Impact | regular Impact | Net Impact")
print("-" * 70)
for r in avg_results:
    net_impact = r['avg_strategic_improvement'] - r['avg_regular_improvement']
    print(f"{r['hashrate']:.1f}     | {r['avg_strategic_improvement']:+.2f}% ± {r['std_strategic_improvement']:.2f}% | {r['avg_regular_improvement']:+.2f}% ± {r['std_regular_improvement']:.2f}% | {net_impact:+.2f}%")

print("\n=== REWARD DISTRIBUTION SUMMARY (100 runs) ===")
print("Hashrate | All regular: Pool0 | Pool1 | Strategic: Pool0 | Pool1")
print("-" * 80)
for r in avg_results:
    regular_p0 = r['avg_regular_percentages'][0]
    regular_p1 = r['avg_regular_percentages'][1]
    strategic_p0 = r['avg_strategic_percentages'][0]
    strategic_p1 = r['avg_strategic_percentages'][1]
    print(f"{r['hashrate']:.1f}     | {regular_p0:.1f}%        | {regular_p1:.1f}% | {strategic_p0:.1f}%        | {strategic_p1:.1f}%")

# Create reward distribution comparison plots
plt.figure(figsize=(15, 10))

# Subplot 1: regular scenario percentages
plt.subplot(2, 2, 1)
regular_pct_pool0 = [r['avg_regular_percentages'][0] for r in avg_results]
regular_pct_pool1 = [r['avg_regular_percentages'][1] for r in avg_results]

plt.plot(hashrates, regular_pct_pool0, 'bo-', linewidth=2, markersize=6, label='Pool 0 (Strategic)')
plt.plot(hashrates, regular_pct_pool1, 'ro-', linewidth=2, markersize=6, label='Pool 1 (regular)')

plt.xlabel('Strategic Pool Hashrate')
plt.ylabel('Reward Percentage (%)')
plt.title('Reward Distribution - All regular Scenario')
plt.grid(True, alpha=0.3)
plt.xticks(hashrates)
plt.legend()
plt.ylim(0, 100)

# Subplot 2: Strategic scenario percentages
plt.subplot(2, 2, 2)
strategic_pct_pool0 = [r['avg_strategic_percentages'][0] for r in avg_results]
strategic_pct_pool1 = [r['avg_strategic_percentages'][1] for r in avg_results]

plt.plot(hashrates, strategic_pct_pool0, 'bo-', linewidth=2, markersize=6, label='Pool 0 (Strategic)')
plt.plot(hashrates, strategic_pct_pool1, 'ro-', linewidth=2, markersize=6, label='Pool 1 (regular)')

plt.xlabel('Strategic Pool Hashrate')
plt.ylabel('Reward Percentage (%)')
plt.title('Reward Distribution - Strategic Scenario')
plt.grid(True, alpha=0.3)
plt.xticks(hashrates)
plt.legend()
plt.ylim(0, 100)

# Subplot 3: Pool 0 comparison (Strategic vs regular)
plt.subplot(2, 2, 3)
plt.plot(hashrates, regular_pct_pool0, 'b--', linewidth=2, markersize=6, label='All regular Scenario')
plt.plot(hashrates, strategic_pct_pool0, 'b-', linewidth=2, markersize=6, label='Strategic Scenario')

plt.xlabel('Strategic Pool Hashrate')
plt.ylabel('Pool 0 Reward Percentage (%)')
plt.title('Pool 0 (Strategic) - Scenario Comparison')
plt.grid(True, alpha=0.3)
plt.xticks(hashrates)
plt.legend()
plt.ylim(0, 100)

# Subplot 4: Pool 1 comparison (Strategic vs regular)
plt.subplot(2, 2, 4)
plt.plot(hashrates, regular_pct_pool1, 'r--', linewidth=2, markersize=6, label='All regular Scenario')
plt.plot(hashrates, strategic_pct_pool1, 'r-', linewidth=2, markersize=6, label='Strategic Scenario')

plt.xlabel('Strategic Pool Hashrate')
plt.ylabel('Pool 1 Reward Percentage (%)')
plt.title('Pool 1 (regular) - Scenario Comparison')
plt.grid(True, alpha=0.3)
plt.xticks(hashrates)
plt.legend()
plt.ylim(0, 100)

plt.tight_layout()
plt.show()

# Create a stacked bar chart showing the reward distribution
plt.figure(figsize=(12, 8))

# Prepare data for stacked bars
x_pos = np.arange(len(hashrates))
width = 0.35

# Create stacked bars for regular scenario
plt.subplot(1, 2, 1)
plt.bar(x_pos, regular_pct_pool0, width, label='Pool 0 (Strategic)', color='blue', alpha=0.7)
plt.bar(x_pos, regular_pct_pool1, width, bottom=regular_pct_pool0, label='Pool 1 (regular)', color='red', alpha=0.7)

plt.xlabel('Strategic Pool Hashrate')
plt.ylabel('Reward Percentage (%)')
plt.title('Reward Distribution - All regular Scenario')
plt.xticks(x_pos, [f'{h:.1f}' for h in hashrates])
plt.legend()
plt.ylim(0, 100)

# Create stacked bars for strategic scenario
plt.subplot(1, 2, 2)
plt.bar(x_pos, strategic_pct_pool0, width, label='Pool 0 (Strategic)', color='blue', alpha=0.7)
plt.bar(x_pos, strategic_pct_pool1, width, bottom=strategic_pct_pool0, label='Pool 1 (regular)', color='red', alpha=0.7)

plt.xlabel('Strategic Pool Hashrate')
plt.ylabel('Reward Percentage (%)')
plt.title('Reward Distribution - Strategic Scenario')
plt.xticks(x_pos, [f'{h:.1f}' for h in hashrates])
plt.legend()
plt.ylim(0, 100)

plt.tight_layout()
plt.show()
