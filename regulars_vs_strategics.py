import random
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# Parameters
blocks = 50000
maturity_period = 4000
reward_percentage = 0.10
base_fee = 0.001
spike_fee = 1.0
num_spikes = 200  # Number of random spikes to generate
num_runs = 50  # Number of simulation runs for statistical analysis

# Pool configuration - multiple regular vs strategic pools
num_regular_pools = 5      # Number of regular pools
num_strategic_pools = 1   # Number of strategic pools
total_pools = num_regular_pools + num_strategic_pools

# Pool hashrates (must sum to 1.0)
# Regular pools get equal hashrate, strategic pools get equal hashrate
regular_hashrate_per_pool = 0.15   # Each regular pool gets 15%
strategic_hashrate_per_pool = 0.25  # Each strategic pool gets 25%

# Verify hashrates sum to 1.0
total_regular = regular_hashrate_per_pool * num_regular_pools
total_strategic = strategic_hashrate_per_pool * num_strategic_pools
if abs(total_regular + total_strategic - 1.0) > 0.001:
    print(f"Warning: Hashrates don't sum to 1.0: {total_regular + total_strategic:.3f}")

pool_hashrates = {}
# Regular pools (0, 1, 2, 3, 4)
for i in range(num_regular_pools):
    pool_hashrates[i] = regular_hashrate_per_pool
# Strategic pools (5, 6, 7)
for i in range(num_strategic_pools):
    pool_hashrates[num_regular_pools + i] = strategic_hashrate_per_pool

# Configure which pools are regular vs strategic
regular_pools = set(range(num_regular_pools))  # Pools 0, 1, 2, 3, 4 are regular
strategic_pools = set(range(num_regular_pools, total_pools))  # Pools 5, 6, 7 are strategic

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
    regular_wins_with_accumulated_fees = 0
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
            if miner in regular_pools:
                # Regular pool only includes fees if matured block was also regular
                # If no matured block yet (matured_block < 0), include fees to avoid infinite withholding
                if matured_block < 0 or (matured_block >= 0 and block_owners[matured_block] in regular_pools):
                    include_fee = True
            elif miner in strategic_pools:
                # Strategic pool only includes fees if THEY THEMSELVES mined the matured block
                # If no matured block yet (matured_block < 0), include fees to avoid infinite withholding
                if matured_block < 0 or (matured_block >= 0 and block_owners[matured_block] == miner):
                    include_fee = True
            else:
                # Default behavior (shouldn't happen)
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
                                # Track if regular/strategic miner won accumulated fees
            if accumulated_fees > 0:
                if matured_miner in regular_pools:
                    regular_wins_with_accumulated_fees += 1
                elif matured_miner in strategic_pools:
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
        'honest_wins_with_accumulated_fees': honest_wins_with_accumulated_fees,
        'dishonest_wins_with_accumulated_fees': dishonest_wins_with_accumulated_fees,
        'total_accumulated_fees_won': total_accumulated_fees_won,
        'honest_accumulation_success_rate': (honest_wins_with_accumulated_fees / blocks_with_accumulated_fees * 100) if blocks_with_accumulated_fees > 0 else 0,
        'dishonest_accumulation_success_rate': (dishonest_wins_with_accumulated_fees / blocks_with_accumulated_fees * 100) if blocks_with_accumulated_fees > 0 else 0,
        'withholding_streaks': withholding_streaks,
        'avg_withholding_streak': avg_streak,
        'min_withholding_streak': min_streak,
        'max_withholding_streak': max_streak,
        'total_streaks': len(withholding_streaks)
    }
    
    return miner_rewards, metrics

def run_hashrate_sweep_single_run(honest_total_hashrate):
    """Run a single simulation for a given total honest hashrate"""
    # Calculate hashrate per pool
    honest_hashrate_per_pool = honest_total_hashrate / num_honest_pools
    dishonest_total_hashrate = 1.0 - honest_total_hashrate
    dishonest_hashrate_per_pool = dishonest_total_hashrate / num_dishonest_pools
    
    # Update pool hashrates
    pool_hashrates_temp = {}
    for i in range(num_honest_pools):
        pool_hashrates_temp[i] = honest_hashrate_per_pool
    for i in range(num_dishonest_pools):
        pool_hashrates_temp[num_honest_pools + i] = dishonest_hashrate_per_pool
    

    
    # Generate miner sequence for this hashrate
    random.seed(int(time.time() * 1000) % (2**32))  # Use time-based seed
    fixed_miner_sequence = []
    for _ in range(blocks):
        rand = random.random()
        cumulative = 0
        for pool_id, hashrate in pool_hashrates_temp.items():
            cumulative += hashrate
            if rand < cumulative:
                fixed_miner_sequence.append(pool_id)
                break
    
    # Run simulations
    honest_rewards, honest_metrics = deterministic_simulation("honest", fixed_miner_sequence)
    strategic_rewards, strategic_metrics = deterministic_simulation("strategic", fixed_miner_sequence)
    
    # Calculate improvement for each pool
    pool_improvements = {}
    for pool_id in pool_hashrates_temp.keys():
        honest_reward = honest_rewards[pool_id]
        strategic_reward = strategic_rewards[pool_id]
        improvement = (strategic_reward - honest_reward) / honest_reward * 100 if honest_reward > 0 else 0
        pool_improvements[pool_id] = improvement
    
    return honest_total_hashrate, pool_improvements

def run_multiple_simulations(num_runs=num_runs):
    """Run multiple simulations and calculate averages for each pool"""
    hashrates = np.arange(0.0, 1.1, 0.1)  # 0.0 to 1.0 (total honest hashrate)
    all_results = {h: {pool_id: [] for pool_id in range(total_pools)} for h in hashrates}
    
    print(f"Running {num_runs} simulations for each hashrate...")
    print(f"Configuration: {num_honest_pools} honest pools, {num_dishonest_pools} dishonest pools")
    print(f"Honest pools: {list(honest_pools)}")
    print(f"Dishonest pools: {list(dishonest_pools)}")
    
    for run in range(num_runs):
        if run % 10 == 0:
            print(f"Run {run + 1}/{num_runs}...")
        
        for honest_total_hashrate in hashrates:
            _, pool_improvements = run_hashrate_sweep_single_run(honest_total_hashrate)
            for pool_id in range(total_pools):
                all_results[honest_total_hashrate][pool_id].append(pool_improvements[pool_id])
    
    # Calculate averages and statistics for each pool
    avg_results = []
    for hashrate in hashrates:
        pool_results = {}
        for pool_id in range(total_pools):
            improvements = all_results[hashrate][pool_id]
            avg_improvement = np.mean(improvements)
            std_improvement = np.std(improvements)
            
            pool_results[pool_id] = {
                'avg_improvement': avg_improvement,
                'std_improvement': std_improvement,
                'min_improvement': min(improvements),
                'max_improvement': max(improvements)
            }
        
        avg_results.append({
            'hashrate': hashrate,
            'pool_results': pool_results
        })
        
        # Calculate hashrates for this configuration
        honest_hashrate_per_pool = hashrate / num_honest_pools
        dishonest_total_hashrate = 1.0 - hashrate
        dishonest_hashrate_per_pool = dishonest_total_hashrate / num_dishonest_pools
        
        # Print results for this hashrate
        print(f"\nTotal Honest Hashrate {hashrate:.1f}:")
        print("  Honest pools:")
        for pool_id in honest_pools:
            result = pool_results[pool_id]
            print(f"    Pool {pool_id} ({honest_hashrate_per_pool*100:.1f}%): {result['avg_improvement']:+.2f}% ± {result['std_improvement']:.2f}%")
        print("  Dishonest pools:")
        for pool_id in dishonest_pools:
            result = pool_results[pool_id]
            print(f"    Pool {pool_id} ({dishonest_hashrate_per_pool*100:.1f}%): {result['avg_improvement']:+.2f}% ± {result['std_improvement']:.2f}%")
    
    return avg_results

# Run the hashrate sweep simulation
print("=== HONEST VS DISHONEST MULTI-POOL STRATEGIC SIMULATION ===")
print(f"Configuration: {num_honest_pools} honest pools, {num_dishonest_pools} dishonest pools")
print(f"Honest pools: {list(honest_pools)} (each with {honest_hashrate_per_pool*100:.1f}% hashrate)")
print(f"Dishonest pools: {list(dishonest_pools)} (each with {dishonest_hashrate_per_pool*100:.1f}% hashrate)")
print()

# Run multiple simulations
avg_results = run_multiple_simulations(num_runs)

print(f"Generated {num_spikes} random spikes across {blocks} blocks")
print()

# Create the plot
hashrates = [r['hashrate'] for r in avg_results]
hashrates = [r['hashrate'] for r in avg_results]

plt.figure(figsize=(16, 12))

# Plot each pool separately
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'H', '8']

# Plot honest pools
for i, pool_id in enumerate(honest_pools):
    avg_improvements = [r['pool_results'][pool_id]['avg_improvement'] for r in avg_results]
    std_improvements = [r['pool_results'][pool_id]['std_improvement'] for r in avg_results]
    
    plt.plot(hashrates, avg_improvements, 
             color=colors[i], marker=markers[i], 
             linewidth=2, markersize=8, 
             label=f'Honest Pool {pool_id}')
    
    # Plot error bars (standard deviation)
    plt.errorbar(hashrates, avg_improvements, yerr=std_improvements, 
                 fmt='none', capsize=3, alpha=0.5, color=colors[i])

# Plot dishonest pools
for i, pool_id in enumerate(dishonest_pools):
    avg_improvements = [r['pool_results'][pool_id]['avg_improvement'] for r in avg_results]
    std_improvements = [r['pool_results'][pool_id]['std_improvement'] for r in avg_results]
    
    plt.plot(hashrates, avg_improvements, 
             color=colors[i + len(honest_pools)], marker=markers[i + len(honest_pools)], 
             linewidth=2, markersize=8, linestyle='--',
             label=f'Dishonest Pool {pool_id}')
    
    # Plot error bars (standard deviation)
    plt.errorbar(hashrates, avg_improvements, yerr=std_improvements, 
                 fmt='none', capsize=3, alpha=0.5, color=colors[i + len(honest_pools)])

plt.xlabel('Total Honest Pool Hashrate')
plt.ylabel('Individual Pool Improvement (%)')
plt.title(f'Honest vs Dishonest Pool Improvements vs Total Honest Hashrate\n({num_honest_pools} Honest Pools vs {num_dishonest_pools} Dishonest Pools, {num_runs} runs)')
plt.grid(True, alpha=0.3)
plt.xticks(hashrates)

# Add zero line for reference
plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='No Improvement')

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('results/honest_vs_dishonest_multi_pool_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved to: results/honest_vs_dishonest_multi_pool_results.png")

# Print summary
print(f"\n=== SUMMARY ({num_runs} runs) ===")
print("Total Honest | Pool Type | Pool ID | Hashrate | Avg Improvement | Std Dev | Min | Max")
print("-" * 100)
for r in avg_results:
    hashrate = r['hashrate']
    # Calculate hashrates for this configuration
    honest_hashrate_per_pool = hashrate / num_honest_pools
    dishonest_total_hashrate = 1.0 - hashrate
    dishonest_hashrate_per_pool = dishonest_total_hashrate / num_dishonest_pools
    
    for pool_id in range(total_pools):
        result = r['pool_results'][pool_id]
        pool_type = "HONEST" if pool_id in honest_pools else "DISHONEST"
        # Determine individual pool hashrate
        if pool_id in honest_pools:
            individual_hashrate = honest_hashrate_per_pool
        else:
            individual_hashrate = dishonest_hashrate_per_pool
        
        print(f"{hashrate:.1f}           | {pool_type:10} | {pool_id:7} | {individual_hashrate*100:6.1f}% | {result['avg_improvement']:+.2f}%        | {result['std_improvement']:.2f}%   | {result['min_improvement']:+.2f}% | {result['max_improvement']:+.2f}%")

print(f"\nConfiguration: {num_honest_pools} honest pools vs {num_dishonest_pools} dishonest pools")
print("Key insights:")
print("- Honest pools cooperate with each other but are strategic against dishonest pools")
print("- Dishonest pools are strategic against both honest and other dishonest pools")
print("- This models a more realistic competitive mining environment")
print("- Results show how different pool types perform in strategic scenarios") 