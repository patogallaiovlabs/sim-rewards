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
num_runs = 50  # Number of simulation runs for statistical analysis

# Pool configuration - multiple strategic pools
num_strategic_pools = 30  # Number of strategic pools
num_honest_pools = 1     # Number of honest pools
total_pools = num_strategic_pools + num_honest_pools

# Pool hashrates (must sum to 1.0)
# Strategic pools get equal hashrate, honest pools get the rest
strategic_hashrate_per_pool = 0.02  # Each strategic pool gets 20%
honest_hashrate = 1.0 - (strategic_hashrate_per_pool * num_strategic_pools)  # Remaining for honest

pool_hashrates = {}
# Strategic pools (0, 1, 2)
for i in range(num_strategic_pools):
    pool_hashrates[i] = strategic_hashrate_per_pool
# Honest pool (3)
pool_hashrates[num_strategic_pools] = honest_hashrate

# Configure which pools are strategic
strategic_pools = set(range(num_strategic_pools))  # Pools 0, 1, 2 are strategic
honest_pools = {num_strategic_pools}  # Pool 3 is honest

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
        if strategy == "honest":
            include_fee = True
        elif strategy == "strategic":
            if miner in strategic_pools:
                # Strategic pool only includes fees if matured block was also strategic
                # If no matured block yet (matured_block < 0), include fees to avoid infinite withholding
                if matured_block < 0 or (matured_block >= 0 and block_owners[matured_block] in strategic_pools):
                    include_fee = True
            else:
                # Honest pools always include fees
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

def run_hashrate_sweep_single_run(strategic_total_hashrate):
    """Run a single simulation for a given total strategic hashrate"""
    # Calculate hashrate per strategic pool
    strategic_hashrate_per_pool = strategic_total_hashrate / num_strategic_pools
    honest_hashrate = 1.0 - strategic_total_hashrate
    
    # Update pool hashrates
    pool_hashrates_temp = {}
    for i in range(num_strategic_pools):
        pool_hashrates_temp[i] = strategic_hashrate_per_pool
    pool_hashrates_temp[num_strategic_pools] = honest_hashrate
    
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
    
    # Calculate improvement for each strategic pool
    pool_improvements = {}
    for pool_id in strategic_pools:
        honest_reward = honest_rewards[pool_id]
        strategic_reward = strategic_rewards[pool_id]
        improvement = (strategic_reward - honest_reward) / honest_reward * 100 if honest_reward > 0 else 0
        pool_improvements[pool_id] = improvement
    
    return strategic_total_hashrate, pool_improvements

def run_multiple_simulations(num_runs=num_runs):
    """Run multiple simulations and calculate averages for each pool"""
    hashrates = np.arange(0.0, 1.1, 0.1)  # 0.0 to 1.0 (total strategic hashrate)
    all_results = {h: {pool_id: [] for pool_id in strategic_pools} for h in hashrates}
    
    print(f"Running {num_runs} simulations for each hashrate...")
    print(f"Configuration: {num_strategic_pools} strategic pools, {num_honest_pools} honest pool")
    print(f"Strategic pools: {list(strategic_pools)} (each with equal hashrate)")
    print(f"Honest pool: {list(honest_pools)}")
    
    for run in range(num_runs):
        if run % 10 == 0:
            print(f"Run {run + 1}/{num_runs}...")
        
        for strategic_total_hashrate in hashrates:
            _, pool_improvements = run_hashrate_sweep_single_run(strategic_total_hashrate)
            for pool_id in strategic_pools:
                all_results[strategic_total_hashrate][pool_id].append(pool_improvements[pool_id])
    
    # Calculate averages and statistics for each pool
    avg_results = []
    for hashrate in hashrates:
        pool_results = {}
        for pool_id in strategic_pools:
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
        
        # Print results for this hashrate
        print(f"\nTotal Strategic Hashrate {hashrate:.1f}:")
        for pool_id in strategic_pools:
            result = pool_results[pool_id]
            print(f"  Pool {pool_id}: {result['avg_improvement']:+.2f}% ± {result['std_improvement']:.2f}%")
    
    return avg_results

# Run simulations
honest_rewards, honest_metrics = deterministic_simulation("honest", fixed_miner_sequence)
strategic_rewards, strategic_metrics = deterministic_simulation("strategic", fixed_miner_sequence)

# Print results
print("=== MULTI-POOL STRATEGIC SIMULATION ===")
print(f"Configuration: {num_strategic_pools} strategic pools, {num_honest_pools} honest pool")
print(f"Strategic pools: {list(strategic_pools)} (each with {strategic_hashrate_per_pool*100:.0f}% hashrate)")
print(f"Honest pool: {list(honest_pools)} ({honest_hashrate*100:.0f}% hashrate)")
print()

print("=== REWARD COMPARISON ===")
for pool_id in pool_hashrates.keys():
    honest_reward = honest_rewards[pool_id]
    strategic_reward = strategic_rewards[pool_id]
    improvement = (strategic_reward - honest_reward) / honest_reward * 100 if honest_reward > 0 else 0
    pool_type = "STRATEGIC" if pool_id in strategic_pools else "honest"
    print(f"Pool {pool_id} ({pool_hashrates[pool_id]*100:.0f}% hashrate, {pool_type}):")
    print(f"  All-honest reward: {honest_reward:.2f}")
    print(f"  Strategic scenario reward: {strategic_reward:.2f}")
    print(f"  Change: {improvement:+.2f}%")
    print()

# Calculate average strategic pool performance
honest_avg_strategic = sum(honest_rewards[i] for i in strategic_pools) / len(strategic_pools)
strategic_avg_strategic = sum(strategic_rewards[i] for i in strategic_pools) / len(strategic_pools)
avg_improvement = (strategic_avg_strategic - honest_avg_strategic) / honest_avg_strategic * 100 if honest_avg_strategic > 0 else 0

print(f"=== AVERAGE STRATEGIC POOL PERFORMANCE ===")
print(f"Average honest reward (strategic pools): {honest_avg_strategic:.2f}")
print(f"Average strategic reward (strategic pools): {strategic_avg_strategic:.2f}")
print(f"Average improvement: {avg_improvement:+.2f}%")
print()

print(f"Generated {num_spikes} random spikes across {blocks} blocks")
print()

# Print fee inclusion metrics
print("=== FEE INCLUSION METRICS ===")
print(f"All-Honest Scenario:")
print(f"  Total blocks: {honest_metrics['total_blocks']}")
print(f"  Fees not included: {honest_metrics['fees_not_included']} ({honest_metrics['fees_not_included_percentage']:.2f}%)")
print(f"  Pool blocks: {honest_metrics['pool_blocks']}")
print()

print(f"Strategic Scenario ({len(strategic_pools)} strategic pools):")
print(f"  Total blocks: {strategic_metrics['total_blocks']}")
print(f"  Fees not included: {strategic_metrics['fees_not_included']} ({strategic_metrics['fees_not_included_percentage']:.2f}%)")
print(f"  Pool blocks: {strategic_metrics['pool_blocks']}")
print(f"  Pool fees not included: {strategic_metrics['pool_fees_not_included']}")
print()

# Print accumulation metrics
print("=== ACCUMULATION STRATEGY METRICS ===")
print(f"All-Honest Scenario:")
print(f"  Blocks with accumulated fees: {honest_metrics['blocks_with_accumulated_fees']}")
print(f"  Strategic wins with accumulated fees: {honest_metrics['strategic_wins_with_accumulated_fees']}")
print(f"  Total accumulated fees won: {honest_metrics['total_accumulated_fees_won']:.2f}")
print()

print(f"Strategic Scenario ({len(strategic_pools)} strategic pools):")
print(f"  Blocks with accumulated fees: {strategic_metrics['blocks_with_accumulated_fees']}")
print(f"  Strategic wins with accumulated fees: {strategic_metrics['strategic_wins_with_accumulated_fees']}")
print(f"  Total accumulated fees won: {strategic_metrics['total_accumulated_fees_won']:.2f}")
print(f"  Accumulation success rate: {strategic_metrics['accumulation_success_rate']:.2f}%")
print()

# Print withholding streak metrics
print("=== WITHHOLDING STREAK METRICS ===")
print(f"All-Honest Scenario:")
print(f"  Total withholding streaks: {honest_metrics['total_streaks']}")
print(f"  Average streak length: {honest_metrics['avg_withholding_streak']:.2f} blocks")
print(f"  Min streak length: {honest_metrics['min_withholding_streak']} blocks")
print(f"  Max streak length: {honest_metrics['max_withholding_streak']} blocks")
print()

print(f"Strategic Scenario ({len(strategic_pools)} strategic pools):")
print(f"  Total withholding streaks: {strategic_metrics['total_streaks']}")
print(f"  Average streak length: {strategic_metrics['avg_withholding_streak']:.2f} blocks")
print(f"  Min streak length: {strategic_metrics['min_withholding_streak']} blocks")
print(f"  Max streak length: {strategic_metrics['max_withholding_streak']} blocks")

# Run multiple simulations
print(f"\nRunning {num_runs} simulations to get average results...")
avg_results = run_multiple_simulations(num_runs)

# Create the plot
hashrates = [r['hashrate'] for r in avg_results]

plt.figure(figsize=(14, 10))

# Plot each strategic pool separately
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'H', '8']

for i, pool_id in enumerate(strategic_pools):
    avg_improvements = [r['pool_results'][pool_id]['avg_improvement'] for r in avg_results]
    std_improvements = [r['pool_results'][pool_id]['std_improvement'] for r in avg_results]
    
    plt.plot(hashrates, avg_improvements, 
             linewidth=2, markersize=8, 
             label=f'Pool {pool_id}')
    
    # Plot error bars (standard deviation)
    plt.errorbar(hashrates, avg_improvements, yerr=std_improvements, 
                 fmt='none', capsize=3, alpha=0.5)

plt.xlabel('Total Strategic Pool Hashrate')
plt.ylabel('Individual Pool Improvement (%)')
plt.title(f'Individual Strategic Pool Improvements vs Total Hashrate ({num_strategic_pools} Strategic Pools, {num_runs} runs)')
plt.grid(True, alpha=0.3)
plt.xticks(hashrates)

# Add zero line for reference
plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='No Improvement')

# Create results directory if it doesn't exist
import os
os.makedirs('results', exist_ok=True)

plt.legend()
plt.tight_layout()
plt.savefig('results/strategic_multi_pool_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved to: results/strategic_multi_pool_results.png")

# Print summary
print(f"\n=== SUMMARY ({num_runs} runs) ===")
print("Total Strategic Hashrate | Pool | Avg Improvement | Std Dev | Min | Max")
print("-" * 80)
for r in avg_results:
    hashrate = r['hashrate']
    for pool_id in strategic_pools:
        result = r['pool_results'][pool_id]
        print(f"{hashrate:.1f}                    | {pool_id}    | {result['avg_improvement']:+.2f}%        | {result['std_improvement']:.2f}%   | {result['min_improvement']:+.2f}% | {result['max_improvement']:+.2f}%")

print(f"\nConfiguration: {num_strategic_pools} strategic pools with equal hashrate distribution")
print("Key insights:")
print("- Multiple strategic pools may have different dynamics than single strategic pool")
print("- Competition between strategic pools may reduce individual gains")
print("- Distributed strategic behavior may be more realistic than concentrated strategic behavior") 