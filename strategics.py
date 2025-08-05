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

# Pool configuration - multiple strategic pools only
num_strategic_pools_list = [2, 3, 7]  # Test different numbers of strategic pools

# Use time-based seed for different results each run
random.seed(int(time.time()))

# Generate random spike positions (deterministic)
spike_positions = set(random.sample(range(blocks), num_spikes))

def setup_pools(num_strategic_pools):
    """Setup pool configuration for given number of strategic pools"""
    total_pools = num_strategic_pools
    
    # Pool hashrates (must sum to 1.0)
    # All pools are strategic and get equal hashrate initially
    strategic_hashrate_per_pool = 1.0 / num_strategic_pools  # Each strategic pool gets equal share
    
    # Verify hashrates sum to 1.0
    total_strategic = strategic_hashrate_per_pool * num_strategic_pools
    if abs(total_strategic - 1.0) > 0.001:
        print(f"Warning: Hashrates don't sum to 1.0: {total_strategic:.3f}")
    
    pool_hashrates = {}
    # Strategic pools (0, 1, 2, ...)
    for i in range(num_strategic_pools):
        pool_hashrates[i] = strategic_hashrate_per_pool
    
    # Configure which pools are strategic
    strategic_pools = set(range(num_strategic_pools))  # All pools are strategic
    
    return pool_hashrates, strategic_pools, total_pools

def deterministic_simulation(strategy: str, miner_sequence, pool_hashrates):
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
            # All pools are strategic - they only include fees if THEY THEMSELVES mined the matured block
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
            # Track if strategic miner won accumulated fees
            if accumulated_fees > 0:
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
        'strategic_accumulation_success_rate': (strategic_wins_with_accumulated_fees / blocks_with_accumulated_fees * 100) if blocks_with_accumulated_fees > 0 else 0,
        'withholding_streaks': withholding_streaks,
        'avg_withholding_streak': avg_streak,
        'min_withholding_streak': min_streak,
        'max_withholding_streak': max_streak,
        'total_streaks': len(withholding_streaks)
    }
    
    return miner_rewards, metrics

def run_hashrate_sweep_single_run(pool_0_hashrate, num_strategic_pools):
    """Run a single simulation for a given hashrate distribution"""
    # For multiple pools, distribute remaining hashrate equally among other pools
    remaining_hashrate = 1.0 - pool_0_hashrate
    other_pools_hashrate = remaining_hashrate / (num_strategic_pools - 1) if num_strategic_pools > 1 else 0
    
    # Update pool hashrates
    pool_hashrates_temp = {0: pool_0_hashrate}
    for i in range(1, num_strategic_pools):
        pool_hashrates_temp[i] = other_pools_hashrate
    
    # Generate miner sequence for this hashrate distribution
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
    
    # Setup pools for this configuration
    pool_hashrates, strategic_pools, total_pools = setup_pools(num_strategic_pools)
    
    # Run simulations
    regular_rewards, regular_metrics = deterministic_simulation("regular", fixed_miner_sequence, pool_hashrates)
    strategic_rewards, strategic_metrics = deterministic_simulation("strategic", fixed_miner_sequence, pool_hashrates)
    
    # Calculate total rewards for percentage calculation
    total_regular_reward = sum(regular_rewards.values())
    total_strategic_reward = sum(strategic_rewards.values())
    
    # Calculate reward percentages for each pool
    pool_reward_percentages = {}
    for pool_id in pool_hashrates_temp.keys():
        regular_percentage = (regular_rewards[pool_id] / total_regular_reward * 100) if total_regular_reward > 0 else 0
        strategic_percentage = (strategic_rewards[pool_id] / total_strategic_reward * 100) if total_strategic_reward > 0 else 0
        pool_reward_percentages[pool_id] = {
            'regular_percentage': regular_percentage,
            'strategic_percentage': strategic_percentage,
            'regular_reward': regular_rewards[pool_id],
            'strategic_reward': strategic_rewards[pool_id]
        }
    
    return pool_0_hashrate, pool_reward_percentages, strategic_metrics

def run_multiple_simulations_for_config(num_strategic_pools, num_runs=num_runs):
    """Run multiple simulations for a specific number of strategic pools"""
    hashrates = np.arange(0.0, 1.1, 0.1)  # Pool 0 hashrate from 0% to 100%
    all_results = {h: {pool_id: {'regular_percentages': [], 'strategic_percentages': []} for pool_id in range(num_strategic_pools)} for h in hashrates}
    all_withholding_stats = {h: {'avg_streak': [], 'min_streak': [], 'max_streak': [], 'total_streaks': []} for h in hashrates}
    
    print(f"Running {num_runs} simulations for {num_strategic_pools} strategic pools...")
    
    for run in range(num_runs):
        if run % 10 == 0:
            print(f"Run {run + 1}/{num_runs}...")
        
        for pool_0_hashrate in hashrates:
            _, pool_reward_percentages, strategic_metrics = run_hashrate_sweep_single_run(pool_0_hashrate, num_strategic_pools)
            for pool_id in range(num_strategic_pools):
                all_results[pool_0_hashrate][pool_id]['regular_percentages'].append(pool_reward_percentages[pool_id]['regular_percentage'])
                all_results[pool_0_hashrate][pool_id]['strategic_percentages'].append(pool_reward_percentages[pool_id]['strategic_percentage'])
            
            # Store withholding statistics
            all_withholding_stats[pool_0_hashrate]['avg_streak'].append(strategic_metrics['avg_withholding_streak'])
            all_withholding_stats[pool_0_hashrate]['min_streak'].append(strategic_metrics['min_withholding_streak'])
            all_withholding_stats[pool_0_hashrate]['max_streak'].append(strategic_metrics['max_withholding_streak'])
            all_withholding_stats[pool_0_hashrate]['total_streaks'].append(strategic_metrics['total_streaks'])
    
    # Calculate averages and statistics for each pool
    avg_results = []
    for hashrate in hashrates:
        pool_results = {}
        for pool_id in range(num_strategic_pools):
            regular_percentages = all_results[hashrate][pool_id]['regular_percentages']
            strategic_percentages = all_results[hashrate][pool_id]['strategic_percentages']
            
            avg_regular_percentage = np.mean(regular_percentages)
            std_regular_percentage = np.std(regular_percentages)
            avg_strategic_percentage = np.mean(strategic_percentages)
            std_strategic_percentage = np.std(strategic_percentages)
            
            pool_results[pool_id] = {
                'avg_regular_percentage': avg_regular_percentage,
                'std_regular_percentage': std_regular_percentage,
                'avg_strategic_percentage': avg_strategic_percentage,
                'std_strategic_percentage': std_strategic_percentage,
                'min_strategic_percentage': min(strategic_percentages),
                'max_strategic_percentage': max(strategic_percentages)
            }
        
        # Calculate withholding statistics for this hashrate
        withholding_stats = all_withholding_stats[hashrate]
        avg_avg_streak = np.mean(withholding_stats['avg_streak'])
        avg_min_streak = np.mean(withholding_stats['min_streak'])
        avg_max_streak = np.mean(withholding_stats['max_streak'])
        avg_total_streaks = np.mean(withholding_stats['total_streaks'])
        
        avg_results.append({
            'hashrate': hashrate,
            'pool_results': pool_results,
            'withholding_stats': {
                'avg_avg_streak': avg_avg_streak,
                'avg_min_streak': avg_min_streak,
                'avg_max_streak': avg_max_streak,
                'avg_total_streaks': avg_total_streaks
            }
        })
        
        # Calculate hashrates for this configuration
        remaining_hashrate = 1.0 - hashrate
        other_pools_hashrate = remaining_hashrate / (num_strategic_pools - 1) if num_strategic_pools > 1 else 0
        
        # Print results for this hashrate
        print(f"\nPool 0 Hashrate {hashrate:.1f} (Others: {other_pools_hashrate:.1f} each):")
        for pool_id in range(num_strategic_pools):
            result = pool_results[pool_id]
            pool_hashrate = hashrate if pool_id == 0 else other_pools_hashrate
            print(f"  Pool {pool_id} ({pool_hashrate*100:.1f}%): Regular {result['avg_regular_percentage']:.2f}% ± {result['std_regular_percentage']:.2f}%, Strategic {result['avg_strategic_percentage']:.2f}% ± {result['std_strategic_percentage']:.2f}%")
        
        # Print withholding statistics
        print(f"  Withholding Streaks - Avg: {avg_avg_streak:.2f}, Min: {avg_min_streak:.2f}, Max: {avg_max_streak:.2f}, Total: {avg_total_streaks:.1f}")
    
    return avg_results

# Run simulations for all configurations
print("=== MULTIPLE STRATEGIC POOLS STRATEGIC SIMULATION ===")
print(f"Testing configurations: {num_strategic_pools_list} strategic pools")
print()

all_config_results = {}

for num_strategic_pools in num_strategic_pools_list:
    print(f"\n{'='*60}")
    print(f"CONFIGURATION: {num_strategic_pools} STRATEGIC POOLS")
    print(f"{'='*60}")
    
    # Run multiple simulations for this configuration
    avg_results = run_multiple_simulations_for_config(num_strategic_pools, num_runs)
    all_config_results[num_strategic_pools] = avg_results

print(f"\nGenerated {num_spikes} random spikes across {blocks} blocks")
print()

# Create plots for each configuration
os.makedirs('results', exist_ok=True)

for num_strategic_pools in num_strategic_pools_list:
    avg_results = all_config_results[num_strategic_pools]
    hashrates = [r['hashrate'] for r in avg_results]
    
    # Create two subplots: one for regular scenario, one for strategic scenario
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 16))
    
    # Plot each pool separately
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'H', '8']
    
    # Plot regular scenario (top subplot)
    for i, pool_id in enumerate(range(num_strategic_pools)):
        avg_regular_percentages = [r['pool_results'][pool_id]['avg_regular_percentage'] for r in avg_results]
        std_regular_percentages = [r['pool_results'][pool_id]['std_regular_percentage'] for r in avg_results]
        
        ax1.plot(hashrates, avg_regular_percentages, 
                 color=colors[i], marker=markers[i], 
                 linewidth=2, markersize=8, linestyle='-',
                 label=f'Pool {pool_id}')
        
        # Plot error bars (standard deviation)
        ax1.errorbar(hashrates, avg_regular_percentages, yerr=std_regular_percentages, 
                     fmt='none', capsize=3, alpha=0.5, color=colors[i])
    
    ax1.set_xlabel('Pool 0 Hashrate (Others = (1 - Pool 0) / (N-1))')
    ax1.set_ylabel('Reward Percentage (%)')
    ax1.set_title(f'{num_strategic_pools} Pools - Regular Scenario\n({num_runs} runs)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(hashrates)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot strategic scenario (bottom subplot)
    for i, pool_id in enumerate(range(num_strategic_pools)):
        avg_strategic_percentages = [r['pool_results'][pool_id]['avg_strategic_percentage'] for r in avg_results]
        std_strategic_percentages = [r['pool_results'][pool_id]['std_strategic_percentage'] for r in avg_results]
        
        ax2.plot(hashrates, avg_strategic_percentages, 
                 color=colors[i], marker=markers[i], 
                 linewidth=2, markersize=8, linestyle='-',
                 label=f'Pool {pool_id}')
        
        # Plot error bars (standard deviation)
        ax2.errorbar(hashrates, avg_strategic_percentages, yerr=std_strategic_percentages, 
                     fmt='none', capsize=3, alpha=0.5, color=colors[i])
    
    ax2.set_xlabel('Pool 0 Hashrate (Others = (1 - Pool 0) / (N-1))')
    ax2.set_ylabel('Reward Percentage (%)')
    ax2.set_title(f'{num_strategic_pools} Pools - Strategic Scenario\n({num_runs} runs)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(hashrates)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'results/strategic_{num_strategic_pools}_pools_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: results/strategic_{num_strategic_pools}_pools_results.png")

# Print summary for all configurations
print(f"\n=== SUMMARY ({num_runs} runs) ===")
for num_strategic_pools in num_strategic_pools_list:
    print(f"\n{num_strategic_pools} STRATEGIC POOLS:")
    print("Pool 0 Hashrate | Pool ID | Hashrate | Regular Reward % | Strategic Reward %")
    print("-" * 80)
    
    avg_results = all_config_results[num_strategic_pools]
    for r in avg_results:
        hashrate = r['hashrate']
        remaining_hashrate = 1.0 - hashrate
        other_pools_hashrate = remaining_hashrate / (num_strategic_pools - 1) if num_strategic_pools > 1 else 0
        
        for pool_id in range(num_strategic_pools):
            result = r['pool_results'][pool_id]
            individual_hashrate = hashrate if pool_id == 0 else other_pools_hashrate
            
            print(f"{hashrate:.1f}           | {pool_id:7} | {individual_hashrate*100:6.1f}% | {result['avg_regular_percentage']:8.2f}% ± {result['std_regular_percentage']:.2f}% | {result['avg_strategic_percentage']:10.2f}% ± {result['std_strategic_percentage']:.2f}%")

print(f"\n=== WITHHOLDING STATISTICS ({num_runs} runs) ===")
for num_strategic_pools in num_strategic_pools_list:
    print(f"\n{num_strategic_pools} STRATEGIC POOLS:")
    print("Pool 0 Hashrate | Avg Streak | Min Streak | Max Streak | Total Streaks")
    print("-" * 80)
    
    avg_results = all_config_results[num_strategic_pools]
    for r in avg_results:
        hashrate = r['hashrate']
        withholding_stats = r['withholding_stats']
        
        print(f"{hashrate:.1f}           | {withholding_stats['avg_avg_streak']:10.2f} | {withholding_stats['avg_min_streak']:10.2f} | {withholding_stats['avg_max_streak']:10.2f} | {withholding_stats['avg_total_streaks']:13.1f}")

print(f"\nTested configurations: {num_strategic_pools_list} strategic pools")
print("Key insights:")
print("- All pools are strategic and strategic against each other")
print("- Each pool only includes fees when they themselves mined the matured block")
print("- Hashrate distribution varies from 0%-100% to 100%-0% for the first pool")
print("- Other pools share the remaining hashrate equally")
print("- This models a competitive environment with no cooperation")
print("- Results show how different numbers of pools and hashrate distributions affect performance")
print("- Withholding streaks show how long fees are withheld before being included") 