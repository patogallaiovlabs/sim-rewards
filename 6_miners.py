import random
import matplotlib.pyplot as plt
import numpy as np
import time

# Three Scenarios Compared:
# All Regular: All miners include transaction fees in every block they mine
# Scenario 1: Only the main miner (40%) uses strategic fee withholding, while the secondary miner (10%) and others remain regular
# Scenario 2: Both the main miner (40%) and secondary miner (10%) use strategic fee withholding, while the other 5 miners remain regular



# Parameters
blocks = 50000
maturity_period = 4000
reward_percentage = 0.10
base_fee = 0.001
spike_fee = 1.0
num_spikes = 200

# Hashrate distribution for the scenarios
# Main miner: 40% (0.4)
# Secondary miner: 10% (0.1) 
# 5 other miners: 10% each (0.1 each)
# Total: 0.4 + 0.1 + 5*0.1 = 1.0

pool_hashrates = {
    0: 0.4,    # Main miner: 40%
    1: 0.1,    # Secondary miner: 10%
    2: 0.1,    # Other miner 1: 10%
    3: 0.1,    # Other miner 2: 10%
    4: 0.1,    # Other miner 3: 10%
    5: 0.1,    # Other miner 4: 10%
    6: 0.1,    # Other miner 5: 10%
}

# Use time-based seed for different results each run
random.seed(int(time.time()))

# Generate random spike positions (deterministic)
spike_positions = set(random.sample(range(blocks), num_spikes))

def deterministic_simulation(strategy: str, miner_sequence, strategic_pools):
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

def run_scenario_comparison():
    """Run comparison between the two scenarios"""
    
    # Generate miner sequence for this hashrate distribution
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
    
    # Scenario 1: Main strategic, Secondary regular, Others regular
    strategic_pools_scenario1 = {0}  # Only main miner is strategic
    regular_rewards, regular_metrics = deterministic_simulation("regular", fixed_miner_sequence, set())
    scenario1_rewards, scenario1_metrics = deterministic_simulation("strategic", fixed_miner_sequence, strategic_pools_scenario1)
    
    # Scenario 2: Main strategic, Secondary strategic, Others regular
    strategic_pools_scenario2 = {0, 1}  # Main and secondary miners are strategic
    scenario2_rewards, scenario2_metrics = deterministic_simulation("strategic", fixed_miner_sequence, strategic_pools_scenario2)
    
    return regular_rewards, scenario1_rewards, scenario2_rewards, regular_metrics, scenario1_metrics, scenario2_metrics

def run_multiple_comparisons(num_runs=100):
    """Run multiple comparisons and calculate averages"""
    all_regular_rewards = {pool_id: [] for pool_id in pool_hashrates.keys()}
    all_scenario1_rewards = {pool_id: [] for pool_id in pool_hashrates.keys()}
    all_scenario2_rewards = {pool_id: [] for pool_id in pool_hashrates.keys()}
    
    print(f"Running {num_runs} comparisons...")
    
    for run in range(num_runs):
        if run % 10 == 0:
            print(f"Run {run + 1}/{num_runs}...")
        
        regular_rewards, scenario1_rewards, scenario2_rewards, _, _, _ = run_scenario_comparison()
        
        # Store rewards for each pool
        for pool_id in pool_hashrates.keys():
            all_regular_rewards[pool_id].append(regular_rewards[pool_id])
            all_scenario1_rewards[pool_id].append(scenario1_rewards[pool_id])
            all_scenario2_rewards[pool_id].append(scenario2_rewards[pool_id])
    
    # Calculate averages and statistics
    avg_results = {}
    for pool_id in pool_hashrates.keys():
        regular_rewards_list = all_regular_rewards[pool_id]
        scenario1_rewards_list = all_scenario1_rewards[pool_id]
        scenario2_rewards_list = all_scenario2_rewards[pool_id]
        
        avg_regular = np.mean(regular_rewards_list)
        avg_scenario1 = np.mean(scenario1_rewards_list)
        avg_scenario2 = np.mean(scenario2_rewards_list)
        
        std_regular = np.std(regular_rewards_list)
        std_scenario1 = np.std(scenario1_rewards_list)
        std_scenario2 = np.std(scenario2_rewards_list)
        
        # Calculate improvements
        scenario1_improvement = (avg_scenario1 - avg_regular) / avg_regular * 100 if avg_regular > 0 else 0
        scenario2_improvement = (avg_scenario2 - avg_regular) / avg_regular * 100 if avg_regular > 0 else 0
        
        avg_results[pool_id] = {
            'avg_regular': avg_regular,
            'avg_scenario1': avg_scenario1,
            'avg_scenario2': avg_scenario2,
            'std_regular': std_regular,
            'std_scenario1': std_scenario1,
            'std_scenario2': std_scenario2,
            'scenario1_improvement': scenario1_improvement,
            'scenario2_improvement': scenario2_improvement
        }
    
    return avg_results

# Run single comparison for detailed analysis
print("=== SINGLE RUN COMPARISON ===")
regular_rewards, scenario1_rewards, scenario2_rewards, regular_metrics, scenario1_metrics, scenario2_metrics = run_scenario_comparison()

print("=== SCENARIO DEFINITIONS ===")
print("Scenario 1: Main miner (40%) strategic, Secondary miner (10%) regular, Others (10% each) regular")
print("Scenario 2: Main miner (40%) strategic, Secondary miner (10%) strategic, Others (10% each) regular")
print()

print("=== REWARD COMPARISON (Single Run) ===")
print("Pool | Hashrate | Type           | All Regular | Scenario 1 | Scenario 2 | S1 Change | S2 Change")
print("-" * 100)
for pool_id in sorted(pool_hashrates.keys()):
    hashrate = pool_hashrates[pool_id]
    pool_type = "Main Strategic" if pool_id == 0 else "Secondary" if pool_id == 1 else "Other Regular"
    
    regular_reward = regular_rewards[pool_id]
    scenario1_reward = scenario1_rewards[pool_id]
    scenario2_reward = scenario2_rewards[pool_id]
    
    scenario1_change = (scenario1_reward - regular_reward) / regular_reward * 100 if regular_reward > 0 else 0
    scenario2_change = (scenario2_reward - regular_reward) / regular_reward * 100 if regular_reward > 0 else 0
    
    print(f"{pool_id:4d} | {hashrate*100:7.0f}% | {pool_type:14s} | {regular_reward:11.2f} | {scenario1_reward:10.2f} | {scenario2_reward:10.2f} | {scenario1_change:+8.2f}% | {scenario2_change:+8.2f}%")

print()

# Run multiple comparisons for statistical analysis
print("=== MULTIPLE RUNS STATISTICAL ANALYSIS ===")
avg_results = run_multiple_comparisons(100)

print("\n=== AVERAGE RESULTS (100 runs) ===")
print("Pool | Hashrate | Type           | Avg Regular | Avg S1 | Avg S2 | S1 Change | S2 Change")
print("-" * 90)
for pool_id in sorted(pool_hashrates.keys()):
    hashrate = pool_hashrates[pool_id]
    pool_type = "Main Strategic" if pool_id == 0 else "Secondary" if pool_id == 1 else "Other Regular"
    
    result = avg_results[pool_id]
    avg_regular = result['avg_regular']
    avg_scenario1 = result['avg_scenario1']
    avg_scenario2 = result['avg_scenario2']
    scenario1_change = result['scenario1_improvement']
    scenario2_change = result['scenario2_improvement']
    
    print(f"{pool_id:4d} | {hashrate*100:7.0f}% | {pool_type:14s} | {avg_regular:11.2f} | {avg_scenario1:7.2f} | {avg_scenario2:7.2f} | {scenario1_change:+8.2f}% | {scenario2_change:+8.2f}%")

print()

# Focus on main and secondary miners as requested
print("=== MAIN AND SECONDARY MINER COMPARISON (100 runs) ===")
print("Miner Type    | Hashrate | Scenario 1 Change | Scenario 2 Change | Difference (S2-S1)")
print("-" * 80)

main_result = avg_results[0]
secondary_result = avg_results[1]

main_s1_change = main_result['scenario1_improvement']
main_s2_change = main_result['scenario2_improvement']
main_diff = main_s2_change - main_s1_change

secondary_s1_change = secondary_result['scenario1_improvement']
secondary_s2_change = secondary_result['scenario2_improvement']
secondary_diff = secondary_s2_change - secondary_s1_change

print(f"Main (40%)     | {pool_hashrates[0]*100:7.0f}% | {main_s1_change:+15.2f}% | {main_s2_change:+15.2f}% | {main_diff:+18.2f}%")
print(f"Secondary (10%) | {pool_hashrates[1]*100:7.0f}% | {secondary_s1_change:+15.2f}% | {secondary_s2_change:+15.2f}% | {secondary_diff:+18.2f}%")

print()

# Create visualization
plt.figure(figsize=(15, 10))

# Subplot 1: Reward comparison for main and secondary miners
plt.subplot(2, 2, 1)
pools = [0, 1]  # Main and secondary
pool_names = ['Main (40%)', 'Secondary (10%)']
scenario1_changes = [avg_results[pool_id]['scenario1_improvement'] for pool_id in pools]
scenario2_changes = [avg_results[pool_id]['scenario2_improvement'] for pool_id in pools]

x = np.arange(len(pools))
width = 0.35

plt.bar(x - width/2, scenario1_changes, width, label='Scenario 1 (Secondary Regular)', color='blue', alpha=0.7)
plt.bar(x + width/2, scenario2_changes, width, label='Scenario 2 (Secondary Strategic)', color='red', alpha=0.7)

plt.xlabel('Miner Type')
plt.ylabel('Reward Change (%)')
plt.title('Main and Secondary Miner Reward Changes')
plt.xticks(x, pool_names)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)

# Subplot 2: Difference between scenarios
plt.subplot(2, 2, 2)
differences = [scenario2_changes[i] - scenario1_changes[i] for i in range(len(pools))]
plt.bar(pool_names, differences, color='green', alpha=0.7)
plt.xlabel('Miner Type')
plt.ylabel('Difference (S2 - S1) (%)')
plt.title('Impact of Secondary Miner Switching to Strategic')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)

# Subplot 3: All miners comparison
plt.subplot(2, 2, 3)
all_pools = sorted(pool_hashrates.keys())
all_scenario1_changes = [avg_results[pool_id]['scenario1_improvement'] for pool_id in all_pools]
all_scenario2_changes = [avg_results[pool_id]['scenario2_improvement'] for pool_id in all_pools]

x = np.arange(len(all_pools))
plt.bar(x - width/2, all_scenario1_changes, width, label='Scenario 1', color='blue', alpha=0.7)
plt.bar(x + width/2, all_scenario2_changes, width, label='Scenario 2', color='red', alpha=0.7)

plt.xlabel('Pool ID')
plt.ylabel('Reward Change (%)')
plt.title('All Miners Reward Changes')
plt.xticks(x, [f'Pool {pid}' for pid in all_pools])
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)

# Subplot 4: Average rewards comparison
plt.subplot(2, 2, 4)
avg_regular_rewards = [avg_results[pool_id]['avg_regular'] for pool_id in all_pools]
avg_scenario1_rewards = [avg_results[pool_id]['avg_scenario1'] for pool_id in all_pools]
avg_scenario2_rewards = [avg_results[pool_id]['avg_scenario2'] for pool_id in all_pools]

x = np.arange(len(all_pools))
plt.bar(x - width, avg_regular_rewards, width, label='All Regular', color='gray', alpha=0.7)
plt.bar(x, avg_scenario1_rewards, width, label='Scenario 1', color='blue', alpha=0.7)
plt.bar(x + width, avg_scenario2_rewards, width, label='Scenario 2', color='red', alpha=0.7)

plt.xlabel('Pool ID')
plt.ylabel('Average Reward')
plt.title('Average Rewards Comparison')
plt.xticks(x, [f'Pool {pid}' for pid in all_pools])
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print detailed statistics
print("=== DETAILED STATISTICS (100 runs) ===")
print("Main Miner (Pool 0):")
print(f"  All Regular: {avg_results[0]['avg_regular']:.2f} ± {avg_results[0]['std_regular']:.2f}")
print(f"  Scenario 1:  {avg_results[0]['avg_scenario1']:.2f} ± {avg_results[0]['std_scenario1']:.2f}")
print(f"  Scenario 2:  {avg_results[0]['avg_scenario2']:.2f} ± {avg_results[0]['std_scenario2']:.2f}")
print(f"  S1 Change:   {avg_results[0]['scenario1_improvement']:+.2f}%")
print(f"  S2 Change:   {avg_results[0]['scenario2_improvement']:+.2f}%")
print()

print("Secondary Miner (Pool 1):")
print(f"  All Regular: {avg_results[1]['avg_regular']:.2f} ± {avg_results[1]['std_regular']:.2f}")
print(f"  Scenario 1:  {avg_results[1]['avg_scenario1']:.2f} ± {avg_results[1]['std_scenario1']:.2f}")
print(f"  Scenario 2:  {avg_results[1]['avg_scenario2']:.2f} ± {avg_results[1]['std_scenario2']:.2f}")
print(f"  S1 Change:   {avg_results[1]['scenario1_improvement']:+.2f}%")
print(f"  S2 Change:   {avg_results[1]['scenario2_improvement']:+.2f}%")
print()

# Calculate total system impact
print("=== SYSTEM-WIDE IMPACT ===")
total_regular = sum(avg_results[pool_id]['avg_regular'] for pool_id in pool_hashrates.keys())
total_scenario1 = sum(avg_results[pool_id]['avg_scenario1'] for pool_id in pool_hashrates.keys())
total_scenario2 = sum(avg_results[pool_id]['avg_scenario2'] for pool_id in pool_hashrates.keys())

total_s1_change = (total_scenario1 - total_regular) / total_regular * 100 if total_regular > 0 else 0
total_s2_change = (total_scenario2 - total_regular) / total_regular * 100 if total_regular > 0 else 0

print(f"Total System Rewards:")
print(f"  All Regular: {total_regular:.2f}")
print(f"  Scenario 1:  {total_scenario1:.2f}")
print(f"  Scenario 2:  {total_scenario2:.2f}")
print(f"  S1 Change:   {total_s1_change:+.2f}%")
print(f"  S2 Change:   {total_s2_change:+.2f}%")
print(f"  Difference:  {total_s2_change - total_s1_change:+.2f}%") 