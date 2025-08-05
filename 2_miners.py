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
num_runs = 100  # Number of simulation runs for statistical analysis

# Pool configuration - one pool to test, others are strategic
num_total_pools = 2  # Total number of pools
test_pool_id = 0  # The pool we want to compare (regular vs strategic)

# Use time-based seed for different results each run
random.seed(int(time.time()))

# Generate random spike positions (deterministic)
spike_positions = set(random.sample(range(blocks), num_spikes))

def setup_pools(test_pool_hashrate, test_pool_strategy="regular", other_pools_strategy="strategic"):
    """Setup pool configuration with one test pool and others with specified strategy"""
    total_pools = num_total_pools
    
    # Calculate hashrate for other pools (equal distribution)
    remaining_hashrate = 1.0 - test_pool_hashrate
    other_pools_hashrate = remaining_hashrate / (num_total_pools - 1) if num_total_pools > 1 else 0
    
    # Pool hashrates
    pool_hashrates = {}
    pool_hashrates[test_pool_id] = test_pool_hashrate
    for i in range(num_total_pools):
        if i != test_pool_id:
            pool_hashrates[i] = other_pools_hashrate
    
    # Configure which pools are strategic
    strategic_pools = set()
    
    # Add test pool to strategic set if it's strategic
    if test_pool_strategy == "strategic":
        strategic_pools.add(test_pool_id)
    
    # Add other pools to strategic set if they're strategic
    if other_pools_strategy == "strategic":
        for i in range(num_total_pools):
            if i != test_pool_id:
                strategic_pools.add(i)
    
    return pool_hashrates, strategic_pools, total_pools

def deterministic_simulation(strategy: str, miner_sequence, pool_hashrates, strategic_pools):
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
                # Strategic pool only includes fees if THEY THEMSELVES mined the matured block
                # If no matured block yet (matured_block < 0), include fees to avoid infinite withholding
                if matured_block < 0 or (matured_block >= 0 and block_owners[matured_block] == miner):
                    include_fee = True
            else:
                # Regular pools always include fees
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

def run_hashrate_sweep_single_run(test_pool_hashrate):
    """Run a single simulation for a given test pool hashrate"""
    # Handle edge case where test pool hashrate is 0
    if test_pool_hashrate == 0:
        # If test pool has 0 hashrate, it gets 0 reward
        return 0, 0, 0, 0, 0, 0, 0, 0, {}, {}, {}, {}, 0, 0, 0, 0
    
    # Generate miner sequence for this hashrate distribution
    random.seed(int(time.time() * 1000) % (2**32))  # Use time-based seed
    fixed_miner_sequence = []
    
    # Calculate hashrates for sequence generation
    remaining_hashrate = 1.0 - test_pool_hashrate
    other_pools_hashrate = remaining_hashrate / (num_total_pools - 1) if num_total_pools > 1 else 0
    
    pool_hashrates_temp = {test_pool_id: test_pool_hashrate}
    for i in range(num_total_pools):
        if i != test_pool_id:
            pool_hashrates_temp[i] = other_pools_hashrate
    
    for _ in range(blocks):
        rand = random.random()
        cumulative = 0
        for pool_id, hashrate in pool_hashrates_temp.items():
            cumulative += hashrate
            if rand < cumulative:
                fixed_miner_sequence.append(pool_id)
                break
    
    # Run simulations for four scenarios:
    # 1. Test pool is regular, others are strategic
    pool_hashrates_regular, strategic_pools_regular, _ = setup_pools(test_pool_hashrate, "regular", "strategic")
    regular_rewards, regular_metrics = deterministic_simulation("strategic", fixed_miner_sequence, pool_hashrates_regular, strategic_pools_regular)
    
    # 2. Test pool is strategic, others are strategic
    pool_hashrates_strategic, strategic_pools_strategic, _ = setup_pools(test_pool_hashrate, "strategic", "strategic")
    strategic_rewards, strategic_metrics = deterministic_simulation("strategic", fixed_miner_sequence, pool_hashrates_strategic, strategic_pools_strategic)
    
    # 3. Test pool is strategic, others are regular
    pool_hashrates_mixed, strategic_pools_mixed, _ = setup_pools(test_pool_hashrate, "strategic", "regular")
    mixed_rewards, mixed_metrics = deterministic_simulation("strategic", fixed_miner_sequence, pool_hashrates_mixed, strategic_pools_mixed)
    
    # 4. Test pool is regular, others are regular (baseline)
    pool_hashrates_baseline, strategic_pools_baseline, _ = setup_pools(test_pool_hashrate, "regular", "regular")
    baseline_rewards, baseline_metrics = deterministic_simulation("regular", fixed_miner_sequence, pool_hashrates_baseline, strategic_pools_baseline)
    
    # Calculate rewards for test pool
    regular_reward = regular_rewards[test_pool_id]
    strategic_reward = strategic_rewards[test_pool_id]
    mixed_reward = mixed_rewards[test_pool_id]
    baseline_reward = baseline_rewards[test_pool_id]
    
    # Calculate improvements
    improvement_vs_regular = (strategic_reward - regular_reward) / regular_reward * 100 if regular_reward > 0 else 0
    improvement_vs_mixed = (strategic_reward - mixed_reward) / mixed_reward * 100 if mixed_reward > 0 else 0
    improvement_vs_baseline = (strategic_reward - baseline_reward) / baseline_reward * 100 if baseline_reward > 0 else 0
    
    # Calculate total rewards in the system
    total_regular_rewards = sum(regular_rewards.values())
    total_strategic_rewards = sum(strategic_rewards.values())
    total_mixed_rewards = sum(mixed_rewards.values())
    total_baseline_rewards = sum(baseline_rewards.values())
    
    return test_pool_hashrate, regular_reward, strategic_reward, mixed_reward, baseline_reward, improvement_vs_regular, improvement_vs_mixed, improvement_vs_baseline, regular_metrics, strategic_metrics, mixed_metrics, baseline_metrics, total_regular_rewards, total_strategic_rewards, total_mixed_rewards, total_baseline_rewards

def run_multiple_simulations(num_runs=num_runs):
    """Run multiple simulations for statistical analysis"""
    hashrates = np.arange(0.0, 1.01, 0.05)  # Test pool hashrate from 0% to 100% in 5% steps
    all_results = {h: {'regular_rewards': [], 'strategic_rewards': [], 'mixed_rewards': [], 'baseline_rewards': [], 'improvements_vs_regular': [], 'improvements_vs_mixed': [], 'improvements_vs_baseline': []} for h in hashrates}
    
    print(f"Running {num_runs} simulations for test pool comparison...")
    
    for run in range(num_runs):
        if run % 10 == 0:
            print(f"Run {run + 1}/{num_runs}...")
        
        for test_pool_hashrate in hashrates:
            _, regular_reward, strategic_reward, mixed_reward, baseline_reward, improvement_vs_regular, improvement_vs_mixed, improvement_vs_baseline, _, _, _, _, total_regular, total_strategic, total_mixed, total_baseline = run_hashrate_sweep_single_run(test_pool_hashrate)
            all_results[test_pool_hashrate]['regular_rewards'].append(regular_reward)
            all_results[test_pool_hashrate]['strategic_rewards'].append(strategic_reward)
            all_results[test_pool_hashrate]['mixed_rewards'].append(mixed_reward)
            all_results[test_pool_hashrate]['baseline_rewards'].append(baseline_reward)
            all_results[test_pool_hashrate]['improvements_vs_regular'].append(improvement_vs_regular)
            all_results[test_pool_hashrate]['improvements_vs_mixed'].append(improvement_vs_mixed)
            all_results[test_pool_hashrate]['improvements_vs_baseline'].append(improvement_vs_baseline)
    
    # Calculate averages and statistics
    avg_results = []
    for hashrate in hashrates:
        regular_rewards = all_results[hashrate]['regular_rewards']
        strategic_rewards = all_results[hashrate]['strategic_rewards']
        mixed_rewards = all_results[hashrate]['mixed_rewards']
        baseline_rewards = all_results[hashrate]['baseline_rewards']
        improvements_vs_regular = all_results[hashrate]['improvements_vs_regular']
        improvements_vs_mixed = all_results[hashrate]['improvements_vs_mixed']
        improvements_vs_baseline = all_results[hashrate]['improvements_vs_baseline']
        
        avg_regular = np.mean(regular_rewards)
        avg_strategic = np.mean(strategic_rewards)
        avg_mixed = np.mean(mixed_rewards)
        avg_baseline = np.mean(baseline_rewards)
        avg_improvement_vs_regular = np.mean(improvements_vs_regular)
        avg_improvement_vs_mixed = np.mean(improvements_vs_mixed)
        avg_improvement_vs_baseline = np.mean(improvements_vs_baseline)
        std_improvement_vs_regular = np.std(improvements_vs_regular)
        std_improvement_vs_mixed = np.std(improvements_vs_mixed)
        std_improvement_vs_baseline = np.std(improvements_vs_baseline)
        
        avg_results.append({
            'hashrate': hashrate,
            'avg_regular_reward': avg_regular,
            'avg_strategic_reward': avg_strategic,
            'avg_mixed_reward': avg_mixed,
            'avg_baseline_reward': avg_baseline,
            'avg_improvement_vs_regular': avg_improvement_vs_regular,
            'avg_improvement_vs_mixed': avg_improvement_vs_mixed,
            'avg_improvement_vs_baseline': avg_improvement_vs_baseline,
            'std_improvement_vs_regular': std_improvement_vs_regular,
            'std_improvement_vs_mixed': std_improvement_vs_mixed,
            'std_improvement_vs_baseline': std_improvement_vs_baseline,
            'min_improvement_vs_regular': min(improvements_vs_regular),
            'max_improvement_vs_regular': max(improvements_vs_regular),
            'min_improvement_vs_mixed': min(improvements_vs_mixed),
            'max_improvement_vs_mixed': max(improvements_vs_mixed),
            'min_improvement_vs_baseline': min(improvements_vs_baseline),
            'max_improvement_vs_baseline': max(improvements_vs_baseline)
        })
        
        # Print results for this hashrate
        print(f"\nTest Pool Hashrate {hashrate:.2f}:")
        print(f"  Regular Strategy: {avg_regular:.2f}")
        print(f"  Strategic Strategy: {avg_strategic:.2f}")
        print(f"  Mixed Strategy (Strategic vs Regular others): {avg_mixed:.2f}")
        print(f"  Baseline Strategy (Regular vs Regular others): {avg_baseline:.2f}")
        print(f"  Improvement vs Regular: {avg_improvement_vs_regular:+.2f}% ± {std_improvement_vs_regular:.2f}%")
        print(f"  Improvement vs Mixed: {avg_improvement_vs_mixed:+.2f}% ± {std_improvement_vs_mixed:.2f}%")
        print(f"  Improvement vs Baseline: {avg_improvement_vs_baseline:+.2f}% ± {std_improvement_vs_baseline:.2f}%")
    
    return avg_results

# Run simulations
print("=== SINGLE POOL SELF-COMPARISON SIMULATION ===")
print(f"Testing pool {test_pool_id} (regular vs strategic) with {num_total_pools} total pools")
print(f"Comparing all four scenarios: regular vs strategic, strategic vs strategic, strategic vs regular, regular vs regular")
print()

avg_results = run_multiple_simulations(num_runs)

print(f"\nGenerated {num_spikes} random spikes across {blocks} blocks")
print()

# Create plots
os.makedirs('results', exist_ok=True)

# Plot 1: Reward comparison
plt.figure(figsize=(12, 8))
hashrates = [r['hashrate'] for r in avg_results]
regular_rewards = [r['avg_regular_reward'] for r in avg_results]
strategic_rewards = [r['avg_strategic_reward'] for r in avg_results]
mixed_rewards = [r['avg_mixed_reward'] for r in avg_results]
baseline_rewards = [r['avg_baseline_reward'] for r in avg_results]

# Calculate expected reward based on hashrate (linear relationship)
# We need to get the total system reward to calculate the expected share
# Use a non-zero hashrate to estimate total system reward
total_system_reward = baseline_rewards[1] / hashrates[1] if len(hashrates) > 1 and hashrates[1] > 0 else 0  # Use second data point to estimate total
expected_rewards = [h * total_system_reward for h in hashrates]

plt.plot(hashrates, regular_rewards, 'b-', linewidth=2, label='Test Pool: Regular, Others: Strategic')
plt.plot(hashrates, strategic_rewards, 'r-', linewidth=2, label='Test Pool: Strategic, Others: Strategic')
plt.plot(hashrates, mixed_rewards, 'orange', linewidth=2, label='Test Pool: Strategic, Others: Regular')
plt.plot(hashrates, baseline_rewards, 'g-', linewidth=2, label='Test Pool: Regular, Others: Regular')
plt.plot(hashrates, expected_rewards, 'k--', linewidth=2, alpha=0.7, label='Expected Reward (Hashrate %)')

plt.xlabel('Test Pool Hashrate')
plt.ylabel('Average Reward')
plt.title(f'Single Pool Strategy Comparison\n({num_runs} runs, {num_total_pools} total pools)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('results/single_pool_self_comparison_rewards.png', dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved to: results/single_pool_self_comparison_rewards.png")

# Plot 2: Improvement percentage
plt.figure(figsize=(12, 8))
improvements_vs_regular = [r['avg_improvement_vs_regular'] for r in avg_results]
improvements_vs_mixed = [r['avg_improvement_vs_mixed'] for r in avg_results]
improvements_vs_baseline = [r['avg_improvement_vs_baseline'] for r in avg_results]
std_improvements_vs_regular = [r['std_improvement_vs_regular'] for r in avg_results]
std_improvements_vs_mixed = [r['std_improvement_vs_mixed'] for r in avg_results]
std_improvements_vs_baseline = [r['std_improvement_vs_baseline'] for r in avg_results]

plt.plot(hashrates, improvements_vs_regular, 'g-', linewidth=2, label='Strategic vs Regular (Others Strategic)')
plt.fill_between(hashrates, 
                 [i - s for i, s in zip(improvements_vs_regular, std_improvements_vs_regular)],
                 [i + s for i, s in zip(improvements_vs_regular, std_improvements_vs_regular)],
                 alpha=0.3, color='green', label='±1 Std Dev (vs Regular)')

plt.plot(hashrates, improvements_vs_mixed, 'orange', linewidth=2, label='Strategic vs Mixed (Others Regular)')
plt.fill_between(hashrates, 
                 [i - s for i, s in zip(improvements_vs_mixed, std_improvements_vs_mixed)],
                 [i + s for i, s in zip(improvements_vs_mixed, std_improvements_vs_mixed)],
                 alpha=0.3, color='orange', label='±1 Std Dev (vs Mixed)')

plt.plot(hashrates, improvements_vs_baseline, 'purple', linewidth=2, label='Strategic vs Baseline (Others Regular)')
plt.fill_between(hashrates, 
                 [i - s for i, s in zip(improvements_vs_baseline, std_improvements_vs_baseline)],
                 [i + s for i, s in zip(improvements_vs_baseline, std_improvements_vs_baseline)],
                 alpha=0.3, color='purple', label='±1 Std Dev (vs Baseline)')

plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='No Improvement')
plt.xlabel('Test Pool Hashrate')
plt.ylabel('Improvement (%)')
plt.title(f'Single Pool Strategy Comparison: Strategic vs Other Strategies\n({num_runs} runs, {num_total_pools} total pools)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('results/single_pool_self_comparison_improvement.png', dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved to: results/single_pool_self_comparison_improvement.png")

# Print summary
print(f"\n=== SUMMARY ({num_runs} runs) ===")
print("Hashrate | Regular | Strategic | Mixed | Baseline | Imp vs Reg | Imp vs Mix | Imp vs Base")
print("-" * 100)
for r in avg_results:
    print(f"{r['hashrate']:.2f}     | {r['avg_regular_reward']:7.2f} | {r['avg_strategic_reward']:9.2f} | {r['avg_mixed_reward']:5.2f} | {r['avg_baseline_reward']:8.2f} | {r['avg_improvement_vs_regular']:+.2f}%      | {r['avg_improvement_vs_mixed']:+.2f}%      | {r['avg_improvement_vs_baseline']:+.2f}%")

print(f"\nKey insights:")
print(f"- Test pool {test_pool_id} is compared across four scenarios:")
print(f"  1. Test pool: Regular, Others: Strategic")
print(f"  2. Test pool: Strategic, Others: Strategic") 
print(f"  3. Test pool: Strategic, Others: Regular")
print(f"  4. Test pool: Regular, Others: Regular (baseline)")
print(f"- Shows the strategic advantage of being strategic in different environments")
print(f"- Positive improvement = strategic strategy is better")
print(f"- Negative improvement = regular strategy is better")

# New plot: Total reward share comparison for first miner
def plot_first_miner_reward_share():
    """Plot the total reward share of the first miner in strategic vs regular scenarios"""
    print("\n=== CREATING FIRST MINER REWARD SHARE PLOT ===")
    
    # Calculate reward shares for each hashrate
    regular_shares = []
    strategic_shares = []
    mixed_shares = []
    baseline_shares = []
    
    for r in avg_results:
        hashrate = r['hashrate']
        regular_reward = r['avg_regular_reward']
        strategic_reward = r['avg_strategic_reward']
        mixed_reward = r['avg_mixed_reward']
        baseline_reward = r['avg_baseline_reward']
        
        # Handle 0 hashrate case
        if hashrate == 0:
            regular_shares.append(0)
            strategic_shares.append(0)
            mixed_shares.append(0)
            baseline_shares.append(0)
            continue
        
        # Run a single simulation to get total system rewards for this hashrate
        _, regular_reward_single, strategic_reward_single, mixed_reward_single, baseline_reward_single, _, _, _, _, _, _, _, total_regular_system, total_strategic_system, total_mixed_system, total_baseline_system = run_hashrate_sweep_single_run(hashrate)
        
        # Calculate reward shares as percentages
        regular_share = (regular_reward / total_regular_system * 100) if total_regular_system > 0 else 0
        strategic_share = (strategic_reward / total_strategic_system * 100) if total_strategic_system > 0 else 0
        mixed_share = (mixed_reward / total_mixed_system * 100) if total_mixed_system > 0 else 0
        baseline_share = (baseline_reward / total_baseline_system * 100) if total_baseline_system > 0 else 0
        
        regular_shares.append(regular_share)
        strategic_shares.append(strategic_share)
        mixed_shares.append(mixed_share)
        baseline_shares.append(baseline_share)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(hashrates, regular_shares, 'b-', linewidth=2, label='Test Pool: Regular, Others: Strategic')
    plt.plot(hashrates, strategic_shares, 'r-', linewidth=2, label='Test Pool: Strategic, Others: Strategic')
    plt.plot(hashrates, mixed_shares, 'orange', linewidth=2, label='Test Pool: Strategic, Others: Regular')
    plt.plot(hashrates, baseline_shares, 'g-', linewidth=2, label='Test Pool: Regular, Others: Regular')
    
    # Add a reference line for expected share based on hashrate (linear relationship)
    expected_shares = [h * 100 for h in hashrates]
    plt.plot(hashrates, expected_shares, 'k--', linewidth=2, alpha=0.7, label='Expected Share (Hashrate %)')
    
    plt.xlabel('Test Pool Hashrate')
    plt.ylabel('Total Reward Share (%)')
    plt.title(f'Test Pool Reward Share: Strategy Comparison\n({num_runs} runs, {num_total_pools} total pools)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/test_pool_reward_share_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved to: results/test_pool_reward_share_comparison.png")
    
    # Print comparison table
    print(f"\n=== TEST POOL REWARD SHARE COMPARISON ===")
    print("Hashrate | Regular | Strategic | Mixed | Baseline | Expected")
    print("-" * 75)
    for i, h in enumerate(hashrates):
        regular_share = regular_shares[i]
        strategic_share = strategic_shares[i]
        mixed_share = mixed_shares[i]
        baseline_share = baseline_shares[i]
        expected = h * 100
        print(f"{h:.2f}     | {regular_share:7.2f}% | {strategic_share:9.2f}% | {mixed_share:5.2f}% | {baseline_share:8.2f}% | {expected:.1f}%")

# Run the new plot
plot_first_miner_reward_share()

# New plot: Difference from baseline (regular/regular)
def plot_difference_from_baseline():
    """Plot the difference of each strategy from the baseline (regular/regular)"""
    print("\n=== CREATING DIFFERENCE FROM BASELINE PLOT ===")
    
    # Calculate differences from baseline for each hashrate
    regular_vs_baseline_diff = []
    strategic_vs_baseline_diff = []
    mixed_vs_baseline_diff = []
    
    for r in avg_results:
        hashrate = r['hashrate']
        regular_reward = r['avg_regular_reward']
        strategic_reward = r['avg_strategic_reward']
        mixed_reward = r['avg_mixed_reward']
        baseline_reward = r['avg_baseline_reward']
        
        # Handle 0 hashrate case
        if hashrate == 0:
            regular_vs_baseline_diff.append(0)
            strategic_vs_baseline_diff.append(0)
            mixed_vs_baseline_diff.append(0)
            continue
        
        # Calculate differences from baseline
        regular_diff = regular_reward - baseline_reward
        strategic_diff = strategic_reward - baseline_reward
        mixed_diff = mixed_reward - baseline_reward
        
        regular_vs_baseline_diff.append(regular_diff)
        strategic_vs_baseline_diff.append(strategic_diff)
        mixed_vs_baseline_diff.append(mixed_diff)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(hashrates, regular_vs_baseline_diff, 'b-', linewidth=2, label='Regular vs Baseline (Others Strategic)')
    plt.plot(hashrates, strategic_vs_baseline_diff, 'r-', linewidth=2, label='Strategic vs Baseline (Others Strategic)')
    plt.plot(hashrates, mixed_vs_baseline_diff, 'orange', linewidth=2, label='Strategic vs Baseline (Others Regular)')
    
    # Add a reference line at zero
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.7, label='Baseline (Regular/Regular)')
    
    plt.xlabel('Test Pool Hashrate')
    plt.ylabel('Reward Difference from Baseline')
    plt.title(f'Reward Difference from Baseline (Regular/Regular)\n({num_runs} runs, {num_total_pools} total pools)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/difference_from_baseline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved to: results/difference_from_baseline.png")
    
    # Print comparison table
    print(f"\n=== REWARD DIFFERENCE FROM BASELINE ===")
    print("Hashrate | Regular-Base | Strategic-Base | Mixed-Base | Baseline")
    print("-" * 70)
    for i, h in enumerate(hashrates):
        regular_diff = regular_vs_baseline_diff[i]
        strategic_diff = strategic_vs_baseline_diff[i]
        mixed_diff = mixed_vs_baseline_diff[i]
        baseline = avg_results[i]['avg_baseline_reward']
        print(f"{h:.2f}     | {regular_diff:+8.2f}    | {strategic_diff:+11.2f}    | {mixed_diff:+8.2f}   | {baseline:.2f}")

# Run the new baseline difference plot
plot_difference_from_baseline() 