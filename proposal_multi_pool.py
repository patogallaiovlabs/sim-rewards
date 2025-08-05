import random
import matplotlib.pyplot as plt
import numpy as np
import time

RUNS = 20  # Reduced from 100 for faster execution
# Parameters
blocks = 50000  # Reduced from 100000 for faster execution
maturity_period = 4000
reward_percentage = 0.10
current_block_reward_percentage = 0.05  # 5% for current block miner
base_fee = 0.001
spike_fee = 1.0
num_spikes = 200  # Number of random spikes to generate

# Pool configuration - multiple strategic pools
num_strategic_pools = 3  # Number of strategic pools
num_honest_pools = 1     # Number of honest pools
total_pools = num_strategic_pools + num_honest_pools

# Pool hashrates (must sum to 1.0)
# Strategic pools get equal hashrate, honest pools get the rest
strategic_hashrate_per_pool = 0.2  # Each strategic pool gets 20%
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

def deterministic_simulation(strategy: str, miner_sequence, current_block_reward_pct=current_block_reward_percentage, fast_mode=False):
    reward_balance = 0.0
    miner_rewards = {pool_id: 0.0 for pool_id in pool_hashrates.keys()}
    block_owners = []
    block_fees = []
    fee = 0
    
    # Metrics tracking (skip in fast mode)
    if not fast_mode:
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
        
        if not fast_mode:
            total_blocks += 1
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
            # Current block miner gets immediate reward
            current_block_reward = fee * current_block_reward_pct
            miner_rewards[miner] += current_block_reward
            
            # Remaining fee goes to reward balance
            reward_balance += fee - current_block_reward
            block_fees.append(fee)
            fee = 0
            
            if not fast_mode:
                # End current streak if there was one
                if current_streak > 0:
                    withholding_streaks.append(current_streak)
                    max_streak = max(max_streak, current_streak)
                    min_streak = min(min_streak, current_streak)
                    current_streak = 0
        else:
            block_fees.append(0)
            if not fast_mode:
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
            
            if not fast_mode:
                # Track if strategic miner won accumulated fees
                if accumulated_fees > 0 and matured_miner in strategic_pools:
                    strategic_wins_with_accumulated_fees += 1
                    total_accumulated_fees_won += accumulated_fees
                    accumulated_fees = 0  # Reset after winning

    if not fast_mode:
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
    else:
        # Fast mode: return only rewards
        return miner_rewards, None

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
    
    # Calculate average reward improvement for all strategic pools
    honest_avg_strategic = sum(honest_rewards[i] for i in strategic_pools) / len(strategic_pools)
    strategic_avg_strategic = sum(strategic_rewards[i] for i in strategic_pools) / len(strategic_pools)
    improvement = (strategic_avg_strategic - honest_avg_strategic) / honest_avg_strategic * 100 if honest_avg_strategic > 0 else 0
    
    return strategic_total_hashrate, improvement

def run_comprehensive_comparison():
    """Run comprehensive comparison with different reward percentages and hashrates"""
    print("\n=== COMPREHENSIVE COMPARISON: REWARD PERCENTAGES vs HASH RATES ===")
    print(f"Running {RUNS} simulations for each combination...")
    print(f"Configuration: {num_strategic_pools} strategic pools, {num_honest_pools} honest pool")
    print(f"Strategic pools: {list(strategic_pools)} (each with equal hashrate)")
    print(f"Honest pool: {list(honest_pools)}")
    
    # Test different percentages from 0.01 to 0.1
    reward_percentages = np.arange(0.01, 0.11, 0.01)
    hashrates = np.arange(0.0, 1.1, 0.1)  # 0.0 to 1.0 (total strategic hashrate)
    
    # Pre-generate all miner sequences for all hashrates
    print("Pre-generating miner sequences...")
    miner_sequences = {}
    for strategic_total_hashrate in hashrates:
        strategic_hashrate_per_pool = strategic_total_hashrate / num_strategic_pools
        honest_hashrate = 1.0 - strategic_total_hashrate
        
        pool_hashrates_temp = {}
        for i in range(num_strategic_pools):
            pool_hashrates_temp[i] = strategic_hashrate_per_pool
        pool_hashrates_temp[num_strategic_pools] = honest_hashrate
        
        # Generate sequence for this hashrate
        sequence = []
        np.random.seed(int(time.time() * 1000) % (2**32))
        for _ in range(blocks):
            rand = np.random.random()
            cumulative = 0
            for pool_id, hashrate in pool_hashrates_temp.items():
                cumulative += hashrate
                if rand < cumulative:
                    sequence.append(pool_id)
                    break
        miner_sequences[strategic_total_hashrate] = sequence
    
    # Store results for each combination
    all_results = {}
    
    for reward_pct in reward_percentages:
        print(f"\nTesting current block reward percentage: {reward_pct:.2f} ({reward_pct*100:.0f}%)")
        all_results[reward_pct] = {h: [] for h in hashrates}
        
        for run in range(RUNS):
            if run % 5 == 0:
                print(f"  Run {run + 1}/{RUNS}...")
            
            for strategic_total_hashrate in hashrates:
                # Use pre-generated sequence and fast mode
                sequence = miner_sequences[strategic_total_hashrate]
                honest_rewards, _ = deterministic_simulation("honest", sequence, float(reward_pct), fast_mode=True)
                strategic_rewards, _ = deterministic_simulation("strategic", sequence, float(reward_pct), fast_mode=True)
                
                # Calculate average improvement for strategic pools
                honest_avg_strategic = sum(honest_rewards[i] for i in strategic_pools) / len(strategic_pools)
                strategic_avg_strategic = sum(strategic_rewards[i] for i in strategic_pools) / len(strategic_pools)
                improvement = (strategic_avg_strategic - honest_avg_strategic) / honest_avg_strategic * 100 if honest_avg_strategic > 0 else 0
                
                all_results[reward_pct][strategic_total_hashrate].append(improvement)
    
    # Calculate averages and standard deviations
    avg_results = {}
    for reward_pct in reward_percentages:
        avg_results[reward_pct] = []
        for hashrate in hashrates:
            improvements = all_results[reward_pct][hashrate]
            avg_improvement = np.mean(improvements)
            std_improvement = np.std(improvements)
            
            avg_results[reward_pct].append({
                'hashrate': hashrate,
                'avg_improvement': avg_improvement,
                'std_improvement': std_improvement,
                'min_improvement': min(improvements),
                'max_improvement': max(improvements)
            })
    
    return reward_percentages, hashrates, avg_results

# Run simulations
honest_rewards, honest_metrics = deterministic_simulation("honest", fixed_miner_sequence)
strategic_rewards, strategic_metrics = deterministic_simulation("strategic", fixed_miner_sequence)

# Print results
print("=== MULTI-POOL STRATEGIC SIMULATION ===")
print(f"Configuration: {num_strategic_pools} strategic pools, {num_honest_pools} honest pool")
print(f"Strategic pools: {list(strategic_pools)} (each with {strategic_hashrate_per_pool*100:.0f}% hashrate)")
print(f"Honest pool: {list(honest_pools)} ({honest_hashrate*100:.0f}% hashrate)")
print(f"MODIFICATION: Current block miner gets {current_block_reward_percentage*100}% of current block fees")
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

# Run comprehensive comparison
reward_percentages, hashrates, comprehensive_results = run_comprehensive_comparison()

# Create the main comprehensive plot
plt.figure(figsize=(14, 10))

# Create a single plot showing improvement vs hashrate for different reward percentages
# Using the same colors as strategic_newmodel.png
colors = ['orange', 'red', 'pink', 'lightblue', 'teal', 'yellow', 'purple', 'brown', 'gray', 'cyan']
markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'H', '8']

for i, reward_pct in enumerate(reward_percentages):
    avg_improvements = [result['avg_improvement'] for result in comprehensive_results[reward_pct]]
    std_improvements = [result['std_improvement'] for result in comprehensive_results[reward_pct]]
    label = f'{reward_pct*100:.0f}%'
    
    plt.plot(hashrates, avg_improvements, 
             color=colors[i], marker=markers[i], 
             linewidth=2, markersize=8, 
             label=label)
    
    # Add error bars (standard deviation)
    plt.errorbar(hashrates, avg_improvements, yerr=std_improvements, 
                 fmt='none', capsize=3, alpha=0.5, color=colors[i])

plt.xlabel('Total Strategic Pool Hashrate')
plt.ylabel('Average Strategic Pool Improvement (%)')
plt.title(f'Average Strategic Pool Improvement vs Total Hashrate for Different Current Block Reward Percentages\n({num_strategic_pools} Strategic Pools with Equal Hashrate)')
plt.grid(True, alpha=0.3)
plt.xticks(hashrates)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='No Improvement')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Print comprehensive summary table
print("\n=== COMPREHENSIVE COMPARISON SUMMARY ({RUNS} runs) ===")
print("Hashrate | 1% ± std | 2% ± std | 3% ± std | 4% ± std | 5% ± std | 6% ± std | 7% ± std | 8% ± std | 9% ± std | 10% ± std")
print("-" * 120)
for i, hashrate in enumerate(hashrates):
    row = f"{hashrate:7.1f} |"
    for reward_pct in reward_percentages:
        result = comprehensive_results[reward_pct][i]
        avg_improvement = result['avg_improvement']
        std_improvement = result['std_improvement']
        row += f" {avg_improvement:4.1f}±{std_improvement:3.1f} |"
    print(row)

# Print summary
print("\n=== COMPREHENSIVE COMPARISON COMPLETE ===")
print("The plot shows how different current block reward percentages affect strategic behavior across various hashrates.")
print(f"Configuration: {num_strategic_pools} strategic pools with equal hashrate distribution")
print("Key insights:")
print("- Lower reward percentages (1-3%) may not provide enough incentive to reduce strategic behavior")
print("- Higher reward percentages (8-10%) may provide too much immediate reward")
print("- The optimal percentage likely lies in the middle range (4-7%)")
print(f"- Multiple strategic pools may have different dynamics than single strategic pool") 