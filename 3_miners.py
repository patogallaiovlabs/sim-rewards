import random
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from mpl_toolkits.mplot3d import Axes3D

# Parameters
blocks = 50000
maturity_period = 4000
reward_percentage = 0.10
base_fee = 0.001
spike_fee = 1.0
num_spikes = 200
num_runs = 10

# Pool configuration - three pools to test
num_total_pools = 3
test_pool_1_id = 0
test_pool_2_id = 1
other_pool_id = 2

random.seed(int(time.time()))
spike_positions = set(random.sample(range(blocks), num_spikes))

def setup_pools(pool1_hashrate, pool2_hashrate, pool1_strategy="regular", pool2_strategy="regular", other_strategy="strategic"):
    """Setup pool configuration with two test pools and one other pool"""
    pool3_hashrate = 1.0 - pool1_hashrate - pool2_hashrate
    
    pool_hashrates = {
        test_pool_1_id: pool1_hashrate,
        test_pool_2_id: pool2_hashrate,
        other_pool_id: pool3_hashrate
    }
    
    strategic_pools = set()
    if pool1_strategy == "strategic":
        strategic_pools.add(test_pool_1_id)
    if pool2_strategy == "strategic":
        strategic_pools.add(test_pool_2_id)
    if other_strategy == "strategic":
        strategic_pools.add(other_pool_id)
    
    return pool_hashrates, strategic_pools

def deterministic_simulation(strategy: str, miner_sequence, pool_hashrates, strategic_pools):
    reward_balance = 0.0
    miner_rewards = {pool_id: 0.0 for pool_id in pool_hashrates.keys()}
    block_owners = []
    block_fees = []
    fee = 0
    
    for block in range(blocks):
        miner = miner_sequence[block]
        block_owners.append(miner)
        
        fee += spike_fee if block in spike_positions else base_fee
        matured_block = block - maturity_period

        include_fee = False
        if strategy == "regular":
            include_fee = True
        elif strategy == "strategic":
            if miner in strategic_pools:
                if matured_block < 0 or (matured_block >= 0 and block_owners[matured_block] == miner):
                    include_fee = True
            else:
                include_fee = True

        if include_fee:
            reward_balance += fee
            block_fees.append(fee)
            fee = 0
        else:
            block_fees.append(0)

        if matured_block >= 0:
            matured_miner = block_owners[matured_block]
            reward = reward_balance * reward_percentage
            reward_balance -= reward
            miner_rewards[matured_miner] += reward * 0.8
    
    return miner_rewards

def run_single_simulation(pool1_hashrate, pool2_hashrate):
    """Run a single simulation for given hashrates"""
    if pool1_hashrate + pool2_hashrate > 1.0:
        return None
    
    # Generate miner sequence
    random.seed(int(time.time() * 1000) % (2**32))
    fixed_miner_sequence = []
    
    pool_hashrates_temp = {
        test_pool_1_id: pool1_hashrate,
        test_pool_2_id: pool2_hashrate,
        other_pool_id: 1.0 - pool1_hashrate - pool2_hashrate
    }
    
    for _ in range(blocks):
        rand = random.random()
        cumulative = 0
        for pool_id, hashrate in pool_hashrates_temp.items():
            cumulative += hashrate
            if rand < cumulative:
                fixed_miner_sequence.append(pool_id)
                break
    
    # Run baseline scenario (all regular) to get Miner C's reward
    pool_hashrates, strategic_pools = setup_pools(pool1_hashrate, pool2_hashrate, "regular", "regular", "regular")
    strategy = "regular"
    rewards = deterministic_simulation(strategy, fixed_miner_sequence, pool_hashrates, strategic_pools)
    
    # Extract rewards for all three miners
    miner_a_reward = rewards[test_pool_1_id]
    miner_b_reward = rewards[test_pool_2_id]
    miner_c_reward = rewards[other_pool_id]
    
    return miner_a_reward, miner_b_reward, miner_c_reward

def run_3d_simulation():
    """Run 3D simulation across hashrate space"""
    print("=== THREE POOL 3D SIMULATION ===")
    
    # Create hashrate grid
    hashrate_step = 0.1
    hashrates = np.arange(0.0, 1.01, hashrate_step)
    
    # Initialize 3D arrays for results
    miner_c_rewards = np.zeros((len(hashrates), len(hashrates)))
    miner_c_reward_percentages = np.zeros((len(hashrates), len(hashrates)))
    
    valid_points = []
    
    print(f"Running simulations for {len(hashrates)}x{len(hashrates)} grid...")
    
    for i, h1 in enumerate(hashrates):
        for j, h2 in enumerate(hashrates):
            if h1 + h2 <= 1.0:  # Valid hashrate combination
                print(f"Processing hashrates: Miner A={h1:.1f}, Miner B={h2:.1f}, Miner C={1.0-h1-h2:.1f}")
                
                # Run multiple simulations for this point
                all_results = []
                for run in range(num_runs):
                    results = run_single_simulation(h1, h2)
                    if results is not None:  # Valid result
                        all_results.append(results)
                
                if all_results:
                    # Calculate averages across all runs
                    all_results_array = np.array(all_results)  # Shape: (num_runs, 3)
                    avg_results = np.mean(all_results_array, axis=0)  # Shape: (3,)
                    
                    # Extract average rewards for each miner
                    miner_a_reward = avg_results[0]
                    miner_b_reward = avg_results[1]
                    miner_c_reward = avg_results[2]
                    
                    # Calculate total system reward
                    total_system_reward = miner_a_reward + miner_b_reward + miner_c_reward
                    
                    # Calculate Miner C's reward percentage
                    miner_c_percentage = (miner_c_reward / total_system_reward * 100) if total_system_reward > 0 else 0
                    
                    # Store results
                    miner_c_rewards[i, j] = miner_c_reward
                    miner_c_reward_percentages[i, j] = miner_c_percentage
                    
                    valid_points.append((h1, h2, miner_c_percentage))
    
    return hashrates, miner_c_rewards, miner_c_reward_percentages, valid_points

def create_3d_plots(hashrates, miner_c_rewards, miner_c_reward_percentages, valid_points):
    """Create 3D plots of the results"""
    os.makedirs('results', exist_ok=True)
    
    # Create meshgrid
    X, Y = np.meshgrid(hashrates, hashrates)
    
    # Plot 1: Miner C Reward Percentages (3D Surface)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, miner_c_reward_percentages.T, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Miner A Hashrate')
    ax.set_ylabel('Miner B Hashrate')
    ax.set_zlabel('Miner C Reward Percentage (%)')
    ax.set_title('Miner C Reward Percentage vs Hashrate Distribution')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig('results/three_pool_3d_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("3D plot saved to: results/three_pool_3d_comparison.png")
    
    # Create contour plot
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, miner_c_reward_percentages.T, levels=20, cmap='viridis')
    ax.set_xlabel('Miner A Hashrate')
    ax.set_ylabel('Miner B Hashrate')
    ax.set_title('Miner C Reward Percentage (%)')
    plt.colorbar(contour, ax=ax)
    plt.savefig('results/three_pool_contour.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Contour plot saved to: results/three_pool_contour.png")

# Run the simulation
if __name__ == "__main__":
    hashrates, miner_c_rewards, miner_c_reward_percentages, valid_points = run_3d_simulation()
    create_3d_plots(hashrates, miner_c_rewards, miner_c_reward_percentages, valid_points)
    
    # Print summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Valid hashrate combinations: {len(valid_points)}")
    print(f"Average Miner C reward percentage: {np.mean(miner_c_reward_percentages[miner_c_reward_percentages != 0]):.2f}%")
    print(f"Max Miner C reward percentage: {np.max(miner_c_reward_percentages):.2f}%")
    print(f"Min Miner C reward percentage: {np.min(miner_c_reward_percentages):.2f}%") 