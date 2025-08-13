from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Miner:
    """
    Represents a miner.

    Attributes:
        id: Stable numeric identifier.
        name: Human-friendly name.
        hashrate: Share of total network hashpower (fraction in [0,1], sum across miners should be 1).
        sibling_propensity: Fraction in [0,1] indicating the miner's *tendency* to produce siblings
            rather than main-chain winners. This does not force exact counts per miner, but sets clear expectations.
            Modeling note:
            - Main block selection uses weights: hashrate * (1 - sibling_propensity)
            - Sibling selection uses weights: hashrate * sibling_propensity
    """
    id: int
    name: str
    hashrate: float
    sibling_propensity: float


@dataclass(frozen=True)
class Block:
    """
    A block height with exactly one main-chain block and zero or more siblings.
    """
    height: int
    main_miner: int
    sibling_miners: List[int]


class ChainSimulator:
    """
    Chain simulator that generates heights with one main block and a Poisson-distributed
    number of siblings per height.

    Design choices:
    - Counts-per-height must be integers; using a Poisson(λ) is the natural discrete
      counterpart when "arrivals" (siblings) are driven by an exponential interarrival process.
      If you pass an 'exponential mean', set mean_siblings to that value; we interpret it
      as the Poisson rate λ (i.e., expected siblings per height).
    - Two independent "propensities":
        (a) Hashrate controls *how often* a miner participates.
        (b) Sibling propensity controls *what role* that participation tends to take.
      Implementation: main miner is sampled from weights ∝ hashrate*(1 - p_sib);
      siblings are sampled from weights ∝ hashrate*p_sib.
      This guarantees exactly one main per height and aligns expected main/sibling shares with
      per-miner propensities without forcing exact quotas.
    """
    def __init__(
        self,
        miners: Sequence[Miner],
        mean_siblings: float,
        sibling_model: str = "poisson",
        seed: Optional[int] = None,
        allow_self_sibling: bool = False,
    ) -> None:
        if mean_siblings < 0:
            raise ValueError("mean_siblings must be non-negative.")
        if sibling_model not in {"poisson", "exp_round"}:
            raise ValueError("sibling_model must be 'poisson' or 'exp_round'.")
        self.miners: List[Miner] = list(miners)
        self.mean_siblings = float(mean_siblings)
        self.sibling_model = sibling_model
        self.rng = np.random.default_rng(seed)
        self.allow_self_sibling = bool(allow_self_sibling)

        self._validate_inputs()

    # ------------------------------- Public API -------------------------------

    def generate_chain(self, num_blocks: int) -> List[Block]:
        """
        Generate a chain with `num_blocks` heights.
        Each height has one main miner and K ~ Poisson(mean_siblings) siblings (or exp-rounded).

        Returns a list of Block entries, one per height (0..num_blocks-1).
        """
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")

        chain: List[Block] = []
        for h in range(num_blocks):
            main_idx = self._choose_main_miner()
            k_siblings = self._sample_sibling_count()
            sibling_miners = self._choose_sibling_miners(k_siblings, main_idx)
            chain.append(Block(height=h, main_miner=main_idx, sibling_miners=sibling_miners))
        return chain

    def summarize_chain(self, chain: Sequence[Block]) -> Dict[str, Dict[int, int]]:
        """
        Summarize per-miner counts: total participations, mains, siblings.
        """
        mains = {m.id: 0 for m in self.miners}
        siblings = {m.id: 0 for m in self.miners}
        for b in chain:
            mains[b.main_miner] += 1
            for s in b.sibling_miners:
                siblings[s] += 1
        totals = {i: mains[i] + siblings[i] for i in mains}
        return {"mains": mains, "siblings": siblings, "totals": totals}

    # Placeholders for reward models you will implement later.
    # Keep signatures stable to plug in reward logic.
    def compute_rewards_current(self, chain: Sequence[Block]) -> Dict[int, float]:
        """
        Current reward model: block reward is split equally among all participants
        (main miner + siblings). This represents the current system where fees are
        distributed among all miners who participated in that block height.
        
        Returns: mapping miner_id -> reward
        """
        rewards = {m.id: 0.0 for m in self.miners}
        
        for block in chain:
            # Count total participants (main + siblings)
            total_participants = 1 + len(block.sibling_miners)
            
            if total_participants > 0:
                # Split the full reward (1.0) equally among all participants
                reward_per_participant = 1.0 / total_participants
                
                # Main miner gets their share
                rewards[block.main_miner] += reward_per_participant
                
                # Siblings get their share
                for sibling_id in block.sibling_miners:
                    rewards[sibling_id] += reward_per_participant
                    
        return rewards

    def compute_rewards_fixed_then_split(
        self, chain: Sequence[Block], fixed_amount: float
    ) -> Dict[int, float]:
        """
        Fixed-then-split reward model: 
        1. Main miner gets a fixed amount per block
        2. Remaining reward (1.0 - fixed_amount) is split equally among all participants
           (main miner + siblings) for that block
        
        Args:
            chain: The blockchain to analyze
            fixed_amount: Fixed reward given to main miner per block (must be <= 1.0)
            
        Returns: mapping miner_id -> reward
        """
        if fixed_amount > 1.0:
            raise ValueError("fixed_amount cannot exceed 1.0")
        if fixed_amount < 0.0:
            raise ValueError("fixed_amount cannot be negative")
            
        rewards = {m.id: 0.0 for m in self.miners}
        
        for block in chain:
            # Main miner gets the fixed amount
            rewards[block.main_miner] += fixed_amount
            
            # Calculate remaining reward to split
            remaining_reward = 1.0 - fixed_amount
            
            # Count total participants (main + siblings)
            total_participants = 1 + len(block.sibling_miners)
            
            if total_participants > 0 and remaining_reward > 0:
                # Split remaining reward equally among all participants
                reward_per_participant = remaining_reward / total_participants
                
                # Main miner gets additional share
                rewards[block.main_miner] += reward_per_participant
                
                # Siblings get their share
                for sibling_id in block.sibling_miners:
                    rewards[sibling_id] += reward_per_participant
                    
        return rewards

    # ------------------------------ Model Internals ---------------------------

    def _validate_inputs(self) -> None:
        if not self.miners:
            raise ValueError("At least one miner is required.")
        if not all(0.0 <= m.hashrate <= 1.0 for m in self.miners):
            raise ValueError("All miner hashrates must be in [0,1].")
        if not np.isclose(sum(m.hashrate for m in self.miners), 1.0, atol=1e-9):
            raise ValueError("Miner hashrates must sum to 1.0.")
        if not all(0.0 <= m.sibling_propensity <= 1.0 for m in self.miners):
            raise ValueError("All sibling_propensity values must be in [0,1].")

    def _sample_sibling_count(self) -> int:
        if self.sibling_model == "poisson":
            # Natural integer-valued count with mean self.mean_siblings
            return int(self.rng.poisson(self.mean_siblings))
        # If you insist on an exponential-based draw, we round it to an int.
        # Note: this will *not* have mean equal to mean_siblings after rounding.
        x = float(self.rng.exponential(self.mean_siblings))
        return int(max(0, int(np.round(x))))

    def _choose_main_miner(self) -> int:
        weights = np.array([m.hashrate * (1.0 - m.sibling_propensity) for m in self.miners], dtype=float)
        s = weights.sum()
        if s <= 0.0:
            # Fallback: if all sibling_propensity==1, we cannot draw a main from these weights.
            # In that degenerate case, fall back to pure hashrate.
            weights = np.array([m.hashrate for m in self.miners], dtype=float)
            s = weights.sum()
        weights /= s
        return int(self.rng.choice(len(self.miners), p=weights))

    def _choose_sibling_miners(self, n: int, main_idx: int) -> List[int]:
        if n <= 0:
            return []
        weights = np.array([m.hashrate * m.sibling_propensity for m in self.miners], dtype=float)
        if not self.allow_self_sibling:
            weights[main_idx] = 0.0
        s = weights.sum()
        if s <= 0.0:
            # Fallback: if no sibling weight (e.g., all propensities=0), nobody makes siblings.
            return []
        weights /= s
        # Sample with replacement; duplicates allowed (a miner may produce multiple siblings at a height).
        return self.rng.choice(len(self.miners), size=int(n), p=weights, replace=True).tolist()


# ------------------------------- Helper Builders -------------------------------

def make_equal_miners(
    count: int,
    sibling_propensity: Union[float, Sequence[float]] = 0.5,
    prefix: str = "miner",
) -> List[Miner]:
    """
    Convenience to create `count` miners with equal hashpower (1/count each).

    sibling_propensity can be:
        - a scalar applied to all miners, or
        - a sequence of length `count` with per-miner values in [0,1].
    """
    if count <= 0:
        raise ValueError("count must be > 0")
    base_rate = 1.0 / float(count)

    if isinstance(sibling_propensity, (list, tuple, np.ndarray)):
        if len(sibling_propensity) != count:
            raise ValueError("sibling_propensity sequence must match miner count.")
        props = list(map(float, sibling_propensity))
    else:
        props = [float(sibling_propensity)] * count

    miners = [
        Miner(id=i, name=f"{prefix}_{i}", hashrate=base_rate, sibling_propensity=props[i])
        for i in range(count)
    ]
    return miners


# ------------------------------- Example usage --------------------------------

def example_usage() -> Tuple[List[Block], Dict[str, Dict[int, int]], ChainSimulator]:
    """
    Example that you can run to sanity-check behavior and reproducibility.
    """
    miners = make_equal_miners(count=7, sibling_propensity=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
    sim = ChainSimulator(
        miners=miners,
        mean_siblings=0.6,       # expected siblings per height
        sibling_model="poisson", # or "exp_round" if you insist on exponential-based rounding
        seed=42,
        allow_self_sibling=False
    )
    chain = sim.generate_chain(num_blocks=10000)
    summary = sim.summarize_chain(chain)
    return chain, summary, sim


def analyze_sibling_distribution(chain: List[Block]) -> None:
    """
    Analyze and plot the distribution of siblings per block.
    """
    import os
    
    sibling_counts = [len(block.sibling_miners) for block in chain]
    mean_siblings = np.mean(sibling_counts)
    
    print(f"\n=== SIBLING DISTRIBUTION ANALYSIS ===")
    print(f"Mean siblings per block: {mean_siblings:.3f}")
    print(f"Min siblings: {min(sibling_counts)}")
    print(f"Max siblings: {max(sibling_counts)}")
    print(f"Standard deviation: {np.std(sibling_counts):.3f}")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(sibling_counts, bins=range(min(sibling_counts), max(sibling_counts) + 2), 
             alpha=0.7, edgecolor='black', align='left')
    plt.axvline(mean_siblings, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_siblings:.3f}')
    plt.xlabel('Number of Siblings per Block')
    plt.ylabel('Frequency')
    plt.title('Distribution of Siblings per Block')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to results folder
    results_dir = "../results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    plot_filename = os.path.join(results_dir, "sibling_distribution.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"Sibling distribution plot saved to: {plot_filename}")





if __name__ == "__main__":
    # Minimal smoke test
    chain, summary, sim = example_usage()
    # Print a brief summary and first few blocks for visual inspection
    # Unified table with all information
    print("\n=== UNIFIED MINER ANALYSIS TABLE ===")
    print(f"{'Miner ID':<8} {'Name':<12} {'Mains':<8} {'Siblings':<10} {'Total':<8} {'Sib%':<6} {'Current':<10} {'Fixed50%':<10} {'Fixed30%':<10} {'Fixed10%':<10}")
    print("-" * 108)
    
    # Compute rewards for all models
    current_rewards = sim.compute_rewards_current(chain)
    fixed_50_rewards = sim.compute_rewards_fixed_then_split(chain, fixed_amount=0.5)
    fixed_30_rewards = sim.compute_rewards_fixed_then_split(chain, fixed_amount=0.3)
    fixed_10_rewards = sim.compute_rewards_fixed_then_split(chain, fixed_amount=0.1)
    
    for miner in sim.miners:
        miner_id = miner.id
        name = miner.name
        mains = summary["mains"][miner_id]
        siblings = summary["siblings"][miner_id]
        total = summary["totals"][miner_id]
        
        # Calculate sibling percentage
        sibling_percentage = (siblings / total * 100) if total > 0 else 0.0
        
        current = current_rewards[miner_id]
        fixed_50 = fixed_50_rewards[miner_id]
        fixed_30 = fixed_30_rewards[miner_id]
        fixed_10 = fixed_10_rewards[miner_id]
        
        print(f"{miner_id:<8} {name:<12} {mains:<8} {siblings:<10} {total:<8} {sibling_percentage:<6.1f} {current:<10.3f} {fixed_50:<10.3f} {fixed_30:<10.3f} {fixed_10:<10.3f}")
    
    # Summary statistics
    total_current = sum(current_rewards.values())
    total_fixed_50 = sum(fixed_50_rewards.values())
    total_fixed_30 = sum(fixed_30_rewards.values())
    total_fixed_10 = sum(fixed_10_rewards.values())
    
    print(f"\nTotal rewards distributed:")
    print(f"Current model: {total_current:.3f}")
    print(f"Fixed50% model: {total_fixed_50:.3f}")
    print(f"Fixed30% model: {total_fixed_30:.3f}")
    print(f"Fixed10% model: {total_fixed_10:.3f}")
    
    # Calculate Gini coefficient for inequality comparison
    def gini_coefficient(values):
        if len(values) == 0:
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0
    
    gini_current = gini_coefficient(list(current_rewards.values()))
    gini_fixed_50 = gini_coefficient(list(fixed_50_rewards.values()))
    gini_fixed_30 = gini_coefficient(list(fixed_30_rewards.values()))
    gini_fixed_10 = gini_coefficient(list(fixed_10_rewards.values()))
    
    print(f"\nInequality (Gini coefficient):")
    print(f"Current model: {gini_current:.3f}")
    print(f"Fixed50% model: {gini_fixed_50:.3f}")
    print(f"Fixed30% model: {gini_fixed_30:.3f}")
    print(f"Fixed10% model: {gini_fixed_10:.3f}")
    
    print(f"\nModel Characteristics:")
    print(f"Current model: Equal distribution among all participants")
    print(f"Fixed50% model: Main miner gets 50% + equal split of remaining 50%")
    print(f"Fixed30% model: Main miner gets 30% + equal split of remaining 70%")
    print(f"Fixed10% model: Main miner gets 10% + equal split of remaining 90%")
    
    print(f"\n=== FIRST 5 HEIGHTS ===")
    for b in chain[:5]:
        print(f"h={b.height} main={b.main_miner} siblings={b.sibling_miners}")
    
    # Analyze and plot sibling distribution
    analyze_sibling_distribution(chain)
