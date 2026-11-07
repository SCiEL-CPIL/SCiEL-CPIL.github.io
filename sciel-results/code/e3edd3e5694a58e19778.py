"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0  # seconds before the 330s kill limit

def _time_budget_remaining():
    return (330.0 - _DEADLINE_MARGIN) - (time.time() - _START_TIME)

def _simulate_aco(demands, capacity, heuristic, n_ants=20, n_iterations=30):
    """
    Fast simulation of the ACO algorithm to evaluate a heuristic.
    Returns the fewest bins used by any ant in the final iteration.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Pheromone matrix
    pheromone = np.ones((n, n))
    
    best_bins_ever = float('inf')
    
    for _ in range(n_iterations):
        best_bins_this_iter = float('inf')
        
        for _ in range(n_ants):
            # State for this ant
            # We need to track which items are placed.
            # Using a boolean mask is efficient for n=120.
            placed = np.zeros(n, dtype=bool)
            
            bins_used = 0
            current_bin_sum = 0.0
            last_item = -1
            
            # 1. Open a bin, place a random fitting item
            # "random fitting item" - usually just random item since capacity is large enough for any single item
            # The problem says "demands are integers from 20 to 99... capacity is 150".
            # So any single item fits.
            idx = np.random.randint(n)
            placed[idx] = True
            current_bin_sum = d[idx]
            last_item = idx
            bins_used = 1
            
            while not np.all(placed):
                # Find fitting items
                # remaining_space = capacity - current_bin_sum
                # fits if d[j] <= remaining_space AND not placed[j]
                remaining_space = capacity - current_bin_sum
                fits_mask = (~placed) & (d <= remaining_space)
                
                if not np.any(fits_mask):
                    # Open new bin
                    bins_used += 1
                    # Place random fitting item in new bin
                    # All items fit in an empty bin
                    available = np.where(~placed)[0]
                    idx = available[np.random.randint(len(available))]
                    placed[idx] = True
                    current_bin_sum = d[idx]
                    last_item = idx
                    continue
                
                # Probabilistic choice
                # probability proportional to pheromone[last_item, j] * heuristic[last_item, j]
                # Only for j in fits_mask
                row_ph = pheromone[last_item, :]
                row_he = heuristic[last_item, :]
                
                scores = row_ph * row_he
                # Mask out non-fitting items
                scores = np.where(fits_mask, scores, 0.0)
                
                # Ensure no negative or zero issues (heuristic is positive, pheromone is positive)
                # If all scores are 0 (shouldn't happen if fits_mask is true and h>0), pick random
                total_score = np.sum(scores)
                if total_score == 0:
                    # Fallback to random
                    candidates = np.where(fits_mask)[0]
                    idx = candidates[np.random.randint(len(candidates))]
                else:
                    probs = scores / total_score
                    # Use multinomial or random choice with p
                    # np.random.choice with p is slow, use cumulative sum for speed?
                    # n=120 is small, np.random.choice is fine
                    idx = np.random.choice(n, p=probs)
                
                placed[idx] = True
                current_bin_sum += d[idx]
                last_item = idx
            
        if bins_used < best_bins_this_iter:
            best_bins_this_iter = bins_used
        
        if best_bins_this_iter < best_bins_ever:
            best_bins_ever = best_bins_this_iter
        
        # Update pheromone
        # "pheromone is multiplied by 0.9 and every entry gains 1/(fewest bins of that iteration)"
        pheromone *= 0.9
        pheromone += 1.0 / best_bins_this_iter
    
    return best_bins_ever

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    
    Strategy: 
    1. Define a parametric family of heuristics based on:
       - Complementarity: How close d[i] + d[j] is to capacity.
       - Residual Utility: How well the remaining space (capacity - d[i] - d[j]) fits the smallest remaining item.
         Since we don't know "remaining" exactly in a static heuristic, we approximate "small remaining space is good" 
         but also "remaining space that can fit small items is good".
         Actually, a common insight is that if d[i]+d[j] leaves a gap G, we want G to be either 0 or large enough to fit common items.
         A simpler proxy: Favor pairs where d[i]+d[j] is close to capacity (high complementarity).
       - Let's try: H[i,j] = (d[i] + d[j])^alpha * (1 + beta * (capacity - d[i] - d[j])^gamma)
         Or simpler: H[i,j] = (d[i] + d[j])^alpha if fits, else 1e-9.
         The seed used alpha=1.
    
    2. Use a grid search over alpha and potentially other parameters.
    3. Evaluate each candidate using the FULL ACO simulation (20 ants, 30 iters).
    4. Return the best heuristic.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Precompute pair sums
    pair_sums = d[:, None] + d[None, :]
    fits = pair_sums <= capacity
    
    # Candidate parameters
    # Let's explore alpha (exponent on sum) and maybe a "gap penalty" term.
    # H[i,j] = (pair_sum)^alpha if fits
    # Or H[i,j] = (pair_sum)^alpha * (1 / (1 + gap)) if fits
    # gap = capacity - pair_sum
    
    best_score = float('inf')
    best_heuristic = None
    
    # Grid of alphas
    alphas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    
    # Maybe a second dimension: "gap sensitivity"
    # H = (sum)^alpha * (1 / (1 + gap * beta))
    betas = [0.0, 0.1, 0.5, 1.0, 2.0]
    
    # We have 10 * 5 = 50 candidates.
    # Each simulation: 20 ants * 30 iters * (n^2 ops approx)
    # n=120. n^2 = 14400.
    # 20*30 = 600 steps per ant? No, 30 iters, 20 ants per iter.
    # Total ant-runs = 600.
    # Each ant-run involves ~n/n_bins steps. n_bins ~ 5-10. So ~10-20 steps.
    # Each step involves O(n) operations (masking, summing).
    # So ~600 * 20 * 120 = 1.4M operations per simulation.
    # 50 simulations = 70M operations.
    # Python is slow, but NumPy vectorization helps.
    # However, the ACO loop is sequential per ant.
    # Let's estimate time. 1.4M ops in Python loops might take ~0.1-0.5s per simulation.
    # 50 simulations -> 5-25s. This is well within the 330s budget.
    
    # Let's refine the heuristic family to be more powerful.
    # Idea: "Complementarity to the smallest item".
    # If we place i, then j, we have gap G = C - d[i] - d[j].
    # If G is small, great.
    # If G is large, it's only good if it can fit a large item.
    # A static heuristic can't know future items.
    
    # Let's stick to:
    # H[i,j] = (d[i] + d[j])^alpha * (1 + beta * (1 - gap/capacity))
    # where gap = capacity - (d[i]+d[j])
    # This rewards high sum and low gap.
    
    # Actually, the seed was (d[i]+d[j])/capacity.
    # Let's try a range of exponents on the fill ratio.
    # Fill ratio F = (d[i]+d[j])/capacity.
    # H = F^alpha.
    
    # Let's also try:
    # H = 1 / (1 + gap)
    # H = (d[i]+d[j]) / (1 + gap)
    
    # I will generate a diverse set of heuristics and test them.
    
    candidates = []
    
    # Type 1: Fill ratio powers
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
        candidates.append(('power', alpha))
        
    # Type 2: Inverse gap
    # H = 1 / (1 + gap)
    # gap = C - (di+dj)
    # If di+dj fits, gap >= 0.
    # H = 1 / (1 + (C - (di+dj)))
    # This is 1/(C-di-dj+1).
    # Max value when di+dj is close to C.
    candidates.append(('inv_gap', 1.0))
    
    # Type 3: Hybrid
    # H = (di+dj)^alpha / (1 + gap)
    for alpha in [1.0, 2.0, 3.0]:
        candidates.append(('hybrid', alpha))
        
    # Type 4: "Best Fit" proxy
    # Favor pairs where di+dj is close to C, but penalize if gap is "unusable".
    # Since we don't know item sizes, let's just use distance to C.
    # H = 1 / (1 + |C - (di+dj)|)
    candidates.append(('dist', 1.0))
    
    best_score = float('inf')
    best_heur = None
    
    # Set seed for reproducibility in simulation?
    # The task says "seeded per instance". But inside heuristics(), we are just evaluating.
    # The evaluator runs the ACO with a specific seed.
    # My simulation should mimic that. But I don't know the seed.
    # I will use a fixed seed for my evaluation to be consistent, or average over a few seeds.
    # Let's use 3 different seeds for evaluation to be robust.
    
    eval_seeds = [42, 123, 456]
    
    for ctype, param in candidates:
        if _time_budget_remaining() < 10.0:
            break
            
        # Construct heuristic
        if ctype == 'power':
            fill = pair_sums / capacity
            h = np.where(fits, fill ** param, 1e-9)
        elif ctype == 'inv_gap':
            gap = capacity - pair_sums
            # gap is >= 0 where fits
            h = np.where(fits, 1.0 / (1.0 + gap), 1e-9)
        elif ctype == 'hybrid':
            gap = capacity - pair_sums
            h = np.where(fits, (pair_sums ** param) / (1.0 + gap), 1e-9)
        elif ctype == 'dist':
            gap = capacity - pair_sums
            # Distance from capacity. Since fits means gap >= 0, dist = gap.
            # Wait, if di+dj > C, it doesn't fit.
            # So for fitting items, dist = C - (di+dj) = gap.
            # So 'dist' is same as 'inv_gap' if we use 1/(1+gap).
            # Let's try squared distance?
            h = np.where(fits, 1.0 / (1.0 + gap ** 2), 1e-9)
        else:
            continue
            
        # Ensure positive and finite
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        
        # Evaluate
        scores = []
        for seed in eval_seeds:
            np.random.seed(seed)
            score = _simulate_aco(d, capacity, h)
            scores.append(score)
        
        avg_score = np.mean(scores)
        
        if avg_score < best_score:
            best_score = avg_score
            best_heur = h.copy()
            
    # If no heuristic was evaluated (time out or error), return seed-like heuristic
    if best_heur is None:
        fill = pair_sums / capacity
        best_heur = np.where(fits, fill, 1e-9)
        best_heur = np.where(np.isfinite(best_heur) & (best_heur > 0), best_heur, 1e-9)
        
    return best_heur

# EVOLVE-BLOCK-END
