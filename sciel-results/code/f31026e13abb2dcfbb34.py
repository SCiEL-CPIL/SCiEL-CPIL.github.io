import itertools
import random
import time
import numpy as np

V = 19
K = 7
T = 4

# Precompute
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

all_blocks = list(itertools.combinations(range(V), K))
N_BLOCKS = len(all_blocks)

# Build M: (N_BLOCKS, N_SUBSETS) boolean matrix
M = np.zeros((N_BLOCKS, N_SUBSETS), dtype=bool)
# Also precompute the 35 subset indices per block for fast redundancy checks
block_subs = []
for i, block in enumerate(all_blocks):
    subs = np.array([subset_to_idx[sub] for sub in itertools.combinations(block, T)], dtype=np.int32)
    block_subs.append(subs)
    M[i, subs] = True

def greedy_construct(covered, deadline, weights=None):
    """Greedy set cover with optional per-subset weights."""
    if weights is None:
        weights = np.ones(N_SUBSETS, dtype=np.float64)
    
    solution = []
    while not covered.all():
        if time.time() > deadline - 15:
            break
        uncovered = ~covered
        # Gain = sum of weights for uncovered subsets in this block
        gains = M @ (uncovered * weights).astype(np.float64)
        best_idx = int(np.argmax(gains))
        if gains[best_idx] < 1e-10:
            break
        solution.append(best_idx)
        covered |= M[best_idx]
    return solution

def local_search(solution, deadline):
    """Local search: remove redundancies, swap blocks, with time-bounded restarts."""
    if not solution:
        return solution
    
    counts = np.zeros(N_SUBSETS, dtype=np.int32)
    for idx in solution:
        counts += M[idx]
    
    best_solution = list(solution)
    best_len = len(solution)
    
    in_sol = [False] * N_BLOCKS
    for idx in solution:
        in_sol[idx] = True
    
    no_improve = 0
    max_no_improve = 80
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Remove redundant blocks (use precomputed block_subs)
        indices = list(range(len(solution)))
        random.shuffle(indices)
        
        for pos in indices:
            if time.time() > deadline:
                break
            if pos >= len(solution):
                continue
            idx = solution[pos]
            # Fast redundancy check using precomputed subset indices
            if np.all(counts[block_subs[idx]] > 1):
                counts -= M[idx]
                in_sol[idx] = False
                solution.pop(pos)
                improved = True
                if len(solution) < best_len:
                    best_len = len(solution)
                    best_solution = list(solution)
                break  # Restart removal
        
        if improved:
            no_improve = 0
            continue
        
        # Phase 2: Try swaps
        if len(solution) == 0:
            break
        
        # Try multiple random swaps in this iteration
        swap_attempts = min(10, len(solution))
        for _ in range(swap_attempts):
            if time.time() > deadline:
                break
            
            i = random.randint(0, len(solution) - 1)
            idx_i = solution[i]
            
            counts -= M[idx_i]
            lost_mask = (counts == 0)
            lost_count = int(np.sum(lost_mask))
            
            if lost_count == 0:
                # Redundant - remove
                in_sol[idx_i] = False
                solution.pop(i)
                improved = True
                if len(solution) < best_len:
                    best_len = len(solution)
                    best_solution = list(solution)
                break
            else:
                lost_indices = np.nonzero(lost_mask)[0]
                
                # Find blocks covering all lost subsets
                subset_M = M[:, lost_indices]
                all_covered = subset_M.all(axis=1)
                
                # Exclude current solution blocks
                all_covered[in_sol] = False
                candidates = np.nonzero(all_covered)[0]
                
                if len(candidates) > 0:
                    # Prefer blocks with higher gain (cover more uncovered-ish subsets)
                    # Since we're swapping, just pick randomly from candidates
                    j = int(random.choice(candidates))
                    in_sol[idx_i] = False
                    in_sol[j] = True
                    solution[i] = j
                    counts += M[j]
                    improved = True
                    break
                else:
                    # Revert
                    counts += M[idx_i]
        
        if improved:
            no_improve = 0
            continue
        
        # No improvement: perturb
        no_improve += 1
        if no_improve >= max_no_improve:
            # Remove 2-3 random blocks and re-greedy the uncovered
            n_remove = min(random.randint(2, 3), len(solution))
            removed = random.sample(range(len(solution)), n_remove)
            removed.sort(reverse=True)
            for pos in removed:
                idx = solution.pop(pos)
                counts -= M[idx]
                in_sol[idx] = False
            
            # Re-greedy to cover lost subsets
            lost_mask = (counts == 0)
            if np.any(lost_mask):
                # Weight lost subsets heavily
                weights = np.ones(N_SUBSETS, dtype=np.float64)
                weights[lost_mask] = 10.0
                new_blocks = greedy_construct(
                    counts > 0, deadline - 2.0, weights
                )
                for nb in new_blocks:
                    if not in_sol[nb]:
                        solution.append(nb)
                        counts += M[nb]
                        in_sol[nb] = True
            no_improve = 0
    
    # Final cleanup pass
    final = list(best_solution)
    final_counts = np.zeros(N_SUBSETS, dtype=np.int32)
    for idx in final:
        final_counts += M[idx]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(final)):
            idx = final[i]
            if np.all(final_counts[block_subs[idx]] > 1):
                final_counts -= M[idx]
                final.pop(i)
                changed = True
                break
    
    return final

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered_set = set(itertools.combinations(b, t))
            if uncovered & covered_set:
                blocks.append(list(b))
                uncovered -= covered_set
        return blocks

    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_overall = None
    best_overall_len = float('inf')
    
    # Multiple greedy initializations with different weightings
    # Weighting strategies: uniform, inverse frequency, random
    strategies = []
    
    # Strategy 1: Uniform greedy
    strategies.append(None)
    
    # Strategy 2: Inverse frequency weighting (rarest subsets first)
    # Count how many blocks cover each subset
    subset_freq = np.zeros(N_SUBSETS, dtype=np.int32)
    for i in range(N_BLOCKS):
        subset_freq[block_subs[i]] += 1
    # Inverse: rare subsets get higher weight
    inv_freq = 1.0 / (subset_freq.astype(np.float64) + 1e-6)
    inv_freq /= inv_freq.max()  # Normalize
    strategies.append(inv_freq)
    
    # Strategy 3: Random perturbation of uniform
    rand_weights = np.random.RandomState(42).rand(N_SUBSETS)
    rand_weights = 0.5 + rand_weights  # [0.5, 1.5]
    strategies.append(rand_weights)
    
    # Strategy 4: Another random
    rand_weights2 = np.random.RandomState(123).rand(N_SUBSETS)
    rand_weights2 = 0.5 + rand_weights2
    strategies.append(rand_weights2)
    
    for strat_idx, weights in enumerate(strategies):
        if time.time() > deadline - 20:
            break
        
        # Allocate time proportionally, but ensure each gets at least 5s
        remaining = deadline - time.time()
        n_strats_left = len(strategies) - strat_idx
        alloc = min(remaining / n_strats_left, 35.0)  # Cap per-strategy at 35s
        strat_deadline = time.time() + alloc
        
        covered = np.zeros(N_SUBSETS, dtype=bool)
        sol = greedy_construct(covered, strat_deadline - 5.0, weights)
        
        # If greedy didn't cover everything, fill in
        if not covered.all():
            # Add blocks to cover remaining
            for i in range(N_BLOCKS):
                if covered.all():
                    break
                if M[i] @ (~covered).astype(np.int8) > 0:
                    sol.append(i)
                    covered |= M[i]
        
        if not covered.all():
            continue
        
        # Local search
        sol = local_search(sol, min(strat_deadline, deadline - 5.0))
        
        if len(sol) < best_overall_len:
            best_overall_len = len(sol)
            best_overall = sol
    
    # If we still have time, do one more local search on best
    if best_overall is not None and time.time() < deadline - 10:
        best_overall = local_search(best_overall, deadline - 2.0)
    
    if best_overall is None:
        # Shouldn't happen, but fallback
        best_overall = greedy_construct(np.zeros(N_SUBSETS, dtype=bool), deadline - 1.0)
    
    return [list(all_blocks[i]) for i in best_overall]
