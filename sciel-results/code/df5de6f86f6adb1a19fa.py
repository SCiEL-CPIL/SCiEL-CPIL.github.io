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
# This takes some time but is one-time cost.
M = np.zeros((N_BLOCKS, N_SUBSETS), dtype=bool)
for i, block in enumerate(all_blocks):
    for sub in itertools.combinations(block, T):
        M[i, subset_to_idx[sub]] = True

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
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 115.0
    
    # Greedy Construction
    covered = np.zeros(N_SUBSETS, dtype=bool)
    solution = []
    
    while not covered.all():
        if time.time() > deadline - 10: # Leave time for local search
            break
        uncovered = ~covered
        # Calculate gain for all blocks
        # M is (N_BLOCKS, N_SUBSETS), uncovered is (N_SUBSETS,)
        # We want count of True in M[i] where uncovered is True
        gains = M @ uncovered.astype(np.int8)
        
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
            
        solution.append(best_idx)
        covered |= M[best_idx]
        
        # Optimization: If we cover everything, break
        if covered.all():
            break
            
    # If greedy didn't finish, add random blocks to cover rest? 
    # Unlikely with 50k blocks. But just in case.
    if not covered.all():
        for i in range(N_BLOCKS):
            if covered.all(): break
            if M[i] @ (~covered).astype(np.int8) > 0:
                solution.append(i)
                covered |= M[i]

    # Local Search
    counts = np.zeros(N_SUBSETS, dtype=np.int32)
    for idx in solution:
        counts += M[idx]
        
    best_solution = solution.copy()
    best_len = len(solution)
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Remove redundant blocks
        # Shuffle solution for diversity in removal order
        indices = list(range(len(solution)))
        random.shuffle(indices)
        
        for pos in indices:
            if time.time() > deadline:
                break
            if pos >= len(solution): continue # safety
            
            idx = solution[pos]
            # Check redundancy
            # Get subsets covered by this block
            # M[idx] is a boolean vector. 
            # We need to check if counts[subset] > 1 for all subset where M[idx, subset] is True.
            # subs = np.nonzero(M[idx])[0]
            # if np.all(counts[subs] > 1):
            
            # Faster: 
            # min_count = np.min(counts[M[idx]])
            # if min_count > 1:
            
            # Note: np.min on empty array errors, but blocks cover 35 subsets, so never empty.
            if np.min(counts[M[idx]]) > 1:
                # Remove
                counts -= M[idx]
                solution.pop(pos)
                improved = True
                if len(solution) < best_len:
                    best_len = len(solution)
                    best_solution = solution.copy()
                break # Restart removal phase
        
        if improved:
            no_improve_count = 0
            continue
            
        # Phase 2: Swap / Perturbation
        if len(solution) == 0:
            break
            
        # Try to swap a block
        # Pick random block to remove
        i = random.randint(0, len(solution) - 1)
        idx_i = solution[i]
        
        # Temporarily remove
        counts -= M[idx_i]
        
        # Find lost subsets
        lost_mask = (counts == 0)
        lost_count = np.sum(lost_mask)
        
        if lost_count == 0:
            # It was redundant, why didn't phase 1 catch it? 
            # Maybe due to order? Anyway, keep it removed.
            solution.pop(i)
            improved = True
            if len(solution) < best_len:
                best_len = len(solution)
                best_solution = solution.copy()
        else:
            # Need to find a block that covers all lost subsets
            # lost_indices = np.nonzero(lost_mask)[0]
            # candidates = np.where(M[:, lost_indices].all(axis=1))[0]
            
            # To speed up, we can just check if any block covers the lost subsets.
            # Since lost_count is small, M[:, lost_indices] is a small slice.
            lost_indices = np.nonzero(lost_mask)[0]
            
            # Check candidates
            # M is (N_BLOCKS, N_SUBSETS)
            # We want rows where all columns in lost_indices are True.
            # subset_M = M[:, lost_indices]
            # all_covered = subset_M.all(axis=1)
            # candidates = np.nonzero(all_covered)[0]
            
            # Filter candidates not in current solution
            # sol_set = set(solution)
            # valid_candidates = [c for c in candidates if c not in sol_set]
            
            # This might be slow if done every time.
            # Optimization: Precompute a set of solution indices?
            # Or just check a few random candidates?
            
            # Let's try to find a replacement.
            # If we can't find one quickly, revert.
            
            # To avoid heavy computation, we can sample a few blocks from the full set
            # or just use the precomputed intersection logic if we had it.
            # But M is dense.
            
            # Let's just do the check. It's C(19,7) * C(lost, 1) ops? No.
            # M[:, lost_indices] is (50388, L). .all(axis=1) is fast in C.
            
            subset_M = M[:, lost_indices]
            all_covered = subset_M.all(axis=1)
            candidates = np.nonzero(all_covered)[0]
            
            # Exclude current solution
            sol_set = set(solution)
            valid_candidates = [c for c in candidates if c not in sol_set]
            
            if valid_candidates:
                # Pick one
                j = random.choice(valid_candidates)
                # Replace i with j
                solution[i] = j
                counts += M[j]
                improved = True
            else:
                # Revert
                counts += M[idx_i]
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation: Add a random block
                # Pick a block that covers many uncovered? No, we are covered.
                # Just pick a random block not in solution to break symmetry.
                c = random.randint(0, N_BLOCKS - 1)
                if c not in solution:
                    solution.append(c)
                    counts += M[c]
                no_improve_count = 0
                improved = True # Force restart of loop to try removals

    # Final cleanup: remove any remaining redundancies
    final_solution = best_solution.copy()
    # Rebuild counts for final solution
    final_counts = np.zeros(N_SUBSETS, dtype=np.int32)
    for idx in final_solution:
        final_counts += M[idx]
        
    changed = True
    while changed:
        changed = False
        for i in range(len(final_solution)):
            idx = final_solution[i]
            if np.min(final_counts[M[idx]]) > 1:
                final_counts -= M[idx]
                final_solution.pop(i)
                changed = True
                break

    return [list(all_blocks[i]) for i in final_solution]
