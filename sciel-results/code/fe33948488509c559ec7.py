import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 145.0  # Safety margin for 150s limit
    
    points = list(range(v))
    
    # 1. Precomputation
    # Generate all k-subsets (potential blocks)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    # Generate all t-subsets
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # Map block -> list of t-subset indices
    block_tsubs = []
    # Also precompute sets for fast membership testing
    block_tsub_sets = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs.append(subs)
        block_tsub_sets.append(set(subs))
        
    # Map t-subset -> list of block indices
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)
            
    def run_greedy(seed):
        random.seed(seed)
        # Initial gains: number of t-subsets covered by each block
        block_gain = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=np.int8)
        uncovered_count = N_t
        selected_indices = []
        
        while uncovered_count > 0 and time.time() < DEADLINE:
            max_gain = np.max(block_gain)
            if max_gain <= 0:
                break
            # Find all blocks with max gain
            candidates = np.where(block_gain == max_gain)[0]
            best_block_idx = random.choice(candidates.tolist())
            
            selected_indices.append(best_block_idx)
            
            # Update coverage and gains
            subs = block_tsubs[best_block_idx]
            for si in subs:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    # Update gains for all blocks covering this t-subset
                    block_gain[tsub_blocks[si]] -= 1
                        
        return selected_indices

    def reduce_and_optimize(selected_indices, time_budget):
        """
        Applies removal and 1-swap local search.
        Returns the optimized list of block indices.
        """
        if not selected_indices:
            return []
        
        # Use a set for O(1) membership check
        current_set = set(selected_indices)
        
        # Compute initial coverage counts
        coverage_count = np.zeros(N_t, dtype=np.int32)
        for g in current_set:
            for si in block_tsubs[g]:
                coverage_count[si] += 1
                
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            # Shuffle order to avoid bias
            items = list(current_set)
            random.shuffle(items)
            
            for g in items:
                if g not in current_set: # Might have been removed in this pass
                    continue
                    
                # Find unique t-subsets for this block
                unique_subs = []
                for si in block_tsubs[g]:
                    if coverage_count[si] == 1:
                        unique_subs.append(si)
                
                if not unique_subs:
                    # Remove block
                    current_set.remove(g)
                    for si in block_tsubs[g]:
                        coverage_count[si] -= 1
                    improved = True
                    continue
                
                # Try to find a replacement
                # Pick a random unique subset to filter candidates
                s0 = random.choice(unique_subs)
                candidates = tsub_blocks[s0]
                
                replacement_found = False
                for g_cand in candidates:
                    if g_cand == g:
                        continue
                    if g_cand in current_set:
                        continue
                    
                    # Check if g_cand covers all unique_subs
                    valid = True
                    # Use the precomputed set for O(1) lookup
                    cand_set = block_tsub_sets[g_cand]
                    for si in unique_subs:
                        if si not in cand_set:
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        current_set.remove(g)
                        current_set.add(g_cand)
                        
                        # Update coverage
                        for si in block_tsubs[g]:
                            coverage_count[si] -= 1
                        for si in block_tsubs[g_cand]:
                            coverage_count[si] += 1
                            
                        improved = True
                        replacement_found = True
                        break
                        
                if replacement_found:
                    # Restart the pass to re-evaluate all blocks with new state
                    break

        return list(current_set)

    # First run: deterministic greedy
    sol_idx = run_greedy(0)
    sol_idx = reduce_and_optimize(sol_idx, DEADLINE - 20.0)
    best_sol = [all_blocks[i] for i in sol_idx]
    best_len = len(best_sol)
    
    # Randomized restarts
    while time.time() < DEADLINE - 10.0:
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy(seed)
        cand_idx = reduce_and_optimize(cand_idx, DEADLINE - 5.0)
        if len(cand_idx) < best_len:
            best_len = len(cand_idx)
            best_sol = [all_blocks[i] for i in cand_idx]
            
    return best_sol
