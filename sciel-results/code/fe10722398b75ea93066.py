import itertools
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Precompute all t-subsets and map to indices
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {s: i for i, s in enumerate(all_ts)}

    # Precompute all k-subsets
    all_ks = list(itertools.combinations(range(v), k))
    n_ks = len(all_ks)
    
    # Number of t-subsets in a k-set
    t_count = int(np.prod([k - i for i in range(t)]) / np.prod([i + 1 for i in range(t)]))
    
    # Precompute for each k-subset, the indices of the t-subsets it contains
    # block_t_indices[i, j] is the index of the j-th t-subset in the i-th k-subset
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = ts_index[sub]

    def greedy_randomized(seed, top_k=10):
        """
        Randomized greedy set cover.
        Picks a block from the top 'top_k' gainers randomly.
        """
        rng = np.random.RandomState(seed)
        covered = np.zeros(n_ts, dtype=bool)
        selected_blocks = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            # Calculate gains: number of uncovered triples covered by each block
            gains = (~covered[block_t_indices]).sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                if not np.all(covered):
                    return None
                else:
                    break
            
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            
            selected_blocks.append(best_idx)
            covered[block_t_indices[best_idx]] = True
            
            if np.all(covered):
                break

        return selected_blocks

    def reduce_blocks(block_indices):
        """
        Remove redundant blocks.
        A block is redundant if every t-subset in it is covered by at least one other block.
        """
        if not block_indices:
            return []
            
        current_blocks = list(block_indices)
        
        while True:
            if time.time() > deadline - 2.0:
                break
                
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current_blocks:
                coverage_count[block_t_indices[b_idx]] += 1
            
            new_blocks = []
            removed_any = False
            
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                
                if np.all(coverage_count[subs_in_block] > 1):
                    coverage_count[subs_in_block] -= 1
                    removed_any = True
                else:
                    new_blocks.append(b_idx)
            
            current_blocks = new_blocks
            
            if not removed_any:
                break
                
        return current_blocks

    def solve_residual_ilp(initial_blocks):
        """
        Solves an ILP to minimize the number of blocks added to initial_blocks
        to cover all remaining uncovered t-subsets.
        """
        if time.time() > deadline - 10.0:
            return initial_blocks

        covered = np.zeros(n_ts, dtype=bool)
        for b_idx in initial_blocks:
            covered[block_t_indices[b_idx]] = True
        
        # Identify uncovered triples
        uncovered_mask = ~covered
        uncovered_indices = np.where(uncovered_mask)[0]
        
        if len(uncovered_indices) == 0:
            return initial_blocks

        # We only need to consider blocks that cover at least one uncovered triple
        # And we want to find the minimum set of such blocks to cover all uncovered triples.
        
        # Let's formulate the ILP:
        # Variables: x_i for each block i (0 or 1)
        # Objective: minimize sum(x_i)
        # Constraints: for each uncovered triple j, sum(x_i for i covering j) >= 1
        
        # To make it faster, we can restrict the variable space to blocks that cover uncovered triples.
        # Let's find which blocks cover at least one uncovered triple.
        
        # Gains for all blocks relative to the current coverage
        gains = (~covered[block_t_indices]).sum(axis=1)
        
        # Candidate blocks are those with gain > 0
        candidate_indices = np.where(gains > 0)[0]
        
        if len(candidate_indices) == 0:
            # Should not happen if there are uncovered triples, unless precomputation error
            return initial_blocks
            
        n_candidates = len(candidate_indices)
        
        # Build the constraint matrix A for the ILP
        # A has rows for each uncovered triple, columns for each candidate block
        # A[j, i] = 1 if candidate block i covers uncovered triple j, else 0
        
        # We can use the precomputed block_t_indices to do this efficiently.
        # But block_t_indices is (n_ks, t_count).
        # We need a sparse matrix or dense matrix for the ILP.
        
        # Since n_candidates might be large (thousands), and n_uncovered might be large,
        # a dense matrix might be too big or slow to build.
        # However, for C(24,7,3), n_ts = 2024.
        # n_ks = 346104.
        # If we have ~76 blocks, covered is mostly full.
        # Uncovered is small.
        # Candidate blocks are those that cover at least one uncovered triple.
        
        # Let's try to build the matrix.
        # n_uncovered = len(uncovered_indices)
        # n_candidates = len(candidate_indices)
        
        # A is (n_uncovered, n_candidates)
        
        # We can iterate over candidate blocks and check which uncovered triples they cover.
        # Or iterate over uncovered triples and check which candidate blocks cover them.
        
        # Let's map global ts index to local uncovered index
        # ts_index_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(uncovered_indices)}
        
        # This might be slow in Python. Let's use numpy.
        # Create a boolean mask for uncovered triples
        # For each candidate block, we have block_t_indices[cand_idx]
        # We need to check if any of these are in uncovered_indices.
        
        # Let's create a set of uncovered indices for fast lookup? No, numpy is better.
        # Let's create a boolean array of size n_ts
        # uncovered_bool = np.zeros(n_ts, dtype=bool)
        # uncovered_bool[uncovered_indices] = True
        
        # Then for each candidate block i:
        # subs = block_t_indices[candidate_indices[i]]
        # covered_subs = subs[uncovered_bool[subs]]
        # These are the local indices in the block's subs array that correspond to uncovered triples.
        # We need the global indices.
        # global_subs = subs
        # uncovered_global_subs = global_subs[uncovered_bool[global_subs]]
        
        # Now we need to set A[local_uncovered_idx, i] = 1 for these.
        # We need to map global_uncovered_idx to local_uncovered_idx.
        
        # Let's use np.searchsorted
        # local_idx = np.searchsorted(uncovered_indices, uncovered_global_subs)
        
        # This is efficient enough.
        
        A = np.zeros((len(uncovered_indices), n_candidates), dtype=np.int8)
        
        for i in range(n_candidates):
            cand_global_idx = candidate_indices[i]
            subs = block_t_indices[cand_global_idx]
            # Filter subs that are uncovered
            # We need a fast way to check membership in uncovered_indices
            # Since uncovered_indices is sorted (from np.where), we can use searchsorted
            # But first we need to know which subs are uncovered.
            
            # Let's create a boolean mask for subs
            # We can use np.in1d or searchsorted
            # searchsorted is O(log N) per element, total O(M log N)
            # M is t_count (35), N is n_uncovered.
            
            # Actually, let's just use a set for uncovered_indices if n_uncovered is small.
            # Or just use the boolean array approach.
            
            # Let's stick to the boolean array approach as it's O(1) per check.
            pass
            
        # Optimization: Build A more efficiently
        # Let's create a boolean array for uncovered
        uncovered_bool = np.zeros(n_ts, dtype=bool)
        uncovered_bool[uncovered_indices] = True
        
        # Precompute mapping from global ts index to local uncovered index
        # This is only needed for the indices that are uncovered.
        # We can use a dictionary or just searchsorted on the fly.
        # Let's try searchsorted.
        
        for i in range(n_candidates):
            cand_global_idx = candidate_indices[i]
            subs = block_t_indices[cand_global_idx]
            
            # Find which subs are uncovered
            # mask = uncovered_bool[subs]
            # local_subs = subs[mask]
            # local_uncovered_idx = np.searchsorted(uncovered_indices, local_subs)
            # A[local_uncovered_idx, i] = 1
            
            # To avoid Python loop overhead if possible, but here n_candidates might be large.
            # Let's just do it.
            
            mask = uncovered_bool[subs]
            if np.any(mask):
                local_subs = subs[mask]
                local_uncovered_idx = np.searchsorted(uncovered_indices, local_subs)
                A[local_uncovered_idx, i] = 1
        
        # Objective: minimize sum(x)
        c = np.ones(n_candidates, dtype=np.float64)
        
        # Constraints: A x >= 1
        # LinearConstraint(A, 1, np.inf)
        constraint = LinearConstraint(A, 1, np.inf)
        
        # Bounds: 0 <= x <= 1
        bounds = Bounds(0, 1)
        
        # Integrality: all variables are integer
        integrality = np.ones(n_candidates, dtype=np.int8)
        
        try:
            res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds)
            
            if res.status == 0:
                # Solution found
                x = res.x
                # x is float array, convert to bool
                selected_new = x > 0.5
                new_block_indices = candidate_indices[selected_new]
                
                # Combine with initial blocks
                # Note: initial_blocks are global indices.
                # new_block_indices are global indices.
                # We need to return the union.
                # But wait, the ILP minimizes the number of NEW blocks.
                # So total blocks = len(initial_blocks) + sum(new_block_indices)
                
                # We should verify that this actually covers everything.
                # And we should check if it's better than current best.
                
                combined_blocks = list(set(initial_blocks + list(new_block_indices)))
                
                # Verify validity
                # (Optional but safe)
                # Let's just return the combined list.
                # The caller will verify/reduce if needed.
                
                return combined_blocks
            else:
                return initial_blocks
                
        except Exception:
            return initial_blocks

    best_blocks = None
    best_size = float('inf')
    
    num_runs = 0
    while time.time() < deadline - 20.0:
        seed = num_runs
        blocks = greedy_randomized(seed)
        
        if blocks is not None:
            # Reduce redundancy
            reduced = reduce_blocks(blocks)
            
            # Try to improve with ILP on the residual
            # Only try ILP if we have time and if the reduced size is not already very small
            if time.time() < deadline - 15.0:
                ilp_blocks = solve_residual_ilp(reduced)
                if ilp_blocks is not None:
                    # The ILP solution might contain redundancies again?
                    # Or it might be minimal for the residual, but combined with initial, there might be overlaps.
                    # Let's reduce again.
                    final_reduced = reduce_blocks(ilp_blocks)
                    
                    if len(final_reduced) < best_size:
                        best_size = len(final_reduced)
                        best_blocks = final_reduced
                    elif len(ilp_blocks) < best_size:
                        # In case reduction removes too much or ILP was better before reduction?
                        # No, reduction only removes redundant blocks, so it should be <=.
                        # But let's be safe.
                        pass
            else:
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    best_blocks = reduced

        num_runs += 1
        
        if num_runs > 100:
            break

    if best_blocks is not None:
        return [list(all_ks[i]) for i in best_blocks]
    
    # Fallback
    blocks = greedy_randomized(0)
    if blocks:
        return [list(all_ks[i]) for i in blocks]
        
    return []
