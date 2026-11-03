import itertools
import numpy as np
import time
import random
import sys

def construct(v, k, t):
    # Deadline guard: evaluator kills at 150s, keep a safety margin
    deadline = time.time() + 118.0
    
    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Precompute subsets_of_block: for each block j, list of t-subset indices
    # and blocks_of_subset: for each t-subset i, list of block indices
    subsets_of_block = []
    blocks_of_subset = [[] for _ in range(n_t)]
    
    for j, B in enumerate(k_subsets):
        sub_indices = []
        for S in itertools.combinations(B, t):
            i = t_index[S]
            sub_indices.append(i)
            blocks_of_subset[i].append(j)
        subsets_of_block.append(np.array(sub_indices, dtype=np.int32))
    
    best_solution = None
    best_size = float('inf')
    
    # Helper: deletion pass to remove redundant blocks
    def deletion_pass(solution_blocks):
        if not solution_blocks:
            return solution_blocks
            
        # Compute cover counts
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in solution_blocks:
            cover_count[subsets_of_block[b_idx]] += 1
            
        # Try to remove blocks
        changed = True
        while changed:
            changed = False
            new_solution = []
            for b_idx in solution_blocks:
                subs_b = subsets_of_block[b_idx]
                # Check if any subset in b_idx has count 1
                can_remove = True
                for s_idx in subs_b:
                    if cover_count[s_idx] == 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    # Remove b_idx
                    cover_count[subs_b] -= 1
                    changed = True
                else:
                    new_solution.append(b_idx)
                    
            solution_blocks = new_solution
            
        return solution_blocks

    # Helper: local search with swaps (inspired by inspiration 1)
    def local_search(blocks, seed=0, max_iterations=200):
        if not blocks:
            return blocks
            
        rng = random.Random(seed)
        reduced = list(blocks)
        best_reduced = list(reduced)
        best_len = len(reduced)
        
        # State
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in reduced:
            for idx in subsets_of_block[b]:
                cover_count[idx] += 1
        
        in_solution = np.zeros(n_k, dtype=bool)
        for b in reduced:
            in_solution[b] = True
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
                
            improved = False
            
            # Phase 1: Remove redundant blocks
            changed = True
            while changed:
                changed = False
                for i in range(len(reduced)):
                    if time.time() > deadline - 5:
                        break
                    
                    b_idx = reduced[i]
                    ts_list = subsets_of_block[b_idx]
                    # Check if all t-subsets in this block are covered by at least 2 blocks
                    if np.all(cover_count[ts_list] >= 2):
                        # Remove block i
                        for idx in ts_list:
                            cover_count[idx] -= 1
                        in_solution[b_idx] = False
                        reduced.pop(i)
                        changed = True
                        improved = True
                        if len(reduced) < best_len:
                            best_len = len(reduced)
                            best_reduced = list(reduced)
                        break  # Restart scan
            
            if len(reduced) < 2:
                break
                
            # Phase 2: Swap using essential t-subsets
            if not improved:
                n = len(reduced)
                order = list(range(n))
                rng.shuffle(order)
                
                for i in order:
                    if time.time() > deadline - 5:
                        break
                        
                    b_idx = reduced[i]
                    ts_list = subsets_of_block[b_idx]
                    
                    # Find essential t-subsets (covered only by this block)
                    essential_mask = (cover_count[ts_list] == 1)
                    essential_t_idxs = ts_list[essential_mask]
                    
                    if len(essential_t_idxs) == 0:
                        # This block is redundant - should have been caught in Phase 1
                        # Remove it
                        for idx in ts_list:
                            cover_count[idx] -= 1
                        in_solution[b_idx] = False
                        reduced.pop(i)
                        improved = True
                        if len(reduced) < best_len:
                            best_len = len(reduced)
                            best_reduced = list(reduced)
                        break
                        
                    # Find candidates: blocks that cover ALL essential t-subsets
                    # Intersection of tsub_blocks for each essential t-subset
                    candidate_set = None
                    for t_idx in essential_t_idxs:
                        covering = blocks_of_subset[t_idx]
                        if candidate_set is None:
                            candidate_set = set(covering)
                        else:
                            candidate_set.intersection_update(covering)
                        
                        if not candidate_set:
                            break
                    
                    if not candidate_set:
                        continue
                    
                    # Filter out blocks already in solution
                    candidates = [b for b in candidate_set if not in_solution[b]]
                    
                    if not candidates:
                        continue
                    
                    # Try each candidate: check if it covers all essential t-subsets
                    # and doesn't break coverage of other t-subsets
                    for cand_b in candidates:
                        cand_ts = subsets_of_block[cand_b]
                        
                        # Check: does cand_b cover all essential t-subsets?
                        # (Already ensured by construction via intersection)
                        
                        # Check: for each t-subset in b_idx that is NOT essential,
                        # is it covered by some other block (not b_idx)?
                        # Actually, we need to check: if we remove b_idx, are all its t-subsets still covered?
                        # Essential ones are covered by cand_b.
                        # Non-essential ones have cover_count >= 2, so after removing b_idx, count >= 1.
                        # So the swap is valid if cand_b covers all essential t-subsets.
                        
                        # Perform swap
                        # Remove b_idx
                        for idx in ts_list:
                            cover_count[idx] -= 1
                        in_solution[b_idx] = False
                        
                        # Add cand_b
                        for idx in cand_ts:
                            cover_count[idx] += 1
                        in_solution[cand_b] = True
                        
                        # Update reduced list
                        reduced[i] = cand_b
                        improved = True
                        break  # Try next block
                    
                    if improved:
                        if len(reduced) < best_len:
                            best_len = len(reduced)
                            best_reduced = list(reduced)
                        break  # Restart scan
            
            if not improved:
                # No improvement in this iteration; could stop or continue
                # Let's continue for a few more iterations to see if noise helps
                pass
                
        return best_reduced

    # Multi-start greedy with local search
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize
        uncovered_mask = np.ones(n_t, dtype=bool)
        gains = np.array([len(subs) for subs in subsets_of_block], dtype=np.float64)
        
        # Add small random noise to break symmetry/ties
        if restart > 0:
            gains += np.random.uniform(0, 0.1, n_k)
            
        solution_blocks = []
        
        while True:
            if time.time() > deadline - 10:
                break
                
            # Find best block
            if np.max(gains) <= 0:
                break
                
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break

            solution_blocks.append(best_idx)
            
            # Update gains
            subs_b = subsets_of_block[best_idx]
            
            # Find which subsets in best_idx are actually uncovered
            newly_covered_mask = uncovered_mask[subs_b]
            
            # Update gains for all blocks sharing subsets with best_idx
            for s_idx in subs_b[newly_covered_mask]:
                for b_prime in blocks_of_subset[s_idx]:
                    gains[b_prime] -= 1.0
                    
            # Update uncovered_mask
            uncovered_mask[subs_b] = False
            
            # Check if all covered
            if not np.any(uncovered_mask):
                break
                
        # Apply deletion pass
        solution_blocks = deletion_pass(solution_blocks)
        
        # Apply local search
        solution_blocks = local_search(solution_blocks, seed=restart, max_iterations=100)
        
        # Update best
        if len(solution_blocks) < best_size:
            best_size = len(solution_blocks)
            best_solution = solution_blocks.copy()
            
            # If we reached a very good size, we might stop early?
            # 151 is the target. If we hit 151, keep searching for 150.
            if best_size <= 151:
                # Continue searching a bit more
                pass

    # Post-processing: Final deletion pass on best_solution
    if best_solution is not None:
        best_solution = deletion_pass(best_solution)
        
    # Construct the output list of blocks
    if best_solution is None:
        # Emergency fallback: simple greedy without optimization
        uncovered = set(range(n_t))
        res_blocks = []
        while uncovered:
            best_b = -1
            max_cov = -1
            for j in range(n_k):
                cnt = 0
                for s in subsets_of_block[j]:
                    if s in uncovered:
                        cnt += 1
                if cnt > max_cov:
                    max_cov = cnt
                    best_b = j
            if best_b == -1 or max_cov == 0:
                break
            res_blocks.append(best_b)
            for s in subsets_of_block[best_b]:
                if s in uncovered:
                    uncovered.remove(s)
        best_solution = res_blocks

    # Return list of lists of ints
    return [list(k_subsets[idx]) for idx in best_solution]
