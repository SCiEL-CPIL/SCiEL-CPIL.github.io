import itertools
import time
import random
import numpy as np
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precomputation
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    num_t = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(points, k))
    num_k = len(all_k_blocks)
    
    # Build sparse matrix M: M[b, s] = 1 if block b covers t-subset s
    rows = []
    cols = []
    for b_idx, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            s_idx = t_subset_to_idx[t_sub]
            rows.append(b_idx)
            cols.append(s_idx)
            
    M = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(num_k, num_t))
    
    # Also keep dense lists of indices for fast access if needed, but sparse is key for gain
    # M is CSR, so M[b] gives a sparse row.
    
    best_blocks = None
    best_score = float('inf')
    
    def reduce_blocks(blocks_idx, coverage):
        """Remove redundant blocks. Returns new list of block indices."""
        changed = True
        while changed:
            changed = False
            new_blocks = []
            # We need to check each block in the current list
            # To make this faster, we can process in random order or just iterate
            for b_idx in blocks_idx:
                # Get indices of t-subsets covered by this block
                # M[b_idx] is a sparse row. We need its non-zero indices.
                # Since M is CSR, M[b_idx].indices gives the column indices.
                idxs = M[b_idx].indices
                # Check if coverage is > 1 for all these idxs
                if np.all(coverage[idxs] > 1):
                    # Safe to remove
                    coverage[idxs] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            blocks_idx = new_blocks
        return blocks_idx

    def greedy_construct(seed, weight_mode=0):
        """
        weight_mode: 0 = pure uncovered count, 1 = add small random perturbation
        """
        random.seed(seed)
        coverage = np.zeros(num_t, dtype=np.int32)
        uncovered_count = num_t
        blocks_idx = []
        
        while uncovered_count > 0:
            if time.time() > deadline - 5.0:
                return None, coverage, blocks_idx
            
            mask = (coverage == 0)
            gains = M.dot(mask.astype(np.int8))
            
            if weight_mode == 1:
                # Add small random noise to break ties more effectively
                noise = np.random.random(num_k) * 0.5
                gains = gains + noise
            
            max_gain = np.max(gains)
            if max_gain == 0:
                return None, coverage, blocks_idx
                
            candidates = np.where(gains == max_gain)[0]
            b_idx = random.choice(candidates)
            
            blocks_idx.append(b_idx)
            
            idxs = M[b_idx].indices
            coverage[idxs] += 1
            uncovered_count = np.sum(coverage == 0)
            
        return blocks_idx, coverage, None

    # Run multiple restarts with varied weight modes
    num_restarts = 80
    for i in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        weight_mode = i % 2
        res = greedy_construct(i, weight_mode)
        if res[0] is None:
            continue
            
        blocks_idx, coverage, _ = res
        
        # Reduce
        blocks_idx = reduce_blocks(blocks_idx, coverage)
        
        current_score = len(blocks_idx)
        if current_score < best_score:
            best_score = current_score
            best_blocks = [list(all_k_blocks[b]) for b in blocks_idx]
            
            # Enhanced Local Search with swap-based operations
            ls_start = time.time()
            ls_deadline = min(deadline - 2.0, ls_start + 15.0)
            
            # Build a set for O(1) membership check
            blocks_set = set(blocks_idx)
            
            # Precompute which blocks cover each t-subset (inverse index)
            # This helps find good swap candidates
            # t_subset_to_blocks[s] = list of blocks that cover t-subset s
            t_subset_to_blocks = [set() for _ in range(num_t)]
            for b_idx in blocks_idx:
                for s_idx in M[b_idx].indices:
                    t_subset_to_blocks[s_idx].add(b_idx)
            
            # Local search: try swaps (remove one, add one) and additions
            improved = True
            while improved and time.time() < ls_deadline:
                improved = False
                
                # Try swap: remove block i, add block j
                # A good swap: removing i exposes some t-subsets, adding j covers them
                # Plus j should cover other things that might allow further reduction
                
                for i in range(len(blocks_idx)):
                    if time.time() > ls_deadline - 1.0:
                        break
                    
                    b_remove = blocks_idx[i]
                    # Find t-subsets that would become uncovered if we remove b_remove
                    # These are t-subsets where coverage == 1 and b_remove covers them
                    idxs_remove = M[b_remove].indices
                    exposed = idxs_remove[coverage[idxs_remove] == 1]
                    
                    if len(exposed) == 0:
                        # Can remove without adding
                        coverage[idxs_remove] -= 1
                        blocks_idx.pop(i)
                        blocks_set.discard(b_remove)
                        improved = True
                        # Re-run reduce after removal
                        blocks_idx = reduce_blocks(blocks_idx, coverage)
                        blocks_set = set(blocks_idx)
                        break
                    else:
                        # Need to find a block that covers all exposed t-subsets
                        # For efficiency, pick the most exposed t-subset and find blocks covering it
                        # Then check if they cover all exposed
                        # Use the first exposed as a filter
                        first_exposed = exposed[0]
                        candidate_blocks = t_subset_to_blocks[first_exposed]
                        
                        # Remove b_remove from candidates (it's being removed)
                        # Actually b_remove covers first_exposed, so it's in candidate_blocks
                        # We want blocks OTHER than b_remove that cover all exposed
                        
                        best_swap = -1
                        best_overlap = -1
                        
                        for b_candidate in candidate_blocks:
                            if b_candidate == b_remove:
                                continue
                            if b_candidate in blocks_set:
                                continue
                            
                            # Check if b_candidate covers all exposed
                            idxs_candidate = M[b_candidate].indices
                            # Count how many exposed are covered
                            # This is expensive; use a quick filter first
                            # If b_candidate doesn't cover even half, skip
                            # Actually let's just check all
                            covered_count = 0
                            for e in exposed:
                                if e in idxs_candidate:
                                    covered_count += 1
                                else:
                                    break
                            
                            if covered_count == len(exposed):
                                # Valid swap candidate
                                # Score by how many additional t-subsets it covers (potential for further reduction)
                                additional = 0
                                for s in idxs_candidate:
                                    if s not in exposed and coverage[s] == 0:
                                        additional += 1
                                if additional > best_overlap:
                                    best_overlap = additional
                                    best_swap = b_candidate
                        
                        if best_swap >= 0:
                            # Perform swap
                            coverage[idxs_remove] -= 1
                            blocks_idx.pop(i)
                            blocks_set.discard(b_remove)
                            
                            idxs_add = M[best_swap].indices
                            coverage[idxs_add] += 1
                            blocks_idx.append(best_swap)
                            blocks_set.add(best_swap)
                            
                            # Update t_subset_to_blocks
                            for s in idxs_remove:
                                if s in t_subset_to_blocks and b_remove in t_subset_to_blocks[s]:
                                    t_subset_to_blocks[s].discard(b_remove)
                            for s in idxs_add:
                                if s in t_subset_to_blocks:
                                    t_subset_to_blocks[s].add(best_swap)
                                else:
                                    t_subset_to_blocks[s] = {best_swap}
                            
                            improved = True
                            # Try to reduce after swap
                            blocks_idx = reduce_blocks(blocks_idx, coverage)
                            blocks_set = set(blocks_idx)
                            # Update t_subset_to_blocks after reduction
                            for s in range(num_t):
                                t_subset_to_blocks[s] = set()
                            for b in blocks_idx:
                                for s in M[b].indices:
                                    t_subset_to_blocks[s].add(b)
                            break
            
            # Final reduction
            blocks_idx = reduce_blocks(blocks_idx, coverage)
            current_score = len(blocks_idx)
            if current_score < best_score:
                best_score = current_score
                best_blocks = [list(all_k_blocks[b]) for b in blocks_idx]
                    
    return best_blocks
