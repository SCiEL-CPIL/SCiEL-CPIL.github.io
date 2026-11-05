import itertools
import time
import random
import numpy as np
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

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
    
    # Precompute block coverage indices for fast access
    block_indices = [M[b].indices for b in range(num_k)]
    
    # Precompute for each t-subset, which blocks cover it
    # This is useful for identifying candidates that cover specific exposed subsets
    t_subset_to_blocks = [set() for _ in range(num_t)]
    for b_idx in range(num_k):
        for s_idx in block_indices[b_idx]:
            t_subset_to_blocks[s_idx].add(b_idx)

    best_blocks = None
    best_score = float('inf')
    
    def reduce_blocks(blocks_idx, coverage):
        """Remove redundant blocks. Returns new list of block indices."""
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                idxs = block_indices[b_idx]
                # Check if all subsets covered by this block are covered by >1 block
                if np.all(coverage[idxs] > 1):
                    coverage[idxs] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            blocks_idx = new_blocks
        return blocks_idx

    def greedy_construct(seed, temp=0.1):
        """
        Greedy construction with temperature-based selection.
        Uses a maintained uncovered_set for fast gain computation via sparse rows.
        """
        random.seed(seed)
        np.random.seed(seed)
        coverage = np.zeros(num_t, dtype=np.int32)
        uncovered_count = num_t
        blocks_idx = []
        
        # Maintain a set of currently uncovered t-subset indices
        uncovered_set = set(range(num_t))
        
        while uncovered_count > 0:
            if time.time() > deadline - 5.0:
                return None, coverage, blocks_idx
            
            # Compute gains: for each block, count how many of its covered
            # t-subsets are currently uncovered.
            # Use sparse row access on the uncovered indices.
            unc_arr = np.fromiter(uncovered_set, dtype=np.int32, count=uncovered_count)
            
            # For each block, gain = number of its indices that are in uncovered_set
            # Efficient: iterate blocks and use np.isin or direct check
            # Since num_k ~ 75582 and avg row length ~ C(8,5)=56, this is ~4.2M checks total per iter
            # But we can do it in a vectorized way per block using boolean indexing
            
            # Alternative faster approach: precompute for each uncovered subset which blocks cover it
            # and count how many times each block appears
            # This is the transpose of what we want: for each block, count uncovered subsets it covers
            # We can use the t_subset_to_blocks mapping: for each uncovered subset, increment count for its covering blocks
            block_gain = np.zeros(num_k, dtype=np.int32)
            for s_idx in uncovered_set:
                for b_idx in t_subset_to_blocks[s_idx]:
                    block_gain[b_idx] += 1
            
            max_gain = np.max(block_gain)
            if max_gain == 0:
                return None, coverage, blocks_idx
                
            # Find candidates with max gain
            candidates = np.where(block_gain == max_gain)[0]
            
            # Temperature-based selection
            if len(candidates) > 1 and temp > 0:
                noise = np.random.random(len(candidates)) * temp
                scores = max_gain + noise
                b_idx = candidates[np.argmax(scores)]
            else:
                b_idx = random.choice(candidates)
            
            blocks_idx.append(b_idx)
            
            idxs = block_indices[b_idx]
            coverage[idxs] += 1
            
            # Update uncovered_set: remove subsets that just became covered
            newly_covered = idxs[coverage[idxs] == 1]
            for s_idx in newly_covered:
                uncovered_set.discard(int(s_idx))
            uncovered_count = len(uncovered_set)
            
        return blocks_idx, coverage, None

    def local_search(blocks_idx, coverage, time_limit):
        """
        Local search using swaps with precomputed sorted block index arrays.
        """
        ls_start = time.time()
        improved = True
        blocks_set = set(blocks_idx)
        
        # Precompute sorted arrays for each block's covered t-subsets for fast isin checks
        # block_indices are already numpy arrays from CSR; ensure they're sorted
        sorted_block_indices = [np.sort(block_indices[b]) for b in range(num_k)]
        
        while improved and (time.time() - ls_start) < time_limit:
            improved = False
            
            order = list(range(len(blocks_idx)))
            random.shuffle(order)
            
            for i in order:
                if time.time() > deadline - 2.0:
                    return blocks_idx, coverage, False
                
                b_remove = blocks_idx[i]
                idxs_remove = block_indices[b_remove]
                
                # Find exposed subsets (covered only by b_remove)
                exposed_mask = (coverage == 1)
                exposed = idxs_remove[exposed_mask[idxs_remove]]
                
                if len(exposed) == 0:
                    coverage[idxs_remove] -= 1
                    blocks_idx.pop(i)
                    blocks_set.discard(b_remove)
                    improved = True
                    blocks_idx = reduce_blocks(blocks_idx, coverage)
                    blocks_set = set(blocks_idx)
                    break
                else:
                    first_exposed = int(exposed[0])
                    candidate_blocks = t_subset_to_blocks[first_exposed]
                    
                    best_swap = -1
                    best_gain = -1
                    exposed_sorted = np.sort(exposed)
                    
                    for b_candidate in candidate_blocks:
                        if b_candidate == b_remove or b_candidate in blocks_set:
                            continue
                        
                        # Check if b_candidate covers all exposed using sorted array
                        cand_sorted = sorted_block_indices[b_candidate]
                        # Use searchsorted for membership check
                        # For each exposed element, check if it's in cand_sorted
                        pos = np.searchsorted(cand_sorted, exposed_sorted)
                        pos = np.clip(pos, 0, len(cand_sorted) - 1)
                        if not np.all(cand_sorted[pos] == exposed_sorted):
                            continue
                        
                        # Calculate gain: number of subsets in b_candidate with coverage == 0
                        cand_idxs = block_indices[b_candidate]
                        new_covers = np.sum(coverage[cand_idxs] == 0)
                        
                        if new_covers > best_gain:
                            best_gain = new_covers
                            best_swap = b_candidate
                    
                    if best_swap >= 0:
                        coverage[idxs_remove] -= 1
                        blocks_idx.pop(i)
                        blocks_set.discard(b_remove)
                        
                        idxs_add = block_indices[best_swap]
                        coverage[idxs_add] += 1
                        blocks_idx.append(best_swap)
                        blocks_set.add(best_swap)
                        
                        improved = True
                        blocks_idx = reduce_blocks(blocks_idx, coverage)
                        blocks_set = set(blocks_idx)
                        break
        
        return blocks_idx, coverage, improved

    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 10.0:
        num_restarts += 1
        # Vary temperature to explore different solution spaces
        temp = 0.05 + 0.1 * (num_restarts % 5)
        res = greedy_construct(num_restarts, temp)
        if res[0] is None:
            continue
            
        blocks_idx, coverage, _ = res
        
        # Reduce
        blocks_idx = reduce_blocks(blocks_idx, coverage)
        
        current_score = len(blocks_idx)
        if current_score < best_score:
            best_score = current_score
            best_blocks = [list(all_k_blocks[b]) for b in blocks_idx]
            
            # Local Search
            time_left = deadline - time.time()
            ls_time = min(20.0, time_left * 0.25)
            if ls_time > 1.0:
                blocks_idx, coverage, _ = local_search(blocks_idx, coverage, ls_time)
                
                # Final reduction
                blocks_idx = reduce_blocks(blocks_idx, coverage)
                current_score = len(blocks_idx)
                if current_score < best_score:
                    best_score = current_score
                    best_blocks = [list(all_k_blocks[b]) for b in blocks_idx]
                    
    if best_blocks is None:
        # Fallback: return a valid (if not optimal) covering if we ran out of time
        # This shouldn't happen if greedy completes at least once
        return None
        
    return best_blocks
