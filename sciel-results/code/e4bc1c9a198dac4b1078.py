import itertools
import random
import time

# Precomputation for the specific case v=19, k=7, t=4
# This allows for very fast greedy and local search steps.

def _precompute_covering_data(v, k, t):
    """
    Precomputes bitmasks for blocks and lists of t-subsets for each block.
    Returns:
    - t_subsets: List of all t-subsets (as tuples)
    - t_to_idx: Dict mapping t-subset tuple to index
    - block_indices: List of all k-block indices (combinations of range(v) of size k)
    - block_masks: List of bitmasks (int) where bit j is set if t_subsets[j] is in the block
    - block_tsub_indices: List of lists of indices of t-subsets contained in the block
    """
    t_subsets = list(itertools.combinations(range(v), t))
    t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    block_combinations = list(itertools.combinations(range(v), k))
    num_blocks = len(block_combinations)
    
    block_masks = [0] * num_blocks
    block_tsub_indices = [[] for _ in range(num_blocks)]
    
    for i, block in enumerate(block_combinations):
        # Find all t-subsets within this block
        mask = 0
        indices = []
        for ts in itertools.combinations(block, t):
            idx = t_to_idx[ts]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_tsub_indices[i] = indices
        
    return t_subsets, t_to_idx, block_combinations, block_masks, block_tsub_indices, num_t

# Global cache for precomputation to avoid re-doing it if called multiple times
# Though construct is called once, it's good practice.
_PRECOMP_CACHE = {}

def construct(v, k, t):
    # Check if we have precomputed data for this specific problem size
    key = (v, k, t)
    if key not in _PRECOMP_CACHE:
        if v > 20 or k > 10: # Fallback for larger problems to avoid memory blowup
            # Simple greedy fallback
            uncovered = set(itertools.combinations(range(v), t))
            blocks = []
            while uncovered:
                # Pick a random uncovered t-subset to start a block? 
                # Or pick the t-subset that is in the fewest blocks? 
                # For simplicity, just pick the first one and greedily expand.
                start_tsub = next(iter(uncovered))
                block = list(start_tsub)
                # Greedily add points to cover more uncovered t-subsets
                while len(block) < k:
                    best_point = None
                    best_gain = -1
                    for p in range(v):
                        if p in block:
                            continue
                        # Calculate gain: how many new t-subsets does adding p cover?
                        # A t-subset is covered if it is a subset of block + {p}
                        # It must contain p and t-1 elements from block.
                        gain = 0
                        for rest in itertools.combinations(block, t-1):
                            ts = tuple(sorted(rest + (p,)))
                            if ts in uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = p
                    if best_point is None:
                        break
                    block.append(best_point)
                block.sort()
                blocks.append(block)
                # Update uncovered
                for ts in itertools.combinations(block, t):
                    uncovered.discard(ts)
            return blocks
            
        _PRECOMP_CACHE[key] = _precompute_covering_data(v, k, t)
    
    t_subsets, t_to_idx, block_combinations, block_masks, block_tsub_indices, num_t = _PRECOMP_CACHE[key]
    num_blocks = len(block_combinations)
    
    # Precompute: for each t-subset index, which blocks cover it
    # This helps with swap-based local search
    t_to_blocks = [[] for _ in range(num_t)]
    for i in range(num_blocks):
        for t_idx in block_tsub_indices[i]:
            t_to_blocks[t_idx].append(i)
    
    # Store total coverage size for each block (number of t-subsets it covers)
    block_sizes = [len(block_tsub_indices[i]) for i in range(num_blocks)]
    
    # Time budget management
    start_time = time.time()
    time_budget = 110.0  # Leave 10s margin
    
    best_sol = None
    best_len = float('inf')
    
    full_mask = (1 << num_t) - 1
    
    # Try many trials with different random seeds
    num_trials = 50
    
    for trial in range(num_trials):
        if time.time() - start_time > time_budget:
            break
            
        random.seed(trial * 7919 + 13)  # Different seed per trial
        
        uncovered = full_mask
        sol = []
        
        # Greedy Phase with tie-breaking: prefer blocks with smaller total size
        # This leaves more flexibility for later
        while uncovered:
            best_block_idx = -1
            max_covered = 0
            best_total_size = float('inf')
            
            for i in range(num_blocks):
                covered = block_masks[i] & uncovered
                if covered:
                    cnt = covered.bit_count()
                    if cnt > max_covered or (cnt == max_covered and block_sizes[i] < best_total_size):
                        max_covered = cnt
                        best_block_idx = i
                        best_total_size = block_sizes[i]
            
            if best_block_idx == -1:
                break
                
            sol.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]
            
        # Local Search Phase 1: Remove Redundant Blocks
        changed = True
        while changed:
            if time.time() - start_time > time_budget:
                break
            changed = False
            # Calculate frequency of each t-subset coverage
            freq = [0] * num_t
            for b_idx in sol:
                for t_idx in block_tsub_indices[b_idx]:
                    freq[t_idx] += 1
            
            new_sol = []
            for b_idx in sol:
                is_redundant = True
                for t_idx in block_tsub_indices[b_idx]:
                    if freq[t_idx] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_sol.append(b_idx)
            
            if len(new_sol) < len(sol):
                sol = new_sol
                changed = True
        
        # Local Search Phase 2: Try to replace one block with another
        # For each block in sol, check if there's a replacement that covers all
        # its unique t-subsets and doesn't increase the total count
        if len(sol) > 1:
            improved = True
            while improved:
                if time.time() - start_time > time_budget:
                    break
                improved = False
                freq = [0] * num_t
                for b_idx in sol:
                    for t_idx in block_tsub_indices[b_idx]:
                        freq[t_idx] += 1
                
                for pos, b_idx in enumerate(sol):
                    # Find t-subsets that are uniquely covered by this block
                    unique_ts = []
                    for t_idx in block_tsub_indices[b_idx]:
                        if freq[t_idx] == 1:
                            unique_ts.append(t_idx)
                    
                    if not unique_ts:
                        continue
                    
                    # Try to find a replacement block that covers all unique_ts
                    # and has good overlap with other blocks
                    best_replace = -1
                    best_score = float('inf')
                    
                    # Only consider blocks that cover all unique t-subsets
                    for candidate in range(num_blocks):
                        if candidate == b_idx:
                            continue
                        # Check if candidate covers all unique t-subsets
                        covers_all = True
                        for t_idx in unique_ts:
                            if not (block_masks[candidate] >> t_idx) & 1:
                                covers_all = False
                                break
                        if not covers_all:
                            continue
                        
                        # Score: how much overlap with existing sol (excluding b_idx)
                        # We want to maximize overlap to minimize "waste"
                        overlap = 0
                        for t_idx in block_tsub_indices[candidate]:
                            if t_idx not in unique_ts and freq[t_idx] > 0:
                                overlap += 1
                        
                        if overlap < best_score:
                            best_score = overlap
                            best_replace = candidate
                    
                    if best_replace != -1:
                        # Check if replacement is valid: after swapping, all t-subsets still covered
                        # Remove b_idx, add best_replace
                        new_sol = sol[:pos] + [best_replace] + sol[pos+1:]
                        
                        # Verify coverage
                        new_freq = [0] * num_t
                        for nb_idx in new_sol:
                            for t_idx in block_tsub_indices[nb_idx]:
                                new_freq[t_idx] += 1
                        
                        valid = True
                        for t_idx in range(num_t):
                            if new_freq[t_idx] == 0:
                                valid = False
                                break
                        
                        if valid and len(new_sol) < len(sol):
                            sol = new_sol
                            improved = True
                            break
                        elif valid:
                            # Same length but maybe better structure, accept it
                            sol = new_sol
                            improved = True
                            break
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_sol = sol.copy()
            
    if best_sol is None:
        return []
        
    # Convert indices back to lists of points
    result_blocks = []
    for b_idx in best_sol:
        result_blocks.append(list(block_combinations[b_idx]))
        
    return result_blocks
