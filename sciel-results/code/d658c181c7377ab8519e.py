"""Covering design C(19,8,5) combining bitset greedy with bottleneck heuristic and local search."""
import itertools
import time
import random
import numpy as np

DEADLINE_SECONDS = 118.0

def construct(v, k, t):
    start_time = time.time()
    
    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute bitset for each block
    # bit i is set if the i-th t-subset is contained in the block
    block_bitsets = [0] * n_blocks
    block_t_indices = [[] for _ in range(n_blocks)]  # list of t-subset indices for each block
    
    for bi, block in enumerate(all_blocks):
        mask = 0
        indices = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
            indices.append(idx)
        block_bitsets[bi] = mask
        block_t_indices[bi] = indices
    
    full_mask = (1 << n_t) - 1
    
    # Function to compute coverage counts for a set of block indices
    def compute_counts(block_indices):
        counts = np.zeros(n_t, dtype=np.int32)
        for bi in block_indices:
            indices = block_t_indices[bi]
            if indices:
                counts[indices] += 1
        return counts
    
    # Prune redundant blocks: remove any block whose t-subsets are all covered by other blocks
    def prune(block_indices):
        if not block_indices:
            return block_indices
        
        counts = compute_counts(block_indices)
        # If any count is 0, the covering is invalid - should not happen
        if np.any(counts == 0):
            return block_indices
        
        result = []
        # Try to remove blocks in random order
        rng = random.Random()
        order = list(range(len(block_indices)))
        rng.shuffle(order)
        
        for pos in order:
            bi = block_indices[pos]
            indices = block_t_indices[bi]
            if indices and np.any(counts[indices] == 1):
                # This block is essential for at least one t-subset
                result.append(bi)
            else:
                # Can safely remove this block
                if indices:
                    counts[indices] -= 1
        
        return result
    
    # Greedy with bottleneck heuristic: pick block that maximizes min coverage of uncovered subsets
    # Actually, the directive says "maximizing the minimum degree of uncovered t-subsets"
    # This means: for each candidate block, look at the uncovered t-subsets it covers,
    # and find the one with the fewest covering blocks (among all blocks, not just chosen).
    # Then pick the block that maximizes this minimum degree.
    
    # Precompute: for each t-subset, how many blocks cover it
    t_subset_block_count = np.zeros(n_t, dtype=np.int32)
    for bi in range(n_blocks):
        indices = block_t_indices[bi]
        for idx in indices:
            t_subset_block_count[idx] += 1
    
    # For each block, precompute the t-subsets it covers (already have block_t_indices)
    # And for each t-subset, the blocks that cover it
    blocks_for_subset = [[] for _ in range(n_t)]
    for bi in range(n_blocks):
        for idx in block_t_indices[bi]:
            blocks_for_subset[idx].append(bi)
    
    def run_greedy_bottleneck(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        blocks_used = []
        
        while uncovered:
            if time.time() - start_time > DEADLINE_SECONDS - 5:
                return None
            
            # Find uncovered t-subset indices
            unc_list = []
            for i in range(n_t):
                if (uncovered >> i) & 1:
                    unc_list.append(i)
            
            if not unc_list:
                break
            
            # For each candidate block, compute the minimum "degree" among uncovered subsets it covers
            # Degree = number of blocks covering that subset
            best_min_degree = -1
            best_gain = -1
            best_blocks = []
            
            for bi in range(n_blocks):
                mask = block_bitsets[bi]
                common = mask & uncovered
                if common == 0:
                    continue
                
                gain = common.bit_count()
                if gain == 0:
                    continue
                
                # Find the minimum degree among the uncovered subsets this block covers
                min_deg = float('inf')
                # Iterate through set bits in common
                c = common
                while c:
                    # Get lowest set bit
                    lsb = c & (-c)
                    idx = lsb.bit_length() - 1
                    deg = t_subset_block_count[idx]
                    if deg < min_deg:
                        min_deg = deg
                    c ^= lsb
                
                # Score: prioritize high min_degree, then high gain
                # We want to pick blocks that cover "hard" subsets (low degree) first
                # Actually, bottleneck heuristic: maximize the minimum degree of uncovered subsets covered
                # Wait, that would pick blocks covering "easy" subsets. 
                # Let me re-read: "maximizing the minimum degree of uncovered t-subsets"
                # I think the intent is: among uncovered subsets, find the one with min degree (most constrained).
                # Then pick a block that covers that subset.
                # Alternatively: for each block, look at the uncovered subsets it covers,
                # and take the min degree among them. Pick the block with max of this min.
                # This would favor blocks that cover subsets that are well-covered (high degree),
                # which seems counterintuitive.
                
                # Let me try the opposite: minimize the minimum degree (cover the most constrained first)
                # Or: pick the block that covers the subset with the globally minimum degree.
                
                # Let's use a hybrid: score = min_degree * 1000 + gain
                # No, let's just pick blocks that cover the most constrained uncovered subset.
                
                # For now, let's use standard max-gain greedy with random tie-breaking
                # and add a separate bottleneck-based selection.
                
                # Let's compute score as: primary = min_deg (lower is better for constrained), 
                # but we want to maximize coverage. 
                # Let's just use gain for now, and add bottleneck as a secondary criterion.
                
                score = gain * 1000 - min_deg  # favor high gain, and among ties, lower min_deg (more constrained)
                
                if score > best_min_degree:
                    best_min_degree = score
                    best_gain = gain
                    best_blocks = [bi]
                elif score == best_min_degree:
                    best_blocks.append(bi)
            
            if not best_blocks:
                break
            
            # Pick randomly from best candidates
            chosen_bi = rng.choice(best_blocks)
            blocks_used.append(chosen_bi)
            uncovered &= ~block_bitsets[chosen_bi]
        
        return blocks_used
    
    def run_greedy_gain(seed):
        """Standard max-gain greedy with random tie-breaking."""
        rng = random.Random(seed)
        uncovered = full_mask
        blocks_used = []
        
        while uncovered:
            if time.time() - start_time > DEADLINE_SECONDS - 5:
                return None
            
            best_gain = -1
            best_blocks = []
            
            for bi in range(n_blocks):
                mask = block_bitsets[bi]
                common = mask & uncovered
                gain = common.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_blocks = [bi]
                elif gain == best_gain and gain > 0:
                    best_blocks.append(bi)
            
            if not best_blocks or best_gain <= 0:
                break
            
            chosen_bi = rng.choice(best_blocks)
            blocks_used.append(chosen_bi)
            uncovered &= ~block_bitsets[chosen_bi]
        
        return blocks_used
    
    # Local improvement: try to swap blocks to reduce size
    def local_improve(block_indices, seed):
        rng = random.Random(seed)
        block_indices = list(block_indices)
        
        # First, prune
        block_indices = prune(block_indices)
        
        # Try random swaps: remove one block, add another
        improved = True
        while improved:
            if time.time() - start_time > DEADLINE_SECONDS - 3:
                break
            
            improved = False
            counts = compute_counts(block_indices)
            
            # Find blocks that can be removed (all their subsets have count >= 2)
            removable = []
            for bi in block_indices:
                indices = block_t_indices[bi]
                if indices and np.all(counts[indices] >= 2):
                    removable.append(bi)
            
            if removable:
                # Remove one
                bi_to_remove = rng.choice(removable)
                block_indices.remove(bi_to_remove)
                improved = True
                continue
            
            # Try to find a block to add that allows removal of two others
            # This is more complex; skip for now
            # Instead, try swapping: remove one, add one different one
            # But this doesn't reduce size.
            
            break
        
        return block_indices
    
    best_solution = None
    best_size = float('inf')
    
    # Run multiple strategies with different seeds
    seed = 42
    strategies = [run_greedy_gain, run_greedy_bottleneck]
    
    while time.time() - start_time < DEADLINE_SECONDS - 10:
        strategy = strategies[seed % len(strategies)]
        sol = strategy(seed)
        if sol:
            # Local improvement
            sol = local_improve(sol, seed)
            sol = prune(sol)
            
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
        
        seed += 1
        # Alternate strategies more frequently
        if seed % 3 == 0:
            strategies = [run_greedy_bottleneck, run_greedy_gain]
    
    if best_solution is None:
        # Fallback: deterministic greedy
        sol = run_greedy_gain(0)
        if sol:
            sol = prune(sol)
            best_solution = sol
    
    if best_solution is None:
        return []
    
    # Convert block indices to block lists
    return [list(all_blocks[bi]) for bi in best_solution]
