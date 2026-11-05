import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a vectorized greedy algorithm
    with multiple randomized restarts and redundancy removal.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Create a mapping from t-subset to index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Total number of possible blocks
    n_blocks = len(list(itertools.combinations(range(v), k)))
    
    def greedy_construct(seed=None):
        """Vectorized greedy covering construction."""
        if seed is not None:
            random.seed(seed)
        
        # uncovered[i] = True if t-subset i is not yet covered
        uncovered = np.ones(n_t_subsets, dtype=bool)
        uncovered_count = n_t_subsets
        blocks = []
        
        while uncovered_count > 0:
            # Find the most constrained uncovered t-subset (random tie-breaking)
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Pick a random uncovered t-subset as the "seed"
            seed_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
            seed_block = list(all_t_subsets[seed_idx])
            
            # Build the block greedily: add points that cover the most uncovered t-subsets
            block = seed_block.copy()
            
            while len(block) < k:
                # For each candidate point, compute how many new t-subsets it would cover
                best_point = None
                best_gain = -1
                
                # Get the (t-1)-subsets of the current block
                block_set = set(block)
                t_minus_1_subsets = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for t1_sub in t_minus_1_subsets:
                        # Form the t-subset by adding x
                        t_sub = tuple(sorted(t1_sub + (x,)))
                        idx = t_subset_to_idx.get(t_sub, -1)
                        if idx != -1 and uncovered[idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is None or best_gain <= 0:
                    # Pick a random point to fill
                    available = [x for x in range(v) if x not in block_set]
                    if available:
                        best_point = random.choice(available)
                    else:
                        break
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered set
            for t_sub in itertools.combinations(block, t):
                idx = t_subset_to_idx.get(tuple(sorted(t_sub)), -1)
                if idx != -1 and uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
        
        return blocks
    
    def remove_redundant(blocks):
        """Remove redundant blocks (those whose t-subsets are all covered by other blocks)."""
        if not blocks:
            return blocks
        
        n = len(blocks)
        # For each block, check if it's redundant
        keep = [True] * n
        
        # Precompute t-subsets for each block
        block_t_subsets = []
        for block in blocks:
            t_subs = set()
            for t_sub in itertools.combinations(block, t):
                t_subs.add(tuple(sorted(t_sub)))
            block_t_subsets.append(t_subs)
        
        # For each block, check if removing it leaves all its t-subsets covered
        for i in range(n):
            if not keep[i]:
                continue
            
            # Check if all t-subsets of block i are covered by other blocks
            redundant = True
            for t_sub in block_t_subsets[i]:
                covered = False
                for j in range(n):
                    if j == i or not keep[j]:
                        continue
                    if t_sub in block_t_subsets[j]:
                        covered = True
                        break
                if not covered:
                    redundant = False
                    break
            
            if redundant:
                keep[i] = False
        
        return [blocks[i] for i in range(n) if keep[i]]
    
    def is_valid(blocks):
        """Check if all t-subsets are covered."""
        covered = set()
        for block in blocks:
            for t_sub in itertools.combinations(block, t):
                covered.add(tuple(sorted(t_sub)))
        return len(covered) == n_t_subsets
    
    # Try multiple seeds and keep the best result
    best_blocks = None
    best_size = float('inf')
    
    # First, try a deterministic greedy
    try:
        blocks = greedy_construct(seed=42)
        if is_valid(blocks):
            blocks = remove_redundant(blocks)
            if len(blocks) < best_size:
                best_size = len(blocks)
                best_blocks = blocks
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        return best_blocks if best_blocks is not None else []
    
    # Try multiple randomized restarts
    n_restarts = 20
    for seed in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        try:
            blocks = greedy_construct(seed=seed + 100)
            if is_valid(blocks):
                blocks = remove_redundant(blocks)
                if len(blocks) < best_size:
                    best_size = len(blocks)
                    best_blocks = blocks
        except Exception:
            continue
    
    if best_blocks is None:
        # Fallback: simple greedy without redundancy removal
        try:
            blocks = greedy_construct(seed=0)
            best_blocks = blocks
        except Exception:
            best_blocks = []
    
    return best_blocks
