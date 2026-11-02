"""Covering design C(24,7,3) - intra-block swap local search with vectorized coverage."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 142.0
    start_time = time.time()
    
    # Precompute all t-subsets and their index
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # For each block (k-subset), precompute which t-subsets it covers (as a list of indices)
    # We'll compute this on the fly but cache for the current set of blocks
    
    def block_to_t_indices(block):
        """Return list of t-subset indices covered by this block."""
        return [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
    
    def greedy_cover(seed=0, time_budget=5.0):
        """Fast greedy covering to get an initial valid solution."""
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        blocks = []
        t0 = time.time()
        
        while uncovered.any():
            if time.time() - t0 > time_budget:
                break
            
            # Pick a random uncovered t-subset
            uncov_indices = np.where(uncovered)[0]
            if len(uncov_indices) == 0:
                break
            start_idx = rng.choice(uncov_indices)
            block = list(all_t_subsets[start_idx])
            
            # Greedily add points to maximize new coverage
            while len(block) < k:
                best_x = None
                best_gain = -1
                # Only consider adding points that create new t-subsets
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = t_subset_to_idx.get(ts)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None or best_gain == 0:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for idx in block_to_t_indices(block):
                uncovered[idx] = False
        
        return blocks
    
    def compute_coverage(blocks):
        """Compute coverage count array and per-block t-index lists."""
        cover_count = np.zeros(n_t, dtype=np.int32)
        block_t_indices = []
        for b in blocks:
            t_idxs = block_to_t_indices(b)
            block_t_indices.append(t_idxs)
            for idx in t_idxs:
                cover_count[idx] += 1
        return cover_count, block_t_indices
    
    def swap_local_search(blocks, seed=0, time_budget=20.0):
        """
        Local search via intra-block swaps.
        Move: pick two blocks A, B and points a in A, b in B.
        Swap: A' = A - {a} + {b}, B' = B - {b} + {a}.
        Accept if covering remains valid (all t-subsets still covered) and total redundancy decreases.
        Redundancy metric: sum(cover_count[i]^2) - n_t (lower is better, means less overlap).
        """
        rng = random.Random(seed)
        blocks = [list(b) for b in blocks]
        n_blocks = len(blocks)
        
        if n_blocks < 2:
            return blocks
        
        cover_count, block_t_indices = compute_coverage(blocks)
        current_redundancy = int(np.sum(cover_count.astype(np.int64) ** 2))
        
        t0 = time.time()
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() - t0 < time_budget and no_improve_count < max_no_improve:
            # Pick two distinct blocks
            i, j = rng.sample(range(n_blocks), 2)
            block_i = blocks[i]
            block_j = blocks[j]
            
            if len(block_i) != k or len(block_j) != k:
                continue
            
            # Pick a point to swap out of block_i and block_j
            a = rng.choice(block_i)
            b = rng.choice(block_j)
            
            if a == b:
                continue
            
            # Create new blocks
            new_block_i = [x for x in block_i if x != a] + [b]
            new_block_j = [x for x in block_j if x != b] + [a]
            
            # Check for duplicates within blocks
            if len(set(new_block_i)) != k or len(set(new_block_j)) != k:
                continue
            
            # Check if new blocks are same as old
            if set(new_block_i) == set(block_i) and set(new_block_j) == set(block_j):
                continue
            
            # Compute new coverage impact
            # Remove old blocks' coverage
            old_i_tidxs = block_t_indices[i]
            old_j_tidxs = block_t_indices[j]
            
            # Temporarily subtract
            for idx in old_i_tidxs:
                cover_count[idx] -= 1
            for idx in old_j_tidxs:
                cover_count[idx] -= 1
            
            # Check if any t-subset becomes uncovered (count drops to 0)
            valid = True
            for idx in old_i_tidxs:
                if cover_count[idx] == 0:
                    valid = False
                    break
            if valid:
                for idx in old_j_tidxs:
                    if cover_count[idx] == 0:
                        valid = False
                        break
            
            if not valid:
                # Restore
                for idx in old_i_tidxs:
                    cover_count[idx] += 1
                for idx in old_j_tidxs:
                    cover_count[idx] += 1
                no_improve_count += 1
                continue
            
            # Add new blocks' coverage
            new_i_tidxs = block_to_t_indices(new_block_i)
            new_j_tidxs = block_to_t_indices(new_block_j)
            
            for idx in new_i_tidxs:
                cover_count[idx] += 1
            for idx in new_j_tidxs:
                cover_count[idx] += 1
            
            # Compute new redundancy
            new_redundancy = int(np.sum(cover_count.astype(np.int64) ** 2))
            
            if new_redundancy < current_redundancy:
                # Accept the swap
                blocks[i] = new_block_i
                blocks[j] = new_block_j
                block_t_indices[i] = new_i_tidxs
                block_t_indices[j] = new_j_tidxs
                current_redundancy = new_redundancy
                no_improve_count = 0
            else:
                # Revert
                for idx in new_i_tidxs:
                    cover_count[idx] -= 1
                for idx in new_j_tidxs:
                    cover_count[idx] -= 1
                for idx in old_i_tidxs:
                    cover_count[idx] += 1
                for idx in old_j_tidxs:
                    cover_count[idx] += 1
                no_improve_count += 1
        
        return blocks
    
    def try_remove_blocks(blocks, time_budget=2.0):
        """Try to remove redundant blocks greedily."""
        blocks = [list(b) for b in blocks]
        t0 = time.time()
        
        while time.time() - t0 < time_budget:
            cover_count, block_t_indices = compute_coverage(blocks)
            removed = False
            for i in range(len(blocks)):
                redundant = True
                for idx in block_t_indices[i]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    del blocks[i]
                    removed = True
                    break
            if not removed:
                break
        
        return blocks
    
    # Main search loop
    best_blocks = None
    best_len = float('inf')
    
    # Initial greedy
    init_blocks = greedy_cover(0, time_budget=8.0)
    if len(init_blocks) > 0:
        best_blocks = init_blocks
        best_len = len(init_blocks)
    
    # Multiple restarts with swap local search
    for seed in range(1, 200):
        if time.time() > deadline - 15.0:
            break
        
        # Generate initial covering
        init_blocks = greedy_cover(seed, time_budget=5.0)
        if len(init_blocks) == 0:
            continue
        
        # Apply swap local search
        ls_blocks = swap_local_search(init_blocks, seed=seed, time_budget=18.0)
        
        # Try to remove redundant blocks
        ls_blocks = try_remove_blocks(ls_blocks, time_budget=2.0)
        
        if len(ls_blocks) < best_len:
            best_len = len(ls_blocks)
            best_blocks = ls_blocks
    
    # Final validation
    if best_blocks is None:
        return []
    
    covered = set()
    for block in best_blocks:
        for ts in itertools.combinations(block, t):
            covered.add(t_subset_to_idx[ts])
    
    if covered != set(range(n_t)):
        # Fallback to a valid greedy
        return greedy_cover(0, time_budget=10.0)
    
    return best_blocks
