import itertools
import time
import random
from collections import Counter
import numpy as np
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_tsubsets)}
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    t_subsets_per_block = int(math.comb(k, t))
    
    # Precompute block_indices: shape (num_blocks, C(k, t))
    block_indices = np.empty((num_blocks, t_subsets_per_block), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        block_indices[i] = [t_subset_to_idx[sub] for sub in itertools.combinations(block, t)]
    
    # Precompute frequency of each t-subset across all blocks
    freq = np.bincount(block_indices.flatten(), minlength=num_tsubsets)
    inv_freq = 1.0 / (freq + 1e-9)
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple strategies/restarts
    seed = 0
    while time.time() < deadline - 20:
        strategy = seed % 4
        
        if strategy == 0:
            blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='greedy')
        elif strategy == 1:
            blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='weighted', inv_freq=inv_freq)
        elif strategy == 2:
            blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='randomized', inv_freq=inv_freq, seed=seed)
        else:
            blocks = _ucb_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, inv_freq=inv_freq, seed=seed)
        
        if blocks is None:
            seed += 1
            continue
            
        # Post-processing: Remove redundant blocks
        blocks = _remove_redundant(blocks, t)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        seed += 1
        
        if best_len <= 76:
            break

    # Phase 2: Local Search on the best solution found
    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = _local_search(best_blocks, all_blocks, block_indices, num_tsubsets, deadline, t)
        
    if best_blocks is None:
        best_blocks = _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='greedy')
        if best_blocks is None:
             best_blocks = []
             
    return best_blocks

def _greedy_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, mode='greedy', inv_freq=None, seed=0):
    uncovered = np.ones(num_tsubsets, dtype=bool)
    selected_indices = []
    
    while np.any(uncovered):
        if time.time() > deadline:
            return None
            
        if mode == 'greedy':
            gains = uncovered[block_indices].sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            uncovered[block_indices[best_idx]] = False
            
        elif mode == 'weighted':
            w = np.where(uncovered, inv_freq, 0.0)
            gains = w[block_indices].sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            uncovered[block_indices[best_idx]] = False
            
        elif mode == 'randomized':
            gains = uncovered[block_indices].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 10:
                rng = random.Random(seed + len(selected_indices))
                candidates = rng.sample(list(candidates), 10)
            
            best_idx = candidates[np.random.choice(len(candidates))]
            selected_indices.append(best_idx)
            uncovered[block_indices[best_idx]] = False
            
    if np.any(uncovered):
        return None

    return [list(all_blocks[i]) for i in selected_indices]

def _ucb_cover_vectorized(all_blocks, block_indices, num_tsubsets, deadline, inv_freq=None, seed=0):
    """
    UCB-based selection: balance exploitation (gain) and exploration (under-used blocks).
    """
    uncovered = np.ones(num_tsubsets, dtype=bool)
    selected_indices = []
    pull_counts = np.zeros(len(all_blocks), dtype=np.int64)
    total_reward = np.zeros(len(all_blocks), dtype=np.float64)
    
    rng = random.Random(seed)
    
    while np.any(uncovered):
        if time.time() > deadline:
            return None
            
        # Compute gains
        gains = uncovered[block_indices].sum(axis=1)
        
        # UCB1: Q(a) + c * sqrt(ln(N) / n(a))
        N = max(1, len(selected_indices))
        # For blocks never pulled, n(a)=0, we treat them as having high exploration value
        # To avoid division by zero, we can use n(a) + 1 or handle separately.
        # Let's use a standard formulation where we only consider blocks with n(a)>0 
        # plus a set of "unexplored" blocks. 
        # Simpler: use n(a) + 1 in denominator to keep it smooth, or just add a large constant for unexplored.
        
        # Let's compute exploration term
        # c = 1.0 is a common default
        c = 1.0
        n = pull_counts.astype(np.float64) + 1.0 # +1 to avoid div by 0 and soften initial exploration
        exploration = c * np.sqrt(np.log(N) / n)
        
        # Reward is the gain.
        # UCB score = gain + exploration
        # But gain is 0 for many blocks. Exploration will drive selection of unexplored blocks.
        ucb_scores = gains + exploration
        
        # To avoid selecting blocks with 0 gain if there are blocks with >0 gain, 
        # we might want to prioritize gain first. 
        # However, UCB is designed to explore. 
        # Let's try: if max gain > 0, we can bias towards high gain.
        # Actually, standard UCB on bandits assumes rewards are stochastic. Here rewards are deterministic given state.
        # So "exploration" in UCB sense means trying blocks that haven't been tried, hoping they might have high gain later?
        # But in set cover, once a block is "used" (selected), it doesn't change. 
        # The "reward" of a block changes as the uncovered set shrinks.
        # So this is a dynamic bandit.
        
        # Let's just pick the argmax of ucb_scores.
        best_idx = np.argmax(ucb_scores)
        
        if gains[best_idx] == 0 and np.any(uncovered):
             # If we picked a block with 0 gain but there are uncovered items, it means all remaining blocks have 0 gain?
             # That shouldn't happen if a valid cover exists.
             # But if we are stuck, break.
             # Actually, if gains[best_idx] == 0, then all gains are 0 (since best_idx is argmax).
             break
             
        selected_indices.append(best_idx)
        pull_counts[best_idx] += 1
        total_reward[best_idx] += gains[best_idx]
        uncovered[block_indices[best_idx]] = False
        
    if np.any(uncovered):
        return None

    return [list(all_blocks[i]) for i in selected_indices]

def _remove_redundant(blocks, t):
    if not blocks:
        return blocks
    
    cover_count = Counter()
    for block in blocks:
        for sub in itertools.combinations(block, t):
            cover_count[sub] += 1
            
    improved = list(blocks)
    n_blocks = len(improved)
    
    for i in range(n_blocks - 1, -1, -1):
        block = improved[i]
        redundant = True
        for sub in itertools.combinations(block, t):
            if cover_count[sub] <= 1:
                redundant = False
                break
        
        if redundant:
            for sub in itertools.combinations(block, t):
                cover_count[sub] -= 1
            improved.pop(i)
            
    return improved

def _local_search(blocks, all_blocks, block_indices, num_tsubsets, deadline, t):
    if not blocks:
        return blocks
        
    current_blocks = list(blocks)
    current_len = len(current_blocks)
    
    # Precompute triples for current blocks for speed?
    # We'll just recompute as needed, it's fast enough for ~80 blocks.
    
    for i in range(len(current_blocks) - 1, -1, -1):
        if time.time() > deadline:
            break
            
        remaining = current_blocks[:i] + current_blocks[i+1:]
        
        covered = set()
        for b in remaining:
            for sub in itertools.combinations(b, t):
                covered.add(sub)
                
        # We need to check if remaining covers everything.
        # Instead of generating all v-choose-t, we can check if the union of triples in remaining is complete?
        # No, we need to know if ANY triple is missing.
        # Efficient way: count coverage of remaining blocks.
        # If any triple has count 0, it's uncovered.
        
        # Let's use the vectorized approach for finding uncovered.
        # Map remaining blocks to indices?
        # We don't have a direct map from block tuple to index in all_blocks quickly without a dict.
        # Let's build a dict for all_blocks indices.
        
        # Actually, let's just use a set of covered triples and check size?
        # No, we need to know WHICH are uncovered to fill them.
        
        # Let's create a boolean array of covered triples.
        # We need the index of each triple in remaining blocks.
        
        # This is getting slow if we do it naively.
        # Let's optimize:
        # 1. Build a mapping from block tuple to index in all_blocks.
        # This is expensive to build every time.
        
        # Alternative: Just use the set difference.
        # all_triples = set(itertools.combinations(range(v), t)) -- this is large (2024 for v=24, t=3).
        # covered_set = set()
        # for b in remaining:
        #    covered_set.update(itertools.combinations(b, t))
        # uncovered_set = all_triples - covered_set
        
        # This is O(num_blocks * C(k,t)). For 80 blocks, 80*35 = 2800 operations. Very fast.
        
        # Let's do this.
        
        # We need all_triples.
        # To avoid regenerating, we can pass it or generate it once.
        # But _local_search doesn't have it.
        # Let's generate it.
        
        all_triples = set(itertools.combinations(range(len(blocks[0][0]) if blocks and blocks[0] else 0, t)) if blocks and blocks[0] else [])
        # Wait, v is not passed. We can infer v from max point in blocks?
        # Or just assume v=24? The task is specific.
        # Let's infer v.
        v = max(max(b) for b in current_blocks) + 1
        
        all_triples = set(itertools.combinations(range(v), t))
        
        covered_set = set()
        for b in remaining:
            covered_set.update(itertools.combinations(b, t))
            
        uncovered_set = all_triples - covered_set
        
        if not uncovered_set:
            current_blocks = remaining
            current_len -= 1
            continue
        
        # Try to cover uncovered_set with minimal new blocks
        new_blocks = list(remaining)
        unc_set = set(uncovered_set)
        
        # Greedy fill
        while unc_set:
            if time.time() > deadline:
                break
                
            best_block = None
            best_gain = 0
            for b in all_blocks:
                gain = 0
                for sub in itertools.combinations(b, t):
                    if sub in unc_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                    
            if best_block is None or best_gain == 0:
                break
                
            new_blocks.append(list(best_block))
            for sub in itertools.combinations(best_block, t):
                unc_set.discard(sub)
                
        if len(new_blocks) < current_len:
            current_blocks = new_blocks
            current_len = len(new_blocks)
            
    return current_blocks
