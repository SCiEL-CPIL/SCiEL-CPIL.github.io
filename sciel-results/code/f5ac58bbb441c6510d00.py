import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140
    
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, list of subset indices it covers
    block_covered = []
    for b in all_blocks:
        idxs = [subset_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_covered.append(idxs)
    
    # For each subset, list of block indices that cover it
    subset_to_blocks = [[] for _ in range(n_subsets)]
    for b_idx, covered_idx in enumerate(block_covered):
        for s_idx in covered_idx:
            subset_to_blocks[s_idx].append(b_idx)
    
    def greedy_fast(seed_offset=0, max_blocks=200):
        """Greedy using subset-to-blocks mapping for fast gain updates."""
        rng = random.Random(seed_offset)
        uncovered_count = [1] * n_subsets  # 1 if uncovered, 0 if covered
        gains = [len(block_covered[i]) for i in range(n_blocks)]  # initial gain
        selected = []
        
        for step in range(max_blocks):
            if time.time() > deadline - 20:
                break
            if max(uncovered_count) == 0:
                break
            
            # Find best block
            max_gain = max(gains)
            if max_gain == 0:
                break
            
            # Collect candidates with near-max gain
            threshold = max(0, max_gain - 4)
            candidates = [i for i in range(n_blocks) if gains[i] >= threshold]
            best_idx = rng.choice(candidates)
            selected.append(best_idx)
            
            # Update: mark covered subsets, update gains
            for s_idx in block_covered[best_idx]:
                if uncovered_count[s_idx] == 1:
                    uncovered_count[s_idx] = 0
                    # This subset was just covered, so all blocks covering it lose 1 from gain
                    for b_idx in subset_to_blocks[s_idx]:
                        gains[b_idx] -= 1
        
        return selected
    
    def check_complete(blocks):
        covered = [0] * n_subsets
        for idx in blocks:
            for s_idx in block_covered[idx]:
                covered[s_idx] = 1
        return all(c == 1 for c in covered)
    
    def try_remove(blocks, i):
        """Check if removing block i still covers everything."""
        # Count coverage
        cov_count = [0] * n_subsets
        for j, idx in enumerate(blocks):
            if j != i:
                for s_idx in block_covered[idx]:
                    cov_count[s_idx] += 1
        return all(c > 0 for c in cov_count)
    
    def local_search(blocks, seed_offset=0, max_iters=300):
        blocks = list(blocks)
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline - 10:
                    break
                if try_remove(blocks, i):
                    blocks.pop(i)
                    improved = True
                    break
            if not improved:
                break
        return blocks
    
    best_result = None
    best_count = float('inf')
    
    n_starts = 40
    for s in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        blocks = greedy_fast(seed_offset=s * 7919)
        
        if not blocks:
            continue
        
        if not check_complete(blocks):
            # Repair
            cov = [0] * n_subsets
            for idx in blocks:
                for s_idx in block_covered[idx]:
                    cov[s_idx] += 1
            uncovered = [i for i in range(n_subsets) if cov[i] == 0]
            while uncovered and len(blocks) < 200:
                if time.time() > deadline - 30:
                    break
                # Find block covering most uncovered
                best_b = -1
                best_gain = -1
                for b_idx in range(n_blocks):
                    gain = sum(1 for s in block_covered[b_idx] if cov[s] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b_idx
                if best_b == -1 or best_gain == 0:
                    break
                blocks.append(best_b)
                for s_idx in block_covered[best_b]:
                    cov[s_idx] += 1
                uncovered = [i for i in range(n_subsets) if cov[i] == 0]
        
        if not check_complete(blocks):
            continue
        
        improved = local_search(blocks, seed_offset=s * 7919)
        
        if len(improved) < best_count:
            best_count = len(improved)
            best_result = improved
    
    if best_result is not None:
        return [list(all_blocks[idx]) for idx in best_result]
    
    # Fallback
    blocks = greedy_fast(seed_offset=123456, max_blocks=250)
    if blocks and check_complete(blocks):
        return [list(all_blocks[idx]) for idx in blocks]
    
    return []
