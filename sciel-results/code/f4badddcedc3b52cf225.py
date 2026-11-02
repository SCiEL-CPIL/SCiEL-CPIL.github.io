import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Generate all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage for each block
    block_covers = []
    triple_to_blocks = defaultdict(list)
    
    for i, b in enumerate(all_blocks):
        covers = []
        for ts in itertools.combinations(b, t):
            idx = tsubset_to_idx[ts]
            covers.append(idx)
            triple_to_blocks[idx].append(i)
        block_covers.append(np.array(covers, dtype=np.int32))
        
    def greedy_cover_random(seed):
        rng = random.Random(seed)
        covered = np.zeros(num_tsubsets, dtype=bool)
        blocks = []
        
        while not covered.all():
            best_block_idx = -1
            best_gain = -1
            
            # Optimization: Only check blocks that cover at least one uncovered triple?
            # Hard to do efficiently without scanning all.
            # But 346k * 35 is ~12M ops. In Python, this is ~1s.
            # We have ~10s per attempt. So ~10 steps? No, greedy takes ~76 steps.
            # 76 * 1s = 76s. Too slow.
            
            # We need faster gain calculation.
            # Vectorize:
            # gains = np.array([np.sum(~covered[c]) for c in block_covers])
            # This is still a Python loop over 346k items.
            
            # Alternative:
            # Maintain a list of uncovered triples.
            # For each block, gain = intersection size with uncovered.
            
            # Let's try to speed up the greedy by only considering a subset of blocks?
            # Or use a faster language? No, must be Python.
            
            # Let's try to optimize the Python loop.
            # Pre-compute block_covers as a list of lists?
            # np.sum is fast, but the loop is the bottleneck.
            
            # Maybe we can use a heuristic:
            # Pick a random uncovered triple.
            # Only consider blocks that cover this triple.
            # This reduces candidates to ~C(21, 6) = 54264? No, blocks containing a specific triple.
            # Total blocks: C(24,7).
            # Blocks containing specific 3 points: C(21, 4) = 5985.
            # This is 1/57th of the total.
            # 5985 * 35 = 210k ops. Very fast.
            
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
                
            # Pick a random uncovered triple
            target_ts_idx = uncovered_indices[rng.randint(0, len(uncovered_indices)-1)]
            
            # Get blocks covering this triple
            candidates = triple_to_blocks[target_ts_idx]
            
            # Evaluate gain for these candidates
            best_gain = -1
            best_block_idx = -1
            
            for i in candidates:
                c = block_covers[i]
                gain = np.sum(~covered[c])
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                elif gain == best_gain and best_gain > 0 and rng.random() < 0.1:
                    best_block_idx = i
                    
            if best_block_idx == -1:
                # Fallback: pick any block with max gain (shouldn't happen)
                break
                
            blocks.append(best_block_idx)
            covered[block_covers[best_block_idx]] = True
            
        return blocks

    def remove_redundant(blocks):
        if not blocks:
            return []
            
        counts = np.zeros(num_tsubsets, dtype=np.int32)
        for b_idx in blocks:
            counts[block_covers[b_idx]] += 1
            
        remaining = []
        for b_idx in blocks:
            c = block_covers[b_idx]
            if np.all(counts[c] > 1):
                counts[c] -= 1
            else:
                remaining.append(b_idx)
                
        return remaining

    def local_search_seeded(initial_blocks, seed, time_limit):
        current_blocks = list(initial_blocks)
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        rng = random.Random(seed)
        
        while time.time() < time_limit:
            current_blocks = remove_redundant(current_blocks)
            
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_blocks = list(current_blocks)
                
            if not current_blocks:
                break
                
            b_old_idx = rng.choice(current_blocks)
            b_old = current_blocks[b_old_idx]
            
            counts = np.zeros(num_tsubsets, dtype=np.int32)
            for b_idx in current_blocks:
                counts[block_covers[b_idx]] += 1
            counts[block_covers[b_old]] -= 1
            
            uncovered = np.where(counts == 0)[0]
            
            if len(uncovered) == 0:
                continue
                
            # Find best block to add
            # Candidates: Blocks that cover at least one uncovered triple.
            # We can union the candidate lists for all uncovered triples.
            # But uncovered might be large.
            # Let's just pick one uncovered triple to restrict candidates, like in greedy.
            # This is a heuristic, but it works well for local search too.
            
            target_ts_idx = uncovered[rng.randint(0, len(uncovered)-1)]
            candidates = triple_to_blocks[target_ts_idx]
            
            best_gain = -1
            best_new_idx = -1
            
            for i in candidates:
                if i in current_blocks:
                    continue
                c = block_covers[i]
                gain = np.sum(counts[c] == 0)
                
                if gain > best_gain:
                    best_gain = gain
                    best_new_idx = i
                elif gain == best_gain and best_gain > 0 and rng.random() < 0.1:
                    best_new_idx = i
                    
            if best_new_idx == -1:
                continue
                
            new_blocks = [b for b in current_blocks if b != b_old]
            new_blocks.append(best_new_idx)
            current_blocks = new_blocks
            
        return best_blocks

    best_blocks = None
    best_count = float('inf')
    
    seed = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        init_blocks = greedy_cover_random(seed)
        
        ls_time = time.time() + min(remaining - 2, 10)
        refined = local_search_seeded(init_blocks, seed, ls_time)
        
        if len(refined) < best_count:
            best_count = len(refined)
            best_blocks = refined
            
        seed += 1
        
    if best_blocks is None:
        return []
        
    return [list(all_blocks[i]) for i in best_blocks]
