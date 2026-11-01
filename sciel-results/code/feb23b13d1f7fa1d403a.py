import time
import random
import itertools
import sys
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Precompute for each t-subset, the list of all blocks (k-subsets) that contain it
    # This is expensive: C(v,t) * C(v-t, k-t) total entries
    # C(19,5) = 11628, C(14,3) = 364, so ~4.2M entries. Might be too slow.
    # Alternative: for each t-subset, compute which blocks contain it on the fly?
    # Or precompute a mapping from t-subset to a list of candidate blocks?
    
    # Let's think differently. 
    # Strategy: Greedy set cover with randomization + local search.
    
    # First, let's precompute the "gain" of each block for a given uncovered set.
    # But we don't want to iterate all C(19,8) = 75582 blocks each time.
    
    # Better approach: 
    # 1. Start with a random valid covering (or greedy).
    # 2. Local search: try to remove redundant blocks, or replace blocks.
    
    # Let's try a different approach:
    # - Generate many random blocks
    # - Use a greedy selection to pick the best covering from a pool of random blocks
    # - Then do local search to improve
    
    # Actually, let's use a smarter greedy:
    # At each step, pick the block that covers the most uncovered t-subsets.
    # To speed this up, we can maintain a count of uncovered t-subsets for each potential block.
    
    # Precompute all possible blocks? C(19,8) = 75582. That's manageable.
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute the set of t-subsets it covers
    block_to_tsubsets = {}
    for b in all_blocks:
        ts_list = list(itertools.combinations(b, t))
        block_to_tsubsets[b] = ts_list
    
    # We'll use a greedy approach with randomization
    best_blocks = None
    best_count = float('inf')
    
    for seed in range(5000):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Find the block that covers the most uncovered t-subsets
            best_gain = -1
            best_block = None
            
            # Randomize the order of blocks to break symmetry
            shuffled_blocks = all_blocks[:]
            random.shuffle(shuffled_blocks)
            
            for b in shuffled_blocks:
                gain = 0
                for ts in block_to_tsubsets[b]:
                    if ts in uncovered:
                        gain += 1
                        if gain > best_gain:
                            if gain == len(block_to_tsubsets[b]):  # Max possible
                                break
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                    if best_gain == len(block_to_tsubsets[b]):
                        break
            
            if best_block is None or best_gain <= 0:
                break  # Should not happen
            
            blocks.append(list(best_block))
            
            for ts in block_to_tsubsets[best_block]:
                uncovered.discard(ts)
        
        if time.time() > deadline:
            break
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [b[:] for b in blocks]
            
            # If we have a good solution, try local search
            if best_count <= 350:
                optimized = local_search(best_blocks, v, k, t, deadline)
                if optimized and len(optimized) < best_count:
                    best_count = len(optimized)
                    best_blocks = optimized
    
    return best_blocks

def local_search(blocks, v, k, t, deadline):
    """
    Local search: 
    1. Remove redundant blocks (blocks where all t-subsets are covered by other blocks)
    2. Try to replace a block with a different one that allows further removals
    """
    if not blocks:
        return None
    
    # Precompute t-subsets for each block
    block_tsubsets = [list(itertools.combinations(b, t)) for b in blocks]
    
    # Count how many blocks cover each t-subset
    cover_count = defaultdict(int)
    for ts_list in block_tsubsets:
        for ts in ts_list:
            cover_count[ts] += 1
    
    n_blocks = len(blocks)
    
    while time.time() < deadline:
        improved = False
        
        # Step 1: Remove redundant blocks
        for i in range(len(blocks)):
            redundant = True
            for ts in block_tsubsets[i]:
                if cover_count[ts] <= 1:
                    redundant = False
                    break
            if redundant:
                for ts in block_tsubsets[i]:
                    cover_count[ts] -= 1
                blocks.pop(i)
                block_tsubsets.pop(i)
                improved = True
                break
        
        if improved:
            continue
        
        # Step 2: Try to replace a block
        # Pick a random block
        i = random.randint(0, len(blocks) - 1)
        old_block = blocks[i]
        old_ts = block_tsubsets[i]
        
        # Find critical t-subsets (covered only by this block)
        critical = [ts for ts in old_ts if cover_count[ts] == 1]
        
        if not critical:
            continue
        
        # Try to find a new block that covers all critical t-subsets
        # The new block must contain the intersection of all critical t-subsets
        core = set(critical[0])
        for ts in critical[1:]:
            core.intersection_update(set(ts))
        
        if len(core) > k:
            continue
        
        # Try random extensions
        found = False
        for _ in range(50):
            remaining = [x for x in range(v) if x not in core]
            ext = random.sample(remaining, k - len(core))
            new_block = tuple(sorted(core.union(ext)))
            
            # Check if new_block covers all critical
            covers = True
            for ts in critical:
                if not set(ts).issubset(set(new_block)):
                    covers = False
                    break
            
            if not covers:
                continue
            
            # Check if new_block is already in blocks
            if new_block in blocks:
                continue
            
            # Replace
            for ts in old_ts:
                cover_count[ts] -= 1
            
            new_ts_list = list(itertools.combinations(new_block, t))
            for ts in new_ts_list:
                cover_count[ts] += 1
            
            blocks[i] = list(new_block)
            block_tsubsets[i] = new_ts_list
            found = True
            break
        
        if not found:
            # Try a different random block
            continue
    
    return blocks

if __name__ == "__main__":
    v, k, t = 19, 8, 5
    blocks = construct(v, k, t)
    print(f"Found covering with {len(blocks)} blocks")
