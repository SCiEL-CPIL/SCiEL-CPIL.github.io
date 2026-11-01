import time
import random
import itertools
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Generate initial covering using a fast greedy
    # We will use a greedy that picks the block covering the most uncovered t-subsets.
    # To speed up, we don't iterate all C(v,k) blocks.
    # We pick a random uncovered t-subset, and try to find a good block containing it.
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy restarts
    for seed in range(10000):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        
        # Shuffle points to break symmetry
        points = list(range(v))
        random.shuffle(points)
        
        # Map original points to shuffled indices for this iteration?
        # No, just shuffle the order of consideration.
        # Actually, for covering design, permuting points is isomorphic.
        # We need to vary the *choice* of blocks.
        
        current_uncovered = set(all_t_subsets)
        blocks = []
        
        # Greedy Phase
        while current_uncovered:
            if time.time() > deadline:
                break
            
            # Pick a target t-subset.
            # Strategy: Pick one that is "hard" to cover.
            # Approximation: Pick random.
            target = random.choice(list(current_uncovered))
            
            fixed = set(target)
            candidates = [x for x in range(v) if x not in fixed]
            
            best_gain = -1
            best_block = None
            
            # Instead of iterating ALL combinations, sample a subset?
            # Or iterate all if k-t is small.
            # k-t = 3. C(16,3) = 560. This is fast.
            
            for comb in itertools.combinations(candidates, k - t):
                block = tuple(sorted(fixed.union(comb)))
                gain = 0
                for ts in itertools.combinations(block, t):
                    if ts in current_uncovered:
                        gain += 1
                        if gain > best_gain:
                            # Early exit not possible for max, but we can break if gain == C(k,t)
                            if gain == 56: # Max possible for k=8, t=5
                                break
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    if best_gain == 56:
                        break
            
            if best_block is None:
                break # Should not happen
                
            blocks.append(list(best_block))
            
            for ts in itertools.combinations(best_block, t):
                current_uncovered.discard(ts)
        
        if time.time() > deadline:
            break
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [b[:] for b in blocks]
            
            # If we hit a very good number, we can spend time on Local Search
            if best_count <= 340:
                # Run Local Search
                optimized_blocks = local_search(best_blocks, v, k, t, deadline)
                if optimized_blocks and len(optimized_blocks) < best_count:
                    best_count = len(optimized_blocks)
                    best_blocks = optimized_blocks
                    
    return best_blocks

def local_search(blocks, v, k, t, deadline):
    """
    Iterated Local Search:
    1. Check for redundant blocks. Remove if found.
    2. Try to replace a block with another that covers its critical subsets.
    """
    if not blocks:
        return None
    
    # Precompute t-subsets for each block
    block_subsets = []
    for b in blocks:
        block_subsets.append(list(itertools.combinations(b, t)))
    
    # Count coverage for each t-subset
    # Use a dict: t_subset -> count
    cover_count = {}
    for subs in block_subsets:
        for s in subs:
            cover_count[s] = cover_count.get(s, 0) + 1
            
    n_blocks = len(blocks)
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to remove redundant blocks
        for i in range(len(blocks)):
            # Check if all subsets of blocks[i] are covered by other blocks
            redundant = True
            for s in block_subsets[i]:
                if cover_count[s] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for s in block_subsets[i]:
                    cover_count[s] -= 1
                blocks.pop(i)
                block_subsets.pop(i)
                improved = True
                break # Restart loop after modification
        
        if improved:
            continue
            
        # 2. Try to replace a block to potentially enable future removals
        # Pick a random block to try to improve
        i = random.randint(0, len(blocks) - 1)
        b_old = blocks[i]
        old_subs = block_subsets[i]
        
        # Find critical subsets: those covered ONLY by b_old
        critical = []
        for s in old_subs:
            if cover_count[s] == 1:
                critical.append(s)
                
        if not critical:
            # Should have been removed in step 1, but maybe race condition or logic gap
            continue
            
        # Try to find a new block b_new that covers all 'critical'
        # b_new must contain the intersection of all critical subsets
        core = set(critical[0])
        for s in critical[1:]:
            core.intersection_update(set(s))
            
        # If core is too large (>k), impossible
        if len(core) > k:
            continue
            
        # We need to choose k - len(core) points from the remaining v - len(core) points
        # such that the resulting block covers all 'critical'
        
        # Heuristic: Try random extensions biased towards points in critical subsets
        found_new_block = None
        
        # Number of attempts
        attempts = 100
        for _ in range(attempts):
            # Pick random points from candidates
            candidates = [x for x in range(v) if x not in core]
            # Bias: pick points that appear in critical subsets
            # Simple: just random from candidates
            ext = random.sample(candidates, k - len(core))
            b_new = tuple(sorted(core.union(ext)))
            
            # Check if b_new covers all critical
            covers_all = True
            for s in critical:
                if not set(s).issubset(set(b_new)):
                    covers_all = False
                    break
                    
            if covers_all:
                # Check if b_new is already in blocks
                # If it is, it's not a change (or it is, but doesn't help redundancy)
                # We want a *different* block to potentially change the structure
                if b_new in blocks:
                    continue
                    
                found_new_block = b_new
                break
                
        if found_new_block:
            # Replace blocks[i] with found_new_block
            # Update cover_count
            # Remove counts for old_subs
            for s in old_subs:
                cover_count[s] -= 1
            # Add counts for new_subs
            new_subs = list(itertools.combinations(found_new_block, t))
            for s in new_subs:
                cover_count[s] = cover_count.get(s, 0) + 1
                
            blocks[i] = list(found_new_block)
            block_subsets[i] = new_subs
            # We don't mark 'improved' because size is same, but structure changed
            # This allows further removals in the next iteration
            
    return blocks

if __name__ == "__main__":
    # Test with task parameters
    v, k, t = 19, 8, 5
    blocks = construct(v, k, t)
    print(f"Found covering with {len(blocks)} blocks")
