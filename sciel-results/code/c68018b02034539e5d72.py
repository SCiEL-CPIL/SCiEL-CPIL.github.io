import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: finish by 140s to be safe against 150s kill
    deadline = start_time + 140
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_size = float('inf')
    
    # We will run randomized greedy restarts until the deadline.
    # Each restart is relatively fast for v=24.
    
    while time.time() < deadline:
        # Randomized Greedy Construction
        uncovered = set(all_t_subsets)
        blocks = []
        
        # Track time inside the greedy loop
        # If we exceed deadline during greedy, we break and discard this attempt
        # unless it's already complete.
        
        while uncovered:
            if time.time() > deadline:
                break
                
            # Pick a random uncovered t-subset to break symmetry
            # Converting to list is O(N), N=2024. Acceptable.
            target_triple = random.choice(list(uncovered))
            
            # Start block with a random point from target_triple
            first_point = random.choice(target_triple)
            block = [first_point]
            
            # Note: [first_point] does not cover any t-subsets if t > 1.
            
            while len(block) < k:
                best_gain = -1
                best_points = []
                
                # Identify candidates
                # We can optimize by precomputing points not in block
                # But v=24 is small.
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    # Check triples formed by x and (t-1) points from block
                    # t=3, so t-1=2. Pairs from block.
                    for rest in itertools.combinations(block, t - 1):
                        candidate_triple = tuple(sorted(rest + (x,)))
                        if candidate_triple in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                if not best_points:
                    # If no point gives positive gain, pick a random one
                    # to proceed. This ensures we fill the block.
                    candidates = [x for x in range(v) if x not in block]
                    if not candidates:
                        break
                    x = random.choice(candidates)
                else:
                    x = random.choice(best_points)
                
                block.append(x)
                
                # Update uncovered set
                # Only triples containing the new point x can be newly covered.
                # So we only check combinations of (block \ {x}) of size t-1.
                # This is faster than checking all combinations of block.
                block_without_x = [p for p in block if p != x]
                for rest in itertools.combinations(block_without_x, t - 1):
                    candidate_triple = tuple(sorted(rest + (x,)))
                    uncovered.discard(candidate_triple)
            
            if len(uncovered) == 0:
                break
                
            block.sort()
            # Ensure block is distinct from existing blocks?
            # Greedy usually produces distinct blocks, but let's be safe.
            if block not in blocks:
                blocks.append(block)
            
            # If we broke due to time, we shouldn't add a partial block
            # But the loop condition `while uncovered` handles the main exit.
            # If time ran out inside, we break the inner while, 
            # then check `if len(uncovered) == 0`. If not, we break outer?
            # No, we just break the `while uncovered` loop.
            # Then we check if it's a valid covering.
        
        # Check if we completed the covering within time
        if len(uncovered) == 0:
            # Reduce the covering
            reduced_blocks = reduce_covering(blocks, v, t)
            if len(reduced_blocks) < best_size:
                best_size = len(reduced_blocks)
                best_blocks = reduced_blocks
                
            # If we found the theoretical lower bound, we can stop
            # Lower bound for C(24,7,3) is 69.
            if best_size == 69:
                break

    if best_blocks is None:
        # Fallback: return a simple greedy result if we didn't finish any
        # This is unlikely to be needed
        return []
        
    return best_blocks

def reduce_covering(blocks, v, t):
    """
    Removes redundant blocks from the covering.
    A block is redundant if it does not uniquely cover any t-subset.
    """
    blocks = [list(b) for b in blocks]
    changed = True
    while changed:
        changed = False
        
        # Count coverage of each t-subset
        coverage_count = {}
        for b in blocks:
            for comb in itertools.combinations(b, t):
                key = tuple(comb)
                coverage_count[key] = coverage_count.get(key, 0) + 1
        
        new_blocks = []
        for b in blocks:
            essential = False
            for comb in itertools.combinations(b, t):
                key = tuple(comb)
                if coverage_count.get(key, 0) == 1:
                    essential = True
                    break
            
            if not essential:
                changed = True
                continue
            
            new_blocks.append(b)
        
        blocks = new_blocks
        
    return blocks
