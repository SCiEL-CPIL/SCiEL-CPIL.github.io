"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 20s safety margin
    
    # Total number of t-subsets
    total_t_subs = math.comb(v, t)
    
    # Greedy covering with beam search for block selection
    # We'll use a multi-pass approach:
    # 1. Fast greedy to get an initial cover
    # 2. Local search / block replacement to improve
    
    # Fast greedy
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered and time.time() < deadline - 5.0:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                cnt = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    if sub in uncovered:
                        cnt += 1
                return cnt
            best_x = None
            best_gain = -1
            for x in range(v):
                if x not in block:
                    g = gain(x)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
            if best_x is not None:
                block.append(best_x)
            else:
                break
        block.sort()
        blocks.append(tuple(block))
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
    
    # If greedy didn't finish, just return what we have (will be invalid)
    if uncovered:
        # Try to finish with a simpler approach
        for x in range(v):
            if time.time() > deadline - 2.0:
                break
            for sub in itertools.combinations([x] + list(range(v)), t):
                if sub in uncovered and time.time() > deadline - 2.0:
                    break
        # Just return what we have
        return [list(b) for b in blocks]
    
    # Now try to improve via local search
    # Count how many times each t-subset is covered
    from collections import Counter
    cover_counts = Counter()
    for b in blocks:
        for sub in itertools.combinations(b, t):
            cover_counts[sub] += 1
    
    # Try to remove redundant blocks
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline - 3.0:
                break
            b = blocks[i]
            # Check if removing this block leaves any t-subset uncovered
            can_remove = True
            for sub in itertools.combinations(b, t):
                if cover_counts[sub] == 1:
                    can_remove = False
                    break
            if can_remove:
                for sub in itertools.combinations(b, t):
                    cover_counts[sub] -= 1
                blocks.pop(i)
                improved = True
                break  # Restart loop as indices changed
    
    # Try block replacement: replace a block with one that covers more uncovered/less-covered subsets
    # This is more complex, so we'll try a simple random perturbation
    
    # Random restarts with different starting points
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    for _ in range(5):
        if time.time() > deadline - 10.0:
            break
        
        # Try a randomized greedy
        random.seed(random.randint(0, 2**31))
        uncovered2 = set(itertools.combinations(range(v), t))
        blocks2 = []
        
        while uncovered2 and time.time() < deadline - 10.0:
            # Pick a random uncovered subset as seed
            seed_sub = random.choice(list(uncovered2))
            block = list(seed_sub)
            
            while len(block) < k:
                def gain2(x):
                    cnt = 0
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        if sub in uncovered2:
                            cnt += 1
                    return cnt
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x not in block:
                        g = gain2(x)
                        if g > best_gain:
                            best_gain = g
                            best_x = x
                if best_x is not None:
                    block.append(best_x)
                else:
                    break
            block.sort()
            blocks2.append(tuple(block))
            for sub in itertools.combinations(block, t):
                uncovered2.discard(sub)
        
        if not uncovered2:
            # Remove redundant blocks
            cover_counts2 = Counter()
            for b in blocks2:
                for sub in itertools.combinations(b, t):
                    cover_counts2[sub] += 1
            
            improved2 = True
            while improved2 and time.time() < deadline - 10.0:
                improved2 = False
                for i in range(len(blocks2)):
                    if time.time() > deadline - 10.0:
                        break
                    b = blocks2[i]
                    can_remove = True
                    for sub in itertools.combinations(b, t):
                        if cover_counts2[sub] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for sub in itertools.combinations(b, t):
                            cover_counts2[sub] -= 1
                        blocks2.pop(i)
                        improved2 = True
                        break
            
            if len(blocks2) < best_count:
                best_blocks = list(blocks2)
                best_count = len(blocks2)
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
