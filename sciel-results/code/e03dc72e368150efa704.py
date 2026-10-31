"""Optimized covering design C(v,k,t) with deadline guard."""
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Total number of t-subsets
    total_t = 1
    for i in range(t):
        total_t *= (v - i)
        total_t //= (i + 1)

    # Represent each t-subset as a bitmask or use a set of tuples
    # For v=19, t=4, C(19,4)=3876 which is manageable
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    blocks = []

    # Precompute for each potential block (k-subset) which t-subsets it covers
    # This is expensive for large v,k, but for v=19,k=7: C(19,7)=50388, each has C(7,4)=35 t-subsets
    # Let's use a greedy approach with efficient gain computation

    # Strategy: Greedy set cover where we pick the block that covers the most uncovered t-subsets
    # But we also need to construct blocks, not just pick from all possible blocks.
    # Two-phase: 1) Greedy construction of blocks, 2) Local improvement

    def time_ok():
        return time.time() < deadline

    # Phase 1: Greedy block construction
    # For each uncovered t-subset, build a block around it
    # To speed up, we use a smarter gain function

    # Precompute: for each point x and each (t-1)-subset s, 
    # the list of t-subsets that contain s and x
    # Actually, let's just do a fast greedy

    while uncovered and time_ok():
        # Pick the most "constrained" uncovered t-subset (one that appears in fewest potential blocks)
        # Or just pick the first one for speed, but let's be a bit smarter
        # Pick an uncovered t-subset that has the fewest extensions to k-subsets
        # For simplicity and speed, pick the first uncovered t-subset
        target = min(uncovered)  # deterministic, but let's randomize for diversity
        
        # Build a block containing target
        block = list(target)
        
        # Greedily add points that maximize coverage of uncovered t-subsets
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_t = tuple(sorted(rest + (x,)))
                    if new_t in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1 or best_gain <= 0:
                # Just add a random point
                candidates = [x for x in range(v) if x not in block]
                if candidates:
                    best_x = random.choice(candidates)
                else:
                    break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Remove covered t-subsets
        for tsub in itertools.combinations(block, t):
            uncovered.discard(tsub)
        
        # Periodic check
        if len(blocks) % 50 == 0 and not time_ok():
            break

    # Phase 2: Local improvement - try to replace blocks with better ones
    if time_ok() and uncovered:
        # Try to fix remaining uncovered t-subsets with new blocks
        for _ in range(100):  # limited iterations
            if not time_ok() or not uncovered:
                break
            target = min(uncovered)
            block = list(target)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_t = tuple(sorted(rest + (x,)))
                        if new_t in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    candidates = [x for x in range(v) if x not in block]
                    if candidates:
                        best_x = random.choice(candidates)
                    else:
                        break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for tsub in itertools.combinations(block, t):
                uncovered.discard(tsub)

    # Remove duplicate blocks
    seen = set()
    unique_blocks = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(list(b))

    return unique_blocks
