"""
Fast greedy covering design C(24,7,3) with redundancy removal.
"""
import time
import itertools
import sys
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # Precompute all t-subsets as frozensets for fast membership
    all_t_subs = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_t_subs)
    
    blocks = []
    
    # Precompute for each pair of points, the list of third points that form a t-subset
    # For t=3: for each pair (a,b), the t-subsets containing both are {a,b,c} for c != a,b
    # We'll use a different approach: for a partial block, compute gain efficiently
    
    while uncovered and time.time() < deadline:
        # Pick the first uncovered t-subset to seed the block
        seed = min(uncovered)
        block = list(seed)
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k:
            block_set = set(block)
            best_gain = -1
            best_point = -1
            
            # For each candidate point not in block, compute how many new t-subsets it covers
            for x in range(v):
                if x in block_set:
                    continue
                
                # Count uncovered t-subsets that include x and (t-1) points from block
                gain = 0
                # Generate all (t-1)-subsets of block
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                break
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Remove all t-subsets covered by this block
        for sub in itertools.combinations(block, t):
            uncovered.discard(frozenset(sub))
    
    # Redundancy removal: try to remove each block if all its t-subsets are still covered
    # Use a counting approach
    if time.time() < deadline - 10:
        # Count how many blocks cover each t-subset
        from collections import Counter
        cover_count = Counter()
        for b in blocks:
            for sub in itertools.combinations(b, t):
                cover_count[frozenset(sub)] += 1
        
        # Try removing blocks in reverse order (later blocks are often more redundant)
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            b = blocks[i]
            # Check if all t-subsets of b are covered by other blocks
            redundant = True
            for sub in itertools.combinations(b, t):
                fs = frozenset(sub)
                if cover_count[fs] <= 1:
                    redundant = False
                    break
            if redundant:
                # Decrement counts
                for sub in itertools.combinations(b, t):
                    cover_count[frozenset(sub)] -= 1
                blocks.pop(i)
    
    return blocks
