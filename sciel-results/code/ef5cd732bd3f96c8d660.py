import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin

    # Try to find a valid graph for n=20, then 19, etc.
    for target_n in [20, 19, 18, 17, 16, 15]:
        if time.time() > deadline - 5:
            break

        num_seeds = 50 if target_n >= 19 else 100
        
        for seed in range(num_seeds):
            if time.time() > deadline - 5:
                break

            rng = random.Random(seed)
            n = target_n
            adj_masks = [0] * n
            edges = []
            
            # Generate random graph
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
                        edges.append([i, j])

            # Check validity
            if _is_valid_fast(adj_masks, n, deadline):
                return {'n': n, 'edges': edges}

    # Fallback: return a trivially valid small graph
    # n=5 is always valid (no subset of size 6)
    return {'n': 5, 'edges': []}

def _is_valid_fast(adj_masks, n, deadline):
    """
    Check if no subset of size >= 6 induces a regular graph.
    Returns True if valid, False if violation found or timeout.
    """
    if n < 6:
        return True

    # Check sizes from 6 to n
    # We check smaller sizes first as they are more numerous? 
    # Actually, larger sizes are fewer. 
    # But smaller sizes are more likely to be regular in a "bad" graph?
    # In a random graph, regular induced subgraphs are rare.
    # We must check ALL to prove validity.
    
    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            # Time check
            if time.time() > deadline:
                return False # Timeout, treat as invalid to try smaller n or next seed
            
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            degs = []
            for v in subset:
                degs.append((adj_masks[v] & mask).bit_count())
            
            if len(set(degs)) == 1:
                return False # Violation found

    return True
