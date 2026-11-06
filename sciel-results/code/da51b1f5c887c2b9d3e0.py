import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Try n=21, 20, 19...
    for n in range(22, 1, -1):
        if time.time() > deadline:
            break
            
        # Try a few random graphs
        max_attempts = 5 if n >= 21 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Generate random graph
            # Use adjacency list of bitmasks
            adj_masks = [0] * n
            
            # Generate edges
            # p=0.5
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            # Check for regular induced subgraph
            # We check sizes from 6 to n
            # If any exists, graph is invalid.
            
            valid = True
            # We can optimize by checking size 6 first.
            # If a graph has no regular subgraph of size 6, it's a strong candidate.
            # It might still have one of size 7+.
            
            for k in range(6, n + 1):
                # Early exit if time is tight
                if time.time() > deadline:
                    valid = False
                    break
                
                # Iterate combinations
                # To speed up, we can sample if n is large? No, we must be sure.
                # But if we just want to find A valid graph, we can stop checking
                # as soon as we find a bad one.
                
                # Optimization: If k is large, there are fewer combinations.
                # Start with k=6.
                
                found_bad = False
                for S in itertools.combinations(range(n), k):
                    # Build mask
                    mask = 0
                    for v in S:
                        mask |= (1 << v)
                    
                    # Check degrees
                    degs = []
                    regular = True
                    first_deg = -1
                    for v in S:
                        d = (adj_masks[v] & mask).bit_count()
                        if first_deg == -1:
                            first_deg = d
                        elif d != first_deg:
                            regular = False
                            break
                    
                    if regular:
                        found_bad = True
                        break
                
                if found_bad:
                    valid = False
                    break
            
            if valid:
                # Extract edges
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj_masks[i] >> j) & 1:
                            edges.append([i, j])
                
                return {'n': n, 'edges': edges}
        
    # Fallback: return a small valid graph
    # n=5 has no subgraph of size 6, so it is trivially valid.
    # But we want to maximize n.
    # If we fail n=20, we return n=5?
    # It's better to return the largest n we *could* verify or a known good one.
    # Since we don't have a hardcoded one, and random search might fail,
    # we should probably just return n=5 to be safe, or try n=20 with more attempts?
    # The loop above tries n=22 down to 2.
    # If it finishes the loop without returning, it means it couldn't find one in time.
    # We should return a valid graph. n=5 is always valid.
    
    return {'n': 5, 'edges': []}
