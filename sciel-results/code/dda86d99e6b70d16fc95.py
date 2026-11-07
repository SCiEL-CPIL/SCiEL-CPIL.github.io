import time
from itertools import combinations

def construct():
    # Start time for deadline guard
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s limit

    # Graph data
    n = 17
    edges = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]]

    # Precompute adjacency bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    best_count = 0
    best_subset = None
    vertices = list(range(n))

    # Iterate over subset sizes from 8 to 17
    for k in range(8, n + 1):
        # Check deadline
        if time.time() - start_time > deadline:
            break

        for combo in combinations(vertices, k):
            # Check deadline
            if time.time() - start_time > deadline:
                break
            
            # Create bitmask for this subset
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            # Calculate degrees within the induced subgraph
            degrees = []
            for v in combo:
                # Count neighbors of v that are also in the subset
                deg = bin(adj_mask[v] & mask).count('1')
                degrees.append(deg)
            
            # Count distinct degrees
            distinct_count = len(set(degrees))
            
            if distinct_count > best_count:
                best_count = distinct_count
                best_subset = combo
                # If we achieve the theoretical maximum (k distinct degrees), we can stop
                if best_count == k:
                    break
        if best_count == k:
            break

    # If no subset was found (should not happen), return a valid default
    if best_subset is None:
        return list(range(8))
    
    return list(best_subset)

# EVOLVE-BLOCK-END
