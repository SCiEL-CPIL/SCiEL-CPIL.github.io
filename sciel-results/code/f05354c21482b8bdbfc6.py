"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Build adjacency matrix for fast lookup
    adj_matrix = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj_matrix[u][v] = True
        adj_matrix[v][u] = True
        
    def get_distinct_degrees(subset):
        degs = []
        for v in subset:
            deg = 0
            for u in subset:
                if u != v and adj_matrix[v][u]:
                    deg += 1
            degs.append(deg)
        return len(set(degs))

    best_subset = None
    best_distinct = 0
    
    # Try all subset sizes from min_v to n
    for size in range(min_v, n + 1):
        # Multiple restarts for local search to find good subsets of this size
        for restart in range(200):
            # Initialize random subset
            subset = set(random.sample(range(n), size))
            
            current_distinct = get_distinct_degrees(subset)
            
            # Local search: try swaps
            no_improve_count = 0
            max_no_improve = 50
            while no_improve_count < max_no_improve:
                # Pick random in and out
                v_in = random.choice(list(subset))
                # Choose v_out from complement
                complement = [v for v in range(n) if v not in subset]
                if not complement:
                    break
                v_out = random.choice(complement)
                
                # Try swap
                new_subset = subset.copy()
                new_subset.remove(v_in)
                new_subset.add(v_out)
                
                new_distinct = get_distinct_degrees(new_subset)
                
                if new_distinct > current_distinct:
                    subset = new_subset
                    current_distinct = new_distinct
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            if current_distinct > best_distinct:
                best_distinct = current_distinct
                best_subset = subset
                
                # Early termination if we hit the theoretical max for this size
                if best_distinct == size:
                    break
        # If we found a perfect distinct count for this size, we might still find a better one for larger size?
        # Max distinct degrees is limited by size. So size=17 can have up to 17 distinct degrees.
        # If we found 8 distinct degrees with size 8, size 9 can have up to 9.
        # So we should continue to larger sizes.
        
    # Ensure we return a list of Python ints
    result = [int(v) for v in best_subset]
    return result
# EVOLVE-BLOCK-END
