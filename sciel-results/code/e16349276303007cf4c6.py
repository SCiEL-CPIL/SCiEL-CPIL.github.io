"""
Combines exact bitmask validity checking with an optimized recursive 
branch-and-bound search using memoization and symmetry breaking.
"""
import time
import sys

# Increase recursion limit just in case, though depth is at most 14
sys.setrecursionlimit(10000)

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    deadline = time.time() + 50.0  # Safety margin for 60s limit
    n = P['n']
    edges = P['edges']
    
    # 1. Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    size = 1 << n
    
    # 2. Precompute is_clique and is_independent for all masks
    # This is O(2^n * n) which is ~ 16M * 14 operations, very fast in Python if optimized, 
    # but let's use the incremental property to make it O(2^n).
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, size):
        lbit = mask & -mask
        v = lbit.bit_length() - 1
        rest = mask ^ lbit
        
        # Check clique: rest is clique AND v connected to all in rest
        if is_clique[rest] and (adj_mask[v] & rest) == rest:
            is_clique[mask] = True
            
        # Check independent: rest is independent AND v not connected to any in rest
        if is_independent[rest] and (adj_mask[v] & rest) == 0:
            is_independent[mask] = True
            
    # 3. Precompute list of valid submasks for each mask? 
    # No, that's memory heavy. Instead, we iterate submasks on the fly in the search.
    # But we can precompute a boolean `valid[mask]` for quick lookup.
    valid = [False] * size
    for mask in range(1, size):
        if is_clique[mask] or is_independent[mask]:
            valid[mask] = True

    # 4. Recursive Search with Memoization and Symmetry Breaking
    # State: remaining_mask
    # We enforce that the part we pick must contain the lowest set bit of remaining_mask.
    # This reduces the search space by a factor of the size of the part on average.
    
    memo = {}
    # best_solution_parts: stores the actual partition for the best score found
    # We will store the mask of the first part chosen in the optimal path for reconstruction.
    parent_part = {} 
    
    def min_parts(rem_mask):
        if time.time() > deadline:
            return -1 # Signal timeout
        
        if rem_mask == 0:
            return 0
            
        if rem_mask in memo:
            return memo[rem_mask]
            
        # Find the lowest set bit
        lbit = rem_mask & -rem_mask
        
        best_val = n + 1 # Upper bound
        
        # Iterate through all submasks of rem_mask that contain lbit
        # sub is the part we are trying to form
        sub = rem_mask
        while sub:
            # Optimization: Only consider submasks that contain the lowest bit
            # to avoid symmetry (order of parts doesn't matter)
            if (sub & lbit) == 0:
                sub = (sub - 1) & rem_mask
                continue
                
            if valid[sub]:
                prev_val = min_parts(rem_mask ^ sub)
                if prev_val != -1:
                    val = prev_val + 1
                    if val < best_val:
                        best_val = val
                        parent_part[rem_mask] = sub
                        # If we found a solution with 1 part, it's optimal
                        if best_val == 1:
                            memo[rem_mask] = 1
                            return 1
            sub = (sub - 1) & rem_mask
            
        if best_val == n + 1:
            # Should not happen for connected logic unless no valid part found, 
            # but single vertex is always valid.
            best_val = n # Fallback
            
        memo[rem_mask] = best_val
        return best_val

    # Run the search
    # Note: min_parts modifies parent_part globally. 
    # We need to be careful if we restart or if the function is called multiple times.
    # Since construct() is called once per evaluation, this is fine.
    
    result_score = min_parts(full_mask)
    
    if result_score == -1:
        # Timeout fallback: Greedy
        parts = []
        for v in range(n):
            parts.append([v])
        return parts

    # 5. Reconstruct the partition
    current = full_mask
    parts_masks = []
    while current > 0:
        p = parent_part.get(current, 0)
        if p == 0:
            # Fallback if parent not recorded (should not happen if score > 0)
            # Take lowest bit as a single vertex
            p = current & -current
        parts_masks.append(p)
        current ^= p
        
    # Convert masks to lists of vertices
    result = []
    for mask in parts_masks:
        verts = []
        for v in range(n):
            if mask & (1 << v):
                verts.append(v)
        result.append(verts)
        
    return result

if __name__ == "__main__":
    # Simple test to ensure it runs
    res = construct()
    assert len(res) > 0
    print(f"Found partition with {len(res)} parts: {res}")
