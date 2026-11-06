import time
import random

N = 9
MAX_V = 1 << N
MAX_TIME = 115.0  # Safety margin before 120s/150s kill

# Precompute bit counts for all possible XOR results (0 to 2^N - 1)
_BIT_COUNT = [i.bit_count() for i in range(MAX_V)]

# Precompute neighbors for all vertices
_NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(MAX_V)]

# Precompute neighbor bitmask for each vertex (bit v is set if v is a neighbor)
_NEIGHBOR_MASK = [0] * MAX_V
for v in range(MAX_V):
    mask = 0
    for b in range(N):
        mask |= 1 << (v ^ (1 << b))
    _NEIGHBOR_MASK[v] = mask

def _warnsdorff_backtrack(deadline):
    """
    Deep backtracking search using Warnsdorff's heuristic and bitmask state.
    """
    best_path = [0]
    best_len = 1
    
    # Initial state
    path = [0]
    # forbidden mask: bit i is 1 if vertex i is forbidden
    # Initially, only vertex 0 is "used" (and thus forbidden for future addition as a new node, 
    # but note: the rule is "distinct vertices". 
    # The "forbidden" set for the *next* move consists of:
    # 1. All vertices already in the path.
    # 2. All neighbors of all vertices in the path EXCEPT the last one.
    # 
    # At start: path=[0]. 
    # Used: {0}. 
    # Non-terminal: {}. 
    # Forbidden for next move: {0} (because it's used). 
    # Neighbors of 0 are NOT forbidden yet because 0 is the terminal.
    forbidden = 1  # bit 0 set
    
    current = 0
    
    # We use a stack to simulate recursion to avoid recursion limit issues, 
    # but Python recursion with sys.setrecursionlimit is often fine for ~200 depth.
    # However, iterative is safer for performance and control.
    # Given the depth is < 512, recursion is acceptable if limit is raised.
    import sys
    sys.setrecursionlimit(10000)
    
    def backtrack(current, forbidden, path):
        nonlocal best_path, best_len
        
        # Time check
        if time.time() > deadline:
            return False
            
        # Update best
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        # Find valid neighbors
        valid_neighbors = []
        for v in _NEIGHBORS[current]:
            # Check if v is forbidden
            if (forbidden >> v) & 1:
                continue
            # v is valid if it's not in the path (covered by forbidden) 
            # and not a neighbor of any non-terminal vertex (covered by forbidden)
            
            # Calculate Warnsdorff score: number of valid neighbors of v
            # new_forbidden if we move to v:
            # forbidden | _NEIGHBOR_MASK[current] | (1 << v)
            new_forbidden = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
            
            count = 0
            for u in _NEIGHBORS[v]:
                if not (new_forbidden >> u) & 1:
                    count += 1
            valid_neighbors.append((count, v))
            
        if not valid_neighbors:
            return False
            
        # Warnsdorff: sort by count ascending (fewest options first)
        # Add a bit of randomness to break ties? 
        # Sorting by (count, random) helps explore different branches.
        # But pure Warnsdorff is often strongest. Let's stick to count, then random index for stability.
        # Actually, just sorting by count is standard.
        valid_neighbors.sort(key=lambda x: x[0])
        
        # To allow exploration of different paths if we restart or if we want to escape 
        # a specific deterministic trap, we could shuffle equal counts.
        # But let's try pure deterministic first. If it's too fast to finish, we can restart.
        
        for count, v in valid_neighbors:
            if time.time() > deadline:
                return False
                
            new_forbidden = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
            path.append(v)
            
            if backtrack(v, new_forbidden, path):
                path.pop()
                return True
                
            path.pop()
            
        return False

    backtrack(0, forbidden, path)
    
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    best_path = [0]
    best_len = 1
    
    # We will run the backtracking search.
    # Since the problem is symmetric, starting at 0 is WLOG for the first run.
    # However, different random seeds or bit permutations might find different local optima.
    # Given the time constraint, a single deep run is usually best for Warnsdorff.
    # But if the first run finishes early, we can try others.
    
    # Run 1: Standard
    p = _warnsdorff_backtrack(deadline)
    if len(p) > best_len:
        best_path = p
        best_len = len(p)
        
    # If we have time left, try running again with a slightly different tie-breaking
    # or starting vertex to see if we can find a longer snake.
    if time.time() < deadline - 5 and best_len < 200:
        # Try starting from a different vertex? 
        # Due to symmetry, starting from any vertex is equivalent if we permute bits.
        # But the *order* of neighbors might matter due to integer representation.
        # Let's just re-run with a modified tie-breaking strategy.
        
        # We can modify the inner loop to use a random tie-breaker.
        # For simplicity, we'll just call the same function again if we have time.
        # It might find the same path. To get diversity, we'd need to inject randomness.
        # Let's implement a randomized Warnsdorff version.
        
        pass # Placeholder for future optimization if needed

    return best_path
