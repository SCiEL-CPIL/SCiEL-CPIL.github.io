"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    N = 256
    # Build conflict graph: edge between a and b if a+b is a perfect square
    # Also diagonal: 2a is a perfect square means a cannot be in set
    
    # Precompute perfect squares up to 2*N
    max_sum = 2 * N
    squares = set()
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
    
    # Diagonal constraint: 2*a must not be in squares
    # Edge constraint: a + b must not be in squares for a < b
    
    # Build adjacency: for each a, which b > a conflict with a
    # Also track which a are forbidden by diagonal
    forbidden_diag = set()
    adj = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden_diag.add(a)
        for b in range(a + 1, N + 1):
            if a + b in squares:
                adj[a].append(b)
                adj[b].append(a)
    
    # Valid vertices: not forbidden by diagonal
    valid_vertices = [a for a in range(1, N + 1) if a not in forbidden_diag]
    
    # Greedy initialization: sort by degree, pick vertices with fewest conflicts
    # Actually, for maximum independent set, we want to pick vertices that
    # don't conflict. Let's use a greedy approach:
    # 1. Start with all valid vertices
    # 2. Remove conflicts iteratively
    
    # Better: use greedy coloring / independent set approach
    # Start with empty set, add vertices in order of increasing degree
    
    def get_degree(v, current_set):
        """Count conflicts with current set"""
        count = 0
        for u in adj[v]:
            if u in current_set:
                count += 1
        return count
    
    # Greedy: pick vertices with fewest conflicts among remaining
    remaining = set(valid_vertices)
    solution = set()
    
    while remaining:
        # Pick vertex with minimum degree in remaining subgraph
        best_v = None
        best_deg = float('inf')
        for v in remaining:
            deg = sum(1 for u in adj[v] if u in remaining)
            if deg < best_deg:
                best_deg = deg
                best_v = v
        solution.add(best_v)
        # Remove best_v and all its neighbors from remaining
        remaining.discard(best_v)
        for u in adj[best_v]:
            remaining.discard(u)
    
    best_solution = set(solution)
    best_size = len(best_solution)
    
    # Local search: try to improve by adding vertices or swapping
    def is_valid(s):
        """Check if set s is a valid independent set"""
        for a in s:
            if 2 * a in squares:
                return False
            for b in s:
                if a < b and a + b in squares:
                    return False
        return True
    
    # Verify initial solution
    if not is_valid(best_solution):
        # Fallback to simple valid set
        best_solution = set()
        for a in valid_vertices:
            if all(a + b not in squares and 2*a not in squares for b in best_solution):
                best_solution.add(a)
        best_size = len(best_solution)
    
    # Local search with simulated annealing
    def try_add(s, v):
        """Try adding v to set s, return True if valid"""
        if v in s:
            return False
        if 2 * v in squares:
            return False
        for u in s:
            if u + v in squares:
                return False
        return True
    
    def try_remove(s, v):
        s.discard(v)
    
    # Hill climbing / local search
    current = set(best_solution)
    current_size = len(current)
    
    # Try single additions
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in valid_vertices:
            if v not in current and try_add(current, v):
                current.add(v)
                current_size += 1
                improved = True
                if len(current) > best_size:
                    best_solution = set(current)
                    best_size = len(current)
    
    # Try swaps: remove one, add one or more
    while time.time() < deadline:
        improved = False
        # Try removing each vertex and then greedily adding
        for v in list(current):
            temp = set(current)
            temp.discard(v)
            # Try to add vertices that were blocked by v
            added = 0
            for u in valid_vertices:
                if u not in temp and try_add(temp, u):
                    temp.add(u)
                    added += 1
            if len(temp) > len(current):
                current = temp
                current_size = len(current)
                improved = True
                if len(current) > best_size:
                    best_solution = set(current)
                    best_size = len(current)
                break
    
        if not improved:
            # Try random restarts / perturbations
            if time.time() > deadline - 5:
                break
            
            # Perturb: remove a random vertex, then greedily re-add
            if len(current) > 1:
                removed = random.choice(list(current))
                current.discard(removed)
                # Greedily add back
                for u in valid_vertices:
                    if u not in current and try_add(current, u):
                        current.add(u)
                if len(current) > best_size:
                    best_solution = set(current)
                    best_size = len(current)
    
    # Final greedy fill
    for v in valid_vertices:
        if v not in best_solution and try_add(best_solution, v):
            best_solution.add(v)
    
    return list(best_solution)
# EVOLVE-BLOCK-END
