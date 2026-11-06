import numpy as np
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Target: try to find a tournament with n < 67
    # Erdos lower bound is 31, so we have room to explore smaller n
    # Strategy: Probabilistic dominator assignment + repair
    
    # We'll try several n values from high to low, starting near 60
    # For each n, we'll do a probabilistic construction and repair
    
    # First, let's define a helper to check if a tournament is valid
    def is_valid_tournament(n, orientation):
        """Check if every 4-set has a common dominator."""
        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=bool)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if orientation[idx] == 0:
                    adj[i, j] = True
                else:
                    adj[j, i] = True
                idx += 1
        
        # Check all 4-sets
        for quad in combinations(range(n), 4):
            # For each vertex in the quad, check if it dominates all others
            found_dominator = False
            for v in quad:
                if all(adj[v, u] for u in quad if u != v):
                    found_dominator = True
                    break
            if not found_dominator:
                return False
        return True
    
    # Faster check using numpy
    def count_violations_fast(n, adj):
        """Count number of 4-sets without a common dominator."""
        violations = 0
        for quad in combinations(range(n), 4):
            found = False
            for v in quad:
                # Check if v dominates all others in quad
                if all(adj[v, u] for u in quad if u != v):
                    found = True
                    break
            if not found:
                violations += 1
        return violations
    
    # Build adjacency from orientation
    def build_adj(n, orientation):
        adj = np.zeros((n, n), dtype=bool)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if orientation[idx] == 0:
                    adj[i, j] = True
                else:
                    adj[j, i] = True
                idx += 1
        return adj
    
    # Orientation from adjacency
    def orientation_from_adj(n, adj):
        orientation = []
        for i in range(n):
            for j in range(i+1, n):
                orientation.append(0 if adj[i, j] else 1)
        return orientation
    
    # Local search: flip bits to reduce violations
    def local_search(n, orientation, max_steps=10000):
        adj = build_adj(n, orientation)
        current_violations = count_violations_fast(n, adj)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if current_violations == 0:
                return orientation, 0
            
            # Pick a random bit to flip
            idx = random.randint(0, len(orientation) - 1)
            # Find i, j from idx
            # idx = i*n - i*(i+1)/2 + (j - i - 1)
            # Simpler: precompute mapping
            # For now, just flip and check
            old_val = orientation[idx]
            orientation[idx] = 1 - old_val
            
            new_adj = build_adj(n, orientation)
            new_violations = count_violations_fast(n, new_adj)
            
            if new_violations < current_violations:
                adj = new_adj
                current_violations = new_violations
            else:
                orientation[idx] = old_val
        
        return orientation, current_violations
    
    # Probabilistic construction: assign each vertex a "dominator role"
    # Idea: create a set D of dominators, and for each 4-set, ensure at least one dominator dominates it
    # Simpler: random tournament with bias, then repair
    
    best_n = None
    best_orientation = None
    
    # Try n from 60 down to 35
    for n in range(60, 34, -1):
        if time.time() > deadline:
            break
        
        # Multiple restarts
        for restart in range(20):
            if time.time() > deadline:
                break
            
            # Random tournament
            orientation = [random.randint(0, 1) for _ in range(n * (n-1) // 2)]
            
            # Local search
            orientation, violations = local_search(n, orientation, max_steps=5000)
            
            if violations == 0:
                best_n = n
                best_orientation = orientation
                break
        
        if best_n is not None:
            break
    
    # If we found a valid tournament, return it
    if best_n is not None:
        return {'n': best_n, 'orientation': best_orientation}
    
    # Fallback: return Paley on 83
    p = 83
    squares = {x * x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'orientation': orientation}
