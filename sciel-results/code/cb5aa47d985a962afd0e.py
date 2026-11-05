"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _has_triangle_circulant_fast(n, S):
    """Check if circulant graph on Z_n with connection set S has a triangle.
    Triangle exists if there exist a,b,c in S (with a,b,c not all equal to 0)
    such that a+b+c ≡ 0 (mod n), where the vertices are 0, a, a+b.
    For a circulant graph, vertex 0 connects to all s in S.
    A triangle 0-a-b exists if a in S, b in S, and b-a in S (or a-b in S).
    """
    S_set = frozenset(S)
    for a in S:
        for b in S:
            if a == b:
                continue
            diff = (b - a) % n
            if diff in S_set:
                return True
    return False

def _max_independent_set_size(n, adj):
    """
    Compute alpha of a graph given its adjacency list.
    Uses branch and bound with bitmasks.
    """
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask
    
    full_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = full_mask ^ (1 << i) ^ adj_mask[i]
    
    best = [0]
    
    def popcount(x):
        return bin(x).count('1')
    
    def dfs(candidates, current_size, depth):
        # Upper bound
        if current_size + popcount(candidates) <= best[0]:
            return
        if candidates == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Pick vertex with highest degree in candidates for better pruning
        best_v = -1
        max_deg = -1
        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = popcount(candidates & adj_mask[v])
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
        
        v = best_v
        # Include v
        dfs(candidates & comp_mask[v], current_size + 1, depth + 1)
        # Exclude v
        dfs(candidates & ~(1 << v), current_size, depth + 1)
    
    dfs(full_mask, 0, 0)
    return best[0]

def _hoffman_bound(n, S):
    """
    Compute Hoffman upper bound on alpha for a regular circulant graph.
    Returns None if not regular or if bound is not useful.
    """
    if not S:
        return 0.0
    
    # Check regularity: for circulant, it's always regular with degree len(S)
    d = len(S)
    if d == 0:
        return float(n)
    
    # Compute eigenvalues
    # lambda_k = sum_{s in S} exp(2*pi*i*k*s/n)
    # Since S is symmetric, lambda_k is real.
    # We can compute using numpy FFT or direct sum.
    # Direct sum is O(n * |S|). n <= 41, |S| <= 20. Fast enough.
    
    lambdas = np.zeros(n)
    for k in range(n):
        s_val = 0.0
        c_val = 0.0
        for s in S:
            angle = 2 * np.pi * k * s / n
            s_val += np.sin(angle)
            c_val += np.cos(angle)
        # lambda_k = c_val + i*s_val. Since symmetric, s_val should be ~0.
        lambdas[k] = c_val
        
    # Find smallest eigenvalue
    lambda_min = np.min(lambdas)
    
    # Hoffman bound: alpha <= n * (-lambda_min) / (d - lambda_min)
    # Note: if d - lambda_min is 0, bound is infinity (disconnected?)
    if d - lambda_min <= 1e-9:
        return float('inf')
        
    bound = n * (-lambda_min) / (d - lambda_min)
    return bound

def _verify_circulant(n, S, use_hoffman=True):
    """Verify circulant graph is triangle-free and alpha < 10."""
    if _has_triangle_circulant_fast(n, S):
        return False, 0
    
    # Try Hoffman bound first
    if use_hoffman:
        try:
            hb = _hoffman_bound(n, S)
            if hb < 9.99: # If upper bound < 10, alpha < 10 is guaranteed
                return True, int(hb) + 1 # Return a conservative alpha estimate
        except:
            pass # Fall back to B&B

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if j != i:
                adj[i].append(j)
    
    # Run exact B&B
    alpha = _max_independent_set_size(n, adj)
    if alpha >= 10:
        return False, alpha
    return True, alpha

def _build_edges(n, S):
    """Build edge list from circulant connection set."""
    edges = []
    S_set = set(S)
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if j > i:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # Try n from 40 down to 25
    for n in range(40, 24, -1):
        if time.time() > deadline - 15:
            break
            
        found = False
        
        # Strategy 1: Systematic enumeration of symmetric S sets
        # For each possible size k, try random subsets of {1, ..., floor(n/2)}
        half = n // 2
        available = list(range(1, half + 1))
        
        # Try different target sizes, from larger to smaller (higher degree = smaller alpha)
        max_k = half  # Maximum possible symmetric S size
        min_k = 3
        
        # Try sizes from larger to smaller
        for k in range(min(max_k, 15), min_k - 1, -1):
            if time.time() > deadline - 10:
                    break
            
            # Number of random attempts for this size
            # Increase attempts if Hoffman bound is used (it's fast)
            num_attempts = 200 if n <= 35 else 100
            
            for attempt in range(num_attempts):
                if time.time() > deadline:
                    break
                
                # Random subset of available of size k
                if k > len(available):
                    continue
                    
                S_half = set(random.sample(available, k))
                # Make symmetric: S = S_half ∪ {-x mod n for x in S_half}
                S = set()
                for x in S_half:
                    S.add(x)
                    S.add((-x) % n)
                
                # Remove duplicates (when x = -x mod n, i.e., x = n/2 for even n)
                S = set(S)
                
                # Quick triangle check
                if _has_triangle_circulant_fast(n, S):
                    continue
                
                # Verify alpha
                valid, alpha = _verify_circulant(n, S)
                if valid:
                    best_n = n
                    best_edges = _build_edges(n, S)
                    found = True
                    break
            
            if found:
                break
        
        # Strategy 2: Try some structured constructions
        if not found and time.time() < deadline - 5:
            # Try S = {1, 2, ..., k} for small k (consecutive offsets)
            for k in range(3, min(half, 8)):
                if time.time() > deadline - 3:
                    break
                S = set(range(1, k + 1)) | set({(-x) % n for x in range(1, k + 1)})
                if _has_triangle_circulant_fast(n, S):
                    continue
                valid, alpha = _verify_circulant(n, S)
                if valid:
                    best_n = n
                    best_edges = _build_edges(n, S)
                    found = True
                    break
            
            # Try S with gaps: {1, 3, 5, ...} (odd offsets)
            if not found:
                odd_offsets = [x for x in range(1, half + 1) if x % 2 == 1]
                for k in range(3, min(len(odd_offsets), 7)):
                    if time.time() > deadline - 3:
                        break
                    S_half = set(odd_offsets[:k])
                    S = set()
                    for x in S_half:
                        S.add(x)
                        S.add((-x) % n)
                    S = set(S)
                    if _has_triangle_circulant_fast(n, S):
                        continue
                    valid, alpha = _verify_circulant(n, S)
                    if valid:
                        best_n = n
                        best_edges = _build_edges(n, S)
                        found = True
                        break
        
        if found:
            break
    
    # Fallback: Known small graphs
    if best_n < 19:
        # Use a known triangle-free graph with alpha < 10
        # The 26-vertex graph from the previous version
        n0 = 26
        best_edges = [[i, j] for i in range(n0) for j in range(i + 1, n0) if (j - i) % n0 % 3 == 1]
        best_n = n0
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
