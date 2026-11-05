"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Build conflict graph using bitmasks
    # adj[i] is a bitmask of vertices j such that (i+1) + (j+1) is a perfect square
    # Note: We include self-loops (a=b) if 2*a is a square
    adj = [0] * N
    
    for i in range(N):
        val = i + 1
        for j in range(N):
            other = j + 1
            if val + other in squares:
                adj[i] |= (1 << j)
    
    # Identify vertices that cannot be in any valid set (self-conflict: 2*a is square)
    # These must be excluded
    valid_mask = 0
    for i in range(N):
        if not (adj[i] & (1 << i)):  # No self-loop
            valid_mask |= (1 << i)
    
    # Branch and bound for Maximum Independent Set
    # We work with the subgraph induced by valid_mask
    
    best_set = set()
    best_size = 0
    
    def popcount(x):
        return bin(x).count('1')
    
    def expand(candidates, current_set, current_size):
        nonlocal best_set, best_size, start_time
        
        if time.time() - start_time > DEADLINE:
            return
        
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Upper bound: current_size + popcount(candidates)
        if current_size + popcount(candidates) <= best_size:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in candidates
        # For simplicity, pick the first set bit
        # Better: pick vertex with max conflicts within candidates
        # Let's try a greedy coloring bound approach or just simple branching
        
        # Simple branching: pick a vertex v from candidates
        # Try including v: remove v and all its neighbors from candidates
        # Try excluding v: remove v from candidates
        
        # To improve efficiency, order vertices by degree
        # Let's pick the vertex with the highest degree in the induced subgraph
        
        # For now, let's just pick the first available vertex
        # Actually, let's try to be smarter: pick vertex with max degree in candidates
        
        # Find all vertices in candidates
        verts_in_cand = []
        temp = candidates
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            verts_in_cand.append(idx)
            temp ^= lsb
        
        if not verts_in_cand:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Pick vertex with highest degree in candidates
        def degree_in_candidates(v):
            return popcount(adj[v] & candidates)
        
        # Sort by degree descending
        verts_in_cand.sort(key=degree_in_candidates, reverse=True)
        v = verts_in_cand[0]
        
        # Branch 1: Include v
        new_candidates = candidates & ~(1 << v) & ~adj[v]
        current_set.append(v)
        expand(new_candidates, current_set, current_size + 1)
        current_set.pop()
        
        # Check deadline
        if time.time() - start_time > DEADLINE:
            return
        
        # Branch 2: Exclude v
        new_candidates = candidates & ~(1 << v)
        expand(new_candidates, current_set, current_size)
    
    # Start with all valid vertices
    initial_candidates = valid_mask
    expand(initial_candidates, [], 0)
    
    if best_size == 0:
        # Fallback: return a simple valid set
        return list(range(1, N + 1, 3))
    
    # Convert best_set (indices) back to values
    result = [i + 1 for i in best_set]
    return result
# EVOLVE-BLOCK-END
