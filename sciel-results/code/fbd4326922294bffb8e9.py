import time
import random
import itertools
import numpy as np

DEADLINE = 330.0
SAFETY_MARGIN = 10.0

def _paley_graph(q):
    """Construct Paley graph on q vertices."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [[False] * q for _ in range(q)]
    for i in range(q):
        for j in range(i+1, q):
            if (j - i) % q in residues:
                adj[i][j] = True
                adj[j][i] = True
    return q, adj

def _get_k3s_and_ik3s(adj, n):
    """Get all K3s and independent 3-sets."""
    k3s = []
    ik3s = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                is_k3 = adj[i][j] and adj[i][k] and adj[j][k]
                if is_k3:
                    k3s.append((i, j, k))
                else:
                    # Check if independent: no edges
                    if not adj[i][j] and not adj[i][k] and not adj[j][k]:
                        ik3s.append((i, j, k))
    return k3s, ik3s

def _get_k4s_and_ik4s(adj, n, k3s, ik3s):
    """Get all K4s and independent 4-sets from K3s and IK3s."""
    k4s = []
    ik4s = []
    # A K4 is a set of 4 vertices where all 4 triples are K3s.
    # We can iterate over K3s and try to extend.
    # Or simply iterate over all 4-combinations. For n=37, C(37,4) ~ 66k, which is fine.
    # But we can be smarter.
    
    # Let's just iterate over combinations for simplicity first, as n is small.
    for combo in itertools.combinations(range(n), 4):
        i, j, k, l = combo
        # Check K4
        if adj[i][j] and adj[i][k] and adj[i][l] and adj[j][k] and adj[j][l] and adj[k][l]:
            k4s.append(combo)
        # Check IK4
        elif not adj[i][j] and not adj[i][k] and not adj[i][l] and not adj[j][k] and not adj[j][l] and not adj[k][l]:
            ik4s.append(combo)
            
    return k4s, ik4s

def _find_valid_neighborhood(n, k4s, ik4s, adj, time_start):
    """
    Find a subset S of {0, ..., n-1} such that:
    1. No K4 is fully contained in S.
    2. No IK4 is fully contained in {0, ..., n-1} \ S.
    Returns S as a list of vertices, or None.
    """
    # We use Simulated Annealing or Random Restart Local Search
    # State: bit vector S of length n
    
    best_s = None
    best_energy = float('inf')
    
    # Pre-filter: If a K4 exists, S cannot contain all its vertices.
    # If an IK4 exists, S must contain at least one of its vertices.
    
    # Heuristic initialization: Random
    # We want S to be roughly half of the vertices.
    
    for restart in range(100):
        if time.time() - time_start > DEADLINE - SAFETY_MARGIN:
            break
            
        # Initialize S randomly
        s = [random.random() < 0.5 for _ in range(n)]
        
        current_energy = _calc_energy(s, k4s, ik4s)
        
        # Local search: flip bits to reduce energy
        improved = True
        while improved and time.time() - time_start < DEADLINE - SAFETY_MARGIN:
            improved = False
            # Try flipping each bit
            for i in range(n):
                old_val = s[i]
                s[i] = not s[i]
                new_energy = _calc_energy(s, k4s, ik4s)
                if new_energy < current_energy:
                    current_energy = new_energy
                    improved = True
                    if current_energy == 0:
                        return [i for i in range(n) if s[i]]
                else:
                    s[i] = old_val
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_s = s[:]
            
    if best_energy == 0:
        return [i for i in range(n) if best_s[i]]
    return None

def _calc_energy(s, k4s, ik4s):
    """Calculate energy: number of violated constraints."""
    energy = 0
    # Constraint 1: No K4 subset of S
    for c in k4s:
        if all(s[v] for v in c):
            energy += 1
    # Constraint 2: No IK4 subset of ~S (i.e., all vertices in IK4 are NOT in S)
    for c in ik4s:
        if all(not s[v] for v in c):
            energy += 1
    return energy

def _to_edges(adj, n):
    """Convert adjacency matrix to list of (i, j) edges."""
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append((i, j))
    return edges

def construct():
    """Construct a (5,5)-free graph. Returns (n, edges)."""
    time_start = time.time()
    
    # Start with Paley graph on 37 vertices
    q, adj = _paley_graph(37)
    n = q
    
    # Initial structures
    k3s, ik3s = _get_k3s_and_ik3s(adj, n)
    k4s, ik4s = _get_k4s_and_ik4s(adj, n, k3s, ik3s)
    
    # Try to add vertices
    max_attempts = 100
    for attempt in range(max_attempts):
        if time.time() - time_start > DEADLINE - 30:
            break
            
        # Find a valid neighborhood for the new vertex
        S = _find_valid_neighborhood(n, k4s, ik4s, adj, time_start)
        
        if S is None:
            # If we fail to find one, we might try a different random seed or break
            # Since the search is stochastic, we can just break if we've tried enough
            if attempt > 10: 
                break
            continue
            
        # Add the new vertex
        new_v = n
        new_n = n + 1
        
        # Update adjacency matrix
        new_adj = [row[:] for row in adj]
        new_adj.append([False] * new_n)
        for i in range(n):
            if i in S:
                new_adj[i].append(True)
                new_adj[new_v][i] = True
        
        # Update K3s, IK3s, K4s, IK4s
        # New K3s involving new_v: (new_v, a, b) where a,b in S and adj[a][b]
        for i in range(n):
            if i in S:
                for j in range(i+1, n):
                    if j in S and adj[i][j]:
                        k3s.append((new_v, i, j))
        
        # New IK3s involving new_v: (new_v, a, b) where a,b NOT in S and not adj[a][b]
        for i in range(n):
            if i not in S:
                for j in range(i+1, n):
                    if j not in S and not adj[i][j]:
                        ik3s.append((new_v, i, j))
                        
        # New K4s involving new_v: (new_v, a, b, c) where {a,b,c} is a K3 in S
        # We need to find K3s in the subgraph induced by S.
        # We can iterate over existing k3s and check if they are in S.
        # But note: k3s contains all K3s of the old graph.
        for c in k3s:
            # c is a tuple (a,b,c_val)
            if c[0] == new_v: continue # Skip ones we just added? No, we added them to k3s list, but they are new.
            # Actually, we should only consider K3s from the OLD graph that are fully in S.
            # The k3s list currently contains old K3s and new ones.
            # We need to be careful. Let's filter.
            pass
        
        # Better approach for K4s:
        # Iterate over all triples in S that form a K3 in the OLD graph.
        # We have the old k3s list. Let's call it old_k3s before updating.
        # Let's refactor to keep old structures separate or update carefully.
        
        # Let's restart the update logic to be cleaner.
        # We will recompute K4s and IK4s from scratch if n is small enough?
        # C(40,4) is ~ 90k. C(42,4) is ~ 110k.
        # Recalculating K4s/IK4s from scratch every step is O(n^4).
        # Total cost: sum_{n=37}^{42} C(n,4) * (cost of check).
        # This is very fast in Python if optimized?
        # 100k * 6 checks = 600k ops.
        # 5 steps * 600k = 3M ops. This is fine.
        
        # So, let's just recompute K4s and IK4s from the new adjacency matrix.
        # And K3s/IK3s can be recomputed too, or updated.
        # Recomputing K3s is O(n^3). C(42,3) ~ 11k. Very fast.
        
        adj = new_adj
        n = new_n
        
        # Recompute all structures
        k3s, ik3s = _get_k3s_and_ik3s(adj, n)
        k4s, ik4s = _get_k4s_and_ik4s(adj, n, k3s, ik3s)
        
        # If we reach 42, we are done (or try to go higher?)
        # R(5,5) is 43-46. So 42 is the target.
        if n >= 42:
            break
            
    edges = _to_edges(adj, n)
    return n, edges

if __name__ == "__main__":
    n, edges = construct()
    print(f"Constructed graph with {n} vertices and {len(edges)} edges")
