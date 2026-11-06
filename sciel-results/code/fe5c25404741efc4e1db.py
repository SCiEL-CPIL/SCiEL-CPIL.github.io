"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _pale_tournament(p):
    """Return adjacency list for Paley tournament on p vertices (p must be prime ≡ 3 mod 4)."""
    squares = {x * x % p for x in range(1, p)}
    adj = [[False] * p for _ in range(p)]
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                adj[i][j] = True
            else:
                adj[j][i] = True
    return adj

def _orientation_from_adj(adj):
    """Convert adjacency matrix to orientation list (one bit per pair i<j in lex order)."""
    n = len(adj)
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            orientation.append(0 if adj[i][j] else 1)
    return orientation

def _check_validity(n, adj):
    """Check if every set of 4 vertices has a common dominator."""
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        found = False
        for v in quad:
            if (adj[v][a] or v == a) and (adj[v][b] or v == b) and \
               (adj[v][c] or v == c) and (adj[v][d] or v == d):
                found = True
                break
        if not found:
            return False
    return True

def _vertex_criticality(n, adj):
    """
    For each vertex, count how many 4-sets have this vertex as the UNIQUE dominator.
    """
    n_critical = [0] * n
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        dominators = []
        for v in quad:
            if (adj[v][a] or v == a) and (adj[v][b] or v == b) and \
               (adj[v][c] or v == c) and (adj[v][d] or v == d):
                dominators.append(v)
        if len(dominators) == 1:
            n_critical[dominators[0]] += 1
    return n_critical

def _remove_vertex(n, adj, vertex):
    """Remove a vertex and return new n and adj matrix."""
    new_n = n - 1
    new_adj = [[False] * new_n for _ in range(new_n)]
    for i in range(new_n):
        for j in range(i + 1, new_n):
            old_i = i if i < vertex else i + 1
            old_j = j if j < vertex else j + 1
            new_adj[i][j] = adj[old_i][old_j]
    return new_n, new_adj

def construct():
    """
    Start from the smallest known valid Paley tournament (p=67) and greedily
    prune vertices that are not unique dominators of any 4-set.
    """
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with Paley tournament on 67 vertices (smallest known valid)
    p = 67
    adj = _pale_tournament(p)
    n = p
    
    # Verify validity first
    if not _check_validity(n, adj):
        # Fallback to 83 if 67 is somehow invalid (shouldn't happen)
        p = 83
        adj = _pale_tournament(p)
        n = p
    
    # Greedy pruning: iteratively remove vertices with zero criticality
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        n_critical = _vertex_criticality(n, adj)
        
        # Sort candidates by criticality (lowest first), then by index for determinism
        candidates = sorted(range(n), key=lambda v: (n_critical[v], v))
        
        for v in candidates:
            if time.time() > deadline - 30:
                break
            if n_critical[v] > 0:
                continue
            
            # Try removing vertex v
            new_n, new_adj = _remove_vertex(n, adj, v)
            if _check_validity(new_n, new_adj):
                n = new_n
                adj = new_adj
                improved = True
                break
    
    return {'n': n, 'orientation': _orientation_from_adj(adj)}
# EVOLVE-BLOCK-END
