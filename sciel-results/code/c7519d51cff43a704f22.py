import time
import random
from itertools import combinations

_START = time.time()
_LIMIT = 140.0  # Safety margin

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _build_adj(n, edges):
    """Build adjacency list as a list of sets, and a bitmask representation."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Bitmask for edges: index (u, v) with u < v
    # Map (u, v) to index
    def edge_index(u, v):
        if u > v: u, v = v, u
        # Simple mapping: u * n + v (not unique if not careful, but u < v ensures uniqueness in range n*n)
        # Actually, just use a dict or a 2D array of indices?
        # For bitmask, we need a single bit position.
        # Total edges <= n*(n-1)/2.
        # We can just use the pair (u,v) as a key in a set for removal, 
        # but for Max Clique, we work on vertices.
        return (u, v)

    return adj

def _max_clique(adj, n):
    """
    Find a maximum clique in the graph defined by adj (list of sets).
    Uses Bron-Kerbosch with pivoting.
    """
    best_clique = []
    
    # Convert adj to bitmasks for speed?
    # n=64, so int is fine.
    adj_mask = [0] * n
    for u in range(n):
        for v in adj[u]:
            adj_mask[u] |= (1 << v)
    
    def bronk(R, P, X):
        nonlocal best_clique
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
            
        # Pivoting
        # Choose pivot u from P union X
        # u should maximize |P & N(u)|
        P_union_X = P | X
        if not P_union_X:
            return
            
        # Find pivot
        # Iterate bits of P_union_X
        pivot = -1
        max_count = -1
        temp = P_union_X
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            # Count common neighbors in P
            count = (P & adj_mask[u]).bit_count()
            if count > max_count:
                max_count = count
                pivot = u
            temp ^= lsb
            
        if pivot == -1:
            # No pivot found, just iterate P
            candidates = P
        else:
            # Candidates are P \ N(pivot)
            candidates = P & ~adj_mask[pivot]
            
        while candidates:
            if not _deadline_ok():
                return
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            candidates ^= lsb
            if not (candidates & (1 << v)): # Already processed? No, we removed it.
                pass
            # R = R U {v}
            new_R = R | (1 << v)
            # P = P \ {v}
            new_P = P & ~(1 << v)
            # X = X \ {v}
            new_X = X & ~(1 << v)
            
            bronk(new_R, new_P & adj_mask[v], new_X & adj_mask[v])
            
            P = new_P
            X = new_X | (1 << v)

    # Initial call
    # P = all vertices, X = empty
    initial_P = (1 << n) - 1
    bronk(0, initial_P, 0)
    
    return best_clique

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    adj = _build_adj(n, edges)
    remaining_edges = set(edges)
    
    cliques = []
    
    while remaining_edges:
        if not _deadline_ok():
            # Fallback: add remaining edges as K2
            for e in remaining_edges:
                cliques.append([e[0], e[1]])
            break
            
        # Build subgraph of remaining edges
        # We need to find Max Clique in the graph formed by remaining_edges.
        # This is not just the original graph.
        # The "graph" for clique finding is the subgraph induced by remaining edges.
        # Two vertices u, v are adjacent in this subgraph if (u,v) in remaining_edges.
        
        sub_adj = [set() for _ in range(n)]
        for u, v in remaining_edges:
            sub_adj[u].add(v)
            sub_adj[v].add(u)
            
        # Find Max Clique in sub_adj
        clique = _max_clique(sub_adj, n)
        
        if not clique:
             # Should not happen if remaining_edges is not empty
             # But if isolated vertices? No, edges exist.
             # Maybe Max Clique size 1?
             # Problem requires cliques with at least 2 vertices.
             # If Max Clique is 1, we must pick an edge (K2).
             # But Max Clique on a graph with edges must have size >= 2.
             pass
             
        if len(clique) < 2:
             # Fallback to edge
             e = next(iter(remaining_edges))
             clique = [e[0], e[1]]
             
        cliques.append(clique)
        
        # Remove edges of clique from remaining
        for u, v in combinations(clique, 2):
            if (min(u,v), max(u,v)) in remaining_edges:
                remaining_edges.remove((min(u,v), max(u,v)))
                
    return cliques

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
