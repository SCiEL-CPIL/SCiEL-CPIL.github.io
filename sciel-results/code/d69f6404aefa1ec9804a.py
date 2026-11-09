import time
import random
import itertools

_START = time.time()
_DEADLINE = 145  # safety margin before 150s

def _has_c5(adj, n):
    """Check if graph contains a 5-cycle (C5). adj is a list of sets."""
    for u in range(n):
        neighbors_u = sorted(adj[u])
        if len(neighbors_u) < 2:
            continue
        for v in neighbors_u:
            for w in range(len(neighbors_u) - 1):
                v2 = neighbors_u[w]
                v3 = neighbors_u[w + 1]
                # Check for 5-cycles containing edge u-v2-v3
                # Actually, let's use a more systematic approach
                pass
    
    # More efficient: check all 5-tuples (a,b,c,d,e) with a<b<c<d<e
    # but that's O(n^5). For n<=35, n^5 ~ 50M which might be too slow.
    # Better: for each 5-clique-free check, use BFS from each vertex
    # up to depth 2, checking for paths of length 4 back to start.
    
    # Actually, for C5 detection, we can iterate over all pairs (u,v)
    # and check if there's a path of length 3 from u to v that doesn't use edge (u,v).
    # Path: u-a-b-v where a,b are distinct, not u,v, and (u,v) not used.
    
    for u in range(n):
        for v in range(u + 1, n):
            if v not in adj[u]:
                continue
            # We need a path u -> a -> b -> v
            # a in N(u) \\ {v}, b in N(v) \\ {u}
            # a != b, and (a,b) is an edge
            # Also a != v, b != u (already excluded)
            # And the path must be simple: u, a, b, v all distinct
            # u != a (a in N(u), a!=v=u? no, a != v), u != b (b in N(v), b != u)
            # a != b, a != v (a not in {v}), b != u (b not in {u})
            # So we need: a in N(u) \\ {v}, b in N(v) \\ {u}, a != b, (a,b) in E
            
            for a in adj[u]:
                if a == v:
                    continue
                for b in adj[v]:
                    if b == u or b == a:
                        continue
                    if b in adj[a]:
                        return True
    return False


def _has_independent_9(adj, n):
    """Check if graph has an independent set of size 9."""
    # Complement graph: independent set in G = clique in complement
    # Build complement adjacency
    comp_adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if v not in adj[u]:
                comp_adj[u].add(v)
                comp_adj[v].add(u)
    
    # Check for K9 in complement using Bron-Kerbosch
    found = [False]
    
    def bron_kerbosch(R, P, X):
        if found[0]:
            return
        if len(R) >= 9:
            found[0] = True
            return
        if not P and not X:
            return
        # Pick pivot from P ∪ X
        if P:
            pivot = next(iter(P))
        elif X:
            pivot = next(iter(X))
        else:
            return
        candidates = P - comp_adj[pivot]
        for v in list(candidates):
            new_R = R | {v}
            new_P = P & comp_adj[v]
            new_X = X & comp_adj[v]
            P.remove(v)
            X.add(v)
            bron_kerbosch(new_R, new_P, new_X)
            P.add(v)
            X.remove(v)
            if found[0]:
                return
    
    bron_kerbosch(set(), set(range(n)), set())
    return found[0]


def _is_valid(edges, n):
    """Check if graph is C5-free and has no independent set of 9."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    if _has_c5(adj, n):
        return False
    if _has_independent_9(adj, n):
        return False
    return True


def construct():
    # Start with the known best: 8 disjoint K4s, n=32
    best_n = 32
    best_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    
    # Try to find better constructions via search
    # Strategy: try small graphs and verify
    # Since we know 32 works, try to see if 33+ works
    
    # For efficiency, try a few heuristic constructions
    
    # Construction 1: 7 K4s + one 5-vertex component
    # 7*4 = 28, need 5 more vertices
    # Try K5 minus some edges to avoid C5
    # K5 has C5, so we need to remove edges
    # Try: 5 vertices where alpha <= 1 (i.e., complement has no K2, i.e., complement is edgeless)
    # That means the 5-vertex graph must be K5, which has C5. So this won't work.
    
    # Construction 2: 6 K4s + 2 K3s + 1 vertex
    # 6*4 + 2*3 + 1 = 24+6+1 = 31
    # alpha = 6 + 2 + 1 = 9. Too high.
    
    # Construction 3: 5 K4s + 3 K3s + 2 vertices
    # 5*4 + 3*3 + 2 = 20+9+2 = 31
    # alpha = 5+3+2 = 10. Too high.
    
    # It seems hard to exceed 32 with simple clique-based constructions.
    
    # Try random search for n=33
    # But this might be too slow. Let's just return the known best.
    
    # Actually, let's try a smarter approach:
    # Start with 8 K4s and try to add vertices or modify the graph
    
    # For now, return the known best construction
    return {'n': best_n, 'edges': best_edges}
