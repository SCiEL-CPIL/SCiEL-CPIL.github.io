"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_k33_free(adj):
    """Check if graph with adjacency list (list of sets) is K3,3-free.
    K3,3 exists iff there exist two disjoint sets A={a1,a2,a3} and B={b1,b2,b3}
    such that every ai is connected to every bj.
    Equivalent: there exist 3 vertices with at least 3 common neighbours.
    """
    n = len(adj)
    # For efficiency, check all pairs of vertices, find common neighbours,
    # then check if any 3 vertices share 3+ common neighbours.
    # Actually the condition is: no 3 vertices have 3 common neighbours.
    # So for each triple of vertices, count common neighbours; if >=3, K3,3 exists.
    # But that's O(n^4) which for n=20 is 20^4 = 160000, fine.
    
    # More efficient: for each pair (u,v), compute common_neighbours(u,v).
    # Then for each common neighbour w, we need to check if there's a triple containing u,v,w
    # that has 3 common neighbours... 
    # 
    # Actually let's think again. K3,3 has parts {a,b,c} and {x,y,z} where all 9 edges exist.
    # This means a,b,c all share common neighbours x,y,z (at least).
    # So: exists triple {a,b,c} such that |N(a) ∩ N(b) ∩ N(c)| >= 3.
    
    # For n=20, checking all C(20,3)=1140 triples, each intersection of 3 sets of size up to 19:
    # That's very fast.
    
    for a, b, c in combinations(range(n), 3):
        common = adj[a] & adj[b] & adj[c]
        if len(common) >= 3:
            return False
    return True


def _edge_count(adj):
    return sum(len(s) for s in adj) // 2


def _add_edge(adj, u, v):
    """Try to add edge (u,v). Returns True if valid."""
    if u in adj[v]:
        return False
    adj[u].add(v)
    adj[v].add(u)
    if not _is_k33_free(adj):
        adj[u].discard(v)
        adj[v].discard(u)
        return False
    return True


def _remove_edge(adj, u, v):
    adj[u].discard(v)
    adj[v].discard(u)


def _greedy_build(n=20, seed=42):
    """Greedy: add edges in random order until no more can be added."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(edges)
    
    added = 0
    for u, v in edges:
        if _add_edge(adj, u, v):
            added += 1
    return adj, added


def _local_search(adj, n=20, max_iters=20000, seed=42, time_limit=100.0):
    """Local search: try swapping edges to improve density while staying K3,3-free.
    
    Moves:
    1. Add a missing edge (if valid).
    2. Remove one edge, add two others (net +1).
    3. Remove two edges, add three others (net +1).
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    # First, try to add any missing edge
    improved = True
    while improved:
        improved = False
        for u in range(n):
            for v in range(u+1, n):
                if v not in adj[u]:
                    if _add_edge(adj, u, v):
                        improved = True
                        if elapsed() > time_limit:
                            return adj
    if elapsed() > time_limit:
        return adj
    
    # Local search with swaps
    for it in range(max_iters):
        if elapsed() > time_limit:
            break
        
        u, v = rng.randint(0, n-1), rng.randint(0, n-1)
        if u == v:
            continue
        if v not in adj[u]:
            continue
        
        # Try to find an alternative: remove (u,v), try to add two edges
        _remove_edge(adj, u, v)
        
        # Generate candidate edges to add
        candidates = []
        for x in range(n):
            for y in range(x+1, n):
                if y not in adj[x]:
                    candidates.append((x, y))
        
        rng.shuffle(candidates)
        
        found = False
        for (x, y) in candidates:
            if _add_edge(adj, x, y):
                # Now try to add one more
                for (x2, y2) in candidates:
                    if (x2, y2) == (x, y):
                        continue
                    if _add_edge(adj, x2, y2):
                        found = True
                        break
            if not found:
                # Undo the first add
                if y in adj[x]:
                    _remove_edge(adj, x, y)
            if found:
                break
        
        if not found:
            # Restore (u,v)
            _add_edge(adj, u, v)
    
    return adj


def _random_restarts(n=20, num_restarts=50, time_limit=110.0):
    """Multiple random restarts of greedy + local search."""
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    best_adj = None
    best_count = 0
    
    for i in range(num_restarts):
        if elapsed() > time_limit:
            break
        
        adj, _ = _greedy_build(n, seed=1000 + i)
        adj = _local_search(adj, n, max_iters=5000, seed=2000 + i, time_limit=110.0)
        
        count = _edge_count(adj)
        if count > best_count:
            best_count = count
            best_adj = [set(s) for s in adj]
    
    if best_adj is None:
        best_adj = _greedy_build(n, seed=42)[0]
    
    return best_adj


def construct():
    """Construct a K3,3-free graph on 20 vertices maximizing edges."""
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    # Quick greedy to get a baseline
    adj, _ = _greedy_build(20, seed=42)
    best_adj = [set(s) for s in adj]
    best_count = _edge_count(best_adj)
    
    # Try random restarts
    start_time = time.time()
    for i in range(200):
        if time.time() > deadline:
            break
        
        adj, _ = _greedy_build(20, seed=10000 + i)
        # Quick local search with limited time per restart
        adj = _local_search(adj, 20, max_iters=2000, seed=20000 + i, time_limit=deadline - start_time)
        
        count = _edge_count(adj)
        if count > best_count:
            best_count = count
            best_adj = [set(s) for s in adj]
    
    # Convert to edge list
    edges = []
    for u in range(20):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    
    return edges

# EVOLVE-BLOCK-END
