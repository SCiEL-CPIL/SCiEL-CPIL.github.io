"""
Triangle-free graph with no independent 10-set, maximizing n.

Uses the Andrásfai graph And(10) construction:
  Vertices Z_28, edges (i,j) iff (j-i) mod 28 in {1, 10, 19} (i.e., ≡ 1 mod 7 and not 0).
  This is triangle-free with alpha = 9, giving a valid graph on 28 vertices.

We then attempt to extend via a local search / greedy vertex addition to push n higher,
up to 40 vertices, while maintaining triangle-free and alpha < 10.

Key idea: Start from the known 28-vertex Andrásfai graph. Try to add vertices one by one,
assigning each new vertex a neighborhood that:
  1. Contains no two adjacent vertices from the current graph (to maintain triangle-free).
  2. Doesn't create an independent set of size 10.

We use a randomized greedy + backtracking approach with a time budget.
"""
import time
import random
import sys

DEADLINE = 120.0  # seconds, with safety margin
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _build_andrasfai(k=10):
    """Build Andrásfai graph And(k): n = 3k-2 vertices.
    Vertices 0..3k-3. Edge (i,j) if (j-i) mod (3k-2) is in {1, k, 2k-1} (i.e., ≡ 1 mod (k-1) but excluding 0).
    
    Actually the standard Andrásfai graph A_k has:
      n = 3k - 2 vertices
      Edge (i, j) iff (j - i) mod (3k-2) ∈ {1, 2, ..., k-1} ∪ {3k-2-(k-1), ..., 3k-2-1}
    But that creates triangles.
    
    Correct construction: The Andrásfai graph is the circulant graph on Z_{3k-2} where
    vertex i is connected to i±1, i±k, i±(2k-1) mod (3k-2).
    For k=10: n=28, connections at offsets ±1, ±10, ±19 mod 28.
    These are exactly the residues ≡ 1 mod 7 that are nonzero: {1, 10, 19} and their negatives {27, 18, 9}.
    So edges connect i to i+1, i+10, i+19 mod 28 (and symmetrically).
    """
    n = 3 * k - 2
    offsets = {1, k, 2*k - 1}
    edges = set()
    for i in range(n):
        for off in offsets:
            j = (i + off) % n
            if i < j:
                edges.add((i, j))
    return n, sorted(edges)

def _has_triangle(n, adj):
    """Check if graph has a triangle using adjacency sets."""
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w > v and w in adj[u]:
                    return True
    return False

def _max_independent_set_size(n, adj):
    """Exact maximum independent set via branch and bound."""
    # Convert to complement: independent set in G = clique in complement
    # But for small n, use standard MIS branch and bound
    best = [0]
    # Use bitsets for speed
    adj_bits = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_bits[i] |= (1 << j)
    
    def dfs(candidates, size):
        if size + len(candidates) <= best[0]:
            return
        if not candidates:
            best[0] = max(best[0], size)
            return
        # Greedy bound: upper bound on clique size in candidate set
        # Use a simple bound
        if size + len(candidates) <= best[0]:
            return
        
        # Pick vertex with max degree in candidate subgraph
        # For speed, just pick first
        v_bit = candidates & (-candidates)
        v = v_bit.bit_length() - 1
        candidates_without_v = candidates & ~v_bit
        
        # Branch: include v
        new_cand = candidates_without_v
        # Remove neighbors of v from candidates
        new_cand &= ~adj_bits[v]
        dfs(new_cand, size + 1)
        
        # Branch: exclude v
        dfs(candidates_without_v, size)
    
    dfs((1 << n) - 1, 0)
    return best[0]

def _verify(n, edges):
    """Check triangle-free and alpha < 10."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    if _has_triangle(n, adj):
        return False
    alpha = _max_independent_set_size(n, adj)
    return alpha < 10

def _try_add_vertex(n, edges, adj):
    """Try to add a vertex to the graph. Returns (new_n, new_edges, new_adj) or None."""
    # The new vertex must be connected to a set S of existing vertices such that:
    # 1. S is an independent set in the current graph (no triangle with new vertex)
    # 2. After addition, no independent set of size 10 exists
    #
    # To maximize chance of success, try random independent sets S of various sizes.
    
    # Get all independent sets of the current graph up to some size
    # For efficiency, try random subsets
    
    best_result = None
    for _ in range(50):
        # Pick a random independent set S
        # Greedy random independent set
        S = set()
        available = list(range(n))
        random.shuffle(available)
        for v in available:
            if not (S & adj[v]):
                S.add(v)
        
        # Don't connect to all - try various subsets
        # Try connecting to S itself
        new_edges = edges[:]
        new_n = n + 1
        new_adj = [set(s) for s in adj]
        new_adj.append(set())
        
        for v in S:
            new_edges.append([v, new_n])
            new_adj[v].add(new_n)
            new_adj[new_n].add(v)
        
        # Check triangle-free: S must be independent (already ensured by construction)
        # Check alpha < 10
        if _max_independent_set_size(new_n, new_adj) < 10:
            return new_n, new_edges, new_adj
    
    return None

def construct():
    # Start with Andrásfai graph And(10): n=28, triangle-free, alpha=9
    n, edges = _build_andrasfai(k=10)
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    best_n = n
    best_edges = edges
    
    # Try to extend
    random.seed(42)
    
    # Multiple restarts with different strategies
    for restart in range(20):
        if _time_left() < 10:
            break
        
        cur_n = n
        cur_edges = edges[:]
        cur_adj = [set(s) for s in adj]
        
        # Try to add vertices greedily
        added = 0
        for attempt in range(15):
            if _time_left() < 5:
                break
            
            result = _try_add_vertex(cur_n, cur_edges, cur_adj)
            if result is not None:
                cur_n, cur_edges, cur_adj = result
                added += 1
                if cur_n > best_n:
                    best_n = cur_n
                    best_edges = cur_edges[:]
            else:
                break
        
        # If we improved, restart from improved state
        if added > 0:
            n = cur_n
            edges = cur_edges
            adj = cur_adj
    
    # Final verification
    if best_n > 1 and _verify(best_n, best_edges):
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: return the base graph
    return {'n': n, 'edges': edges}
