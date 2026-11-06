import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def _remaining():
        return deadline - time.time()

    def _creates_c4(adj, n, u, v):
        """Check if adding edge (u,v) creates a C4 (i.e., violates C4-free property)."""
        # Condition: no two vertices may have 2+ common neighbors.
        # After adding (u,v):
        # 1. u and v must have < 2 common neighbors (they gain each other, but that's not a common neighbor)
        #    Actually u and v's common neighbors don't change when we add (u,v).
        #    Wait: common neighbors of u and v are vertices w such that w~u and w~v. Adding (u,v) doesn't change that.
        #    So we just need |N(u) ∩ N(v)| <= 1.
        # 2. For any w ~ u (w != v): after adding (u,v), u becomes a common neighbor of w and v (since w~u and u~v).
        #    So |N(w) ∩ N(v) ∪ {u}| must be <= 1. Since u ∉ N(v) before, and u ∈ N(w),
        #    N(w) ∩ N(v)_new = (N(w) ∩ N(v)) ∪ ({u} if u ∈ N(w) else empty)
        #    Since w~u, u ∈ N(w), so this is (N(w) ∩ N(v)) ∪ {u}.
        #    We need |N(w) ∩ N(v) ∪ {u}| <= 1.
        #    Since u ∉ N(v) (edge not yet added), u is not in N(w) ∩ N(v).
        #    So |N(w) ∩ N(v)| + 1 <= 1 => |N(w) ∩ N(v)| = 0.
        # 3. Similarly for w ~ v (w != u): |N(w) ∩ N(u)| = 0.
        
        # Check 1
        if len(adj[u] & adj[v]) >= 2:
            return True
        
        # Check 2: for each neighbor w of u (w != v), N(w) ∩ N(v) must be empty
        for w in adj[u]:
            if w == v:
                continue
            if adj[w] & adj[v]:
                return True
        
        # Check 3: for each neighbor w of v (w != u), N(w) ∩ N(u) must be empty
        for w in adj[v]:
            if w == u:
                continue
            if adj[w] & adj[u]:
                return True
        
        return False

    def _is_c4_free(adj, n):
        """Verify the graph is C4-free."""
        for u in range(n):
            for v in range(u+1, n):
                if len(adj[u] & adj[v]) >= 2:
                    return False
        return True

    def _has_independent_set(adj, n, target):
        """Check if there's an independent set of size 'target' using backtracking."""
        if n < target:
            return False
        
        # Use a branch-and-bound approach
        # Order vertices by degree (higher degree first for pruning)
        order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        
        found = [False]
        
        def backtrack(candidates, current_size):
            if found[0]:
                return
            if current_size == target:
                found[0] = True
                return
            if len(candidates) < target - current_size:
                return
            if time.time() > deadline - 5:
                return
            
            # Pick a vertex from candidates
            v = candidates[0]
            rest = candidates[1:]
            
            # Include v: remove v and its neighbors from candidates
            new_candidates = [x for x in rest if x not in adj[v]]
            backtrack(new_candidates, current_size + 1)
            
            # Exclude v
            if not found[0]:
                backtrack(rest, current_size)
        
        backtrack(order, 0)
        return found[0]

    def _greedy_build_c4_free(n, rng):
        """Build a C4-free graph greedily by adding edges in random order."""
        adj = [set() for _ in range(n)]
        edges = []
        
        all_edges = list(combinations(range(n), 2))
        rng.shuffle(all_edges)
        
        for u, v in all_edges:
            if not _creates_c4(adj, n, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
        
        return adj, edges

    def _refine_graph(adj, n, edges, rng, max_steps=5000):
        """Try to improve the graph by adding/removing edges to reduce independence number."""
        # This is a local search: try swapping edges to improve alpha
        # For now, just return the graph as-is (the greedy construction is the main approach)
        return adj, edges

    def _verify(n, edges):
        """Full verification of the graph."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        if not _is_c4_free(adj, n):
            return False
        
        if _has_independent_set(adj, n, 11):
            return False
        
        return True

    # Try to construct a C4-free graph on n vertices with alpha < 11
    # Start with larger n and work down
    
    best_result = None
    best_n = 0
    
    # Try n = 43 down to 35
    for n in [43, 42, 41, 40, 39, 38, 37, 36, 35]:
        if _remaining() < 10:
            break
        
        found = False
        # Number of attempts based on remaining time
        max_attempts = max(5, int(_remaining() / 2))
        
        for attempt in range(max_attempts):
            if _remaining() < 5:
                break
            
            rng = random.Random(attempt * 7919 + n * 104729)
            
            adj, edges = _greedy_build_c4_free(n, rng)
            
            # Check if alpha < 11
            if not _has_independent_set(adj, n, 11):
                # Verify fully
                if _verify(n, edges):
                    best_result = {'n': n, 'edges': edges}
                    best_n = n
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: 30-vertex graph made of 10 triangles (definitely C4-free, alpha = 10)
    edges = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges.append([a, b])
        edges.append([a, c])
        edges.append([b, c])
    
    return {'n': 30, 'edges': edges}
