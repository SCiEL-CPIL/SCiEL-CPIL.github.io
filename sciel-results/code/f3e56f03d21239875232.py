"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _creates_c6_fast(adj, n, u, v):
    """Fast check if adding edge (u,v) creates a C6 using meet-in-the-middle.
    
    A C6: u - a - b - c - d - v (all distinct, edges exist).
    Meet-in-the-middle: find all 2-step paths from u (u-a-b) and all 2-step paths to v (v-d-c).
    Then check if any pair shares a middle vertex b==c with valid distinctness.
    """
    adj_u = adj[u]
    adj_v = adj[v]
    
    # Build set of vertices reachable from u in exactly 2 steps: {b : exists a, u~a, a~b, a!=v, b!=u}
    # And track which 'a' led to each b
    # Two-step from u: b is reachable if exists a in adj[u], a != v, and b in adj[a], b != u
    # Two-step to v: d is reachable if exists c in adj[v], c != u, and d in adj[c], d != v
    
    # For efficiency: enumerate 2-step from u, store b -> set of a's
    # Then for each 2-step from v (c -> d where d in adj_v... wait let me rethink)
    
    # Path: u - a - b - c - d - v
    # 2-step from u: a, b (u~a, a~b)
    # 2-step from v: d, c (v~d, d~c)
    # Need: b == c, a != d, a != v, d != u, b != u, b != v, a != b, c != d (but c=b so b!=d)
    
    # Simpler: enumerate all (a,b) pairs for 2-step from u
    # For each (a,b): b must not be u or v
    # Then check if there exists (c,d) with c=b, d in adj[v], d != u, d != a, d != b
    # i.e., b in adj[d] for some d in adj[v] where d != u, d != a, d != b
    
    # Even faster: for each b, precompute if b has a neighbor in adj[v] excluding {u, a, b}
    # Let's just do: for each 2-step (a,b) from u, check if any neighbor of b (call it d) 
    # is also in adj[v], with d != u, d != a, d != b, and d != v (d in adj[v] so d != v unless self-loop, not possible)
    
    # Actually d is in adj[v] means v~d. We need d in adj[b] AND d in adj[v] AND d != u AND d != a AND d != b
    # So: common = adj[b] & adj[v], check if any element in common is not in {u, a, b}
    
    for a in adj_u:
        if a == v:
            continue
        for b in adj[a]:
            if b == u or b == v or b == a:
                continue
            # Check if b shares a neighbor with v (excluding u, a, b)
            common = adj[b] & adj_v
            for d in common:
                if d != u and d != a and d != b:
                    return True
    return False

def _greedy_build(n, rng, deadline):
    """Build a C6-free graph greedily with a given seed."""
    adj = [set() for _ in range(n)]
    edges = set()
    
    # Start with 8 disjoint K5s for a good base
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = 5 * k + a, 5 * k + b
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
    
    # Generate candidate edges and sort by a heuristic
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edges:
                # Heuristic: prefer edges between vertices with lower degrees
                candidates.append((len(adj[u]) + len(adj[v]), rng.random(), u, v))
    
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    for _, _, u, v in candidates:
        if time.time() > deadline:
            break
        if not _creates_c6_fast(adj, n, u, v):
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
    
    return edges, adj

def _local_search(edges, adj, n, deadline, rng):
    """Local search: try edge-addition and swap moves with the fast C6 check."""
    current_count = len(edges)
    best_count = current_count
    best_edges = edges.copy()
    best_adj = [s.copy() for s in adj]
    
    # Build list of non-edges for quick random sampling
    all_pairs = []
    for u in range(n):
        for v in range(u + 1, n):
            all_pairs.append((u, v))
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Strategy: try to add a random non-edge first (free improvement)
        added = None
        for _ in range(50):
            u, v = rng.choice(all_pairs)
            if (u, v) in edges:
                continue
            if not _creates_c6_fast(adj, n, u, v):
                added = (u, v)
                break
        
        if added is not None:
            # Free addition - always accept
            edges.add(added)
            adj[added[0]].add(added[1])
            adj[added[1]].add(added[0])
            no_improve = 0
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges.copy()
                best_adj = [s.copy() for s in adj]
            continue
        
        # No free addition found; try a swap: remove one edge, add another
        removed = None
        for _ in range(50):
            ru, rv = rng.choice(list(edges))
            # Try to find an edge that can replace it
            for _ in range(100):
                au, av = rng.choice(all_pairs)
                if (au, av) in edges:
                    continue
                if (au, av) == (ru, rv):
                    continue
                # Check if we can remove (ru,rv) and add (au,av)
                # First verify (au,av) doesn't create C6 in current graph
                if _creates_c6_fast(adj, n, au, av):
                    continue
                # Remove (ru,rv), add (au,av), check if (au,av) still ok
                adj[ru].discard(rv)
                adj[rv].discard(ru)
                edges.discard((ru, rv))
                
                # Check again after removal (graph changed)
                if not _creates_c6_fast(adj, n, au, av):
                    edges.add((au, av))
                    adj[au].add(av)
                    adj[av].add(au)
                    removed = (ru, rv)
                    break
                else:
                    # Revert
                    edges.add((ru, rv))
                    adj[ru].add(rv)
                    adj[rv].add(ru)
                    continue
            
            if removed is not None:
                break
        
        if removed is None:
            no_improve += 1
            continue
        
        # Swap was successful, count stays same but structure changed
        # Check if we can now add more
        no_improve = max(0, no_improve - 1)
    
    return best_edges, best_adj, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n = 40
    best_overall = 0
    best_edges_overall = set()
    
    # Multi-start: greedy build + local search
    num_starts = 20
    time_per_start = (deadline - start_time) / num_starts
    
    for s in range(num_starts):
        if time.time() > deadline:
            break
        
        seed_start = 42 + s * 7919
        rng = random.Random(seed_start)
        
        # Greedy build with a shorter deadline to leave time for local search
        greedy_deadline = start_time + time_per_start * 0.4
        edges, adj = _greedy_build(n, rng, greedy_deadline)
        
        if len(edges) > best_overall:
            best_overall = len(edges)
            best_edges_overall = edges.copy()
        
        # Local search with remaining time for this start
        ls_deadline = min(deadline - 3, start_time + time_per_start * (s + 1))
        edges, adj, count = _local_search(edges, adj, n, ls_deadline, rng)
        
        if count > best_overall:
            best_overall = count
            best_edges_overall = edges.copy()
    
    # Final verification: make sure no C6 exists
    final_adj = [set() for _ in range(n)]
    for u, v in best_edges_overall:
        final_adj[u].add(v)
        final_adj[v].add(u)
    
    # Verify C6-free by checking all potential 6-cycles
    has_c6 = False
    for u in range(n):
        if has_c6:
            break
        for v in final_adj[u]:
            if v <= u:
                continue
            # Temporarily remove edge and check
            final_adj[u].discard(v)
            final_adj[v].discard(u)
            if _creates_c6_fast(final_adj, n, u, v):
                has_c6 = True
                break
            final_adj[u].add(v)
            final_adj[v].add(u)
    
    if has_c6:
        # This shouldn't happen, but fallback to just the K5s
        fallback = set()
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u, v = 5 * k + a, 5 * k + b
                    fallback.add((u, v))
        return [list(e) for e in fallback]
    
    return [list(e) for e in best_edges_overall]

# EVOLVE-BLOCK-END
