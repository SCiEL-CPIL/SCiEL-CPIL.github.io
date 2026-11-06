import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin from 150s limit

    # Build the known valid 32-vertex graph: 8 disjoint K4s
    # This is C5-free and has alpha = 8 < 9
    n_base = 32
    edges_base = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges_base.append([base + i, base + j])

    best_n = n_base
    best_edges = edges_base

    # Try to find a valid graph on 33 vertices using a local search approach
    # Strategy: Start from the 32-vertex graph and try to add a 33rd vertex
    # with various edge patterns, or modify the existing graph

    def has_c5(n, adj):
        """Check if graph has a 5-cycle."""
        # For each vertex v, for each pair of neighbors u, w of v,
        # check if there's a path u-x-y-w in G-v where x,y are distinct
        # and x != w, y != u, x != y
        for v in range(n):
            neighbors = list(adj[v])
            for i in range(len(neighbors)):
                u = neighbors[i]
                for j in range(i + 1, len(neighbors)):
                    w = neighbors[j]
                    # Look for path u-x-y-w in G - {v}
                    # x in N(u) \ {v, w}, y in N(w) \ {v, u}
                    # x != y, (x,y) in E
                    for x in adj[u]:
                        if x == v or x == w:
                            continue
                        for y in adj[w]:
                            if y == v or y == u or y == x:
                                continue
                            if y in adj[x]:
                                # Cycle: v-u-x-y-w-v
                                # Check all vertices distinct:
                                # v, u, x, y, w
                                # u != w (different indices)
                                # x != u (simple graph, x in N(u))
                                # x != v (checked)
                                # x != w (checked)
                                # y != w (simple graph, y in N(w))
                                # y != v (checked)
                                # y != u (checked)
                                # y != x (checked)
                                return True
        return False

    def alpha_lower_bound(n, adj):
        """Greedy maximal independent set to get a lower bound on alpha."""
        # Use greedy with random ordering
        vertices = list(range(n))
        random.shuffle(vertices)
        used = set()
        indep = []
        for v in vertices:
            if v in used:
                continue
            # Check if v is independent from all in indep
            ok = True
            for u in indep:
                if u in adj[v]:
                    ok = False
                    break
            if ok:
                indep.append(v)
                used.add(v)
                # Mark neighbors as used
                for u in adj[v]:
                    used.add(u)
        return len(indep)

    def alpha_exact_small(n, adj):
        """Exact alpha check for small n using branch and bound."""
        # This is exponential, only use for final verification or small n
        # For n=33, this might be too slow in Python
        # Use a simple backtracking with pruning
        best = [0]
        
        def dfs(available, current_size):
            if current_size > best[0]:
                best[0] = current_size
            if not available or current_size == len(available):
                return
            # Upper bound: current_size + len(available)
            if current_size + len(available) <= best[0]:
                return
            # Pick a vertex with minimum degree in available subgraph
            min_deg = float('inf')
            pivot = None
            for v in available:
                deg = sum(1 for u in available if u != v and u in adj[v])
                if deg < min_deg:
                    min_deg = deg
                    pivot = v
            
            # Branch 1: Include pivot
            new_available = [u for u in available if u != pivot and u not in adj[pivot]]
            dfs(new_available, current_size + 1)
            
            # Branch 2: Exclude pivot
            new_available = [u for u in available if u != pivot]
            dfs(new_available, current_size)
        
        dfs(list(range(n)), 0)
        return best[0]

    def is_c5_free(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return not has_c5(n, adj)

    # Try to find a 33-vertex C5-free graph with alpha < 9
    # Approach: Try various modifications of the 8-K4 structure
    
    if time.time() < deadline - 30:
        # Try adding a 33rd vertex connected to specific patterns
        # For a vertex to not create a C5 with the K4s, it needs careful connection
        
        # Try: vertex 32 connected to one vertex from each K4
        for pattern in range(4**8):  # 65536 patterns, might be too many
            if time.time() > deadline - 5:
                break
            edges_try = list(edges_base)
            adj_try = [set() for _ in range(33)]
            for u, v in edges_base:
                adj_try[u].add(v)
                adj_try[v].add(u)
            
            # Build the connection pattern
            for b in range(8):
                # Extract which vertex in K4 to connect to
                choice = (pattern >> (3*b)) & 7  # 0-3 for valid, skip invalid
                if choice > 3:
                    break
                target = 4*b + choice
                edges_try.append([32, target])
                adj_try[32].add(target)
                adj_try[target].add(32)
            else:
                # All 8 connections made
                if is_c5_free(33, edges_try):
                    # Check alpha
                    lb = alpha_lower_bound(33, adj_try)
                    if lb < 9:
                        # Need exact check
                        if alpha_exact_small(33, adj_try) < 9:
                            best_n = 33
                            best_edges = edges_try
                            break
    else:
        pass

    # If we found n=33, try n=34
    if best_n == 33 and time.time() < deadline - 20:
        # Try adding vertex 33 to the 33-vertex graph
        adj_33 = [set() for _ in range(33)]
        for u, v in best_edges:
            adj_33[u].add(v)
            adj_33[v].add(u)
        
        found_34 = False
        for attempt in range(1000):
            if time.time() > deadline - 5 or found_34:
                break
            edges_try = list(best_edges)
            adj_try = [set(adj_33[i]) for i in range(33)]
            adj_try.append(set())
            
            # Connect vertex 33 to random subset
            num_connect = random.randint(2, 6)
            targets = random.sample(range(33), num_connect)
            for t in targets:
                edges_try.append([33, t])
                adj_try[33].add(t)
                adj_try[t].add(33)
            
            if is_c5_free(34, edges_try):
                lb = alpha_lower_bound(34, adj_try)
                if lb < 9:
                    if alpha_exact_small(34, adj_try) < 9:
                        best_n = 34
                        best_edges = edges_try
                        found_34 = True
    
    return {'n': best_n, 'edges': best_edges}
