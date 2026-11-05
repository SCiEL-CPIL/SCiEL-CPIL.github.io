import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflicts(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflicts = [set() for _ in range(m)]
        for i, (u, v) in enumerate(edges):
            for x in adj[u]:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        conflicts[i].add(edge_to_idx[e2])
            for x in adj[v]:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        conflicts[i].add(edge_to_idx[e2])
            conflicts[i].discard(i)
        return [list(c) for c in conflicts]

    def greedy_color(conflicts, order):
        m = len(conflicts)
        color_of = [-1] * m
        used = 0
        for i in order:
            forbidden = set()
            for j in conflicts[i]:
                if color_of[j] != -1:
                    forbidden.add(color_of[j])
            c = 0
            while c in forbidden:
                c += 1
            color_of[i] = c
            if c + 1 > used:
                used = c + 1
        return color_of, used

    def local_search_reduce(conflicts, colors, max_steps=5000):
        m = len(conflicts)
        colors = colors[:]
        improved = True
        steps = 0
        while improved and steps < max_steps:
            improved = False
            for i in range(m):
                current = colors[i]
                forbidden = set()
                for j in conflicts[i]:
                    forbidden.add(colors[j])
                for c in range(current):
                    if c not in forbidden:
                        colors[i] = c
                        improved = True
                        steps += 1
                        break
        return colors, max(colors) + 1

    def clique_number(conflicts):
        """Greedy clique lower bound via maximal cliques."""
        m = len(conflicts)
        if m == 0:
            return 0
        conflict_sets = [set(c) for c in conflicts]
        best = 0
        # Try multiple random orderings
        rng = random.Random(7)
        for _ in range(5):
            order = list(range(m))
            rng.shuffle(order)
            clique = []
            candidates = set(order)
            for v in order:
                if v not in candidates:
                    continue
                # Check if v is connected to all in clique
                if all(v in conflict_sets[u] for u in clique):
                    clique.append(v)
                    candidates = candidates & conflict_sets[v]
            if len(clique) > best:
                best = len(clique)
        return best

    def branch_and_bound(conflicts, upper_bound, deadline):
        """Exact branch-and-bound to find minimum number of colours."""
        m = len(conflicts)
        if m == 0:
            return [], 0

        # Order edges by conflict degree descending for better pruning
        order = list(range(m))
        order.sort(key=lambda i: len(conflicts[i]), reverse=True)
        # Map: position in order -> edge index
        pos_to_edge = order
        edge_to_pos = {order[i]: i for i in range(m)}

        # Conflict matrix as sets for fast lookup
        conflict_sets = [set(c) for c in conflicts]

        best_colors = None
        best_k = upper_bound

        # State: color assignment array (by edge index)
        color_of = [-1] * m
        # For each colour class, the set of edges in it
        colour_classes = [set() for _ in range(upper_bound)]

        # Precompute: for each edge, which positions come after it
        # We assign in order of `order` (position 0, 1, 2, ...)

        # Forward checking: maintain for each unassigned edge, the set of available colours
        # Actually, let's use a simpler approach: at each step, try assigning colour c to next edge
        # and check consistency.

        def solve(pos, k_used):
            nonlocal best_colors, best_k
            if time.time() > deadline:
                return False

            if pos == m:
                # All edges coloured with k_used colours
                if k_used < best_k:
                    best_k = k_used
                    best_colors = color_of[:]
                return True

            edge = pos_to_edge[pos]
            # Determine available colours: 0..k_used-1 (and possibly k_used if we want to try extending)
            # For minimization, we first try colours 0..best_k-1, but we can also try up to best_k-1
            # Actually we should try colours 0..best_k-1 (since best_k is current upper bound)
            
            # Find forbidden colours from already-coloured conflicts
            forbidden = set()
            for j in conflict_sets[edge]:
                if color_of[j] != -1:
                    forbidden.add(color_of[j])

            # Try colours in order of most constrained (fewest options) - but here we just go 0..best_k-1
            # To be more effective, try colours that appear most in the colour classes (to reuse)
            # Simple approach: try 0, 1, ..., best_k-1
            for c in range(best_k):
                if c in forbidden:
                    continue
                # Assign colour c to edge
                color_of[edge] = c
                colour_classes[c].add(edge)

                # Recurse
                if solve(pos + 1, max(k_used, c + 1)):
                    colour_classes[c].discard(edge)
                    color_of[edge] = -1
                    return True

                colour_classes[c].discard(edge)
                color_of[edge] = -1

            return False

        # Start with best_k = upper_bound
        # We can also try to reduce upper_bound by trying k = upper_bound, upper_bound-1, etc.
        # But branch-and-bound will find the minimum.
        # To speed up, we can first try to find a solution with upper_bound colours,
        # then try upper_bound-1, etc.
        
        # Actually, let's just run the solver and let it find the minimum.
        # But we need to be careful: the solver as written will find SOME solution,
        # not necessarily the minimum, because it returns on first success.
        # Let's modify: we want the MINIMUM number of colours.
        
        # Better approach: binary search or iterative deepening on k.
        # Try k = lower_bound, lower_bound+1, ..., upper_bound
        # For each k, check if a valid k-colouring exists.
        
        # First, get a lower bound
        lb = clique_number(conflicts)
        if lb < 2:
            lb = 2  # At least 2 colours needed if there are edges
        
        # Try increasing k from lb to upper_bound
        for k in range(lb, upper_bound + 1):
            if time.time() > deadline:
                break
            
            # Reset state
            color_of = [-1] * m
            found = False
            
            def solve_k(pos, k_val):
                nonlocal found
                if found:
                    return True
                if time.time() > deadline:
                    return False
                
                if pos == m:
                    found = True
                    best_colors = color_of[:]
                    best_k = k_val
                    return True
                
                edge = pos_to_edge[pos]
                forbidden = set()
                for j in conflict_sets[edge]:
                    if color_of[j] != -1:
                        forbidden.add(color_of[j])
                
                # Try colours 0..k_val-1
                # Heuristic: try colours in order of increasing class size (to balance)
                # Or just 0..k_val-1
                for c in range(k_val):
                    if c in forbidden:
                        continue
                    color_of[edge] = c
                    if solve_k(pos + 1, k_val):
                        return True
                    color_of[edge] = -1
                
                return False
            
            if solve_k(0, k):
                break
        
        if best_colors is not None:
            return best_colors, best_k
        return None, upper_bound

    def color_graph(edges, adj, seed_offset=0):
        m = len(edges)
        if m == 0:
            return [], 0
        
        conflicts = build_conflicts(edges, adj)
        
        # Get initial upper bound via greedy + local search
        best_ub = float('inf')
        best_colors = None
        
        # Multiple greedy orderings
        rng = random.Random(seed_offset)
        for _ in range(10):
            order = list(range(m))
            rng.shuffle(order)
            colors, score = greedy_color(conflicts, order)
            if score < best_ub:
                best_ub = score
                best_colors = colors[:]
        
        # Also try degree-sorted orderings
        order_desc = list(range(m))
        order_desc.sort(key=lambda i: len(conflicts[i]), reverse=True)
        colors, score = greedy_color(conflicts, order_desc)
        if score < best_ub:
            best_ub = score
            best_colors = colors[:]
        
        order_asc = list(range(m))
        order_asc.sort(key=lambda i: len(conflicts[i]))
        colors, score = greedy_color(conflicts, order_asc)
        if score < best_ub:
            best_ub = score
            best_colors = colors[:]
        
        # Local search to reduce
        if best_colors is not None:
            ls_colors, ls_score = local_search_reduce(conflicts, best_colors, max_steps=3000)
            if ls_score < best_ub:
                best_ub = ls_score
                best_colors = ls_colors
        
        # Now try exact branch-and-bound to find the true minimum
        if time.time() < deadline - 15.0:
            exact_colors, exact_k = branch_and_bound(conflicts, best_ub, deadline - 15.0)
            if exact_colors is not None and exact_k < best_ub:
                best_ub = exact_k
                best_colors = exact_colors
        
        return best_colors, best_ub

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    graphs = []
    for spec in specs:
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        graphs.append((edges, adj))
    
    best_results = []
    best_scores = [float('inf')] * len(graphs)
    
    for idx, (edges, adj) in enumerate(graphs):
        if time.time() > deadline - 5.0:
            break
        
        # Run with a couple of seed offsets for the greedy initialization
        for offset in range(5):
            if time.time() > deadline - 10.0:
                break
            colors, score = color_graph(edges, adj, seed_offset=offset)
            if score < best_scores[idx]:
                best_scores[idx] = score
                best_results.append(colors) if len(best_results) <= idx else None
                while len(best_results) < len(graphs):
                    best_results.append([])
                best_results[idx] = colors
        
        if time.time() > deadline - 5.0:
            break
    
    for i in range(len(graphs)):
        if not best_results[i]:
            colors, _ = color_graph(graphs[i][0], graphs[i][1], seed_offset=0)
            best_results[i] = colors

    return best_results
