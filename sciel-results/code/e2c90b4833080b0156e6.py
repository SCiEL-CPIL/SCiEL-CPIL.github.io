"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict
from random import Random

def construct():
    # Deadline guard: finish well before the 150s wall-clock kill.
    _t0 = time.time()
    _DEADLINE = 138.0  # leave ~12s margin

    def _instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def _conflict(u1, v1, u2, v2, adj):
        """Check if two edges (u1,v1) and (u2,v2) conflict (share vertex or are joined)."""
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            return True
        if u2 in adj[u1] or v2 in adj[u1]:
            return True
        if u2 in adj[v1] or v2 in adj[v1]:
            return True
        return False

    def _conflict_fast(i, j, edges, adj):
        """Faster conflict check using precomputed edge endpoints."""
        u1, v1 = edges[i]
        u2, v2 = edges[j]
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            return True
        if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
            return True
        return False

    def _build_conflict_sets(edges, adj):
        """Build conflict sets for all edges."""
        n = len(edges)
        conflict_sets = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if _conflict_fast(i, j, edges, adj):
                    conflict_sets[i].add(j)
                    conflict_sets[j].add(i)
        return conflict_sets

    def _greedy_color(edges, adj, order=None, conflict_sets=None):
        """
        Greedy coloring with a given edge order.
        Returns list of colours (one per edge in original sorted order).
        """
        n_edges = len(edges)
        if n_edges == 0:
            return []
        
        if order is None:
            order = list(range(n_edges))
        
        colour = [-1] * n_edges
        colour_to_edges = defaultdict(set)
        max_c = 0
        
        for i in order:
            for c in range(max_c + 1):
                ok = True
                for j in colour_to_edges[c]:
                    if conflict_sets is not None:
                        if j in conflict_sets[i]:
                            ok = False
                            break
                    else:
                        if _conflict(edges[i][0], edges[i][1], edges[j][0], edges[j][1], adj):
                            ok = False
                            break
                if ok:
                    colour[i] = c
                    colour_to_edges[c].add(i)
                    if c == max_c:
                        max_c += 1
                    break
        
        return colour

    def _advanced_coloring(edges, adj, time_limit=10.0):
        """
        Advanced coloring using multiple strategies and local search.
        """
        t_start = time.time()
        n_edges = len(edges)
        if n_edges == 0:
            return []
        
        conflict_sets = _build_conflict_sets(edges, adj)
        
        best_colours = None
        best_num_colors = float('inf')
        
        # Try multiple orderings
        orders = []
        
        # Natural order
        orders.append(list(range(n_edges)))
        
        # Degree-based order
        order_deg = list(range(n_edges))
        order_deg.sort(key=lambda i: len(adj[edges[i][0]]) + len(adj[edges[i][1]]), reverse=True)
        orders.append(order_deg)
        
        # Reverse natural
        orders.append(list(range(n_edges - 1, -1, -1)))
        
        # Random orders
        rng = Random(42)
        for _ in range(5):
            order_rand = list(range(n_edges))
            rng.shuffle(order_rand)
            orders.append(order_rand)
        
        for order in orders:
            if time.time() - t_start > time_limit:
                break
            colours = _greedy_color(edges, adj, order, conflict_sets)
            num = max(colours) + 1 if colours else 0
            if num < best_num_colors:
                best_colours = colours
                best_num_colors = num
        
        # Local search improvement
        if time.time() - t_start < time_limit - 2:
            best_colours = _local_search_advanced(best_colours, edges, adj, conflict_sets, time_limit=time_limit - 2)
        
        return best_colours

    def _local_search_advanced(colours, edges, adj, conflict_sets, time_limit=8.0):
        """
        Improve coloring using conflict sets for fast checks.
        """
        t_start = time.time()
        n_edges = len(edges)
        if n_edges == 0:
            return colours
        
        col = colours[:]
        colour_to_edges = defaultdict(set)
        for i, c in enumerate(col):
            colour_to_edges[c].add(i)
        
        def _can_assign(i, c):
            for j in colour_to_edges[c]:
                if j == i:
                    continue
                if j in conflict_sets[i]:
                    return False
            return True
        
        improved = True
        while improved:
            improved = False
            if time.time() - t_start > time_limit:
                break
            
            for i in range(n_edges):
                cur_c = col[i]
                if cur_c == 0:
                    continue
                for c in range(cur_c):
                    if _can_assign(i, c):
                        colour_to_edges[cur_c].discard(i)
                        colour_to_edges[c].add(i)
                        col[i] = c
                        improved = True
                        break
            
            # Compact colors
            if improved:
                present = sorted(set(col))
                if present != list(range(len(present))):
                    mapping = {old: new for new, old in enumerate(present)}
                    for i in range(n_edges):
                        col[i] = mapping[col[i]]
                    new_cte = defaultdict(set)
                    for i, c in enumerate(col):
                        new_cte[c].add(i)
                    colour_to_edges = new_cte
        
        return col

    def _reduced_coloring(colours, edges, adj, conflict_sets, time_limit=5.0):
        """
        Try to reduce colors by removing a color class and reassigning.
        """
        t_start = time.time()
        n_edges = len(edges)
        if n_edges == 0:
            return colours
        
        col = colours[:]
        num_colors = max(col) + 1 if col else 0
        
        for target_c in range(num_colors - 1, -1, -1):
            if time.time() - t_start > time_limit:
                break
            
            # Try to remove color target_c
            edges_in_color = [i for i, c in enumerate(col) if c == target_c]
            if not edges_in_color:
                continue
            
            # Try to reassign each edge in target_c to another color
            success = True
            for i in edges_in_color:
                found = False
                for c in range(target_c):
                    # Check if i can be assigned to c
                    ok = True
                    for j in [k for k, cc in enumerate(col) if cc == c and k != i]:
                        if j in conflict_sets[i]:
                            ok = False
                            break
                    if ok:
                        col[i] = c
                        found = True
                        break
                if not found:
                    success = False
                    break
            
            if success:
                # Compact
                present = sorted(set(col))
                if present != list(range(len(present))):
                    mapping = {old: new for new, old in enumerate(present)}
                    for i in range(n_edges):
                        col[i] = mapping[col[i]]
        
        return col

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = _instance(n, d, seed)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        time_left = _DEADLINE - time.time()
        if time_left < 5:
            # Fallback: quick greedy
            colours = _greedy_color(edges, adj, None)
            out.append(colours)
            continue
        
        # Use advanced coloring
        best_colours = _advanced_coloring(edges, adj, time_limit=max(5, time_left - 10))
        
        # Try reduced coloring if time permits
        if time.time() < _DEADLINE - 5:
            conflict_sets = _build_conflict_sets(edges, adj)
            best_colours = _reduced_coloring(best_colours, edges, adj, conflict_sets, time_limit=min(5, _DEADLINE - time.time() - 2))
        
        out.append(best_colours)

    return out
# EVOLVE-BLOCK-END
