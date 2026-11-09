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
        # Check if any endpoint of first edge is adjacent to any endpoint of second edge
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
            u, v = edges[i]
            # Try colors 0, 1, 2, ...
            for c in range(max_c + 1):
                ok = True
                for j in colour_to_edges[c]:
                    if _conflict(u, v, edges[j][0], edges[j][1], adj):
                        ok = False
                        break
                if ok:
                    colour[i] = c
                    colour_to_edges[c].add(i)
                    if c == max_c:
                        max_c += 1
                    break
        
        return colour

    def _local_search(colours, edges, adj, time_limit=8.0):
        """
        Improve coloring by moving edges to lower colors and compacting.
        Uses multiple passes with random restarts for stuck edges.
        """
        t_start = time.time()
        n_edges = len(edges)
        if n_edges == 0:
            return colours
        
        # Work on a copy
        col = colours[:]
        
        # Build colour_to_edges
        colour_to_edges = defaultdict(set)
        for i, c in enumerate(col):
            colour_to_edges[c].add(i)
        
        def _can_assign(i, c):
            u, v = edges[i]
            for j in colour_to_edges[c]:
                if j == i:
                    continue
                if _conflict(u, v, edges[j][0], edges[j][1], adj):
                    return False
            return True
        
        improved = True
        while improved:
            improved = False
            if time.time() - t_start > time_limit:
                break
            
            # Pass 1: try to move each edge to the lowest possible color
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

    def _improve_with_swap(colours, edges, adj, time_limit=5.0):
        """
        Try swapping colors of pairs of edges to reduce total colors.
        """
        t_start = time.time()
        n_edges = len(edges)
        if n_edges == 0:
            return colours
        
        col = colours[:]
        colour_to_edges = defaultdict(set)
        for i, c in enumerate(col):
            colour_to_edges[c].add(i)
        
        max_c = max(col) if col else 0
        
        improved = True
        while improved:
            improved = False
            if time.time() - t_start > time_limit:
                break
            
            for i in range(n_edges):
                cur_c = col[i]
                if cur_c == 0:
                    continue
                u, v = edges[i]
                
                # Try moving edge i to a lower color
                for c in range(cur_c):
                    if _can_assign_swap(i, c, col, colour_to_edges, edges, adj):
                        colour_to_edges[cur_c].discard(i)
                        colour_to_edges[c].add(i)
                        col[i] = c
                        improved = True
                        break
                
                # Try swapping with an edge in a lower color
                if not improved:
                    for c in range(cur_c):
                        for j in list(colour_to_edges[c]):
                            a, b = edges[j]
                            # Check if swapping works
                            # Remove i from cur_c and j from c temporarily
                            colour_to_edges[cur_c].discard(i)
                            colour_to_edges[c].discard(j)
                            
                            # Check if i can go to c and j can go to cur_c
                            ok_i = True
                            for k in colour_to_edges[c]:
                                if _conflict(u, v, edges[k][0], edges[k][1], adj):
                                    ok_i = False
                                    break
                            if ok_i:
                                ok_j = True
                                for k in colour_to_edges[cur_c]:
                                    if _conflict(a, b, edges[k][0], edges[k][1], adj):
                                        ok_j = False
                                        break
                            
                            if ok_i and ok_j:
                                # Do the swap
                                colour_to_edges[c].add(i)
                                colour_to_edges[cur_c].add(j)
                                col[i] = c
                                col[j] = cur_c
                                improved = True
                            else:
                                # Undo
                                colour_to_edges[cur_c].add(i)
                                colour_to_edges[c].add(j)
                            break
                        if improved:
                            break
            
            # Compact
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

    def _can_assign_swap(i, c, col, colour_to_edges, edges, adj):
        u, v = edges[i]
        for j in colour_to_edges[c]:
            if j == i:
                continue
            if _conflict(u, v, edges[j][0], edges[j][1], adj):
                return False
        return True

    def _smart_order(edges, adj):
        """
        Order edges by degree (sum of endpoint degrees) descending.
        High-degree edges first tend to use fewer colors.
        """
        n_edges = len(edges)
        order = list(range(n_edges))
        # Sort by sum of endpoint degrees descending
        order.sort(key=lambda i: len(adj[edges[i][0]]) + len(adj[edges[i][1]]), reverse=True)
        return order

    def _advanced_coloring(edges, adj):
        """
        Advanced coloring using multiple strategies and local search.
        """
        n_edges = len(edges)
        if n_edges == 0:
            return []
        
        # Build conflict sets for faster checking
        conflict_sets = _build_conflict_sets(edges, adj)
        
        # Try multiple initial orderings
        orderings = [
            list(range(n_edges)),  # natural order
            list(range(n_edges - 1, -1, -1)),  # reverse
            _smart_order(edges, adj),  # degree-based
        ]
        
        # Also try a random ordering
        rng = Random(42)
        rand_order = list(range(n_edges))
        rng.shuffle(rand_order)
        orderings.append(rand_order)
        
        best_colours = None
        best_num_colors = float('inf')
        
        for order in orderings:
            colours = _greedy_color(edges, adj, order, conflict_sets)
            num = max(colours) + 1 if colours else 0
            if num < best_num_colors:
                best_colours = colours[:]
                best_num_colors = num
        
        # Apply local search
        best_colours = _local_search(best_colours, edges, adj, time_limit=10.0)
        
        # Apply swap-based improvement
        best_colours = _improve_with_swap(best_colours, edges, adj, time_limit=8.0)
        
        return best_colours

    def _reduced_coloring(edges, adj, target_colors):
        """
        Try to color with at most target_colors using backtracking with pruning.
        Returns coloring if found, None otherwise.
        """
        n_edges = len(edges)
        if n_edges == 0:
            return []
        
        # Build conflict sets
        conflict_sets = _build_conflict_sets(edges, adj)
        
        # Order edges by number of conflicts descending (most constrained first)
        order = list(range(n_edges))
        order.sort(key=lambda i: len(conflict_sets[i]), reverse=True)
        
        colour = [-1] * n_edges
        colour_to_edges = [set() for _ in range(target_colors)]
        
        def backtrack(pos):
            if pos == len(order):
                return True
            
            i = order[pos]
            u, v = edges[i]
            
            for c in range(target_colors):
                # Check if edge i can be assigned color c
                ok = True
                for j in colour_to_edges[c]:
                    if _conflict(u, v, edges[j][0], edges[j][1], adj):
                        ok = False
                        break
                
                if ok:
                    colour[i] = c
                    colour_to_edges[c].add(i)
                    
                    if backtrack(pos + 1):
                        return True
                    
                    # Undo
                    colour_to_edges[c].discard(i)
                    colour[i] = -1
            
            return False
        
        if backtrack(0):
            return colour
        return None

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = _instance(n, d, seed)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        # First, get a good coloring with advanced method
        colours = _advanced_coloring(edges, adj)
        num_colors = max(colours) + 1 if colours else 0
        
        # Try to reduce further with backtracking if we have time
        remaining_time = _DEADLINE - time.time()
        if remaining_time > 20:
            # Try target colors from current best down to a reasonable lower bound
            # The Erdős-Nešetřil conjecture gives 5d^2/4
            conjecture_bound = 5 * d * d // 4
            for target in range(num_colors - 1, max(conjecture_bound - 1, 0), -1):
                if time.time() > _DEADLINE - 10:
                    break
                result = _reduced_coloring(edges, adj, target)
                if result is not None:
                    colours = result
                    num_colors = target
                    break
        
        out.append(colours)

    return out
# EVOLVE-BLOCK-END
