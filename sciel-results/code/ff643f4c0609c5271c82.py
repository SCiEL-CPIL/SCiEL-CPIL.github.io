"""Improved strong edge colouring with bitmask conflict sets and iterative local search."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict
from random import Random

def construct():
    _t0 = time.time()
    _DEADLINE = 140.0

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

    def _build_bitmasks(edges, adj):
        """Precompute conflict bitmasks for each edge."""
        n_edges = len(edges)
        masks = [0] * n_edges
        for i in range(n_edges):
            u1, v1 = edges[i]
            for j in range(i + 1, n_edges):
                u2, v2 = edges[j]
                if (u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2 or
                    u2 in adj[u1] or v2 in adj[u1] or
                    u2 in adj[v1] or v2 in adj[v1]):
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
        return masks

    def _greedy_color(edges, masks, order=None):
        n_edges = len(edges)
        if n_edges == 0:
            return []
        if order is None:
            order = list(range(n_edges))
        
        colour = [-1] * n_edges
        colour_masks = []  # colour_masks[c] = bitmask of edges with colour c
        max_c = 0
        
        for i in order:
            mi = masks[i]
            for c in range(max_c):
                if colour_masks[c] & mi == 0:
                    colour[i] = c
                    colour_masks[c] |= (1 << i)
                    break
            else:
                colour[i] = max_c
                colour_masks.append(1 << i)
                max_c += 1
        
        return colour

    def _local_search(colours, edges, masks, time_limit=5.0):
        t_start = time.time()
        n_edges = len(edges)
        if n_edges == 0:
            return colours
        
        col = colours[:]
        colour_masks = []
        max_c = max(col) + 1 if col else 0
        for c in range(max_c):
            m = 0
            for i, cc in enumerate(col):
                if cc == c:
                    m |= (1 << i)
            colour_masks.append(m)
        
        improved = True
        while improved:
            improved = False
            if time.time() - t_start > time_limit:
                break
            
            for i in range(n_edges):
                cur_c = col[i]
                if cur_c == 0:
                    continue
                mi = masks[i]
                for c in range(cur_c):
                    if colour_masks[c] & mi == 0:
                        colour_masks[cur_c] &= ~(1 << i)
                        colour_masks[c] |= (1 << i)
                        col[i] = c
                        improved = True
                        break
            
            # Compact
            if improved:
                present = sorted(set(col))
                if present != list(range(len(present))):
                    mapping = {old: new for new, old in enumerate(present)}
                    for i in range(n_edges):
                        col[i] = mapping[col[i]]
                    new_cm = []
                    for c in range(len(present)):
                        m = 0
                        for i, cc in enumerate(col):
                            if cc == c:
                                m |= (1 << i)
                        new_cm.append(m)
                    colour_masks = new_cm
        
        return col

    def _try_remove_color(colours, edges, masks, target_c, time_limit=2.0):
        """Try to eliminate color target_c by reassigning its edges."""
        t_start = time.time()
        n_edges = len(edges)
        col = colours[:]
        num_colors = max(col) + 1 if col else 0
        
        if target_c >= num_colors:
            return colours
        
        edges_in_color = [i for i, c in enumerate(col) if c == target_c]
        if not edges_in_color:
            return colours
        
        # Build colour masks
        colour_masks = []
        for c in range(num_colors):
            m = 0
            for i, cc in enumerate(col):
                if cc == c:
                    m |= (1 << i)
            colour_masks.append(m)
        
        # Remove target_c from consideration
        for i in edges_in_color:
            colour_masks[target_c] &= ~(1 << i)
        
        # Try to reassign each edge to a lower color
        for i in edges_in_color:
            if time.time() - t_start > time_limit:
                return colours
            mi = masks[i]
            found = False
            for c in range(target_c):
                if colour_masks[c] & mi == 0:
                    colour_masks[c] |= (1 << i)
                    col[i] = c
                    found = True
                    break
            if not found:
                return colours
        
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
        if time_left < 3:
            # Fallback
            masks = _build_bitmasks(edges, adj)
            colours = _greedy_color(edges, masks)
            out.append(colours)
            continue
        
        masks = _build_bitmasks(edges, adj)
        n_edges = len(edges)
        
        best_colours = None
        best_num = float('inf')
        
        # Try multiple orderings
        orders = [list(range(n_edges))]
        
        # Degree-based
        order_deg = list(range(n_edges))
        order_deg.sort(key=lambda i: len(adj[edges[i][0]]) + len(adj[edges[i][1]]), reverse=True)
        orders.append(order_deg)
        
        # Random orders
        rng = Random(42)
        for _ in range(10):
            order_rand = list(range(n_edges))
            rng.shuffle(order_rand)
            orders.append(order_rand)
        
        for order in orders:
            if time.time() - _t0 > _DEADLINE - 15:
                break
            colours = _greedy_color(edges, masks, order)
            num = max(colours) + 1 if colours else 0
            if num < best_num:
                best_colours = colours
                best_num = num
        
        # Local search
        if time.time() - _t0 < _DEADLINE - 10:
            best_colours = _local_search(best_colours, edges, masks, time_limit=min(8, _DEADLINE - time.time() - 8))
        
        # Try to remove colors from highest to lowest
        if time.time() - _t0 < _DEADLINE - 5:
            num_colors = max(best_colours) + 1 if best_colours else 0
            for target_c in range(num_colors - 1, -1, -1):
                if time.time() - _t0 > _DEADLINE - 3:
                    break
                new_colours = _try_remove_color(best_colours, edges, masks, target_c, time_limit=min(1.5, _DEADLINE - time.time() - 2))
                if new_colours is not best_colours:
                    if max(new_colours) + 1 < max(best_colours) + 1:
                        best_colours = new_colours
        
        out.append(best_colours)

    return out
# EVOLVE-BLOCK-END
