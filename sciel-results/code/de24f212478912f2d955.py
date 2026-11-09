"""Strong edge colouring with bitmask conflict sets, DSATUR-like greedy, and swap-based local search."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict
from random import Random

def construct():
    _t0 = time.time()
    _DEADLINE = 140.0  # Safety margin before 150s kill

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
            
            # Pass 1: Simple moves (move edge to lower color if possible)
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
            
            # Pass 2: Swaps (try to swap edge i (color c1) with edge j (color c2) 
            # such that i moves to c2 and j moves to c1, potentially reducing max color)
            # We focus on reducing the highest color class.
            if max_c > 0:
                top_color = max_c - 1
                edges_in_top = [i for i in range(n_edges) if col[i] == top_color]
                for i in edges_in_top:
                    if time.time() - t_start > time_limit:
                        break
                    mi = masks[i]
                    # Try to move i to a lower color c2
                    for c2 in range(top_color):
                        if colour_masks[c2] & mi == 0:
                            # Direct move possible
                            colour_masks[top_color] &= ~(1 << i)
                            colour_masks[c2] |= (1 << i)
                            col[i] = c2
                            improved = True
                            break
                        else:
                            # Check for swap
                            # Find edges in c2 that conflict with i
                            conflicting_in_c2 = colour_masks[c2] & mi
                            if conflicting_in_c2:
                                # Iterate through conflicting edges in c2
                                # We can only have a few due to degree constraints
                                # Extract indices
                                temp_mask = conflicting_in_c2
                                while temp_mask:
                                    # Get lowest set bit
                                    lsb = temp_mask & -temp_mask
                                    j = lsb.bit_length() - 1
                                    temp_mask ^= lsb
                                    
                                    mj = masks[j]
                                    # Check if j can move to top_color
                                    # j conflicts with i, but i is moving away.
                                    # We need to check if j conflicts with any other edge in top_color
                                    # excluding i.
                                    # mask of top_color edges excluding i:
                                    top_mask_excl_i = colour_masks[top_color] & ~(1 << i)
                                    
                                    if (mj & top_mask_excl_i) == 0:
                                        # Swap i and j
                                        # i: top_color -> c2
                                        # j: c2 -> top_color
                                        
                                        # Update masks
                                        colour_masks[top_color] &= ~(1 << i)
                                        colour_masks[top_color] |= (1 << j)
                                        
                                        colour_masks[c2] &= ~(1 << j)
                                        colour_masks[c2] |= (1 << i)
                                        
                                        col[i] = c2
                                        col[j] = top_color
                                        
                                        improved = True
                                        break
                                if improved:
                                    break
                                    
            # Compact colors if improved
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
                    max_c = len(present)
                else:
                    # Update max_c just in case
                    max_c = max(col) + 1 if col else 0
                    # Rebuild masks if compaction didn't happen but colors changed? 
                    # Actually, if present == range, masks are still valid structure-wise 
                    # but we might have empty colors? No, compaction removes empty.
                    # If no compaction, masks are still correct for the indices.
        
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
        
        # Degree-based (High conflict first)
        order_deg = list(range(n_edges))
        # Approximate conflict count by sum of degrees of endpoints
        order_deg.sort(key=lambda i: len(adj[edges[i][0]]) + len(adj[edges[i][1]]), reverse=True)
        orders.append(order_deg)
        
        # Random orders
        rng = Random(42)
        for _ in range(5):
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
        
        # Local search with time limit
        time_for_ls = min(10.0, _DEADLINE - time.time() - 5.0)
        if time_for_ls > 1.0:
            best_colours = _local_search(best_colours, edges, masks, time_limit=time_for_ls)
        
        out.append(best_colours)

    return out
# EVOLVE-BLOCK-END
