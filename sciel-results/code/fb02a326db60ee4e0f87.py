import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = 256

    # Build the conflict graph: edge between a,b if (a*b) % (a+b) == 0
    adj = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # comp_adj[i] = bitmask of vertices that do NOT conflict with i (excluding i itself)
    comp_adj = [0] * (n + 1)
    all_mask = 0
    for i in range(1, n + 1):
        all_mask |= (1 << i)
    for i in range(1, n + 1):
        comp_adj[i] = all_mask & ~(adj[i] | (1 << i))

    best_clique = []
    best_size = 0

    def greedy_color_order(P):
        """
        Greedy coloring of P in the complement graph.
        Returns (color_of, color_order) where color_of[v] is the color (1-indexed)
        and color_order is P sorted by color ascending, then by degree descending.
        The max color is an upper bound on max clique size in P.
        """
        if not P:
            return {}, []
        
        remaining = set(P)
        color_of = {}
        
        color_idx = 0
        while remaining:
            color_idx += 1
            color_class = []
            for v in list(remaining):
                # v can join this color class if it's not adjacent (in comp graph) to any in color_class
                ok = True
                for u in color_class:
                    if (comp_adj[v] >> u) & 1:
                        ok = False
                        break
                if ok:
                    color_class.append(v)
                    color_of[v] = color_idx
            remaining -= set(color_class)
        
        # Sort by color ascending, then by degree in P descending
        def sort_key(v):
            deg = bin(comp_adj[v] & all_mask).count('1')
            return (color_of[v], -deg)
        
        color_order = sorted(P, key=sort_key)
        return color_of, color_order

    def expand(R, P):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        if len(R) + len(P) <= best_size:
            return
        
        color_of, color_order = greedy_color_order(P)
        max_color = max(color_of.values()) if color_of else 0
        
        if len(R) + max_color <= best_size:
            return
        
        P_set = set(P)
        
        # Process in reverse color_order (highest color first)
        for v in reversed(color_order):
            if time.time() > deadline:
                return
            
            if v not in P_set:
                continue
            
            # Upper bound: len(R) + color_of[v] (since v has this color, and 
            # any clique through v can use at most color_of[v] vertices from P
            # that are in color classes 1..color_of[v])
            if len(R) + color_of[v] <= best_size:
                break
            
            R.append(v)
            
            # P_new = P_set ∩ N_comp(v) where N_comp(v) = neighbors of v in complement graph
            P_new = [u for u in P_set if u != v and (comp_adj[v] >> u) & 1]
            
            expand(R, P_new)
            
            R.pop()
            P_set.discard(v)

    P_initial = list(range(1, n + 1))
    expand([], P_initial)

    if not best_clique:
        return list(range(1, n + 1, 2))
    
    return list(best_clique)
