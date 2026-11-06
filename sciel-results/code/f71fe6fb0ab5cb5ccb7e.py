import time
import math
import numpy as np
from random import Random

def construct():
    def instance(n, d, seed):
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

    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]

    def compress(color):
        m = len(color)
        if m == 0:
            return []
        mapping = {}
        new = [0] * m
        nxt = 0
        for i, c in enumerate(color):
            if c not in mapping:
                mapping[c] = nxt
                nxt += 1
            new[i] = mapping[c]
        return new

    def build_conflict(n, edges):
        """Build conflict graph as bitsets for fast operations."""
        m = len(edges)
        if m == 0:
            return np.zeros(0, dtype=np.uint64), 0
        
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)
        
        # Conflict: share vertex OR joined by an edge
        conflict = [0] * m  # Each entry is a Python int (arbitrary precision bitset)
        
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                # Share vertex?
                if a == c or a == dd or b == c or b == dd:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                else:
                    # Joined by an edge?
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)
        
        return conflict, m

    def max_independent_set_bnb(conflict, active_mask, time_limit, rng):
        """
        Branch and bound to find maximum independent set.
        Uses bitset representation for speed.
        """
        best_set = [0]  # Best independent set as bitmask
        best_size = [0]
        
        def popcount(x):
            return bin(x).count('1')
        
        def upper_bound(mask):
            """Simple upper bound: number of vertices in mask."""
            return popcount(mask)
        
        def expand(iset, mask):
            nonlocal best_set, best_size
            
            if time.time() > time_limit:
                return
            
            if mask == 0:
                if popcount(iset) > best_size[0]:
                    best_size[0] = popcount(iset)
                    best_set[0] = iset
                return
            
            # Upper bound pruning
            if popcount(iset) + upper_bound(mask) <= best_size[0]:
                return
            
            # Choose vertex with minimum degree in remaining subgraph for better branching
            # Find a vertex to branch on
            v = -1
            min_deg = float('inf')
            temp_mask = mask
            while temp_mask:
                lsb = temp_mask & (-temp_mask)
                v_candidate = lsb.bit_length() - 1
                # Count neighbors in mask
                deg = popcount(conflict[v_candidate] & mask)
                if deg < min_deg:
                    min_deg = deg
                    v = v_candidate
                temp_mask ^= lsb
            
            if v == -1:
                return
            
            # Branch 1: include v
            new_iset = iset | (1 << v)
            # Remove v and its neighbors from mask
            new_mask = mask & ~(conflict[v] | (1 << v))
            expand(new_iset, new_mask)
            
            # Branch 2: exclude v
            new_mask2 = mask & ~(1 << v)
            expand(iset, new_mask2)
        
        # Convert active_mask (Python int) to work with
        if active_mask == 0:
            return 0
        
        expand(0, active_mask)
        return best_set[0]

    def greedy_large_is(conflict, active_mask, rng, num_restarts=200):
        """
        Find a large independent set using greedy with randomization.
        Faster than B&B, used as heuristic.
        """
        if active_mask == 0:
            return 0
        
        def popcount(x):
            return bin(x).count('1')
        
        best_is = 0
        best_size = 0
        
        for _ in range(num_restarts):
            if time.time() > time_limit_global:
                break
            
            iset = 0
            mask = active_mask
            
            while mask:
                # Find min-degree vertex in remaining subgraph
                v = -1
                min_deg = float('inf')
                temp_mask = mask
                candidates = []
                while temp_mask:
                    lsb = temp_mask & (-temp_mask)
                    v_candidate = lsb.bit_length() - 1
                    deg = popcount(conflict[v_candidate] & mask)
                    if deg < min_deg:
                        min_deg = deg
                        candidates = [v_candidate]
                    elif deg == min_deg:
                        candidates.append(v_candidate)
                    temp_mask ^= lsb
                
                if not candidates:
                    break
                
                # Random choice among min-degree candidates
                v = rng.choice(candidates)
                iset |= (1 << v)
                mask &= ~(conflict[v] | (1 << v))
            
            if popcount(iset) > best_size:
                best_size = popcount(iset)
                best_is = iset
        
        return best_is

    def peeling_color(conflict, m, time_limit, rng):
        """
        Peeling approach: repeatedly extract large independent sets.
        """
        colors = [-1] * m
        active = (1 << m) - 1
        current_color = 0
        
        while active and time.time() < time_limit:
            # Try to find a large independent set
            # Use greedy first, then try B&B if time permits
            is_mask = greedy_large_is(conflict, active, rng, 100)
            
            # If we have time, try to improve with B&B on the active set
            remaining = time_limit - time.time()
            if remaining > 0.5 and popcount(active) <= 50:
                bnb_mask = max_independent_set_bnb(conflict, active, time_limit - 0.1, rng)
                if popcount(bnb_mask) > popcount(is_mask):
                    is_mask = bnb_mask
            
            if is_mask == 0:
                # Fallback: pick any vertex
                lsb = active & (-active)
                is_mask = lsb
            
            # Assign color
            temp = is_mask
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                colors[v] = current_color
                temp ^= lsb
            
            active &= ~is_mask
            current_color += 1
        
        # Handle any uncolored vertices
        if -1 in colors:
            next_c = current_color
            for i in range(m):
                if colors[i] == -1:
                    colors[i] = next_c
                    next_c += 1
        
        return colors

    def local_search_reduce(colors, conflict, m, time_limit):
        """
        Try to reduce the number of colors by moving vertices to existing colors.
        """
        num_colors = max(colors) + 1 if m > 0 else 0
        if num_colors == 0:
            return colors, 0
        
        # Build color classes
        color_classes = [set() for _ in range(num_colors)]
        for i in range(m):
            color_classes[colors[i]].add(i)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try to remove each color class
            for c in range(num_colors - 1, -1, -1):
                if not color_classes[c]:
                    continue
                
                # Try to reassign each vertex in this color class to another color
                temp_classes = [set(cls) for cls in color_classes]
                success = True
                
                for v in sorted(color_classes[c]):
                    # Try to assign v to an existing color
                    placed = False
                    for c2 in range(num_colors):
                        if c2 == c:
                            continue
                        # Check if v can be added to color class c2
                        # v must not conflict with any vertex in c2
                        can_add = True
                        for u in temp_classes[c2]:
                            if conflict[v] & (1 << u):
                                can_add = False
                                break
                        if can_add:
                            temp_classes[c2].add(v)
                            placed = True
                            break
                    
                    if not placed:
                        success = False
                        break
                
                if success:
                    # Apply the change
                    color_classes = temp_classes
                    # Rebuild colors array
                    new_colors = [-1] * m
                    for c2 in range(num_colors):
                        for v in color_classes[c2]:
                            new_colors[v] = c2
                    colors = new_colors
                    num_colors -= 1
                    improved = True
                    break
        
        return colors, num_colors

    def solve_instance(n, d, seed, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        conflict, m = build_conflict(n, edges)
        
        def popcount(x):
            return bin(x).count('1')
        
        rng = Random(seed * 7919 + 42)
        
        best_coloring = None
        best_num_colors = float('inf')
        
        # Multiple attempts with different strategies
        attempt = 0
        while time.time() < deadline - 0.5:
            attempt += 1
            attempt_rng = Random(rng.random())
            
            # Get initial coloring via peeling
            coloring = peeling_color(conflict, m, deadline - 0.5, attempt_rng)
            num_colors = max(coloring) + 1 if m > 0 else 0
            
            # Try to reduce colors via local search
            coloring, num_colors = local_search_reduce(coloring, conflict, m, deadline - 0.5)
            
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_coloring = coloring[:]
                
            # Check if we've reached a good bound
            en_bound = int(5 * d * d / 4)
            if best_num_colors <= en_bound:
                break
        
        if best_coloring is None:
            # Fallback: simple greedy
            active = (1 << m) - 1
            colors = [-1] * m
            c = 0
            while active:
                lsb = active & (-active)
                v = lsb.bit_length() - 1
                colors[v] = c
                active &= ~(conflict[v] | (1 << v))
                c += 1
            best_coloring = colors
        
        return compress(best_coloring)

    try:
        start = time.time()
        total_budget = 140.0
        per_instance = total_budget / len(specs)

        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = start + total_budget - time.time()
            share = min(per_instance, remaining / (len(specs) - idx))
            share = max(share, 5.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))

        return out
    except Exception:
        # Fallback: simple greedy
        out = []
        for spec in specs:
            n, d, seed = spec
            edges = instance(n, d, seed)
            if not edges:
                out.append([])
                continue
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj.get(x, set())}
                taken = set()
                for f, c in colour.items():
                    if set(f) & near:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                colour[e] = c
            out.append([colour[e] for e in edges])
        return out
