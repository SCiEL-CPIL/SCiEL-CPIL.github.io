import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin against 90s limit

    N = 256
    values = list(range(2, N + 1))
    n = len(values)
    val_to_idx = {v: i for i, v in enumerate(values)}

    # Precompute conflict: conflict if lcm(a,b) <= 256
    # Use bitmasks for efficient intersection
    conflict = [0] * n
    for i in range(n):
        a = values[i]
        for j in range(i + 1, n):
            b = values[j]
            if a * b <= 256 * gcd(a, b):
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)

    # Weight for each node (reciprocal)
    weight = [Fraction(1, v) for v in values]

    # Greedy coloring for upper bound
    # Color the candidate set such that nodes of same color don't conflict
    # (i.e., they can't both be in a clique)
    def greedy_color(candidates):
        """
        candidates: list of indices
        Returns list of colors, one per candidate.
        Color i means at most one node of that color can be in any clique.
        So max clique size <= number of distinct colors.
        """
        if not candidates:
            return []
        
        colors = [-1] * len(candidates)
        # Sort candidates by decreasing degree within candidate set (for better coloring)
        # Actually for coloring bound, we want to color so that adjacent vertices get different colors
        # Then max independent set in conflict graph = max clique in compatibility graph <= max color
        # Wait, let me think again.
        
        # We want an upper bound on the max clique in the compatibility graph (edges where lcm > 256).
        # Equivalently, max independent set in the conflict graph.
        # Greedy coloring of the conflict graph: assign colors so that conflicting nodes get different colors.
        # Then any independent set can have at most one node of each color.
        # So max independent set size <= number of colors used.
        
        # For a weighted version, we can use the sum of weights of nodes with the smallest color indices
        # that are "dominating" in some sense.
        
        # Standard approach: color the conflict graph, then for each color, take the node with max weight.
        # Upper bound = sum over colors of (max weight in that color class).
        
        # Let's sort candidates by weight descending for better coloring
        order = sorted(range(len(candidates)), key=lambda k: weight[candidates[k]], reverse=True)
        
        # Try multiple colorings (different orderings) and take min
        best_bound = None
        best_coloring = None
        
        # Coloring 1: by weight descending
        for attempt_order in [order]:
            color_assign = [0] * len(candidates)
            color_class_members = []  # color_class_members[c] = list of indices in candidates
        
            for pos, k in enumerate(attempt_order):
                node = candidates[k]
                # Find which colors are forbidden (used by conflicting neighbors)
                forbidden = set()
                for j in range(n):
                    if j in candidates_set and (conflict[node] >> j) & 1:
                        # j conflicts with node, so j's color is forbidden
                        forbidden.add(color_assign[j])
                
                # Find smallest available color
                c = 0
                while c in forbidden:
                    c += 1
                color_assign[k] = c
                if c >= len(color_class_members):
                    color_class_members.append([])
                color_class_members[c].append(k)
            
            # Compute upper bound: for each color, take max weight
            bound_parts = []
            for c in range(len(color_class_members)):
                if color_class_members[c]:
                    max_w = max(weight[candidates[k]] for k in color_class_members[c])
                    bound_parts.append(max_w)
            
            bound = sum(bound_parts)
            if best_bound is None or bound < best_bound:
                best_bound = bound
                best_coloring = color_assign[:]
        
        # Also try coloring by degree
        deg_order = sorted(range(len(candidates)), 
                          key=lambda k: bin(conflict[candidates[k]] & candidates_set).count('1'), 
                          reverse=True)
        color_assign2 = [0] * len(candidates)
        color_class_members2 = []
        for pos, k in enumerate(deg_order):
            node = candidates[k]
            forbidden = set()
            for j in range(n):
                if j in candidates_set and (conflict[node] >> j) & 1:
                    forbidden.add(color_assign2[j])
            c = 0
            while c in forbidden:
                c += 1
            color_assign2[k] = c
            if c >= len(color_class_members2):
                color_class_members2.append([])
            color_class_members2[c].append(k)
        
        bound2_parts = []
        for c in range(len(color_class_members2)):
            if color_class_members2[c]:
                max_w = max(weight[candidates[k]] for k in color_class_members2[c])
                bound2_parts.append(max_w)
        bound2 = sum(bound2_parts)
        
        if best_bound is None or bound2 < best_bound:
            best_bound = bound2
            best_coloring = color_assign2[:]
        
        return best_bound, best_coloring

    # Branch and bound
    best_value = Fraction(0)
    best_set = []
    
    def bnb(candidates_set, current_set, current_weight):
        nonlocal best_value, best_set
        
        if time.time() - start_time > deadline - 2:
            return
        
        if not candidates_set:
            if current_weight > best_value:
                best_value = current_weight
                best_set = current_set[:]
            return
        
        # Upper bound using greedy coloring
        # candidates_set is a bitmask
        candidates_list = []
        mask = candidates_set
        while mask:
            i = (mask & -mask).bit_length() - 1
            candidates_list.append(i)
            mask &= mask - 1
        
        if not candidates_list:
            if current_weight > best_value:
                best_value = current_weight
                best_set = current_set[:]
            return
        
        bound, coloring = greedy_color(candidates_list)
        
        if current_weight + bound <= best_value:
            return
        
        # If adding all candidates would exceed time, just return
        if time.time() - start_time > deadline - 2:
            return
        
        # Pick a pivot vertex for branching
        # Choose vertex with highest weight among candidates
        pivot = max(candidates_list, key=lambda i: weight[i])
        
        # Branch 1: include pivot
        # Find all candidates compatible with pivot
        compatible = candidates_set & ~conflict[pivot] & ~(1 << pivot)
        current_set.append(pivot)
        new_weight = current_weight + weight[pivot]
        if new_weight > best_value:
            best_value = new_weight
            best_set = current_set[:]
        bnb(compatible, current_set, new_weight)
        current_set.pop()
        
        # Branch 2: exclude pivot
        bnb(candidates_set & ~(1 << pivot), current_set, current_weight)
    
    # Initial greedy solution
    def greedy_solve():
        sel = []
        used = 0
        # Try multiple orderings
        orders = []
        
        # Order by weight descending
        orders.append(sorted(range(n), key=lambda i: weight[i], reverse=True))
        
        # Order by weight / (1 + degree)
        degs = [bin(conflict[i]).count('1') for i in range(n)]
        orders.append(sorted(range(n), key=lambda i: weight[i] / (1 + degs[i]), reverse=True))
        
        # Order by weight / sqrt(1 + degree)
        orders.append(sorted(range(n), key=lambda i: weight[i] / (1 + degs[i])**0.5, reverse=True))
        
        best_sel = []
        best_val = Fraction(0)
        
        for order in orders:
            sel = []
            used = 0
            for idx in order:
                if used & conflict[idx]:
                    continue
                sel.append(idx)
                used |= (1 << idx)
            val = sum(weight[i] for i in sel)
            if val > best_val:
                best_val = val
                best_sel = sel[:]
        
        return best_sel, best_val
    
    # Get initial solution
    initial_sel, initial_val = greedy_solve()
    if initial_val > best_value:
        best_value = initial_val
        best_set = initial_sel[:]
    
    # Local search improvement
    def local_search(sel):
        if not sel:
            return sel, Fraction(0)
        
        sel = sel[:]
        used = 0
        for i in sel:
            used |= (1 << i)
        val = sum(weight[i] for i in sel)
        
        improved = True
        while improved and time.time() - start_time < deadline - 5:
            improved = False
            
            # Try adding
            for i in range(n):
                if (used >> i) & 1:
                    continue
                if used & conflict[i]:
                    continue
                sel.append(i)
                used |= (1 << i)
                val += weight[i]
                improved = True
            
            # Try 1-swap
            non_sel = [i for i in range(n) if not ((used >> i) & 1)]
            for s_idx, s in enumerate(sel):
                used_without = used & ~(1 << s)
                best_gain = Fraction(0)
                best_new = None
                for new_i in non_sel:
                    if used_without & conflict[new_i]:
                        continue
                    gain = weight[new_i] - weight[s]
                    if gain > best_gain:
                        best_gain = gain
                        best_new = new_i
                if best_new is not None:
                    sel[s_idx] = best_new
                    used = used_without | (1 << best_new)
                    val += best_gain
                    improved = True
                    non_sel = [i for i in range(n) if not ((used >> i) & 1)]
                    break
        
        return sel, val
    
    if best_set and time.time() - start_time < deadline - 10:
        polished, polished_val = local_search(best_set)
        if polished_val > best_value:
            best_value = polished_val
            best_set = polished
    
    # Run branch and bound with remaining time
    if time.time() - start_time < deadline - 5:
        # Use all nodes as initial candidates
        all_candidates = (1 << n) - 1
        # But exclude nodes that conflict with nothing? No, include all.
        # Actually, let's be smart: only consider nodes that could potentially improve
        # A node can improve if its weight > 0 (always true) and it doesn't conflict with best_set
        # But we need to explore from scratch for B&B
        
        bnb(all_candidates, [], Fraction(0))
    
    # Final polish
    if best_set and time.time() - start_time < deadline - 2:
        polished, polished_val = local_search(best_set)
        if polished_val > best_value:
            best_value = polished_val
            best_set = polished
    
    if not best_set:
        return [256]
    
    return [values[i] for i in best_set]
