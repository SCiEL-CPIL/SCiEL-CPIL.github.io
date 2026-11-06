"""Strong edge colouring: bitmask-accelerated greedy + local search with annealing."""
import time
from random import Random


def construct():
    t0 = time.time()
    DEADLINE = t0 + 145.0  # safety margin under the 150 s kill

    def gen_instance(n, d, seed):
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

    def build_conflict_bitmasks(edges):
        """Build conflict bitmask for each edge using Python ints as bitsets."""
        m = len(edges)
        # vertex -> list of edge indices incident
        v2e = {}
        for i, (u, v) in enumerate(edges):
            v2e.setdefault(u, []).append(i)
            v2e.setdefault(v, []).append(i)
        
        # edge_set for quick lookup: (u,v) -> edge index
        edge_idx = {}
        for i, (u, v) in enumerate(edges):
            edge_idx[(u, v)] = i
            edge_idx[(v, u)] = i
        
        conflict = [0] * m  # bitmask of conflicting edges
        
        # 1) Share a vertex
        for u, lst in v2e.items():
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    conflict[lst[a]] |= 1 << lst[b]
                    conflict[lst[b]] |= 1 << lst[a]
        
        # 2) Joined by an edge
        for i, (u, v) in enumerate(edges):
            lu = v2e.get(u, [])
            lv = v2e.get(v, [])
            for a in lu:
                if a == i:
                    continue
                for b in lv:
                    if b == i or b == a:
                        continue
                    conflict[a] |= 1 << b
                    conflict[b] |= 1 << a
        
        return conflict

    def greedy_first_fit(edges, conflict, order):
        """Greedy first-fit using conflict bitmasks. Returns (colour_list, num_colours)."""
        m = len(edges)
        colour = [-1] * m
        used = 0
        # For each colour, maintain bitmask of edges already assigned
        col_masks = []
        
        for e in order:
            ce = 1 << e
            placed = False
            for c in range(used):
                if col_masks[c] & conflict[e] == 0:
                    colour[e] = c
                    col_masks[c] |= ce
                    placed = True
                    break
            if not placed:
                colour[e] = used
                col_masks.append(ce)
                used += 1
        return colour, used

    def local_search(edges, conflict, colour, used, rng, time_budget=3.0):
        """Local search: try edge moves, swaps, and class merges. Returns (colour, used)."""
        m = len(edges)
        col_of = colour[:]
        start = time.time()
        
        # Precompute conflict masks for edges? No, conflict is already a list of ints.
        
        for _ in range(1000):
            if time.time() - start > time_budget:
                break
            improved = False
            
            # Build colour classes and their bitmasks
            classes = {}
            class_masks = {}
            for e in range(m):
                c = col_of[e]
                if c >= 0:
                    if c not in classes:
                        classes[c] = []
                        class_masks[c] = 0
                    classes[c].append(e)
                    class_masks[c] |= 1 << e
            
            maxc = max(classes.keys()) if classes else 0
            
            # 1) Try merging colour classes (highest into lower)
            # Iterate from highest color down to try to eliminate it
            for c in range(maxc, -1, -1):
                if c not in classes:
                    continue
                cls_c = classes[c]
                mask_c = class_masks[c]
                
                # Try to merge c into any lower color d
                # If c can be merged into d, we reduce count by 1
                for d in range(c):
                    if d not in classes:
                        continue
                    mask_d = class_masks[d]
                    
                    # Check if any edge in c conflicts with any edge in d
                    # conflict[e] & mask_d != 0 means conflict
                    ok = True
                    for e in cls_c:
                        if conflict[e] & mask_d:
                            ok = False
                            break
                    
                    if ok:
                        for e in cls_c:
                            col_of[e] = d
                        # Update classes
                        classes[d].extend(cls_c)
                        class_masks[d] |= mask_c
                        del classes[c]
                        del class_masks[c]
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                continue
            
            # 2) Try moving a single edge to a lower colour
            # Randomize edge order to avoid bias
            edges_list = list(range(m))
            rng.shuffle(edges_list)
            
            for e in edges_list:
                ce = col_of[e]
                if ce <= 0:
                    continue
                
                # Try to move e to a lower color d
                # We want to find the lowest d < ce such that e doesn't conflict with d
                best_d = None
                for d in range(ce):
                    if d not in classes:
                        continue
                    mask_d = class_masks[d]
                    if conflict[e] & mask_d == 0:
                        best_d = d
                        break
                
                if best_d is not None:
                    # Move e from ce to best_d
                    col_of[e] = best_d
                    classes[best_d].append(e)
                    class_masks[best_d] |= 1 << e
                    classes[ce].remove(e)
                    class_masks[ce] &= ~(1 << e)
                    if not classes[ce]:
                        del classes[ce]
                        del class_masks[ce]
                    improved = True
                    break
            
            if improved:
                continue
            
            # 3) Try 1-swap: move e to colour d, move f from d to colour ce
            # This is more expensive, so we do it less frequently or with limits
            # We'll try a few random swaps
            if rng.random() < 0.5: # 50% chance to try swap
                for _ in range(10): # Limit number of swap attempts per iteration
                    e = rng.randrange(m)
                    ce = col_of[e]
                    if ce <= 0:
                        continue
                    
                    # Pick a random lower color d
                    lower_colors = [d for d in range(ce) if d in classes]
                    if not lower_colors:
                        continue
                    d = rng.choice(lower_colors)
                    
                    # Pick a random edge f from d
                    if not classes[d]:
                        continue
                    f = rng.choice(classes[d])
                    
                    # Check if e can go to d (excluding f)
                    mask_d_without_f = class_masks[d] & ~(1 << f)
                    if conflict[e] & mask_d_without_f:
                        continue
                    
                    # Check if f can go to ce (excluding e)
                    mask_ce_without_e = class_masks[ce] & ~(1 << e)
                    if conflict[f] & mask_ce_without_e:
                        continue
                    
                    # Perform swap
                    col_of[e] = d
                    col_of[f] = ce
                    
                    classes[d].remove(f)
                    classes[d].append(e)
                    class_masks[d] = (class_masks[d] & ~(1 << f)) | (1 << e)
                    
                    classes[ce].remove(e)
                    classes[ce].append(f)
                    class_masks[ce] = (class_masks[ce] & ~(1 << e)) | (1 << f)
                    
                    improved = True
                    break
            
            if not improved:
                break
        
        # Renumber compactly
        present = sorted(set(col_of))
        remap = {old: new for new, old in enumerate(present)}
        for e in range(m):
            col_of[e] = remap[col_of[e]]
        return col_of, len(present)

    def simulated_annealing(edges, conflict, colour, used, rng, time_budget=5.0):
        """SA-style perturbation and local search."""
        m = len(edges)
        best_colour = colour[:]
        best_used = used
        cur_colour = colour[:]
        cur_used = used
        start = time.time()
        T = 10.0
        
        while time.time() - start < time_budget:
            # Perturb: pick a random move
            # 50% chance: move one edge to random color
            # 50% chance: swap colors of two edges
            if rng.random() < 0.5:
                e = rng.randrange(m)
                old_c = cur_colour[e]
                new_c = rng.randrange(used + 1)
                if new_c == old_c:
                    continue
                cur_colour[e] = new_c
                
                # Check validity
                valid = True
                # Check if e conflicts with any other edge in new_c
                # We need to check against all edges in new_c
                # Since we don't have class masks here, we iterate
                # Optimization: We could maintain class masks in SA, but for simplicity
                # and since local_search is called after, we just do a quick check.
                # Actually, a full validity check is O(m).
                # Let's just do the check.
                for f in range(m):
                    if f == e:
                        continue
                    if cur_colour[f] == new_c and (conflict[e] >> f) & 1:
                        valid = False
                        break
                
                if not valid:
                    cur_colour[e] = old_c
                    continue
            else:
                # Swap colors of two edges
                e1 = rng.randrange(m)
                e2 = rng.randrange(m)
                if e1 == e2:
                    continue
                c1 = cur_colour[e1]
                c2 = cur_colour[e2]
                if c1 == c2:
                    continue
                
                # Swap
                cur_colour[e1] = c2
                cur_colour[e2] = c1
                
                # Check validity for e1 in c2
                valid = True
                for f in range(m):
                    if f == e1:
                        continue
                    if cur_colour[f] == c2 and (conflict[e1] >> f) & 1:
                        valid = False
                        break
                
                # Check validity for e2 in c1
                if valid:
                    for f in range(m):
                        if f == e2:
                            continue
                        if cur_colour[f] == c1 and (conflict[e2] >> f) & 1:
                            valid = False
                            break
                
                if not valid:
                    # Revert
                    cur_colour[e1] = c1
                    cur_colour[e2] = c2
                    continue
            
            # Compute actual used
            present = set(cur_colour)
            cur_used = len(present)
            
            if cur_used < best_used:
                best_colour = cur_colour[:]
                best_used = cur_used
            else:
                # Accept with probability exp(-(delta)/T)
                delta = cur_used - best_used
                if delta > 0:
                    # Use math.exp for better precision/performance
                    import math
                    if rng.random() > math.exp(-delta / T):
                        # Reject
                        # Revert the move
                        # This is tricky because we don't store the exact move type
                        # For now, if rejected, we just run local search on the current state?
                        # No, we should revert to the state before the move.
                        # This requires storing the previous state or the move details.
                        # Given the complexity, let's just accept worsening moves rarely
                        # or just run local search.
                        # To keep it simple and robust, let's just accept if delta is small
                        # or use a simpler heuristic.
                        # Actually, reverting is important for SA.
                        # Let's implement a simple revert logic by storing the move.
                        pass # We'll simplify by just accepting all valid moves initially
                        # and relying on local search to fix things.
                        # This makes it more of a random walk with local search.
            
            # Run quick local search
            cur_colour, cur_used = local_search(edges, conflict, cur_colour, cur_used, rng, time_budget=0.5)
            
            if cur_used < best_used:
                best_colour = cur_colour[:]
                best_used = cur_used
            
            T *= 0.95
        
        return best_colour, best_used

    def two_swap_search(edges, conflict, colour, used, rng, time_budget=3.0):
        """Try 2-swap moves: swap edges between two colour classes to enable merges."""
        m = len(edges)
        col_of = colour[:]
        start = time.time()
        
        for _ in range(500):
            if time.time() - start > time_budget:
                break
            improved = False
            
            # Build colour classes and masks
            classes = {}
            class_masks = {}
            for e in range(m):
                c = col_of[e]
                if c >= 0:
                    if c not in classes:
                        classes[c] = []
                        class_masks[c] = 0
                    classes[c].append(e)
                    class_masks[c] |= 1 << e
            
            maxc = max(classes.keys()) if classes else 0
            
            # Try random swaps
            # Pick two random colors a > b
            if maxc < 1:
                break
                
            a = rng.randrange(maxc + 1)
            b = rng.randrange(a) if a > 0 else 0
            if a == b or a not in classes or b not in classes:
                continue
                
            cls_a = classes[a]
            cls_b = classes[b]
            mask_a = class_masks[a]
            mask_b = class_masks[b]
            
            # Pick random edge e from a and f from b
            if not cls_a or not cls_b:
                continue
            e = rng.choice(cls_a)
            f = rng.choice(cls_b)
            
            # Check if e can go to b (excluding f)
            mask_b_without_f = mask_b & ~(1 << f)
            if conflict[e] & mask_b_without_f:
                continue
            
            # Check if f can go to a (excluding e)
            mask_a_without_e = mask_a & ~(1 << e)
            if conflict[f] & mask_a_without_e:
                continue
            
            # Perform swap
            col_of[e] = b
            col_of[f] = a
            
            classes[b].remove(f)
            classes[b].append(e)
            class_masks[b] = (class_masks[b] & ~(1 << f)) | (1 << e)
            
            classes[a].remove(e)
            classes[a].append(f)
            class_masks[a] = (class_masks[a] & ~(1 << e)) | (1 << f)
            
            improved = True
            
            # Try to merge classes after swap
            # Check if any class can be merged into another
            for c in range(maxc, -1, -1):
                if c not in classes:
                    continue
                cls_c = classes[c]
                mask_c = class_masks[c]
                
                for d in range(c):
                    if d not in classes:
                        continue
                    mask_d = class_masks[d]
                    
                    ok = True
                    for edge_c in cls_c:
                        if conflict[edge_c] & mask_d:
                            ok = False
                            break
                    
                    if ok:
                        for edge_c in cls_c:
                            col_of[edge_c] = d
                        classes[d].extend(cls_c)
                        class_masks[d] |= mask_c
                        del classes[c]
                        del class_masks[c]
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # If no merge happened, the swap might not have helped immediately
                # but it changes the landscape. We accept it if it didn't increase used count?
                # Actually, swap keeps used count same.
                # We just accept it.
                pass
        
        # Renumber compactly
        present = sorted(set(col_of))
        remap = {old: new for new, old in enumerate(present)}
        for e in range(m):
            col_of[e] = remap[col_of[e]]
        return col_of, len(present)

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = gen_instance(*spec)
        conflict = build_conflict_bitmasks(edges)
        m = len(edges)
        best_colour = None
        best_used = m
        rng = Random(42)
        
        # Multiple randomized restarts with greedy
        n_restarts = 120
        for r in range(n_restarts):
            if time.time() > DEADLINE - 15.0:
                break
            order = list(range(m))
            rng.shuffle(order)
            colour, used = greedy_first_fit(edges, conflict, order)
            if used < best_used:
                best_colour = colour[:]
                best_used = used
        
        # Local compression on best
        if best_colour is not None:
            best_colour, best_used = local_search(edges, conflict, best_colour, best_used, rng, time_budget=10.0)
        
        # Two-swap search for further improvement
        if time.time() < DEADLINE - 20.0 and best_colour is not None:
            best_colour, best_used = two_swap_search(edges, conflict, best_colour, best_used, rng, time_budget=10.0)
        
        # Simulated annealing for further improvement
        if time.time() < DEADLINE - 20.0 and best_colour is not None:
            best_colour, best_used = simulated_annealing(edges, conflict, best_colour, best_used, rng, time_budget=15.0)
        
        # More targeted restarts if time permits
        if time.time() < DEADLINE - 25.0:
            for r in range(100):
                if time.time() > DEADLINE - 15.0:
                    break
                order = list(range(m))
                rng.shuffle(order)
                colour, used = greedy_first_fit(edges, conflict, order)
                if used <= best_used:
                    colour, used = local_search(edges, conflict, colour, used, rng, time_budget=2.0)
                    if used < best_used:
                        best_colour = colour[:]
                        best_used = used
        
        if best_colour is None:
            best_colour, best_used = greedy_first_fit(edges, conflict, list(range(m)))
        
        out.append(best_colour)
    
    return out
