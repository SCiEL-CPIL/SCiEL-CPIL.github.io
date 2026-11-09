import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

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

    def solve_instance(n, d, seed, time_limit):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Graph adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Conflict sets (bitset as Python int for speed)
        # Two edges conflict if they share a vertex or are joined by an edge
        conflict = [0] * m
        for i in range(m):
            ui, vi = edges[i]
            for j in range(i + 1, m):
                uj, vj = edges[j]
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                else:
                    if ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)

        # ---- Greedy initial coloring (DSATUR-like) ----
        def greedy_color():
            colors = [-1] * m
            uncolored = set(range(m))
            for _ in range(m):
                best_i, best_key = None, (-1, -1)
                for i in uncolored:
                    # saturation: number of distinct colors among neighbors
                    seen = set()
                    mask = conflict[i]
                    for j in uncolored:
                        if mask & (1 << j):
                            cj = colors[j]
                            if cj != -1:
                                seen.add(cj)
                    # degree among uncolored
                    deg = 0
                    mask = conflict[i]
                    for j in uncolored:
                        if mask & (1 << j):
                            deg += 1
                    key = (len(seen), deg)
                    if key > best_key:
                        best_key = key
                        best_i = i
                i = best_i
                taken = set()
                mask = conflict[i]
                for j in uncolored:
                    if mask & (1 << j):
                        cj = colors[j]
                        if cj != -1:
                            taken.add(cj)
                c = 0
                while c in taken:
                    c += 1
                colors[i] = c
                uncolored.discard(i)
            return colors

        init_colors = greedy_color()
        init_count = max(init_colors) + 1 if m > 0 else 0

        # ---- MaxIS on conflict graph using bitsets (Tomita-style bounded) ----
        # Returns the largest independent set (as list of indices) from a given
        # candidate set, bounded by 'bound' (we only care about sets >= bound).
        # Independent set in conflict graph = induced matching in original.
        
        # Precompute for speed: for each vertex, its non-neighbor mask within a set
        # We'll implement a recursive bounded MaxIS.

        def max_is_bounded(candidates, bound, deadline_local):
            """Find an independent set of size >= bound in candidates (a set of ints).
            Returns the set as a list, or None if no such set exists."""
            if len(candidates) < bound:
                return None
            if time.time() > deadline_local:
                return None
            
            # Convert to list for iteration
            cand_list = list(candidates)
            
            # Greedy lower bound: try to build a large IS greedily
            def greedy_is(cand_set):
                result = []
                remaining = set(cand_set)
                while remaining:
                    # Pick vertex with minimum degree within remaining
                    best_v = None
                    best_deg = float('inf')
                    for v in remaining:
                        # Count conflicts within remaining
                        mask = conflict[v]
                        deg = 0
                        for u in remaining:
                            if u != v and (mask & (1 << u)):
                                deg += 1
                        if deg < best_deg:
                            best_deg = deg
                            best_v = v
                        if best_deg == 0:
                            break
                    if best_v is None:
                        break
                    result.append(best_v)
                    # Remove best_v and its neighbors from remaining
                    new_remaining = set()
                    mask = conflict[best_v]
                    for u in remaining:
                        if u != best_v and not (mask & (1 << u)):
                            new_remaining.add(u)
                    remaining = new_remaining
                return result
            
            greedy_sol = greedy_is(candidates)
            if len(greedy_sol) >= bound:
                return greedy_sol
            
            # Exact bounded search
            best_found = None
            best_size = 0
            
            def expand(P, X, current_set, depth):
                nonlocal best_found, best_size
                if time.time() > deadline_local:
                    return
                if len(P) + len(current_set) <= best_size:
                    return
                if not P:
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_found = current_set[:]
                    return
                
                # Pick a vertex from P with maximum degree in P
                # (for IS, we want to branch on vertices)
                # Actually for MaxIS, standard approach: pick vertex, branch include/exclude
                # But with bitsets it's tricky. Let's use a simpler approach:
                # Since graphs are small (m <= ~240), use recursive selection
                
                # Choose vertex with max degree in P
                v = None
                max_deg = -1
                for u in P:
                    deg = 0
                    mask = conflict[u]
                    for w in P:
                        if w != u and (mask & (1 << w)):
                            deg += 1
                    if deg > max_deg:
                        max_deg = deg
                        v = u
                
                # Branch 1: include v
                new_P = set()
                mask_v = conflict[v]
                for u in P:
                    if u != v and not (mask_v & (1 << u)):
                        new_P.add(u)
                current_set.append(v)
                expand(new_P, X | {v}, current_set, depth + 1)
                current_set.pop()
                
                # Branch 2: exclude v
                P.remove(v)
                expand(P, X | {v}, current_set, depth + 1)
                P.add(v)
            
            expand(set(candidates), set(), [], 0)
            
            if best_found and len(best_found) >= bound:
                return best_found
            return None

        # ---- Iterative MaxIS decomposition to minimize colors ----
        def decompose_by_maxis(init_cols, time_budget):
            """Starting from a valid coloring, try to reduce colors by 
            repartitioning using MaxIS extraction."""
            colors = init_cols[:]
            current_count = max(colors) + 1 if colors else 0
            
            # Build color classes
            color_classes = defaultdict(set)
            for i, c in enumerate(colors):
                color_classes[c].add(i)
            
            # We'll try to find a better coloring by:
            # 1. Taking the largest independent set from all edges
            # 2. Repeating on the remainder
            # This gives a coloring where each color class is a maximum IS
            # (not necessarily globally optimal but often good)
            
            uncolored = set(range(m))
            new_colors = [-1] * m
            color_idx = 0
            deadline_local = time.time() + time_budget
            
            while uncolored and time.time() < deadline_local:
                # Find a large independent set in uncolored
                # Use greedy first for speed
                greedy_sol = greedy_is(uncolored)
                if not greedy_sol:
                    # All remaining are mutually conflicting? Shouldn't happen
                    # Assign each its own color
                    for i in uncolored:
                        new_colors[i] = color_idx
                        color_idx += 1
                    break
                
                # Try to improve with bounded exact search
                # Target: at least len(greedy_sol)
                bound = len(greedy_sol)
                if bound > 1:
                    exact_sol = max_is_bounded(uncolored, bound, deadline_local)
                    if exact_sol and len(exact_sol) > len(greedy_sol):
                        greedy_sol = exact_sol
                
                # Assign this color class
                for i in greedy_sol:
                    new_colors[i] = color_idx
                    uncolored.discard(i)
                color_idx += 1
            
            if color_idx < current_count:
                return new_colors, color_idx
            return colors, current_count

        # ---- Local search refinement: swap edges between color classes to reduce count ----
        def local_refine(colors_in, time_budget):
            """Try to reduce number of colors by moving edges between classes."""
            rng = random.Random(seed + 42)
            colors = colors_in[:]
            max_c = max(colors) if colors else -1
            current_count = max_c + 1
            
            # Build membership
            members = defaultdict(set)
            for i, c in enumerate(colors):
                members[c].add(i)
            
            best_count = current_count
            best_colors = colors[:]
            deadline_local = time.time() + time_budget
            
            while time.time() < deadline_local and best_count > 1:
                progress = False
                # Try to eliminate the highest color class
                if best_count > 1:
                    top_color = best_count - 1
                    if top_color in members and members[top_color]:
                        # Try to redistribute edges from top_color into lower colors
                        for i in list(members[top_color]):
                            # Try each lower color
                            for c in range(top_color):
                                if c not in members or not members[c]:
                                    continue
                                # Check if i can be added to color c
                                ok = True
                                mask = conflict[i]
                                for j in members[c]:
                                    if mask & (1 << j):
                                        ok = False
                                        break
                                if ok:
                                    # Move i from top_color to c
                                    members[top_color].discard(i)
                                    members[c].add(i)
                                    colors[i] = c
                                    progress = True
                                    break
                            if progress:
                                break
                        if not members[top_color]:
                            # Eliminate this color
                            for c in range(top_color, -1, -1):
                                if c in members and members[c]:
                                    best_count = c + 1
                                    break
                            best_colors = colors[:]
            
            return best_colors, best_count

        # ---- Main solving loop ----
        best_colors = init_colors
        best_count = init_count
        
        # Phase 1: MaxIS decomposition
        if time.time() < deadline - 10:
            t1 = min(15.0, deadline - time.time() - 10)
            new_c, new_cnt = decompose_by_maxis(init_colors, t1)
            if new_cnt < best_count:
                best_colors = new_c
                best_count = new_cnt
        
        # Phase 2: Local refinement
        if time.time() < deadline - 5:
            t2 = min(15.0, deadline - time.time() - 5)
            new_c, new_cnt = local_refine(best_colors, t2)
            if new_cnt < best_count:
                best_colors = new_c
                best_count = new_cnt
        
        # Phase 3: Re-run MaxIS decomposition on the refined coloring
        if time.time() < deadline - 5:
            t3 = min(15.0, deadline - time.time() - 5)
            new_c, new_cnt = decompose_by_maxis(best_colors, t3)
            if new_cnt < best_count:
                best_colors = new_c
                best_count = new_cnt
        
        # Phase 4: More local refinement
        if time.time() < deadline - 2:
            t4 = min(10.0, deadline - time.time() - 2)
            new_c, new_cnt = local_refine(best_colors, t4)
            if new_cnt < best_count:
                best_colors = new_c
                best_count = new_cnt
        
        # Verify final coloring is valid
        def is_valid(cols):
            cc = defaultdict(set)
            for i, c in enumerate(cols):
                cc[c].add(i)
            for c, ms in cc.items():
                ms_list = list(ms)
                for a in range(len(ms_list)):
                    for b in range(a + 1, len(ms_list)):
                        if conflict[ms_list[a]] & (1 << ms_list[b]):
                            return False
            return True
        
        if not is_valid(best_colors):
            # Fallback to initial
            best_colors = init_colors
        
        return best_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        remaining = deadline - time.time()
        if remaining <= 1.0:
            # Emergency greedy
            edges = instance(n, d, seed)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            cols = {}
            for e in edges:
                near = set(e)
                for x in e:
                    near |= adj[x]
                taken = set()
                for f, c in cols.items():
                    if set(f) & near:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                cols[e] = c
            results.append([cols[e] for e in edges])
            continue
        tl = max(1.0, remaining - 1.0)
        results.append(solve_instance(n, d, seed, tl))
    return results
