import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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

        # Precompute adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Precompute conflict sets
        conflict_sets = [set() for _ in range(m)]
        for i in range(m):
            ui, vi = edges[i]
            for j in range(i + 1, m):
                uj, vj = edges[j]
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict_sets[i].add(j)
                    conflict_sets[j].add(i)
                else:
                    if ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                        conflict_sets[i].add(j)
                        conflict_sets[j].add(i)

        # Greedy DSATUR initial coloring
        def greedy_dsatur():
            colors = [-1] * m
            uncolored = set(range(m))
            while uncolored:
                best_i, best_score = None, (-1, -1)
                for i in uncolored:
                    neighbor_colors = set()
                    for j in conflict_sets[i]:
                        if colors[j] != -1:
                            neighbor_colors.add(colors[j])
                    sat = len(neighbor_colors)
                    deg = sum(1 for j in conflict_sets[i] if j in uncolored)
                    score = (sat, deg)
                    if score > best_score:
                        best_score = score
                        best_i = i
                i = best_i
                taken = set()
                for j in conflict_sets[i]:
                    if colors[j] != -1:
                        taken.add(colors[j])
                c = 0
                while c in taken:
                    c += 1
                colors[i] = c
                uncolored.remove(i)
            return colors

        init_colors = greedy_dsatur()
        best_colors = init_colors[:]
        best_num_colors = max(init_colors) + 1 if init_colors else 0

        # Exact B&B for Max Independent Set
        # State: remaining nodes (R), chosen nodes (C)
        # Objective: Maximize |C| + |MIS(R)|
        
        # Greedy Lower Bound for MIS(R)
        def greedy_mis_lower_bound(R):
            if not R:
                return 0
            R = set(R)
            count = 0
            while R:
                # Pick node with min degree in R
                min_deg = float('inf')
                chosen = None
                for v in R:
                    deg = sum(1 for u in R if u != v and u in conflict_sets[v])
                    if deg < min_deg:
                        min_deg = deg
                        chosen = v
                count += 1
                R.discard(chosen)
                # Remove neighbors of chosen from R
                to_remove = set()
                for u in R:
                    if u in conflict_sets[chosen]:
                        to_remove.add(u)
                R -= to_remove
            return count

        # B&B Solver
        # We want to find the size of the maximum independent set.
        # If size is S, then we need m/S colors? No, that's packing.
        # We want to pack edges into color classes. Each class is an independent set in the conflict graph.
        # So we want to partition V into k independent sets.
        # This is graph coloring.
        # B&B on Graph Coloring:
        # Try to color with K colors.
        # Backtrack search.
        
        # Let's try to find the minimum K such that a valid coloring exists.
        # We can binary search K or just increment K.
        # Upper bound: best_num_colors
        # Lower bound: max degree in conflict graph + 1? Or clique size?
        
        # Let's use the B&B approach for graph coloring directly.
        # Order nodes by degree (descending) or saturation.
        # Let's stick to the node ordering from DSATUR for consistency or just degree.
        
        # Actually, the prompt inspiration 3 suggested "exact minimum graph coloring via a maximum independent set (MaxIS)".
        # This usually means:
        # 1. Find MaxIS in G. Remove it. Repeat.
        # This gives a coloring, but not necessarily optimal.
        # However, we can use B&B to find the *minimum number* of colors.
        
        # Let's implement a standard B&B for graph coloring (branching on the next vertex to color).
        # To make it fast, we use the "DSATUR" ordering dynamically or statically.
        # Static ordering by degree is simpler and often good enough for these sizes (m ~ 80-100).
        
        # Let's try a hybrid:
        # Use the DSATUR greedy result as the upper bound (UB).
        # Try to find a coloring with K < UB.
        # Use B&B with pruning.
        
        # Precompute static order: nodes sorted by degree in conflict graph (descending)
        degrees = [len(conflict_sets[i]) for i in range(m)]
        order = sorted(range(m), key=lambda i: -degrees[i])
        
        # Map original index to position in order
        # Or just iterate through `order` in the recursion.
        
        # B&B State:
        # idx: current index in `order`
        # colors: list of colors assigned to nodes in `order` so far (or just a dict/array of size m)
        # used_colors: set of colors used so far? Or just track max color?
        # Actually, we want to minimize the number of colors.
        # Let's fix K (number of colors allowed) and see if we can color with K.
        # If yes, try K-1.
        
        # But B&B for "is there a K-coloring?" is exponential.
        # We can use the lower bound from greedy MIS to prune?
        # If we have a set of uncolored nodes, and we know they form a clique, we need that many new colors.
        
        # Let's try a simpler, robust B&B:
        # 1. Start with K = best_num_colors - 1.
        # 2. Try to find a valid coloring with K colors.
        # 3. If found, decrement K and repeat.
        # 4. If K=0 and no coloring found, return best_num_colors.
        
        # To speed up "try K colors":
        # Use backtracking with forward checking.
        
        def try_coloring(K):
            # colors[i] = color of node i (0..K-1) or -1
            colors = [-1] * m
            # available_colors[i] = set of available colors for node i
            # Initially all 0..K-1
            # But we can prune: if a node has no available colors, fail.
            
            # We process nodes in `order`.
            # To make it faster, we can use a recursive function that takes the next node to color.
            # We need to track which colors are "blocked" for future nodes?
            # Forward checking: when we assign color c to node v, for each neighbor u of v (uncolored),
            # we remove c from available_colors[u]. If available_colors[u] becomes empty, fail.
            
            # This is expensive to maintain if we copy sets.
            # Alternative: simple backtracking with constraint checking at assignment time.
            
            # Let's use a more efficient B&B:
            # Branch on the node with the smallest number of available colors (MRV heuristic) among uncolored nodes.
            
            # This requires dynamic calculation of available colors.
            # m is small (<= 100).
            
            def backtrack(uncolored, color_counts):
                if not uncolored:
                    return True
                
                # MRV: pick uncolored node with fewest available colors
                best_node = None
                min_avail = K + 1
                best_avail_colors = None
                
                for u in uncolored:
                    avail = []
                    for c in range(K):
                        # Check if color c is available for u
                        # i.e., no neighbor of u has color c
                        # We can precompute neighbors
                        ok = True
                        for v in conflict_sets[u]:
                            if colors[v] == c:
                                ok = False
                                break
                        if ok:
                            avail.append(c)
                    
                    if len(avail) < min_avail:
                        min_avail = len(avail)
                        best_node = u
                        best_avail_colors = avail
                        if min_avail == 0:
                            return False
                        if min_avail == 1:
                            # Can't get better than 1
                            pass # Still check others? No, 0 is fail, 1 is best possible.
                
                if best_node is None:
                    return True # Should not happen if uncolored not empty
                
                # Try each available color for best_node
                for c in best_avail_colors:
                    colors[best_node] = c
                    # Recurse
                    new_uncolored = set(uncolored)
                    new_uncolored.discard(best_node)
                    if backtrack(new_uncolored, color_counts):
                        return True
                    colors[best_node] = -1
                return False
            
            # Initialize
            # We need to be careful with performance. m=100, K=20.
            # MRV loop is O(m * K * deg). 100 * 20 * 20 = 40k ops per node.
            # Depth 100. 4M ops per leaf. Too slow if branching factor is high.
            
            # Optimization: Use bitmasks for colors? K <= 20, fits in int.
            # But checking neighbors still requires iterating.
            
            # Let's try a different pruning:
            # If the uncolored subgraph has a clique of size > K - (max color used so far)?
            # Too complex.
            
            # Given the time limit, let's rely on the fact that the DSATUR greedy is very good.
            # The gap between greedy and optimal is usually small (1 or 2).
            # So we only need to test K = best, best-1, best-2.
            # If we can't find best-1 quickly, we give up.
            
            # To make `backtrack` faster, we can use a simple heuristic ordering first.
            # Let's just run it and see.
            
            # Actually, a standard DSATUR backtracking is quite powerful.
            # Let's implement DSATUR backtracking specifically for testing if K colors suffice.
            
            # State:
            # colors: array of size m, -1 if uncolored
            # uncolored: set
            # For each uncolored node, we compute saturation (number of different colors in neighbors)
            # and degree (number of uncolored neighbors).
            # Pick node with max (saturation, degree).
            
            def backtrack_dsatur(uncolored):
                if not uncolored:
                    return True
                
                best_i = None
                best_sat = -1
                best_deg = -1
                
                for i in uncolored:
                    neighbor_colors = set()
                    deg = 0
                    for j in conflict_sets[i]:
                        if colors[j] != -1:
                            neighbor_colors.add(colors[j])
                        else:
                            deg += 1
                    sat = len(neighbor_colors)
                    if sat > best_sat or (sat == best_sat and deg > best_deg):
                        best_sat = sat
                        best_deg = deg
                        best_i = i
                
                i = best_i
                taken = set()
                for j in conflict_sets[i]:
                    if colors[j] != -1:
                        taken.add(colors[j])
                
                # Try colors 0..K-1 not in taken
                # Order of trying colors? Smallest available first?
                for c in range(K):
                    if c not in taken:
                        colors[i] = c
                        if backtrack_dsatur(uncolored - {i}):
                            return True
                        colors[i] = -1
                return False
            
            # Start
            colors = [-1] * m
            uncolored = set(range(m))
            return backtrack_dsatur(uncolored)

        # Try to improve
        # We have best_num_colors from greedy.
        # Let's try K = best_num_colors - 1, - 2, etc.
        # We must be careful with time.
        
        current_best = best_num_colors
        # Limit the number of K values we try to avoid timeout
        # e.g. try up to 3 values below current best
        for k in range(current_best - 1, 0, -1):
            if time.time() > time_limit:
                break
            
            # Reset colors array for the attempt
            # Note: try_coloring modifies `colors` internally if we pass it, but here it creates a new one?
            # No, my `try_coloring` above uses a closure `colors` which is local to it?
            # Wait, in the code above, `colors` was defined inside `try_coloring`?
            # No, I defined `colors = [-1]*m` inside `try_coloring` in the draft?
            # Let's check the draft structure.
            # Ah, I put `colors = [-1] * m` inside `try_coloring` in the thought process, but in the code block I need to be precise.
            
            # Let's refine the `try_coloring` function to be self-contained and fast.
            # The MRV version was too slow. DSATUR backtracking is better.
            
            # To prevent infinite loops or extreme slowness, we add a node limit to the B&B.
            # If we visit too many nodes, we abort.
            
            nodes_visited = 0
            max_nodes = 500000 # Safety limit per K
            
            def limited_backtrack_dsatur(uncolored):
                nonlocal nodes_visited
                if nodes_visited > max_nodes:
                    return False # Abort
                nodes_visited += 1
                if time.time() > time_limit:
                    return False # Abort
                
                if not uncolored:
                    return True
                
                best_i = None
                best_sat = -1
                best_deg = -1
                
                for i in uncolored:
                    neighbor_colors = set()
                    deg = 0
                    for j in conflict_sets[i]:
                        if colors[j] != -1:
                            neighbor_colors.add(colors[j])
                        else:
                            deg += 1
                    sat = len(neighbor_colors)
                    if sat > best_sat or (sat == best_sat and deg > best_deg):
                        best_sat = sat
                        best_deg = deg
                        best_i = i
                
                i = best_i
                taken = set()
                for j in conflict_sets[i]:
                    if colors[j] != -1:
                        taken.add(colors[j])
                
                for c in range(k):
                    if c not in taken:
                        colors[i] = c
                        if limited_backtrack_dsatur(uncolored - {i}):
                            return True
                        colors[i] = -1
                return False

            colors = [-1] * m
            uncolored = set(range(m))
            if limited_backtrack_dsatur(uncolored):
                current_best = k
                best_colors = colors[:]
                # If we found a solution with k, we can try k-1
                # But we must reset the time check or continue loop
                continue
            else:
                # If we aborted due to time/node limit, we might have missed a solution.
                # But if we finished the search (nodes_visited <= max_nodes and not time out), then k is not possible.
                # Distinguishing abort from true failure is hard.
                # If we aborted, we should probably stop trying lower k as well, or just assume k is not possible?
                # Actually, if we aborted, we don't know.
                # Let's check if we hit the limit.
                if time.time() > time_limit or nodes_visited >= max_nodes:
                    # We aborted, so we can't be sure.
                    # However, if we found a solution for current_best, that's valid.
                    # If we didn't, we keep the previous best.
                    break

        # Verify best_colors
        # The `colors` array from the last successful `limited_backtrack_dsatur` is in `best_colors` if we updated it.
        # If we didn't update it, we use `init_colors`.
        
        if current_best < best_num_colors:
            return best_colors
        else:
            return init_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        remaining = deadline - time.time()
        if remaining <= 0:
            edges = instance(n, d, seed)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colors = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colors.items() if set(f) & near}
                c = 0
                while c in taken:
                    c += 1
                colors[e] = c
            results.append([colors[e] for e in edges])
            continue
        time_limit = max(1.0, remaining - 1.0)
        result = solve_instance(n, d, seed, time_limit)
        results.append(result)
    return results
