import time
import random
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

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
        # Two edges conflict if they share a vertex or are connected by an edge (induced matching condition)
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

        # 1. Fast Greedy Initial Solution (DSATUR-like)
        # This ensures we always have a valid answer to return if search fails or times out.
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

        # 2. Backtracking Search for Exact Minimum Coloring
        # We try to find a solution with K colors, starting from best_num_colors - 1 downwards.
        
        # Precompute static ordering based on degree in conflict graph
        # Higher degree vertices are more constrained, so we should assign them first?
        # Actually, for backtracking, we often want to pick the variable with the smallest domain first (dynamic).
        # But a static good order helps. Let's sort by degree descending for the initial stack.
        # However, DSATUR is dynamic. Let's implement a recursive backtracking with dynamic variable selection.
        
        def try_backtracking(K, time_limit):
            # Returns a valid coloring with K colors or None
            # We use a recursive backtracking with forward checking.
            
            # To make it efficient, we represent the state as:
            # colors: list of length m, -1 if uncolored
            # domains: list of sets of available colors for each vertex
            
            colors = [-1] * m
            # Initial domains: all colors 0..K-1
            domains = [set(range(K)) for _ in range(m)]
            
            # We need a way to track time and abort
            # Using a nonlocal variable or a class attribute for time check
            start_ts = time.time()
            
            # Recursive helper
            def backtrack():
                if time.time() - start_ts > time_limit:
                    return False
                
                # Find the uncolored vertex with the fewest available colors (MRV heuristic)
                # Tie-breaker: highest degree among uncolored neighbors (DSATUR-like)
                min_size = K + 1
                best_vertex = -1
                best_saturation = -1
                
                for i in range(m):
                    if colors[i] != -1:
                        continue
                    # Calculate current domain size
                    # We can maintain domains dynamically or recalculate.
                    # Recalculating is O(m*K) per step, which might be slow.
                    # Let's try maintaining domains.
                    # But for simplicity and correctness first, let's recalculate available colors.
                    # Available colors = {c in 0..K-1 | c not in colors of neighbors}
                    
                    used_by_neighbors = set()
                    for j in conflict_sets[i]:
                        if colors[j] != -1:
                            used_by_neighbors.add(colors[j])
                    
                    avail = [c for c in range(K) if c not in used_by_neighbors]
                    size = len(avail)
                    
                    if size == 0:
                        return False # Dead end
                    
                    # Heuristic: Minimize size, Maximize saturation (number of different colors used by neighbors)
                    # Saturation is effectively size of used_by_neighbors intersected with valid colors?
                    # No, saturation is usually the number of distinct colors in the neighborhood.
                    sat = len(used_by_neighbors)
                    
                    # Priority: Smallest size, then largest saturation
                    if size < min_size or (size == min_size and sat > best_saturation):
                        min_size = size
                        best_vertex = i
                        best_saturation = sat
                        
                if best_vertex == -1:
                    return True # All colored
                    
                i = best_vertex
                used_by_neighbors = set()
                for j in conflict_sets[i]:
                    if colors[j] != -1:
                        used_by_neighbors.add(colors[j])
                
                # Try assigning each available color
                for c in range(K):
                    if c not in used_by_neighbors:
                        colors[i] = c
                        
                        # Forward Check: Check if any neighbor becomes uncolorable
                        # We don't need to explicitly check domains if we recalculate in next step,
                        # but we can do a quick check for "naked singles" or contradictions.
                        # Simple forward check:
                        valid = True
                        for j in conflict_sets[i]:
                            if colors[j] != -1:
                                continue
                            # Check if j has any available colors
                            used_by_j = set()
                            for k in conflict_sets[j]:
                                if colors[k] != -1:
                                    used_by_j.add(colors[k])
                            if all(c2 in used_by_j for c2 in range(K)):
                                valid = False
                                break
                            
                        if valid:
                            if backtrack():
                                return True
                        colors[i] = -1
                        
                return False

            return backtrack()

        # Try to improve using Backtracking
        current_best = best_num_colors
        # We try K = current_best - 1, then -2, etc.
        # Limit the number of attempts to avoid spending too much time on infeasible Ks
        max_attempts = 5
        
        for k in range(current_best - 1, 0, -1):
            if time.time() > deadline - 5.0:
                break
            remaining_time = deadline - 5.0 - time.time()
            if remaining_time <= 1.0:
                break
            
            # Allocate time for this specific K attempt
            # If K is close to current_best, it's likely feasible, give it more time?
            # Or give equal time?
            # Let's give a fixed slice, e.g., 10s per attempt, but not exceeding remaining.
            time_for_attempt = min(remaining_time, 10.0)
            
            result = try_backtracking(k, time_for_attempt)
            if result:
                current_best = k
                best_colors = colors[:]
                # Continue to try k-1
            else:
                # If it failed, it might be infeasible or timeout.
                # If timeout, we don't know.
                # If infeasible, we can stop.
                # Since we can't easily distinguish, and lower K is harder,
                # it's safer to stop if we used significant time.
                # If it returned quickly, it's likely infeasible.
                # Let's check how much time was used.
                # If it used > 80% of allocated time, assume timeout and stop.
                # Else assume infeasible and stop (since K-1 is harder).
                # Actually, if K is infeasible, K-1 is definitely infeasible.
                # So we can always break if it fails?
                # Wait, if it timed out, K might be feasible.
                # But if we break, we keep current_best.
                # If we continue, we might waste time.
                # Given the small size of K reduction, let's just break on failure.
                # The greedy solution is a good baseline.
                break

        return best_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        remaining = deadline - time.time()
        if remaining <= 0:
            # Fallback: simple greedy
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
        
        # Allocate time proportionally or equally?
        # The graphs grow in size. Let's give more time to larger graphs.
        # n=40 (m~80), n=60 (m~120), n=80 (m~160).
        # Let's just use the remaining time for the last instance, split for others?
        # Better: Give each instance a fixed slice, e.g., 40s, 40s, 40s?
        # Total 120s. Deadline is 140s.
        # Let's give 45s to each.
        time_limit = 45.0
        if len(results) == 2: # Last instance
             time_limit = remaining - 2.0
        
        result = solve_instance(n, d, seed, time_limit)
        results.append(result)
        
        # Check time
        if time.time() > deadline - 2.0:
             # If we are running out of time, ensure we return valid results for remaining
             # by using the fallback greedy for any unsolved instances.
             # But the loop handles this by checking remaining at the top.
             pass

    return results
