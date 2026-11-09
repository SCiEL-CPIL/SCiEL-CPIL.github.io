"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys

def construct():
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

    def solve_instance(edges, n, time_limit=35):
        E = len(edges)
        if E == 0:
            return []
        
        # Build adjacency sets
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Precompute conflict matrix
        # conflict[i][j] = True if edge i and j cannot share a color
        conflict = [[False] * E for _ in range(E)]
        conflict_count = [0] * E # Degree in conflict graph
        
        for i in range(E):
            u1, v1 = edges[i]
            for j in range(i + 1, E):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i][j] = True
                    conflict[j][i] = True
                    conflict_count[i] += 1
                    conflict_count[j] += 1
                else:
                    if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                        conflict[i][j] = True
                        conflict[j][i] = True
                        conflict_count[i] += 1
                        conflict_count[j] += 1

        # Greedy fallback to ensure we always have a valid solution
        def greedy_coloring():
            # Order by conflict degree descending
            order = list(range(E))
            order.sort(key=lambda x: conflict_count[x], reverse=True)
            colors = [-1] * E
            color_classes = []
            for idx in order:
                placed = False
                for c in range(len(color_classes)):
                    ok = True
                    for j in color_classes[c]:
                        if conflict[idx][j]:
                            ok = False
                            break
                    if ok:
                        color_classes[c].append(idx)
                        colors[idx] = c
                        placed = True
                        break
                if not placed:
                    color_classes.append([idx])
                    colors[idx] = len(color_classes) - 1
            
            # Compact
            final_colors = [0] * E
            mapping = {}
            next_c = 0
            for c in range(len(color_classes)):
                if color_classes[c]:
                    mapping[c] = next_c
                    next_c += 1
            for i in range(E):
                final_colors[i] = mapping[colors[i]]
            return final_colors

        best_valid_colors = greedy_coloring()
        best_k = len(set(best_valid_colors))
        
        # Simulated Annealing to find smaller k
        # We try k from a lower bound up to best_k - 1
        # Lower bound: max conflict degree + 1 is a weak bound. 
        # Let's start from max(conflict_count) + 1? Or maybe d+1?
        # For induced matching, the clique number of the conflict graph is a better LB.
        # But calculating clique number is hard.
        # Let's just try values.
        
        start_time = time.time()
        
        # Determine search range
        # The conjecture says 5d^2/4. 
        # We found a valid coloring with best_k.
        # We try to see if we can do better.
        
        # To be efficient, we can try a few random restarts of SA for a fixed k.
        # If we find a valid coloring for k, we update best_k and best_valid_colors.
        
        # We will iterate k from LB to best_k - 1.
        # LB estimation:
        # A simple LB is the maximum size of an induced matching? No.
        # The conflict graph must be colored.
        # Let's use LB = max(conflict_count) + 1? No, that's for standard coloring.
        # Actually, just start from 1? No, too slow.
        # Let's use LB = int(0.5 * best_k) or something?
        # Let's try to be aggressive.
        
        # We have time_limit.
        # Let's allocate time per k attempt.
        
        # Heuristic LB:
        # The graph H (conflict graph) has max degree max_conf_deg.
        # LB = max_conf_deg + 1 is valid for standard coloring, but here we are coloring H.
        # So LB = max(conflict_count) + 1 is a valid lower bound for the number of colors?
        # Yes, if a vertex in H has degree D, it needs D+1 distinct colors from its neighbors?
        # No, in H, adjacent vertices must have different colors.
        # So if a vertex in H has degree D, it conflicts with D others.
        # It doesn't force D+1 colors globally, but it's a decent heuristic start.
        
        max_conf_deg = max(conflict_count) if conflict_count else 0
        LB = max_conf_deg + 1
        
        # If LB is very low, we might waste time.
        # Let's cap the number of SA attempts.
        
        # We try k from LB to best_k - 1.
        # If time is tight, we prioritize smaller k (better score).
        
        # To avoid spending too much time on a k that is impossible,
        # we run SA for a short duration.
        
        # Total time budget for SA is time_limit - 5 (reserve for finalization)
        sa_budget = time_limit - 5
        if sa_budget < 1:
            return best_valid_colors
            
        # Number of k values to try
        num_k_to_try = best_k - LB + 1
        if num_k_to_try <= 0:
            return best_valid_colors
            
        time_per_k = sa_budget / num_k_to_try
        
        # If time_per_k is very small, we might not be able to try many k's.
        # Let's limit the number of k's we actually try to avoid overhead.
        # If best_k is 20 and LB is 5, we have 16 k's.
        
        for k in range(LB, best_k):
            if time.time() - start_time > sa_budget:
                break
                
            # Run SA for k colors
            # Time allowed for this k
            t_limit = time_per_k
            # If we have lots of time, give more to smaller k?
            # Let's just use uniform time for now.
            
            # SA Parameters
            T_start = 10.0
            T_end = 0.01
            # We want to minimize conflicts.
            
            # Initialize colors randomly
            colors = [random.randrange(k) for _ in range(E)]
            
            # Calculate initial conflicts
            def count_conflicts(col):
                cnt = 0
                # Optimize: only check pairs in same color?
                # Or just iterate all pairs? E is small (~200). E^2 = 40000. Fast.
                # But we can do better.
                # For each color c, let S_c be set of edges with color c.
                # For each pair in S_c, check conflict.
                # This is sum |S_c|^2. Worst case k=1, E^2.
                # Average case E^2/k.
                # Let's just do the full E^2 check for simplicity and correctness, 
                # but optimized with precomputed conflict matrix.
                
                # Actually, we can maintain a count.
                # But for SA, we need delta.
                # Let's just recalculate from scratch if E is small enough?
                # E=200, E^2=40000. 40000 ops is fast.
                # If we do 10000 iterations, 400M ops. Might be slow in Python.
                
                # Better: Incremental update.
                # When changing color of edge i from c_old to c_new:
                # 1. Remove i from c_old.
                #    Conflicts removed: sum(conflict[i][j] for j in S_c_old if j != i)
                # 2. Add i to c_new.
                #    Conflicts added: sum(conflict[i][j] for j in S_c_new)
                # Delta = Added - Removed.
                
                return cnt
            
            # Let's implement incremental SA.
            
            # State: colors list, and sets of edges per color.
            color_sets = [set() for _ in range(k)]
            for i in range(E):
                color_sets[colors[i]].add(i)
                
            # Initial conflict count
            current_conflicts = 0
            for i in range(E):
                c = colors[i]
                for j in color_sets[c]:
                    if j > i and conflict[i][j]:
                        current_conflicts += 1
                        
            best_conflicts = current_conflicts
            best_colors = colors[:]
            
            if current_conflicts == 0:
                # Found valid coloring immediately
                # Compact and update best
                # This k is valid
                compacted = best_colors[:] 
                # We need to map 0..k-1 to 0..num_used-1
                used_colors = set(compacted)
                mapping = {c: idx for idx, c in enumerate(sorted(used_colors))}
                final_compacted = [mapping[c] for c in compacted]
                if len(used_colors) < best_k:
                    best_k = len(used_colors)
                    best_valid_colors = final_compacted
                # Since we are iterating k upwards, and we found a valid one,
                # we can break? 
                # Wait, we are looking for the MINIMUM k.
                # If we found a valid coloring for k, we should record it.
                # But maybe k-1 was also valid? We tried k-1 in the previous iteration.
                # So if we are at k, and we succeed, it means k-1 failed (or we didn't try it? No, we iterate upwards).
                # So this is the smallest k found so far.
                # We can break the k loop?
                # Yes, because we are increasing k. If k works, any k' > k also works (by just not using the extra colors? No, SA uses exactly k colors? No, it uses colors from 0..k-1. It might use fewer. But the "capacity" is k).
                # Actually, if we find a valid coloring with max color < k, it's also a valid coloring for k-1?
                # No. If the coloring uses colors {0, 1}, it's valid for k=2.
                # If we are testing k=5, and we find a coloring using colors {0,1}, it is valid for k=2.
                # So if we find a valid coloring for k, we should check the actual number of colors used.
                # Let num_used = len(set(colors)).
                # If num_used < best_k, update.
                # And we can stop searching for larger k?
                # No, we want the MINIMUM number of colors.
                # If we found a valid coloring with num_used colors, then num_used is a candidate for the answer.
                # Since we are iterating k from LB upwards, the first valid coloring we find might use fewer colors than k.
                # But we should continue to see if we can find a coloring with even fewer colors?
                # No, if we found a valid coloring using `num_used` colors, then the chromatic number is <= num_used.
                # We are currently at k. If num_used < k, it means the previous k's (LB..k-1) failed to find a valid coloring?
                # Not necessarily. They might have failed due to SA limitations.
                # So we should continue?
                # But if we found a valid coloring, we have a valid solution.
                # The score is the number of colors used.
                # So if we have a valid solution with 10 colors, and we are testing k=15, and we find a solution with 12 colors, we update best to 10? No, 10 is better.
                # So we should keep the best valid solution found so far.
                # We should break the k loop if we find a valid solution?
                # No, because we might find a better one (fewer colors) with a larger k?
                # Impossible. If a coloring uses C colors, it is valid for any k >= C.
                # So if we find a valid coloring for k, the number of colors used is <= k.
                # We want to minimize this number.
                # So if we find a valid coloring for k=10 using 8 colors, best_k becomes 8.
                # Then we try k=9? No, we are iterating k upwards.
                # We started at LB.
                # If LB=5, we try k=5, 6, 7...
                # If k=5 fails, k=6 fails... k=10 succeeds with 8 colors.
                # Best is 8.
                # Should we try k=11? It might succeed with 7 colors?
                # No, if a coloring uses 7 colors, it is valid for k=7.
                # So if k=7 failed, it means SA didn't find a 7-coloring.
                # But a 7-coloring might exist.
                # So finding a valid coloring for k=10 doesn't prove that 8 is the minimum.
                # It just gives us a candidate.
                # We should continue trying larger k?
                # No, that's counter-intuitive. We want smaller k.
                # The fact that k=10 worked suggests the answer is <= 10.
                # We should go back and try k=9, 8, 7... with more time?
                # Or just accept the best found?
                # Given time constraints, let's just keep the best valid solution found.
                # And break if we find a valid solution?
                # If we break, we might miss a better solution.
                # But running SA for larger k is unlikely to yield a coloring with FEWER colors than the current best, because the space of k-colorings includes (k-1)-colorings (by ignoring one color).
                # Wait, if we allow k colors, we can always choose to use only k-1 of them.
                # So the set of valid k-colorings is a superset of valid (k-1)-colorings.
                # So if a (k-1)-coloring exists, a k-coloring exists.
                # So if we find a valid k-coloring, it doesn't help us find a smaller one.
                # We need to find a valid (k-1)-coloring.
                # So we should iterate k from LB upwards.
                # If we find a valid coloring for k, we record the number of colors used (which might be < k).
                # Then we can stop?
                # If we stop at the first valid k, we have a valid coloring.
                # The number of colors used might be less than k.
                # Is it possible that a valid coloring with fewer colors exists but we missed it?
                # Yes, if SA failed for smaller k.
                # But if we continue to larger k, we won't find a coloring with fewer colors than the one we just found?
                # Actually, we might.
                # Example: Best found so far is 10 colors (from k=12).
                # We try k=13. We might find a coloring using 9 colors?
                # Yes, because the search space is larger.
                # So we should continue?
                # But time is limited.
                # Let's just run SA for a fixed set of k values, or just one k?
                
                # Alternative Strategy:
                # 1. Get greedy solution with K_g colors.
                # 2. Try to improve it.
                # 3. Use SA to minimize the number of colors directly?
                #    Objective: Minimize max color index + 1?
                #    This is hard to define in SA.
                
                # Let's stick to the "Try k from LB to Best_K-1" approach.
                # If we find a valid coloring for k, we update best.
                # We do NOT break, because we might find a better one?
                # No, if we find a valid coloring for k, the number of colors used is C.
                # We want to minimize C.
                # If we continue to k+1, we might find a coloring with C' < C.
                # This is possible.
                # So we should continue.
                
                pass # Logic for updating best is below

            # SA Loop
            T = T_start
            cooling = (T_start - T_end) / max(1, int(time_per_k * 1000)) # Rough estimate of steps
            # Better: determine steps based on time.
            # Let's just run until time_per_k is exceeded.
            
            sa_start = time.time()
            while time.time() - sa_start < time_per_k:
                # Pick random edge
                i = random.randrange(E)
                c_old = colors[i]
                
                # Pick random new color
                c_new = random.randrange(k)
                
                if c_old == c_new:
                    continue
                    
                # Calculate delta
                # Remove i from c_old
                # Conflicts with c_old neighbors:
                removed_conf = 0
                for j in color_sets[c_old]:
                    if j != i and conflict[i][j]:
                        removed_conf += 1
                        
                # Add i to c_new
                added_conf = 0
                for j in color_sets[c_new]:
                    if conflict[i][j]:
                        added_conf += 1
                        
                delta = added_conf - removed_conf
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    # Accept
                    color_sets[c_old].remove(i)
                    color_sets[c_new].add(i)
                    colors[i] = c_new
                    current_conflicts += delta
                    
                    if current_conflicts == 0:
                        # Valid coloring found
                        used_colors = set(colors)
                        num_used = len(used_colors)
                        if num_used < best_k:
                            best_k = num_used
                            # Compact colors
                            mapping = {c: idx for idx, c in enumerate(sorted(used_colors))}
                            best_valid_colors = [mapping[c] for c in colors]
                        # We found a valid coloring. 
                        # Should we break?
                        # If we break, we stop trying this k.
                        # We can move to next k.
                        break
                        
                # Cool down
                # Simple exponential cooling
                T *= 0.9999
                if T < T_end:
                    T = T_end

            # End of SA for this k

        # End of k loop
        
        return best_valid_colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        colors = solve_instance(edges, n, time_limit=35)
        out.append(colors)
    return out
# EVOLVE-BLOCK-END
