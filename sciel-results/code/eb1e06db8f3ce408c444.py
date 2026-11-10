# EVOLVE-BLOCK-START
import math
import time
import random
from fractions import Fraction

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Precompute conflict matrix: conflict[i][j] = True if lcm(i+2, j+2) <= 256
    N = 255  # numbers 2..256
    nums = list(range(2, 257))
    conflict = [[False] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if math.lcm(nums[i], nums[j]) <= 256:
                conflict[i][i + 1 if False else j] = True
                conflict[j][i] = True

    # Precompute: for each vertex, list of conflicting vertices
    conf_adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if conflict[i][j]:
                conf_adj[i].append(j)
                conf_adj[j].append(i)

    # Greedy: pick vertices in order of 1/value (smallest first = largest reciprocal)
    def greedy(order):
        selected = set()
        for v in order:
            if all(v not in conf_adj[u] for u in selected):
                selected.add(v)
        return selected

    # 2-opt local search: try removing one and adding another, or swapping
    def local_search(selected, time_limit=5.0):
        sel = set(selected)
        best_val = sum(1.0 / nums[v] for v in sel)
        t_start = time.time()
        improved = True
        while improved and time.time() - t_start < time_limit:
            improved = False
            # Try removing one element
            for v in list(sel):
                new_sel = sel - {v}
                # Try adding each non-conflicting vertex not in new_sel
                for u in range(N):
                    if u in new_sel:
                        continue
                    if any(conflict[u][w] for w in new_sel):
                        continue
                    new_val = best_val - 1.0 / nums[v] + 1.0 / nums[u]
                    if new_val > best_val + 1e-12:
                        sel = new_sel | {u}
                        best_val = new_val
                        improved = True
                        break
                if improved:
                    break
            # Try removing two and adding one or two (simplified: just single swaps above)
        return sel, best_val

    # Upper bound: fractional LP relaxation
    # For each prime p, let S_p = {v : p divides nums[v]}
    # Any independent set can contain at most 1 from each S_p
    # Also, any independent set has size <= N
    # Upper bound on reciprocal sum: solve a relaxation
    # Simpler bound: for each vertex v, the max reciprocal sum of an independent set
    # containing v is bounded by sum of 1/nums[u] for u not conflicting with v,
    # but that's loose. Use a better bound.
    
    # Better upper bound: partition vertices into groups where each group is a clique.
    # The max independent set picks at most 1 from each clique.
    # Upper bound = sum over cliques of max(1/nums[v] for v in clique)
    # We can use: for each prime p, the set of multiples of p in [2,256] is a clique.
    # But these cliques overlap, so we need a proper clique cover.
    # Simplest valid bound: for each vertex v, ub(v) = 1/nums[v] + sum of 1/nums[u] for u
    # that don't conflict with v, but this overcounts.
    
    # Use a greedy coloring-based bound: color the conflict graph, then
    # UB = sum over colors of max(1/nums[v] for v with that color)
    # Actually, for an independent set, it picks at most one vertex per color class
    # only if each color class is a clique. In a proper coloring, same-color vertices
    # are NOT adjacent, so that doesn't work.
    
    # Alternative: use the fact that for any vertex v, any independent set containing v
    # can only use vertices from V \ (N(v) ∪ {v}). So:
    # UB = max over v in I of [1/nums[v] + UB(V \ (N(v)∪{v}))]
    # This is recursive and expensive. Use a simpler bound.
    
    # Simple but useful bound: 
    # UB = sum of top-k reciprocals where k is the max independent set size
    # Max independent set size <= N / (min degree + 1) by Turan, but that's loose.
    # Actually, let's just use: UB = sum of all 1/nums[v] (trivial, always valid)
    # And a tighter one: for each prime p, at most 1 multiple of p can be in I.
    # So UB = sum over primes p of max(1/nums[v] : p|v, v in [2,256])
    # But this overcounts because a number can be in multiple prime groups.
    # However, it's still a valid upper bound if we take the max over all such groupings.
    
    # Let me use a different approach: B&B with a simple bound
    # UB for a partial solution = current_sum + sum of 1/nums[v] for v not yet decided
    #   and not conflicting with selected vertices
    # This is the "remaining potential" bound.
    
    def branch_and_bound(order, time_limit=20.0):
        """B&B on vertices in given order. At each step, decide include/exclude."""
        best_sel = set()
        best_val = 0.0
        t_start = time.time()
        
        # Sort by reciprocal value descending (try high-value vertices first)
        # order is list of vertex indices
        selected = set()
        
        def _bb(idx, cur_sum, cur_sel):
            nonlocal best_sel, best_val
            if time.time() - t_start > time_limit:
                return
            if idx >= len(order):
                if cur_sum > best_val:
                    best_val = cur_sum
                    best_sel = set(cur_sel)
                return
            
            # Upper bound: current sum + sum of reciprocals of all remaining non-conflicting
            # This is expensive to compute each time. Use a precomputed suffix sum.
            # Simpler: just bound by cur_sum + suffix_sum[idx]
            # where suffix_sum[i] = sum of 1/nums[order[j]] for j>=i
            # But we need to exclude those conflicting with cur_sel.
            # For speed, use the loose bound and prune only when clearly worse.
            
            # Tighter bound: for each remaining vertex, check if it conflicts with cur_sel
            # If not, add its reciprocal. This is O(N) per node.
            ub = cur_sum
            for j in range(idx, len(order)):
                v = order[j]
                if v in cur_sel:
                    continue
                if any(conflict[v][u] for u in cur_sel):
                    continue
                ub += 1.0 / nums[v]
            
            if ub <= best_val + 1e-12:
                return
            
            v = order[idx]
            
            # Try including v
            if not any(conflict[v][u] for u in cur_sel):
                _bb(idx + 1, cur_sum + 1.0 / nums[v], cur_sel | {v})
            
            # Try excluding v
            _bb(idx + 1, cur_sum, cur_sel)
        
        # Use a subset of vertices for B&B to keep it tractable
        # Take top 60 by reciprocal value
        top_verts = sorted(range(N), key=lambda v: -1.0/nums[v])[:60]
        _bb(0, 0.0, set())
        
        return best_sel, best_val

    # Multi-start: different greedy orderings
    best_result = set()
    best_score = 0.0

    # Start 1: natural order (2,3,4,...,256)
    sel1 = greedy(list(range(N)))
    sel1, val1 = local_search(sel1, time_limit=8.0)
    if val1 > best_score:
        best_score = val1
        best_result = sel1

    # Start 2: reverse order (256, 255, ..., 2)
    sel2 = greedy(list(range(N - 1, -1, -1)))
    sel2, val2 = local_search(sel2, time_limit=8.0)
    if val2 > best_score:
        best_score = val2
        best_result = sel2

    # Start 3: random orderings
    rng = random.Random(42)
    for _ in range(5):
        if time.time() - t0 > DEADLINE - 30:
            break
        order = list(range(N))
        rng.shuffle(order)
        sel = greedy(order)
        sel, val = local_search(sel, time_limit=5.0)
        if val > best_score:
            best_score = val
            best_result = sel

    # B&B on top candidates
    if time.time() - t0 < DEADLINE - 25:
        # B&B on top 50 vertices
        top50 = sorted(range(N), key=lambda v: -1.0/nums[v])[:50]
        bb_sel, bb_val = branch_and_bound(top50, time_limit=15.0)
        if bb_val > best_score:
            best_score = bb_val
            best_result = bb_sel

    # Final local search on best result
    if time.time() - t0 < DEADLINE - 5:
        best_result, best_score = local_search(best_result, time_limit=10.0)

    # Convert to list of actual numbers
    result = sorted(nums[v] for v in best_result)
    return result if result else [2]
# EVOLVE-BLOCK-END
