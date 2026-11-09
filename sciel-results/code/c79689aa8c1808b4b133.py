import time
import random
import sys

def construct():
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    sys.setrecursionlimit(10000)

    def is_valid(subset):
        reachable = 1
        for x in subset:
            new_reachable = reachable | (reachable << x)
            if (new_reachable >> 200) & 1:
                return False
            reachable = new_reachable
        return True

    def greedy(order, existing=None):
        if existing is None:
            existing = set()
        reachable = 1
        selected = set(existing)
        for x in order:
            if x in selected:
                continue
            new_reachable = reachable | (reachable << x)
            if not ((new_reachable >> 200) & 1):
                selected.add(x)
                reachable = new_reachable
        return selected

    def local_search(best, deadline):
        """Improve via add, swap, and 2-opt moves"""
        best = set(best)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding elements
            for x in range(1, 101):
                if x in best:
                    continue
                candidate = best | {x}
                if is_valid(candidate):
                    best = candidate
                    improved = True
                    break
            if time.time() >= deadline:
                break
            # Try 1-swap: remove one, add one
            for remove_x in sorted(best):
                if time.time() >= deadline:
                    break
                remaining = best - {remove_x}
                for add_x in range(1, 101):
                    if add_x in remaining:
                        continue
                    candidate = remaining | {add_x}
                    if is_valid(candidate):
                        best = candidate
                        improved = True
                        break
                if improved:
                    break
            if time.time() >= deadline:
                break
            # Try 2-opt: remove two, add one or two
            best_list = sorted(best)
            n = len(best_list)
            for i in range(n):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() >= deadline:
                        break
                    remaining = set(best_list[:i] + best_list[i+1:j] + best_list[j+1:])
                    # Try adding one element
                    for add_x in range(1, 101):
                        if add_x in remaining:
                            continue
                        candidate = remaining | {add_x}
                        if is_valid(candidate):
                            best = candidate
                            improved = True
                            break
                    if improved:
                        break
                    # Try adding two elements
                    for add_x in range(1, 101):
                        if add_x in remaining:
                            continue
                        for add_y in range(add_x + 1, 101):
                            if add_y in remaining:
                                continue
                            candidate = remaining | {add_x, add_y}
                            if is_valid(candidate):
                                best = candidate
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        return best

    def build_conflict_graph():
        """
        Build a conflict graph where vertices are 1-100.
        Edge (u,v) exists if u+v=200, or if there exists w in [1,100]
        such that u+v+w=200 (meaning u,v,w can't all be in a valid set).
        """
        n = 100
        conflict = [[False] * (n + 1) for _ in range(n + 1)]
        
        # Direct pairs: u + v = 200
        for u in range(1, n + 1):
            v = 200 - u
            if 1 <= v <= n and v != u:
                conflict[u][v] = True
                conflict[v][u] = True
        
        # Pairs that can form 200 with one more element: u + v + w = 200
        for u in range(1, n + 1):
            for v in range(u + 1, n + 1):
                w = 200 - u - v
                if 1 <= w <= n and w != u and w != v:
                    conflict[u][v] = True
                    conflict[v][u] = True
        
        return conflict

    def greedy_min_degree(conflict, deadline):
        """
        Greedy minimum-degree heuristic for maximum independent set.
        """
        remaining = set(range(1, 101))
        selected = set()
        
        while remaining and time.time() < deadline:
            # Find vertex with minimum degree in remaining
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = sum(1 for u in remaining if u != v and conflict[v][u])
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            
            if best_v is None:
                break
            
            selected.add(best_v)
            neighbors = {u for u in remaining if conflict[best_v][u]}
            remaining.discard(best_v)
            remaining -= neighbors
        
        return selected

    def repair(subset):
        """Removes elements from subset until it is valid."""
        subset = set(subset)
        while not is_valid(subset):
            # Find a bad subset using DP
            reachable = [False] * 201
            parent = [None] * 201
            reachable[0] = True
            for x in subset:
                for s in range(200 - x, -1, -1):
                    if reachable[s] and not reachable[s + x]:
                        reachable[s + x] = True
                        parent[s + x] = (s, x)
            
            if not reachable[200]:
                break
            
            # Backtrack to find bad subset
            s = 200
            bad = set()
            while parent[s] is not None:
                prev_s, x = parent[s]
                bad.add(x)
                s = prev_s
            
            # Remove the largest element in the bad subset
            x = max(bad)
            subset = subset - {x}
        
        return subset

    # Generate diverse initial solutions
    random.seed(42)
    candidates = []

    # Strategy 1: Greedy from top
    candidates.append(greedy(range(100, 0, -1)))

    # Strategy 2: Greedy from bottom
    candidates.append(greedy(range(1, 101)))

    # Strategy 3: Greedy from middle out
    mid_order = list(range(50, 0, -1)) + list(range(51, 101))
    candidates.append(greedy(mid_order))

    # Strategy 4: Greedy prioritizing large numbers first
    candidates.append(greedy(sorted(range(1, 101), key=lambda x: -x)))

    # Strategy 5: Modular construction (Mod 3, non-zero)
    mod3_set = set(x for x in range(1, 101) if x % 3 != 0)
    candidates.append(repair(mod3_set))

    # Strategy 6: Modular construction (Mod 4, 1,2)
    mod4_set = set(x for x in range(1, 101) if x % 4 in [1, 2])
    candidates.append(repair(mod4_set))

    # Strategy 7-16: Random orderings
    for i in range(10):
        order = list(range(1, 101))
        random.shuffle(order)
        candidates.append(greedy(order))

    # Strategy 17: Graph-based minimum degree heuristic
    if time.time() < DEADLINE - 10.0:
        conflict = build_conflict_graph()
        graph_solution = greedy_min_degree(conflict, DEADLINE - 8.0)
        if is_valid(graph_solution):
            candidates.append(graph_solution)
        else:
            candidates.append(repair(graph_solution))

    # Pick best initial
    best = max(candidates, key=len)
    best_size = len(best)

    # Local search to improve
    best = local_search(best, DEADLINE - 2.0)

    # Try multiple local search restarts from different initial solutions
    if time.time() < DEADLINE - 5.0:
        for cand in sorted(candidates, key=len, reverse=True)[:8]:
            if time.time() >= DEADLINE - 3.0:
                break
            if len(cand) < best_size:
                continue
            improved = local_search(set(cand), DEADLINE - 2.0)
            if len(improved) > len(best):
                best = improved
                best_size = len(best)

    # Final attempt: try to add any remaining elements
    for x in range(1, 101):
        if time.time() >= DEADLINE - 1.0:
            break
        if x in best:
            continue
        candidate = best | {x}
        if is_valid(candidate):
            best = candidate

    # Second pass: try adding elements again after first pass
    for x in range(1, 101):
        if time.time() >= DEADLINE - 0.5:
            break
        if x in best:
            continue
        candidate = best | {x}
        if is_valid(candidate):
            best = candidate

    # Final verification
    assert len(best) > 0
    assert all(1 <= x <= 100 for x in best)
    assert len(best) == len(set(best))

    # Verify no subset sums to 200
    reachable = 1
    for x in sorted(best):
        new_reachable = reachable | (reachable << x)
        if (new_reachable >> 200) & 1:
            return list(best)
        reachable = new_reachable

    return list(best)
