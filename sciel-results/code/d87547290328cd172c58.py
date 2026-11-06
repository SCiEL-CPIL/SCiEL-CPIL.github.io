import time
import random
import numpy as np

def construct():
    start = time.time()
    DEADLINE = 140.0

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def count_conflicts(S):
        """Return number of (a,b,c) triples with a+b=c in S."""
        Sset = set(S)
        cnt = 0
        for a in S:
            for b in S:
                if a + b in Sset:
                    cnt += 1
        return cnt

    def can_add(x, Sset, S):
        for a in S:
            if (x + a) in Sset or (x - a) in Sset:
                return False
        return True

    def repair_sum_free(S, time_limit, start_time):
        """Remove elements until S is sum-free, removing the most-conflicting element each time."""
        S = list(set(S))
        Sset = set(S)
        while not is_sum_free(S):
            if time.time() - start_time > time_limit:
                break
            n = len(S)
            if n == 0:
                break
            conflicts = [0] * n
            s_to_idx = {v: i for i, v in enumerate(S)}
            for i in range(n):
                a = S[i]
                for j in range(i, n):
                    b = S[j]
                    s = a + b
                    if s in Sset:
                        conflicts[i] += 1
                        if j != i:
                            conflicts[j] += 1
                        k = s_to_idx.get(s)
                        if k is not None:
                            conflicts[k] += 1
            if max(conflicts) == 0:
                break
            max_conf = max(conflicts)
            idx = conflicts.index(max_conf)
            S.pop(idx)
            Sset = set(S)
        return S

    def build_conflict_graph(V):
        """Build directed conflict graph: edge u->v means u+v in V (v is the sum).
        Returns adjacency list and in-degree counts."""
        Vset = set(V)
        n = len(V)
        idx_map = {v: i for i, v in enumerate(V)}
        out_adj = [[] for _ in range(n)]  # out_adj[i] = list of j where V[i]+V[j] in V
        in_deg = [0] * n
        for i in range(n):
            a = V[i]
            for j in range(n):
                s = a + V[j]
                if s in Vset:
                    k = idx_map[s]
                    out_adj[i].append(k)
                    in_deg[k] += 1
        return out_adj, in_deg

    def greedy_independent_set(V, out_adj, in_deg, time_limit, start_time):
        """Greedy: repeatedly pick node with lowest out-degree (fewest conflicts it creates),
        remove it and all its neighbors from consideration."""
        n = len(V)
        active = [True] * n
        out_adj_copy = [list(adj) for adj in out_adj]
        in_deg_copy = list(in_deg)
        selected = []
        
        while any(active):
            if time.time() - start_time > time_limit:
                break
            # Find active node with minimum out-degree among active neighbors
            best_i = -1
            best_deg = float('inf')
            for i in range(n):
                if not active[i]:
                    continue
                # Count active out-neighbors
                cnt = 0
                for j in out_adj_copy[i]:
                    if active[j]:
                        cnt += 1
                if cnt < best_deg:
                    best_deg = cnt
                    best_i = i
                if best_deg == 0:
                    break
            if best_i == -1:
                break
            selected.append(best_i)
            active[best_i] = False
            # Remove all neighbors (both directions) from active
            for j in out_adj_copy[best_i]:
                if active[j]:
                    active[j] = False
            # Also remove nodes that point TO best_i (they'd form a+b=c with c removed, but
            # actually if best_i is removed, edges pointing to it don't matter for sum-freeness
            # of remaining set. But edges FROM best_i matter. Let's also clear in-edges.)
            # Actually we need to be careful: if we remove node i, then for any node j where
            # V[j] + V[k] = V[i], the triple (j, k, i) is broken since i is gone.
            # But for node j where V[i] + V[j] = V[k], the triple (i, j, k) is broken since i is gone.
            # So removing i only breaks triples involving i. The remaining set is safe.
            # However, we should also remove j if V[j] + V[best_i] would create a conflict... 
            # No - we already removed best_i, so no conflict from best_i.
            # But we need to remove nodes that CANNOT coexist with selected nodes.
            # Let me reconsider: we want an independent set in the "conflict" sense.
            # Two nodes conflict if adding both creates a sum-free violation.
            # This is more complex. Let's use a simpler approach.
            pass
        
        return [V[i] for i in selected]

    def greedy_conflict_prune(V, time_limit, start_time):
        """Iteratively remove highest out-degree nodes, then greedily add back."""
        V = list(V)
        Vset = set(V)
        n = len(V)
        if n == 0:
            return []
        
        # Build conflict structure: for each element, how many pairs (a,b) in V give a+b = this element
        # and how many pairs give element + b = c in V
        out_count = [0] * n  # number of b such that V[i]+b in V
        in_count = [0] * n   # number of pairs (a,b) such that a+b = V[i]
        idx_map = {v: i for i, v in enumerate(V)}
        
        for i in range(n):
            a = V[i]
            for j in range(n):
                s = a + V[j]
                if s in Vset:
                    k = idx_map[s]
                    out_count[i] += 1
                    in_count[k] += 1
        
        # Iteratively remove node with highest total conflict degree
        active = [True] * n
        out_count_cur = list(out_count)
        in_count_cur = list(in_count)
        removed = []
        
        max_removals = min(n // 3, 50)  # Remove at most n/3 elements
        for _ in range(max_removals):
            if time.time() - start_time > time_limit:
                break
            # Find active node with highest out_count + in_count
            best_i = -1
            best_deg = -1
            for i in range(n):
                if not active[i]:
                    continue
                deg = out_count_cur[i] + in_count_cur[i]
                if deg > best_deg:
                    best_deg = deg
                    best_i = i
            if best_i == -1 or best_deg <= 0:
                break
            active[best_i] = False
            removed.append(best_i)
            # Update counts: removing V[best_i] affects counts of others
            a = V[best_i]
            for j in range(n):
                if not active[j]:
                    continue
                s = a + V[j]
                if s in Vset:
                    k = idx_map[s]
                    if active[k]:
                        out_count_cur[j] -= 1
                        in_count_cur[k] -= 1
                s2 = V[j] + a
                if s2 in Vset:
                    k = idx_map[s2]
                    if active[k]:
                        out_count_cur[j] -= 1
                        in_count_cur[k] -= 1
                # Also if V[j] + something = a, that's in_count of a which we don't care about
                # But we need to check: is there x,y active such that x+y = V[j]?
                # That doesn't change. We only need to update counts involving best_i.
        
        # Now greedily add back from removed + remaining active
        remaining = [V[i] for i in range(n) if active[i]]
        S = list(remaining)
        Sset = set(S)
        
        # Try to add back removed elements
        for i in removed:
            if time.time() - start_time > time_limit:
                break
            x = V[i]
            if can_add(x, Sset, S):
                S.append(x)
                Sset.add(x)
        
        # Now try to add any other elements not in S (shouldn't be any, but just in case)
        for x in V:
            if x not in Sset and can_add(x, Sset, S):
                S.append(x)
                Sset.add(x)
        
        return S

    def greedy_add(S_base, V, time_limit, start_time):
        """Greedily add elements from V \\ S_base to S_base, maintaining sum-freeness."""
        S = list(set(S_base))
        Sset = set(S)
        remaining = [x for x in V if x not in Sset]
        best_S = S
        best_size = len(S)
        orders = [
            lambda lst: sorted(lst, key=lambda x: abs(x), reverse=True),
            lambda lst: sorted(lst, key=lambda x: x),
            lambda lst: sorted(lst, key=lambda x: -x),
        ]
        for order_fn in orders:
            if time.time() - start_time > time_limit:
                break
            S_try = list(best_S) if len(best_S) >= len(S) else list(S)
            Sset_try = set(S_try)
            rem = [x for x in V if x not in Sset_try]
            rem = order_fn(rem)
            for x in rem:
                if time.time() - start_time > time_limit:
                    break
                if can_add(x, Sset_try, S_try):
                    S_try.append(x)
                    Sset_try.add(x)
            if len(S_try) > best_size:
                best_size = len(S_try)
                best_S = S_try
        return best_S

    def local_search(S_init, V, time_limit, start_time):
        """Local search with add, remove, and swap moves to maximize sum-free subset size."""
        S = list(set(S_init))
        Sset = set(S)
        Vset = set(V)
        best_S = list(S)
        best_size = len(S)
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # Try adding elements
            for x in V:
                if time.time() - start_time > time_limit:
                    break
                if x not in Sset and can_add(x, Sset, S):
                    S.append(x)
                    Sset.add(x)
                    improved = True
            
            # Try swap moves: remove one element, add another
            if not improved and len(S) > 0:
                for i in range(len(S)):
                    if time.time() - start_time > time_limit:
                        break
                    removed = S[i]
                    S_temp = S[:i] + S[i+1:]
                    Sset_temp = set(S_temp)
                    
                    for x in V:
                        if x in Sset_temp:
                            continue
                        if can_add(x, Sset_temp, S_temp):
                            # Swap: remove S[i], add x
                            S = S_temp + [x]
                            Sset = Sset_temp | {x}
                            improved = True
                            break
                    if improved:
                        break
            
            if len(S) > best_size:
                best_size = len(S)
                best_S = list(S)
                no_improve_count = 0
            elif improved:
                no_improve_count += 1
            else:
                no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                break
        
        return best_S

    def direct_odds(V):
        return [x for x in V if x % 2 != 0]

    def direct_upper_magnitude(V):
        if not V:
            return []
        max_abs = max(abs(x) for x in V)
        threshold = max_abs / 2.0
        return [x for x in V if abs(x) > threshold]

    def direct_upper_third(V):
        if not V:
            return []
        max_abs = max(abs(x) for x in V)
        threshold = max_abs * 2.0 / 3.0
        return [x for x in V if abs(x) > threshold]

    def direct_modular(V, M):
        if not V or M <= 0:
            return []
        min_v = min(V)
        lower = M // 3 + 1
        upper = (2 * M) // 3
        result = []
        for x in V:
            r = (x - min_v) % M
            if lower <= r <= upper:
                result.append(x)
        return result

    def direct_positive(V):
        return [x for x in V if x > 0]

    def direct_negative(V):
        return [x for x in V if x < 0]

    def solve_instance(V, time_limit, start_time):
        """Solve one instance, return the best sum-free subset."""
        candidates = [
            direct_odds(V),
            direct_upper_magnitude(V),
            direct_upper_third(V),
            direct_positive(V),
            direct_negative(V),
        ]
        max_abs = max(abs(x) for x in V) if V else 0
        for M in [max_abs, max_abs * 2, max(3, len(V)), max(3, len(V)) * 2]:
            if M > 0:
                candidates.append(direct_modular(V, M))

        best_set = set()
        best_size = 0

        # First try conflict-pruning approach on the full set
        if time.time() - start_time < time_limit - 5:
            pruned = greedy_conflict_prune(V, time_limit - 10, start_time)
            if is_sum_free(pruned):
                expanded = greedy_add(set(pruned), V, time_limit - (time.time() - start_time) * 0.2, start_time)
                if is_sum_free(expanded):
                    if len(expanded) > best_size:
                        best_size = len(expanded)
                        best_set = set(expanded)
                if len(pruned) > best_size:
                    best_size = len(pruned)
                    best_set = set(pruned)

        for cand in candidates:
            if time.time() - start_time > time_limit:
                break
            cand_set = set(cand)
            if not is_sum_free(cand_set):
                cand_list = repair_sum_free(list(cand_set), 1.0, time.time())
                cand_set = set(cand_list)
            if time.time() - start_time > time_limit - 0.5:
                if len(cand_set) > best_size:
                    best_size = len(cand_set)
                    best_set = cand_set
                continue
            expanded = greedy_add(cand_set, V, time_limit - (time.time() - start_time) * 0.3, start_time)
            if is_sum_free(expanded):
                if len(expanded) > best_size:
                    best_size = len(expanded)
                    best_set = set(expanded)

        # Apply local search to the best candidate found
        if time.time() - start_time < time_limit - 1:
            remaining_time = time_limit - (time.time() - start_time) - 0.5
            if remaining_time > 1:
                ls_result = local_search(list(best_set), V, remaining_time, start_time)
                if is_sum_free(ls_result) and len(ls_result) > best_size:
                    best_size = len(ls_result)
                    best_set = set(ls_result)

        if not is_sum_free(best_set):
            best_set = set(direct_odds(V))
            if not is_sum_free(best_set):
                best_set = set()
                for x in V:
                    if can_add(x, set(best_set), list(best_set)):
                        best_set.add(x)

        return sorted(best_set)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        elapsed = time.time() - start
        remaining = DEADLINE - elapsed
        if remaining < 10:
            results.append(direct_odds(V))
            continue
        # Allocate more time to larger instances
        if size == 240:
            alloc = remaining * 0.5
        elif size == 120:
            alloc = remaining * 0.3
        else:
            alloc = remaining * 0.2
        alloc = max(alloc, 2.0)
        results.append(solve_instance(V, alloc, start))

    return results
