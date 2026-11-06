import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118  # 32s safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if len(S) < 3:
            return True
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(S_set, candidate):
        """Check if candidate can be added to S_set to keep it sum-free."""
        if candidate in S_set:
            return False
        for x in S_set:
            if (candidate + x) in S_set:
                return False
            if (candidate - x) in S_set:
                return False
        return True

    def greedy_add(S_list, candidates, S_set=None):
        if S_set is None:
            S_set = set(S_list)
        result = list(S_list)
        for c in candidates:
            if c in S_set:
                continue
            if can_add(S_set, c):
                result.append(c)
                S_set.add(c)
        return result, S_set

    def local_search(S_list, values, time_limit=5.0):
        """Improve S_list by trying to add elements and perform swaps."""
        S_set = set(S_list)
        values_set = set(values)
        start = time.time()
        S_list = list(S_list)

        # Phase 1: Try adding any missing element
        missing = list(values_set - S_set)
        if missing:
            S_list, S_set = greedy_add(S_list, missing, S_set)

        # Phase 2: Iterative improvement - remove one element, greedily add all possible
        improved = True
        while improved and (time.time() - start) < time_limit:
            improved = False
            S_copy = list(S_list)
            S_set_copy = set(S_set)

            for x in S_copy:
                if (time.time() - start) >= time_limit:
                    break

                # Remove x
                S_test = S_set_copy - {x}
                S_test_list = [y for y in S_copy if y != x]

                # Try to add all missing elements greedily
                cand_list = [y for y in values if y not in S_test]
                S_new_list, S_new_set = greedy_add(S_test_list, cand_list, S_test)

                if len(S_new_list) > len(S_list):
                    S_list = S_new_list
                    S_set = S_new_set
                    improved = True
                    break

        # Phase 3: Try swap - remove one, add up to 2 specific elements
        if (time.time() - start) < time_limit:
            improved = True
            while improved and (time.time() - start) < time_limit:
                improved = False
                S_copy = list(S_list)
                S_set_copy = set(S_set)

                for x in S_copy:
                    if (time.time() - start) >= time_limit:
                        break

                    # Remove x
                    S_test = S_set_copy - {x}
                    S_test_list = [y for y in S_copy if y != x]

                    # Try adding pairs of missing elements
                    missing = list(values_set - S_test)
                    added = 0
                    for i, a in enumerate(missing):
                        if (time.time() - start) >= time_limit:
                            break
                        if a in S_test:
                            continue
                        if can_add(S_test, a):
                            S_test.add(a)
                            S_test_list.append(a)
                            added += 1
                            if added >= 2:
                                break

                    if len(S_test_list) > len(S_list):
                        S_list = S_test_list
                        S_set = S_test
                        improved = True
                        break

        # Phase 4: Final greedy add pass
        missing = list(values_set - S_set)
        if missing:
            S_list, S_set = greedy_add(S_list, missing, S_set)

        return S_list

    def build_conflict_graph(values):
        """Build adjacency list for conflict graph.
        Edge between i,j if values[i]+values[j] is in values, or 2*values[i]==values[j].
        A sum-free set is an independent set in this graph."""
        n = len(values)
        val_to_idx = {v: i for i, v in enumerate(values)}
        adj = [set() for _ in range(n)]
        
        for i in range(n):
            for j in range(i+1, n):
                vi, vj = values[i], values[j]
                # Check if vi + vj is in values
                s = vi + vj
                if s in val_to_idx:
                    k = val_to_idx[s]
                    adj[i].add(k)
                    adj[k].add(i)
                    adj[j].add(k)
                    adj[k].add(j)
                # Check if 2*vi == vj or 2*vj == vi
                if 2 * vi == vj:
                    adj[i].add(j)
                    adj[j].add(i)
                elif 2 * vj == vi:
                    adj[i].add(j)
                    adj[j].add(i)
        
        return adj

    def greedy_independent_set(adj, order):
        """Greedy: add vertices in given order if no neighbor already added."""
        n = len(adj)
        in_set = [False] * n
        result = []
        for v in order:
            if in_set[v]:
                continue
            # Check if any neighbor is already in set
            conflict = False
            for nb in adj[v]:
                if in_set[nb]:
                    conflict = True
                    break
            if not conflict:
                in_set[v] = True
                result.append(v)
        return result

    def spectral_partition(adj, values):
        """Use spectral bisection to find a large independent set.
        Compute Laplacian, get Fiedler vector, partition by sign."""
        n = len(adj)
        if n == 0:
            return []
        
        # Build degree matrix and Laplacian
        deg = [len(adj[i]) for i in range(n)]
        # L = D - A
        L = np.zeros((n, n))
        for i in range(n):
            L[i, i] = deg[i]
            for j in adj[i]:
                L[i, j] = -1
        
        # Get eigenvalues/eigenvectors
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            # Fiedler vector is eigenvector of second smallest eigenvalue
            fiedler = eigvecs[:, 1]
        except:
            return []
        
        # Partition: positive side vs negative side
        pos_indices = [i for i in range(n) if fiedler[i] >= 0]
        neg_indices = [i for i in range(n) if fiedler[i] < 0]
        
        # Greedy independent set on each side
        best = []
        for side in [pos_indices, neg_indices]:
            # Order by degree (low degree first)
            side_sorted = sorted(side, key=lambda v: deg[v])
            indep = greedy_independent_set(adj, side_sorted)
            if len(indep) > len(best):
                best = indep
        
        # Also try the other assignment
        pos_indices2 = [i for i in range(n) if fiedler[i] <= 0]
        neg_indices2 = [i for i in range(n) if fiedler[i] > 0]
        for side in [pos_indices2, neg_indices2]:
            side_sorted = sorted(side, key=lambda v: deg[v])
            indep = greedy_independent_set(adj, side_sorted)
            if len(indep) > len(best):
                best = indep
        
        return [values[i] for i in best]

    def max_cut_heuristic(adj, values, time_limit=2.0):
        """Iterative max-cut style: start with random partition, flip vertices to increase independent set size."""
        n = len(adj)
        if n == 0:
            return []
        
        start = time.time()
        rng = random.Random(123)
        
        best_size = 0
        best_set = set()
        
        for trial in range(3):
            if time.time() - start > time_limit:
                break
            
            # Random initial partition
            partition = [rng.choice([0, 1]) for _ in range(n)]
            
            # Compute size of independent set in each color class
            def get_independent_set(color):
                verts = [i for i in range(n) if partition[i] == color]
                # Greedy: add if no neighbor in set
                in_iset = set()
                for v in sorted(verts, key=lambda x: deg[x]):
                    if all(nb not in in_iset for nb in adj[v]):
                        in_iset.add(v)
                return in_iset
            
            iset0 = get_independent_set(0)
            iset1 = get_independent_set(1)
            current_size = max(len(iset0), len(iset1))
            
            # Iterative improvement: flip vertices
            improved = True
            while improved and time.time() - start < time_limit:
                improved = False
                for v in range(n):
                    if time.time() - start > time_limit:
                        break
                    old_color = partition[v]
                    partition[v] = 1 - old_color
                    
                    iset0_new = get_independent_set(0)
                    iset1_new = get_independent_set(1)
                    new_size = max(len(iset0_new), len(iset1_new))
                    
                    if new_size >= current_size:
                        current_size = new_size
                        improved = True
                    else:
                        partition[v] = old_color
            
            # Get best independent set
            iset0 = get_independent_set(0)
            iset1 = get_independent_set(1)
            best_iset = iset0 if len(iset0) >= len(iset1) else iset1
            
            if len(best_iset) > best_size:
                best_size = len(best_iset)
                best_set = best_set | best_iset  # union to track
        
        return [values[i] for i in best_set]

    def solve_one(values, time_budget=15.0):
        if time.time() > deadline:
            return []

        n = len(values)
        if n == 0:
            return []

        V_set = set(values)
        max_v = max(values)
        min_v = min(values)

        best = []
        best_set = set()

        # Generate multiple initial solutions
        candidates = []

        # Strategy 1: Odd numbers
        s1 = [x for x in values if x % 2 != 0]
        candidates.append(s1)

        # Strategy 2: Large positive numbers (above max/2)
        thresh_pos = max_v / 2.0
        s2 = [x for x in values if x > thresh_pos]
        candidates.append(s2)

        # Strategy 3: Large negative numbers (below min/2)
        thresh_neg = min_v / 2.0
        s3 = [x for x in values if x < thresh_neg]
        candidates.append(s3)

        # Strategy 4: Union of large pos and neg
        s4 = sorted(set(s2).union(s3))
        candidates.append(s4)

        # Strategy 5: Greedy by absolute value (largest first)
        s5_list, s5_set = greedy_add([], sorted(values, key=abs, reverse=True))
        candidates.append(s5_list)

        # Strategy 6: Greedy by value (smallest first)
        s6_list, s6_set = greedy_add([], sorted(values))
        candidates.append(s6_list)

        # Strategy 7: Greedy by value (largest first)
        s7_list, s7_set = greedy_add([], sorted(values, reverse=True))
        candidates.append(s7_list)

        # Strategy 8: Random shuffles
        rng = random.Random(42)
        for trial in range(5):
            shuffled = list(values)
            rng.shuffle(shuffled)
            sl, ss = greedy_add([], shuffled)
            candidates.append(sl)

        # Strategy 9: Graph-based spectral partition
        if n <= 300:
            try:
                adj = build_conflict_graph(values)
                spec_result = spectral_partition(adj, values)
                if len(spec_result) > 0:
                    candidates.append(spec_result)
            except:
                pass

        # Strategy 10: Max-cut heuristic on conflict graph
        if n <= 200 and time.time() < deadline - 10:
            try:
                adj = build_conflict_graph(values)
                mc_result = max_cut_heuristic(adj, values, time_limit=1.5)
                if len(mc_result) > 0:
                    candidates.append(mc_result)
            except:
                pass

        # Evaluate and keep best
        for c in candidates:
            if not is_sum_free(c):
                # Repair: remove conflicting elements
                c_valid = list(c)
                c_set = set(c_valid)
                changed = True
                while changed:
                    changed = False
                    i = 0
                    while i < len(c_valid):
                        x = c_valid[i]
                        conflict = False
                        for y in c_set:
                            if y == x:
                                continue
                            if (x + y) in c_set:
                                conflict = True
                                break
                            if (x - y) in c_set:
                                conflict = True
                                break
                        if conflict:
                            c_valid.pop(i)
                            c_set.remove(x)
                            changed = True
                        else:
                            i += 1
                c = c_valid

            if len(c) > len(best):
                best = c
                best_set = set(c)

        # Local search on best
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            ts = min(time_budget, remaining_time - 3)
            improved = local_search(best, values, time_limit=ts)
            if len(improved) > len(best):
                best = improved
                best_set = set(best)

        # Try additional random restarts if time permits
        remaining_time = deadline - time.time()
        if remaining_time > 8:
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                shuffled = list(values)
                rng.shuffle(shuffled)
                trial_list, trial_set = greedy_add([], shuffled)

                if len(trial_list) >= len(best):
                    ts = min(3.0, deadline - time.time() - 3)
                    if ts > 1:
                        improved = local_search(trial_list, values, time_limit=ts)
                        if len(improved) > len(best):
                            best = improved
                            best_set = set(best)

        return best

    results = []
    # Allocate time proportionally to problem size, with minimum time for small ones
    time_alloc = {60: 10.0, 120: 22.0, 240: 75.0}

    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            results.append([])
            continue

        values = instance(size, seed)
        budget = time_alloc.get(size, 15.0)
        # Don't exceed remaining time
        remaining = deadline - time.time()
        budget = min(budget, max(2.0, remaining - 3.0))

        subset = solve_one(values, time_budget=budget)

        # Final verification
        if not is_sum_free(subset):
            # Fallback to odds
            subset = [x for x in values if x % 2 != 0]

        results.append(subset)

    return results
