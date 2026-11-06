import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin

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

    def build_schur_graph(values):
        """Build adjacency matrix for the Schur graph.
        Vertices are elements of values.
        Edge between i and j if values[i] + values[j] is in values, 
        or 2*values[i] == values[j] (i.e., one is double the other).
        """
        n = len(values)
        val_to_idx = {v: i for i, v in enumerate(values)}
        adj = np.zeros((n, n), dtype=np.int8)
        
        for i in range(n):
            for j in range(i+1, n):
                s = values[i] + values[j]
                if s in val_to_idx:
                    adj[i, j] = 1
                    adj[j, i] = 1
                # Also check if one is double the other
                if 2 * values[i] == values[j]:
                    adj[i, j] = 1
                    adj[j, i] = 1
                elif 2 * values[j] == values[i]:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        return adj

    def spectral_independent_set(adj, values, target_size=None, time_limit=2.0):
        """Find a large independent set using spectral bisection.
        Uses the fact that independent sets in a graph correspond to
        subsets with no edges between them.
        """
        start = time.time()
        n = len(values)
        
        if n <= 2:
            return list(range(n))
        
        # Compute Laplacian
        deg = adj.sum(axis=1)
        L = np.diag(deg) - adj.astype(float)
        
        # Get eigenvalues/eigenvectors
        try:
            evals, evecs = np.linalg.eigh(L)
        except:
            return []
        
        # The second eigenvector (Fiedler vector) gives a good bipartition
        # For finding independent sets, we want to separate vertices
        # that have edges between them.
        
        # Strategy: use the Fiedler vector to create a partition,
        # then take the larger part and iteratively remove vertices
        # with high degree within that part.
        
        fiedler = evecs[:, 1]  # Second eigenvector
        
        # Create initial partition based on median
        median_val = np.median(fiedler)
        partition_A = set(i for i in range(n) if fiedler[i] > median_val)
        partition_B = set(i for i in range(n) if fiedler[i] <= median_val)
        
        # Try both partitions and keep the one that gives larger independent set
        best_indep = set()
        
        for part in [partition_A, partition_B]:
            if not part:
                continue
            
            # Start with the partition and remove vertices to make it independent
            current = set(part)
            
            # Iteratively remove the vertex with highest internal degree
            improved = True
            while improved and (time.time() - start) < time_limit:
                improved = False
                if not current:
                    break
                
                # Compute internal degrees
                max_deg = -1
                max_vertex = None
                for v in current:
                    internal_deg = sum(1 for u in current if u != v and adj[v, u] == 1)
                    if internal_deg > max_deg:
                        max_deg = internal_deg
                        max_vertex = v
                
                if max_deg == 0:
                    break  # Already an independent set
                
                if max_vertex is not None:
                    current.discard(max_vertex)
                    improved = True
            
            if len(current) > len(best_indep):
                best_indep = current
        
        # Also try: start with all vertices and greedily remove
        # vertices with highest degree
        current_all = set(range(n))
        improved = True
        while improved and (time.time() - start) < time_limit:
            improved = False
            if not current_all:
                break
            
            max_deg = -1
            max_vertex = None
            for v in current_all:
                internal_deg = sum(1 for u in current_all if u != v and adj[v, u] == 1)
                if internal_deg > max_deg:
                    max_deg = internal_deg
                    max_vertex = v
            
            if max_deg == 0:
                break
            
            if max_vertex is not None:
                current_all.discard(max_vertex)
                improved = True
        
        if len(current_all) > len(best_indep):
            best_indep = current_all
        
        return list(best_indep)

    def spectral_greedy_hybrid(values, time_limit=3.0):
        """Combine spectral initial solution with greedy refinement."""
        start = time.time()
        
        if len(values) < 3:
            return list(values)
        
        # Build Schur graph
        adj = build_schur_graph(values)
        
        # Get spectral independent set
        time_per_method = time_limit / 3.0
        spec_indices = spectral_independent_set(adj, values, time_limit=time_per_method)
        spec_set = set(values[i] for i in spec_indices)
        
        # Refine with greedy add
        result, result_set = greedy_add(list(spec_set), values, set(spec_set))
        
        # If time remains, try local search
        remaining = time_limit - (time.time() - start)
        if remaining > 0.5:
            improved = local_search(result, values, time_limit=remaining)
            if len(improved) > len(result):
                result = improved
        
        return result

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
        
        # Phase 2: Iterative improvement - try removing one element and adding others
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
        
        # Phase 3: Try swap - remove one, add two specific elements
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
        
        return S_list

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
        for trial in range(3):
            shuffled = list(values)
            rng.shuffle(shuffled)
            sl, ss = greedy_add([], shuffled)
            candidates.append(sl)
        
        # Strategy 9: Spectral method
        remaining_for_spec = time_budget * 0.4
        if remaining_for_spec > 1.0:
            spec_result = spectral_greedy_hybrid(values, time_limit=remaining_for_spec)
            candidates.append(spec_result)
        
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
            for _ in range(8):
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
    time_alloc = {60: 12.0, 120: 25.0, 240: 70.0}
    
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
