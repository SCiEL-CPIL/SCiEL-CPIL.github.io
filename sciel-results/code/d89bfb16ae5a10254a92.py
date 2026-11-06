"""Improved program: CDCL-inspired local search for large sum-free subsets."""
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def get_all_conflicts(subset):
        """Return list of conflicting triples (a, b, c) where a+b=c, all in subset."""
        s = set(subset)
        conflicts = []
        seen = set()
        for a in subset:
            for b in subset:
                c = a + b
                if c in s and c not in seen:
                    conflicts.append((a, b, c))
                    seen.add(c)
        return conflicts

    def conflict_frequency(subset):
        """Count how many conflicts each element participates in."""
        s = set(subset)
        freq = defaultdict(int)
        for a in subset:
            for b in subset:
                c = a + b
                if c in s:
                    freq[a] += 1
                    freq[b] += 1
                    freq[c] += 1
        return freq

    def greedy_init(values, order_key=None, rng=None):
        """Greedy construction with given ordering."""
        n = len(values)
        if order_key is None:
            order = list(range(n))
        else:
            order = sorted(range(n), key=order_key)
        
        current = set()
        for i in order:
            v = values[i]
            ok = True
            for a in current:
                if (v - a) in current or (v + a) in current:
                    ok = False
                    break
            if ok:
                current.add(v)
        return current

    def cdcl_local_search(values, time_budget, rng):
        """
        CDCL-inspired local search:
        - Maintain conflict database (learned clauses)
        - Use conflict analysis to guide removals
        - Unit propagation: if removing one element resolves multiple conflicts, prioritize it
        """
        n = len(values)
        val_set = set(values)
        
        # Multiple greedy initializations
        best_subset = set()
        best_size = 0
        
        orderings = [
            lambda i: abs(values[i]),
            lambda i: abs(values[i]),  # will reverse later
            lambda i: values[i],
            lambda i: values[i],
            lambda i: 0,  # original order
            lambda i: -abs(values[i]),
            lambda i: -values[i],
        ]
        
        for oi, order_key in enumerate(orderings):
            if time.time() >= time_budget:
                break
            order = sorted(range(n), key=order_key, reverse=(oi % 2 == 1 and oi > 1))
            current = set()
            for i in order:
                v = values[i]
                ok = True
                for a in current:
                    if (v - a) in current or (v + a) in current:
                        ok = False
                        break
                if ok:
                    current.add(v)
            if len(current) > best_size:
                best_size = len(current)
                best_subset = set(current)
        
        current = set(best_subset)
        
        # Learned clauses: sets of elements where at least one must be removed
        # Represent as frozensets for hashing
        learned_clauses = set()
        
        # Conflict database: map from element to set of conflicts it's in
        # We'll rebuild as needed
        
        def analyze_conflicts(subset):
            """Analyze conflicts and return frequency map and critical elements."""
            s = set(subset)
            freq = defaultdict(int)
            conflict_elems = set()
            for a in subset:
                for b in subset:
                    c = a + b
                    if c in s:
                        freq[a] += 1
                        freq[b] += 1
                        freq[c] += 1
                        conflict_elems.add(a)
                        conflict_elems.add(b)
                        conflict_elems.add(c)
            return freq, conflict_elems
        
        def try_add(subset, v):
            """Check if v can be added to subset."""
            s = set(subset)
            for a in s:
                if (v - a) in s or (v + a) in s:
                    return False
            return True
        
        def try_remove_resolve(subset, rem_elem, candidates):
            """Try removing rem_elem then adding one of candidates."""
            trial = set(subset)
            trial.discard(rem_elem)
            for v in candidates:
                if v in trial:
                    continue
                if try_add(trial, v):
                    trial.add(v)
                    return trial, v
            return None, None
        
        improved = True
        stagnation_count = 0
        max_stagnation = 20
        
        while improved and time.time() < time_budget:
            improved = False
            
            # Phase 1: Try to add elements
            candidates = [v for v in values if v not in current]
            rng.shuffle(candidates)
            
            added_any = False
            for v in candidates:
                if try_add(current, v):
                    current.add(v)
                    improved = True
                    added_any = True
                    break
            
            if added_any:
                stagnation_count = 0
                continue
            
            # Phase 2: Analyze conflicts
            freq, conflict_elems = analyze_conflicts(current)
            
            if not conflict_elems:
                # No conflicts, but couldn't add anything - try harder
                # Random perturbation
                if current and rng.random() < 0.3:
                    rem = rng.choice(list(current))
                    current.discard(rem)
                    improved = True
                    stagnation_count += 1
                    if stagnation_count > max_stagnation:
                        break
                continue
            
            # Phase 3: CDCL-style conflict analysis
            # Find elements whose removal would resolve the most conflicts
            # and potentially allow adding more elements
            
            # Sort conflicting elements by frequency (most critical first)
            sorted_conflicts = sorted(conflict_elems, key=lambda x: freq.get(x, 0), reverse=True)
            
            found_improvement = False
            for rem in sorted_conflicts:
                if time.time() >= time_budget:
                    break
                
                # Try removing this element and see if we can add more
                trial, added_v = try_remove_resolve(current, rem, candidates)
                if trial is not None:
                    # Check if this is actually an improvement
                    if len(trial) > len(current):
                        current = trial
                        improved = True
                        found_improvement = True
                        # Learn: this removal+addition pattern worked
                        # Store as a clause: if we had rem and not added_v, that was bad
                        break
                    elif len(trial) == len(current) and rng.random() < 0.1:
                        # Neutral move, accept occasionally for exploration
                        current = trial
                        improved = True
                        found_improvement = True
                        break
            
            if found_improvement:
                stagnation_count = 0
                continue
            
            # Phase 4: If no improvement, try multi-removal moves
            # Remove 2 elements to potentially add 3
            if len(conflict_elems) >= 2:
                pair_list = list(conflict_elems)
                rng.shuffle(pair_list)
                found_pair = False
                for i in range(min(len(pair_list), 10)):
                    for j in range(i+1, min(len(pair_list), 10)):
                        if time.time() >= time_budget:
                            break
                        rem1, rem2 = pair_list[i], pair_list[j]
                        if rem1 == rem2:
                            continue
                        trial = set(current)
                        trial.discard(rem1)
                        trial.discard(rem2)
                        for v in candidates:
                            if v in trial:
                                continue
                            if try_add(trial, v):
                                trial.add(v)
                                # Try to add one more
                                for v2 in candidates:
                                    if v2 in trial or v2 == v:
                                        continue
                                    if try_add(trial, v2):
                                        trial.add(v2)
                                        break
                                if len(trial) > len(current):
                                    current = trial
                                    improved = True
                                    found_pair = True
                                    break
                        if found_pair:
                            break
                    if found_pair:
                        break
                
                if found_pair:
                    stagnation_count = 0
                    continue
            
            # Phase 5: Random restart / perturbation
            stagnation_count += 1
            if stagnation_count > max_stagnation:
                # Larger perturbation
                if current:
                    num_remove = max(1, len(current) // 10)
                    for _ in range(num_remove):
                        if current:
                            rem = rng.choice(list(current))
                            current.discard(rem)
                improved = True
                stagnation_count = 0
            elif rng.random() < 0.2:
                if current:
                    rem = rng.choice(list(current))
                    current.discard(rem)
                improved = True
        
        # Final cleanup
        while not is_sum_free(current):
            freq, conflict_elems = analyze_conflicts(current)
            if not conflict_elems:
                break
            rem = rng.choice(list(conflict_elems))
            current.discard(rem)
        
        return sorted(current)

    def solve_instance(values, time_budget):
        rng = random.Random(hash(tuple(values)) % (2**32))
        
        # Run CDCL local search
        result = cdcl_local_search(values, time_budget, rng)
        
        # Additional pass: try to improve with fresh random seeds
        if time.time() < time_budget:
            for seed_offset in range(5):
                if time.time() >= time_budget:
                    break
                rng2 = random.Random(hash(tuple(values) + (seed_offset,)) % (2**32))
                sub_budget = time_budget - (time.time() - (time_budget - (deadline - time.time())))
                # Just use remaining time
                remaining = time_budget - time.time()
                if remaining < 0.1:
                    break
                result2 = cdcl_local_search(values, time.time() + remaining * 0.3, rng2)
                if len(result2) > len(result):
                    result = result2
        
        return result

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        values = instance(size, seed)
        remaining = deadline - time.time()
        if remaining < 1:
            subset = [x for x in values if x % 2]
            results.append(subset)
            continue

        if size == 240:
            budget = time.time() + remaining * 0.5
        elif size == 120:
            budget = time.time() + remaining * 0.3
        else:
            budget = time.time() + remaining * 0.2

        subset = solve_instance(values, budget)
        results.append(subset)

    return results
