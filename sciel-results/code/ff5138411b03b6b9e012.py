"""Improved program: greedy multi-start + enhanced local search with remove-and-retry."""
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # wall-clock limit; keep margin
    MARGIN = 5.0      # safety margin seconds

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c for a,b,c in S."""
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def can_add(x, chosen_set):
        """Check if x can be added to chosen_set while maintaining sum-free."""
        for a in chosen_set:
            if (x - a) in chosen_set:
                return False
            if a + x in chosen_set:
                return False
        if 2 * x in chosen_set:
            return False
        return True

    def greedy_build(S, order, chosen_set=None):
        """Greedy: add elements in given order if sum-free is maintained."""
        if chosen_set is None:
            chosen = []
            chosen_set = set()
        else:
            chosen = list(chosen_set)
        for idx in order:
            x = S[idx]
            if x in chosen_set:
                continue
            if can_add(x, chosen_set):
                chosen.append(x)
                chosen_set.add(x)
        return chosen

    def compute_conflicts(S):
        """For each element, compute how many other elements it conflicts with."""
        Sset = set(S)
        n = len(S)
        conflicts = [0] * n
        for i in range(n):
            x = S[i]
            for j in range(i + 1, n):
                y = S[j]
                # x and y conflict if x+y in Sset, or 2x=y, or 2y=x
                if x + y in Sset or 2 * x == y or 2 * y == x:
                    conflicts[i] += 1
                    conflicts[j] += 1
        return conflicts

    def local_search_remove_retry(S, initial, max_iterations=100):
        """Local search: remove one element, then greedily add back as many as possible."""
        current = list(initial)
        current_set = set(current)
        best = len(current)
        n = len(S)
        
        # Precompute conflict info
        Sset = set(S)
        
        for iteration in range(max_iterations):
            if time.time() - start_time > DEADLINE - MARGIN:
                break
            
            improved = False
            # Try removing each element and see if we can add more back
            for i in range(len(current)):
                if time.time() - start_time > DEADLINE - MARGIN:
                    break
                x = current[i]
                # Remove x
                new_set = set(current)
                new_set.discard(x)
                
                # Now try to add back elements not in new_set, greedily
                # Sort candidates by how few conflicts they have with new_set
                candidates = [y for y in S if y not in new_set]
                
                # Score candidates: prefer those that don't conflict with new_set
                def score(y):
                    cnt = 0
                    for a in new_set:
                        if (y - a) in new_set or a + y in new_set:
                            cnt += 1
                    if 2 * y in new_set:
                        cnt += 1
                    return cnt
                
                # Try adding candidates in order of lowest conflict score
                candidates.sort(key=score)
                
                trial_set = set(new_set)
                added = 0
                for y in candidates:
                    if can_add(y, trial_set):
                        trial_set.add(y)
                        added += 1
                
                if len(trial_set) > best:
                    best = len(trial_set)
                    current = list(trial_set)
                    current_set = set(current)
                    improved = True
                    break
            
            if not improved:
                break
        
        return current

    def local_search_swap(S, initial, max_iterations=50):
        """Local search: try swapping one element out for one in."""
        current = list(initial)
        current_set = set(current)
        best = len(current)
        n = len(S)
        
        for iteration in range(max_iterations):
            if time.time() - start_time > DEADLINE - MARGIN:
                break
            
            improved = False
            # Try each element in current
            for i in range(len(current)):
                if time.time() - start_time > DEADLINE - MARGIN:
                    break
                x = current[i]
                # Remove x
                new_set = set(current)
                new_set.discard(x)
                
                # Try adding each element not in new_set
                for y in S:
                    if y in new_set:
                        continue
                    if can_add(y, new_set):
                        # We can add y without removing x, so just add
                        new_set.add(y)
                        if len(new_set) > best:
                            best = len(new_set)
                            current = list(new_set)
                            current_set = set(current)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
        
        return current

    def solve_instance(S, time_budget):
        """Solve one instance: multiple greedy starts + local search."""
        n = len(S)
        best_result = []
        best_size = 0
        
        # Baseline: odd numbers
        odds = [x for x in S if x % 2 != 0]
        if is_sum_free(odds):
            best_result = odds
            best_size = len(odds)
        
        # Compute conflicts for ordering
        conflicts = compute_conflicts(S)
        
        # Try multiple greedy orderings
        import random
        orderings = []
        
        # Natural order
        orderings.append(list(range(n)))
        # Reverse order
        orderings.append(list(range(n - 1, -1, -1)))
        
        # Sorted by conflict count (least conflicted first)
        orderings.append(sorted(range(n), key=lambda i: conflicts[i]))
        # Sorted by conflict count (most conflicted first)
        orderings.append(sorted(range(n), key=lambda i: -conflicts[i]))
        
        # Random shuffles
        rng = random.Random(42)
        for _ in range(30):
            perm = list(range(n))
            rng.shuffle(perm)
            orderings.append(perm)
        
        # Sorted by absolute value
        orderings.append(sorted(range(n), key=lambda i: abs(S[i])))
        # Sorted by value
        orderings.append(sorted(range(n), key=lambda i: S[i]))
        # Sorted by -value
        orderings.append(sorted(range(n), key=lambda i: -S[i]))
        
        # Run greedy on all orderings
        for order in orderings:
            if time.time() - start_time > DEADLINE - MARGIN:
                break
            result = greedy_build(S, order)
            if len(result) > best_size:
                best_size = len(result)
                best_result = result
        
        # Local search on best result
        if time.time() - start_time < DEADLINE - MARGIN:
            ls_result = local_search_remove_retry(S, best_result, max_iterations=200)
            if len(ls_result) > best_size:
                best_size = len(ls_result)
                best_result = ls_result
        
        # Second round of local search
        if time.time() - start_time < DEADLINE - MARGIN:
            ls_result2 = local_search_swap(S, best_result, max_iterations=100)
            if len(ls_result2) > best_size:
                best_size = len(ls_result2)
                best_result = ls_result2
        
        # Third round: another remove-retry
        if time.time() - start_time < DEADLINE - MARGIN:
            ls_result3 = local_search_remove_retry(S, best_result, max_iterations=300)
            if len(ls_result3) > best_size:
                best_size = len(ls_result3)
                best_result = ls_result3
        
        return best_result

    # Solve all three instances
    instances = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        instances.append(S)
    
    results = []
    for S in instances:
        if time.time() - start_time > DEADLINE - MARGIN:
            # Fallback to odd numbers
            results.append([x for x in S if x % 2 != 0])
        else:
            result = solve_instance(S, DEADLINE - MARGIN - (time.time() - start_time))
            # Verify
            if not is_sum_free(result):
                result = [x for x in S if x % 2 != 0]
            results.append(result)
    
    return results
