import time
import random
import numpy as np
import bisect

N = 256

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit

    # Precompute divisibility constraints for efficiency
    # For each pair (a, b) with a < b, compute which c >= b would violate
    # a | (b + c) iff c % a == (-b) % a
    
    # Precompute for all a in [1, N], b in [a+1, N]:
    # forbidden_c[a, b] = set of c >= b such that a | (b+c)
    # But this is too large. Instead, we'll use a faster validity check.

    def can_add_fast(s_array, candidate):
        """Fast check if candidate can be added to s_array (sorted numpy array)."""
        s = s_array
        n = len(s)
        if n == 0:
            return True
        
        # Find position
        pos = bisect.bisect_left(s, candidate)
        
        # candidate as a: for each b > candidate in s, check no c >= b with c % candidate == (-b) % candidate
        upper = s[pos:]  # all > candidate
        for b in upper:
            target = (-b) % candidate
            # Check if any element in upper from position of b onward has remainder target mod candidate
            # upper elements are all > candidate, so we need to check all c in upper where c >= b
            # Since upper is sorted, elements from the position of b onward
            idx_in_upper = bisect.bisect_left(upper, b)
            if idx_in_upper < len(upper):
                remainders = upper[idx_in_upper:] % candidate
                if np.any(remainders == target):
                    return False
        
        # candidate as b: for each a < candidate in s, check no c >= candidate with (candidate + c) % a == 0
        lower = s[:pos]  # all < candidate
        for a in lower:
            if a == 0:
                continue
            # c ranges over {candidate} union s[pos:]
            # Check candidate itself: (candidate + candidate) % a
            if (2 * candidate) % a == 0:
                return False
            # Check c in s[pos:]
            if pos < n:
                upper_part = s[pos:]
                targets = (candidate + upper_part) % a
                if np.any(targets == 0):
                    return False
        
        # candidate as c: for each a < b <= candidate in s, check (b + candidate) % a != 0
        for i in range(n):
            a = s[i]
            if a == 0:
                continue
            for j in range(i + 1, n):
                b = s[j]
                if b > candidate:
                    break
                if (b + candidate) % a == 0:
                    return False
        
        return True

    def ac3_prune(s_array, candidate_pool):
        """
        AC-3 style arc consistency: iteratively remove elements from candidate_pool
        that cannot coexist with any element in s_array (the core set).
        Returns the pruned candidate pool.
        """
        if len(s_array) == 0:
            return candidate_pool[:]
        
        s = np.array(s_array)
        pool = candidate_pool[:]
        changed = True
        while changed:
            changed = False
            new_pool = []
            for cand in pool:
                if can_add_fast(s, cand):
                    new_pool.append(cand)
                else:
                    changed = True
            pool = new_pool
        return pool

    def backtracking_search(core_set, candidate_pool, depth=0, max_depth=30):
        """
        Randomized backtracking search to maximize set size.
        core_set: current set (sorted list)
        candidate_pool: elements that could potentially be added
        """
        if time.time() - start_time > deadline:
            return core_set[:]
        
        if depth >= max_depth:
            return core_set[:]
        
        if len(candidate_pool) == 0:
            return core_set[:]
        
        # AC-3 style pruning
        pruned_pool = ac3_prune(core_set, candidate_pool)
        
        if len(pruned_pool) == 0:
            return core_set[:]
        
        # Select element with smallest "domain" - here we use heuristic:
        # Try elements that conflict with fewer others in the pool
        # For simplicity, try random order with some bias toward larger values
        random.shuffle(pruned_pool)
        
        best_result = core_set[:]
        
        for cand in pruned_pool:
            if time.time() - start_time > deadline:
                break
            
            # Try adding cand
            new_core = core_set[:]
            pos = bisect.bisect_left(new_core, cand)
            new_core.insert(pos, cand)
            new_core_array = np.array(new_core)
            
            # Remove cand from pool and prune
            remaining_pool = [x for x in pruned_pool if x != cand]
            if remaining_pool:
                pruned_remaining = ac3_prune(new_core, remaining_pool)
            else:
                pruned_remaining = []
            
            # Recurse
            result = backtracking_search(new_core, pruned_remaining, depth + 1, max_depth)
            if len(result) > len(best_result):
                best_result = result
            
            # Also try NOT adding cand (just skip it)
            # This is implicit in trying other candidates
        
        return best_result

    def greedy_with_pruning(starting_set, max_candidates=50):
        """
        Greedy approach with AC-3 pruning between additions.
        """
        core = sorted(starting_set)
        core_array = np.array(core)
        
        # Build initial candidate pool
        all_elements = set(range(1, N + 1))
        core_set = set(core)
        candidate_pool = [x for x in range(1, N + 1) if x not in core_set]
        
        # Prune initially
        candidate_pool = ac3_prune(core, candidate_pool)
        
        # Greedily add elements
        while candidate_pool and time.time() - start_time < deadline:
            # Select best candidate (try all, pick one that allows most further additions)
            best_cand = None
            best_score = -1
            
            # Sample a subset for efficiency
            sample_size = min(len(candidate_pool), max_candidates)
            if len(candidate_pool) > sample_size:
                sample = random.sample(candidate_pool, sample_size)
            else:
                sample = candidate_pool[:]
            
            for cand in sample:
                pos = bisect.bisect_left(core, cand)
                new_core = core[:]
                new_core.insert(pos, cand)
                new_core_array = np.array(new_core)
                
                # Count how many remaining candidates can still be added
                remaining = [x for x in candidate_pool if x != cand]
                if remaining:
                    pruned = ac3_prune(new_core, remaining)
                    score = len(pruned)
                else:
                    score = 0
                
                if score > best_score:
                    best_score = score
                    best_cand = cand
            
            if best_cand is None:
                break
            
            # Add best_cand
            pos = bisect.bisect_left(core, best_cand)
            core.insert(pos, best_cand)
            core_array = np.array(core)
            
            # Update pool
            candidate_pool = [x for x in candidate_pool if x != best_cand]
            if candidate_pool:
                candidate_pool = ac3_prune(core, candidate_pool)
        
        return core

    # Main strategy: Try multiple starting configurations
    best_set = []
    
    # Strategy 1: Start with upper range
    for min_start in [128, 144, 160, 171, 180, 192, 200, 210, 220]:
        if time.time() - start_time > deadline:
            break
        
        starting = list(range(min_start, N + 1))
        
        # Repair conflicts by removing elements
        changed = True
        while changed:
            changed = False
            for i in range(len(starting)):
                if time.time() - start_time > deadline:
                    break
                a = starting[i]
                for j in range(i + 1, len(starting)):
                    b = starting[j]
                    target = (-b) % a
                    for k in range(j, len(starting)):
                        if starting[k] % a == target:
                            # Remove a (smallest, most constrained)
                            starting.pop(i)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
        
        # Now use greedy with pruning to add more elements
        result = greedy_with_pruning(starting)
        
        if len(result) > len(best_set):
            best_set = result[:]
    
    # Strategy 2: Start empty and build up
    if time.time() - start_time < deadline:
        # Start with a reasonable seed
        seed_set = list(range(128, N + 1))
        # Repair
        changed = True
        while changed:
            changed = False
            for i in range(len(seed_set)):
                if time.time() - start_time > deadline:
                    break
                a = seed_set[i]
                for j in range(i + 1, len(seed_set)):
                    b = seed_set[j]
                    target = (-b) % a
                    for k in range(j, len(seed_set)):
                        if seed_set[k] % a == target:
                            seed_set.pop(i)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
        
        result = greedy_with_pruning(seed_set)
        if len(result) > len(best_set):
            best_set = result[:]
    
    # Local search improvement
    if len(best_set) > 0:
        for iteration in range(100):
            if time.time() - start_time > deadline:
                break
            
            s = best_set[:]
            s_set = set(s)
            
            # Try removing a random element
            idx = random.randint(0, len(s) - 1)
            removed = s.pop(idx)
            
            # Try to add back removed or other elements using greedy with pruning
            # First, try to add back the removed element
            if can_add_fast(np.array(s), removed):
                pos = bisect.bisect_left(s, removed)
                s.insert(pos, removed)
                s_set.add(removed)
            
            # Try adding other elements
            candidates = [x for x in range(1, N + 1) if x not in s_set]
            if candidates:
                pruned = ac3_prune(s, candidates)
                random.shuffle(pruned)
                for cand in pruned[:20]:  # Try at most 20
                    if time.time() - start_time > deadline:
                        break
                    if can_add_fast(np.array(s), cand):
                        pos = bisect.bisect_left(s, cand)
                        s.insert(pos, cand)
                        s_set.add(cand)
            
            if len(s) > len(best_set):
                best_set = s[:]
    
    return best_set
