"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Find a maximum-size Sidon set (B_2 set) in [1, 284] that must contain
    the prescribed elements {1, 2, 5, 11}.
    
    A Sidon set has all pairwise sums a+b (a<=b) distinct.
    This is equivalent to a Golomb ruler where differences are distinct,
    but here we require sums to be distinct (which is the same condition
    for the set of all pairwise sums including doubles).
    
    Strategy:
    1. Start with the prescribed elements {1, 2, 5, 11}.
    2. Use a combination of greedy construction, local search (simulated annealing),
       and randomized restarts to find a large valid Sidon set.
    3. The known maximum without constraints is 20 (Golomb ruler length 283).
       With the prescribed elements, we aim to get close to 20.
    """
    deadline = time.time() + 145  # 5s safety margin before 150s kill
    
    def valid_sidon(s):
        """Check if set s is a valid Sidon set (all pairwise sums distinct)."""
        s = sorted(s)
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sm = s[i] + s[j]
                if sm in sums:
                    return False
                sums.add(sm)
        return True
    
    def sum_set(s):
        """Return the set of all pairwise sums for set s."""
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        return sums
    
    def greedy_construct(prescribed, rng):
        """
        Greedy construction starting from prescribed elements.
        Try to add elements in order (or randomized order) that maintain Sidon property.
        """
        a = list(prescribed)
        sums = sum_set(a)
        
        # Try adding elements from 12 to 284
        candidates = list(range(12, 285))
        rng.shuffle(candidates)
        
        for x in candidates:
            if x in a:
                continue
            new_sums = {x + y for y in a} | {2 * x}
            if not (new_sums & sums):
                a.append(x)
                sums |= new_sums
        
        return sorted(a)
    
    def local_search(s, rng, time_limit=2.0):
        """
        Local search via simulated annealing to try to grow the Sidon set.
        Operations:
        - Add a new element if possible
        - Swap an element for another
        - Remove an element to make room for two others
        """
        a = list(s)
        sums = sum_set(a)
        best = len(a)
        best_set = list(a)
        
        start = time.time()
        temp = 10.0
        cooling = 0.9995
        
        while time.time() - start < time_limit and temp > 0.01:
            # Random operation
            op = rng.random()
            
            if op < 0.4:
                # Try to add a random element
                candidates = [x for x in range(1, 285) if x not in a]
                if not candidates:
                    temp *= cooling
                    continue
                x = rng.choice(candidates)
                new_sums = {x + y for y in a} | {2 * x}
                conflicts = new_sums & sums
                if not conflicts:
                    a.append(x)
                    sums |= new_sums
                    if len(a) > best:
                        best = len(a)
                        best_set = list(a)
                else:
                    # Penalty-based: accept with some probability
                    penalty = len(conflicts)
                    if rng.random() < temp / (penalty * 10):
                        # This would require removing conflicting elements, 
                        # but for simplicity, just skip
                        pass
                temp *= cooling
                
            elif op < 0.7:
                # Try swap: remove one element, add another
                if len(a) < 2:
                    temp *= cooling
                    continue
                idx = rng.randrange(len(a))
                removed = a[idx]
                a.pop(idx)
                # Recompute sums
                sums = sum_set(a)
                
                candidates = [x for x in range(1, 285) if x not in a]
                if candidates:
                    x = rng.choice(candidates)
                    new_sums = {x + y for y in a} | {2 * x}
                    if not (new_sums & sums):
                        a.append(x)
                        sums |= new_sums
                        if len(a) > best:
                            best = len(a)
                            best_set = list(a)
                    else:
                        # Revert
                        a.insert(idx, removed)
                        sums = sum_set(a)
                else:
                    a.insert(idx, removed)
                    sums = sum_set(a)
                temp *= cooling
                
            else:
                # Try to add two elements by removing one (net +1)
                if len(a) < 3:
                    temp *= cooling
                    continue
                idx = rng.randrange(len(a))
                removed = a[idx]
                a.pop(idx)
                sums = sum_set(a)
                
                # Try to find two elements to add
                added_any = False
                for _ in range(2):
                    candidates = [x for x in range(1, 285) if x not in a]
                    if not candidates:
                        break
                    x = rng.choice(candidates)
                    new_sums = {x + y for y in a} | {2 * x}
                    if not (new_sums & sums):
                        a.append(x)
                        sums |= new_sums
                        added_any = True
                
                if added_any and len(a) > best:
                    best = len(a)
                    best_set = list(a)
                elif not added_any:
                    # Revert
                    a.insert(idx, removed)
                    sums = sum_set(a)
                temp *= cooling
        
        return best_set
    
    def hill_climb_grow(s, rng):
        """
        Aggressively try to grow the set by finding any valid addition.
        """
        a = list(s)
        sums = sum_set(a)
        
        # Try all possible additions
        for x in range(1, 285):
            if x in a:
                continue
            new_sums = {x + y for y in a} | {2 * x}
            if not (new_sums & sums):
                a.append(x)
                sums |= new_sums
        
        return sorted(a)
    
    # Main search
    prescribed = [1, 2, 5, 11]
    best_result = list(prescribed)
    best_size = len(prescribed)
    
    rng = random.Random(42)
    
    # Phase 1: Multiple greedy constructions with different random seeds
    for trial in range(50):
        if time.time() > deadline:
            break
        result = greedy_construct(prescribed, rng)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
    
    # Phase 2: Local search from best and other good solutions
    for trial in range(30):
        if time.time() > deadline:
            break
        if trial == 0:
            start_set = best_result
        else:
            start_set = greedy_construct(prescribed, rng)
        
        result = local_search(start_set, rng, time_limit=3.0)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
    
    # Phase 3: Hill climbing from best
    if time.time() < deadline - 10:
        result = hill_climb_grow(best_result, rng)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
    
    # Phase 4: More aggressive local search with more iterations
    if time.time() < deadline - 20:
        for trial in range(20):
            if time.time() > deadline:
                break
            start_set = list(best_result)
            # Add some random perturbation
            if len(start_set) > 5:
                for _ in range(3):
                    idx = rng.randrange(len(start_set))
                    if start_set[idx] not in prescribed:
                        old = start_set[idx]
                        candidates = [x for x in range(1, 285) if x not in start_set]
                        if candidates:
                            start_set[idx] = rng.choice(candidates)
                            # Check validity
                            if not valid_sidon(start_set):
                                start_set[idx] = old
            
            result = local_search(start_set, rng, time_limit=4.0)
            if len(result) > best_size:
                best_size = len(result)
                best_result = result
    
    # Phase 5: Final hill climb
    if time.time() < deadline - 5:
        result = hill_climb_grow(best_result, rng)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
    
    # Verify final result
    assert valid_sidon(best_result), "Final result is not a valid Sidon set"
    assert set(prescribed).issubset(set(best_result)), "Missing prescribed elements"
    assert all(1 <= x <= 284 for x in best_result), "Elements out of range"
    
    return sorted(best_result)
# EVOLVE-BLOCK-END
