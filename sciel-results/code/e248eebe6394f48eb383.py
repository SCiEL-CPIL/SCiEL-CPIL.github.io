import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin for 90s limit
    
    n = 128
    MIN_SUM = 128
    MAX_SUM = 256
    
    # Precompute: for each target t in [128,256], the list of unordered pairs (a,b) with a<=b, a+b=t
    # This is used for incremental evaluation
    target_pairs = {}
    for t in range(MIN_SUM, MAX_SUM + 1):
        pairs = []
        for a in range(0, 129):
            b = t - a
            if b < a:
                continue
            if 0 <= b <= 128:
                pairs.append((a, b))
        target_pairs[t] = pairs
    
    def evaluate(A_set):
        """
        Fast evaluation using convolution.
        Returns (max_mult, num_uncovered).
        """
        if not A_set:
            return 1000, 129
        
        v = np.zeros(257, dtype=np.int32)
        for x in A_set:
            if 0 <= x <= 128:
                v[x] = 1
        
        conv = np.convolve(v, v, mode='full')
        
        max_mult = 0
        num_uncovered = 0
        
        for t in range(MIN_SUM, MAX_SUM + 1):
            ordered = conv[t]
            diag = 1 if (t % 2 == 0 and (t // 2) in A_set) else 0
            unordered = (ordered + diag) // 2
            
            if unordered == 0:
                num_uncovered += 1
            elif unordered > max_mult:
                max_mult = unordered
                
        return max_mult, num_uncovered
    
    def incremental_evaluate(A_set, added=None, removed=None):
        """
        Faster evaluation by leveraging that we know which elements changed.
        But for simplicity and correctness, we'll use the convolution method.
        The key speedup is in the search strategy, not just evaluation.
        """
        return evaluate(A_set)
    
    def cost(A_set):
        """
        Cost function: heavily penalize uncovered targets, then minimize max multiplicity.
        """
        m, u = evaluate(A_set)
        if u > 0:
            return u * 10000 + m
        return m
    
    def greedy_cover(A_set):
        """
        Greedily add elements to ensure coverage.
        For each uncovered target, find a pair (a,b) with a+b=t and add the element
        that creates the most coverage for other uncovered targets.
        """
        current_A = set(A_set)
        
        # Check which targets are uncovered
        m, u = evaluate(current_A)
        if u == 0:
            return current_A
        
        # Find uncovered targets
        v = np.zeros(257, dtype=np.int32)
        for x in current_A:
            v[x] = 1
        conv = np.convolve(v, v, mode='full')
        
        uncovered = []
        for t in range(MIN_SUM, MAX_SUM + 1):
            ordered = conv[t]
            diag = 1 if (t % 2 == 0 and (t // 2) in current_A) else 0
            unordered = (ordered + diag) // 2
            if unordered == 0:
                uncovered.append(t)
        
        # For each uncovered target, try to add an element that helps
        # We'll do a few passes
        for _ in range(3):
            if not uncovered:
                break
            
            # For each uncovered target, find candidate elements to add
            best_add = None
            best_gain = 0
            best_target = None
            
            for t in uncovered:
                pairs = target_pairs[t]
                for a, b in pairs:
                    # If we add a (or b), does it help cover t?
                    # Adding a helps if b is already in A
                    # Adding b helps if a is already in A
                    # Adding both helps if neither is in A
                    if a not in current_A and b in current_A:
                        # Adding a covers t
                        # Check how many other uncovered targets this also helps
                        gain = 1
                        for t2 in uncovered:
                            if t2 == t:
                                continue
                            pairs2 = target_pairs[t2]
                            for a2, b2 in pairs2:
                                if (a2 == a and b2 in current_A) or (b2 == a and a2 in current_A):
                                    gain += 1
                                    break
                        if gain > best_gain:
                            best_gain = gain
                            best_add = a
                            best_target = t
                    elif b not in current_A and a in current_A:
                        gain = 1
                        for t2 in uncovered:
                            if t2 == t:
                                continue
                            pairs2 = target_pairs[t2]
                            for a2, b2 in pairs2:
                                if (a2 == b and b2 in current_A) or (b2 == b and a2 in current_A):
                                    gain += 1
                                    break
                        if gain > best_gain:
                            best_gain = gain
                            best_add = b
                            best_target = t
            
            if best_add is not None:
                current_A.add(best_add)
                # Recalculate uncovered
                m, u = evaluate(current_A)
                if u == 0:
                    return current_A
                v = np.zeros(257, dtype=np.int32)
                for x in current_A:
                    v[x] = 1
                conv = np.convolve(v, v, mode='full')
                uncovered = []
                for t in range(MIN_SUM, MAX_SUM + 1):
                    ordered = conv[t]
                    diag = 1 if (t % 2 == 0 and (t // 2) in current_A) else 0
                    unordered = (ordered + diag) // 2
                    if unordered == 0:
                        uncovered.append(t)
            else:
                # Can't fix with single additions; add t//2 as fallback
                if uncovered:
                    t = uncovered[0]
                    mid = t // 2
                    if 0 <= mid <= 128:
                        current_A.add(mid)
                    if t % 2 == 1 and mid + 1 <= 128:
                        current_A.add(mid + 1)
                    m, u = evaluate(current_A)
                    if u == 0:
                        return current_A
        
        return current_A
    
    def simulated_annealing(A_init, time_budget, seed=None):
        """
        Simulated annealing with smart moves.
        """
        if seed is not None:
            random.seed(seed)
        
        current_A = set(A_init)
        current_A = greedy_cover(current_A)
        current_cost = cost(current_A)
        
        best_A = set(current_A)
        best_cost = current_cost
        
        T = 20.0
        T_min = 0.5
        cooling = 0.9998
        
        # Track recent moves to avoid cycles
        recent_states = {}
        
        iter_count = 0
        while time.time() < time_budget:
            iter_count += 1
            
            # Generate a move
            move_type = random.random()
            
            if move_type < 0.5 and len(current_A) > 10:
                # Remove an element
                # Prefer removing elements that contribute less to coverage
                candidates = list(current_A)
                random.shuffle(candidates)
                
                best_remove = None
                best_remove_cost = float('inf')
                
                # Try a few candidates
                for x in candidates[:5]:
                    test_A = current_A - {x}
                    if not test_A:
                        continue
                    tc = cost(test_A)
                    if tc < best_remove_cost:
                        best_remove_cost = tc
                        best_remove = x
                
                if best_remove is not None:
                    new_A = current_A - {best_remove}
                else:
                    x = random.choice(candidates)
                    new_A = current_A - {x}
                    
            elif move_type < 0.8:
                # Add an element
                # Prefer adding elements that help with high-multiplicity targets
                candidates = [x for x in range(129) if x not in current_A]
                if not candidates:
                    continue
                random.shuffle(candidates)
                
                best_add = None
                best_add_cost = float('inf')
                
                for x in candidates[:5]:
                    test_A = current_A | {x}
                    tc = cost(test_A)
                    if tc < best_add_cost:
                        best_add_cost = tc
                        best_add = x
                
                if best_add is not None:
                    new_A = current_A | {best_add}
                else:
                    x = random.choice(candidates)
                    new_A = current_A | {x}
                    
            else:
                # Swap: remove one, add another
                if len(current_A) < 2:
                    continue
                x = random.choice(list(current_A))
                y = random.randint(0, 128)
                if y == x:
                    continue
                new_A = (current_A - {x}) | {y}
                if not new_A:
                    continue
            
            if len(new_A) == 0:
                continue
            
            new_cost = cost(new_A)
            delta = new_cost - current_cost
            
            # Acceptance criterion
            if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                current_A = new_A
                current_cost = new_cost
                
                # Update best
                if current_cost < best_cost:
                    best_A = set(current_A)
                    best_cost = current_cost
                
                # Track state
                state_key = frozenset(current_A)
                recent_states[state_key] = iter_count
            else:
                # Rejection: try a different move
                pass
            
            T = max(T * cooling, T_min)
            
            # Periodic restart if stuck
            if iter_count % 500 == 0:
                # Check if we've been stuck
                recent_keys = list(recent_states.keys())
                if len(recent_keys) > 10:
                    # Sort by recency
                    recent_keys.sort(key=lambda k: recent_states[k], reverse=True)
                    if recent_keys[0] - recent_keys[-1] < 100:
                        # Stuck, do a random perturbation
                        for _ in range(3):
                            if random.random() < 0.5 and len(current_A) > 10:
                                x = random.choice(list(current_A))
                                current_A.discard(x)
                            else:
                                x = random.randint(0, 128)
                                current_A.add(x)
                        current_A = greedy_cover(current_A)
                        current_cost = cost(current_A)
                        if current_cost < best_cost:
                            best_A = set(current_A)
                            best_cost = current_cost
        
        return best_A, best_cost
    
    def local_polish(A_set, time_budget):
        """
        Final local polish: try removing redundant elements and small swaps.
        """
        current_A = set(A_set)
        current_cost = cost(current_A)
        best_A = set(current_A)
        best_cost = current_cost
        
        # Try removing elements one by one
        changed = True
        while changed and time.time() < time_budget:
            changed = False
            for x in sorted(current_A, reverse=True):
                if time.time() > time_budget:
                    break
                test_A = current_A - {x}
                if not test_A:
                    continue
                tc = cost(test_A)
                if tc <= current_cost:
                    current_A = test_A
                    current_cost = tc
                    if current_cost < best_cost:
                        best_A = set(current_A)
                        best_cost = current_cost
                    changed = True
        
        # Try small swaps
        for _ in range(50):
            if time.time() > time_budget:
                break
            if len(current_A) < 2:
                break
            x = random.choice(list(current_A))
            y = random.randint(0, 128)
            if y == x:
                continue
            test_A = (current_A - {x}) | {y}
            if not test_A:
                continue
            tc = cost(test_A)
            if tc < current_cost:
                current_A = test_A
                current_cost = tc
                if current_cost < best_cost:
                    best_A = set(current_A)
                    best_cost = current_cost
        
        return best_A, best_cost
    
    # Main search: multiple restarts with diverse initializations
    best_overall_A = set()
    best_overall_cost = float('inf')
    
    # Structured initializations
    inits = []
    
    # Full set
    inits.append(set(range(0, 129)))
    
    # Half: [64, 128]
    inits.append(set(range(64, 129)))
    
    # Even numbers
    inits.append(set(range(0, 129, 2)))
    
    # Multiples of 3
    inits.append(set(range(0, 129, 3)))
    
    # Multiples of 4
    inits.append(set(range(0, 129, 4)))
    
    # Random sets of various sizes
    random.seed(42)
    for size in [30, 40, 50, 60, 70, 80, 90, 100, 110]:
        s = set(random.sample(range(0, 129), size))
        inits.append(s)
    
    # Ensure 128 is in each init (needed for 256 = 128+128)
    for i in range(len(inits)):
        inits[i].add(128)
    
    # Run simulated annealing from each initialization
    for i, init_A in enumerate(inits):
        if time.time() > deadline - 5:
            break
        
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        # Use more time for earlier, more structured inits
        if i < 3:
            time_budget = min(remaining - 2, 15)
        elif i < 6:
            time_budget = min(remaining - 2, 10)
        else:
            time_budget = min(remaining - 2, 7)
        
        if time_budget < 1:
            break
        
        result_A, result_cost = simulated_annealing(init_A, time_budget, seed=42 + i)
        
        if result_cost < best_overall_cost:
            best_overall_cost = result_cost
            best_overall_A = result_A
    
    # Final polish
    if best_overall_A:
        polished_A, polished_cost = local_polish(best_overall_A, deadline - 2)
        if polished_cost < best_overall_cost:
            best_overall_A = polished_A
            best_overall_cost = polished_cost
    
    # Fallback if no valid solution found
    if not best_overall_A or best_overall_cost >= 10000:
        # Use full set as fallback
        best_overall_A = set(range(0, 129))
    
    # Final check: ensure coverage
    m, u = evaluate(best_overall_A)
    if u > 0:
        # Greedily fix coverage
        best_overall_A = greedy_cover(best_overall_A)
    
    return sorted(best_overall_A)
