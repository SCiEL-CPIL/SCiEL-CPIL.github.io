"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

_DEADLINE = 87.0  # seconds, safety margin before 90s limit

def _check_x_is_mean_of_subset(x, S):
    """
    Check if x is the mean of any subset of at least 2 elements in S.
    Uses bitset DP on differences with sorted-bounds pruning.
    """
    n = len(S)
    if n < 2:
        return False
    
    zs = [y - x for y in S]
    
    # Calculate max absolute sum for offset
    max_sum = sum(abs(z) for z in zs)
    if max_sum == 0:
        return False  # All equal to x, impossible with distinct elements
        
    offset = max_sum
    mask = (1 << (2 * max_sum + 1)) - 1
    
    # dp[k] is a bitset where bit s is set if sum s-offset is achievable with exactly k elements
    dp = [0] * (n + 1)
    dp[0] = 1 << offset
    
    for z in zs:
        for k in range(n, 0, -1):
            if dp[k-1] == 0:
                continue
            if z >= 0:
                shifted = dp[k-1] << z
            else:
                shifted = dp[k-1] >> (-z)
            
            dp[k] |= (shifted & mask)
            
            # Early exit: if sum 0 is achievable with k >= 2 elements
            if k >= 2 and (dp[k] & (1 << offset)):
                return True
                
    return False

def _check_existing_violation(x, S):
    """
    Check if adding x to S creates any violation where an element a in S
    is the mean of a subset containing x and at least one other element from S.
    """
    n = len(S)
    if n < 2:
        return False
    
    for a in S:
        U = [t for t in S if t != a]
        if len(U) < 1:
            continue
        
        target_val = a - x
        
        ws = [t - a for t in U]
        max_abs = sum(abs(w) for w in ws)
        if max_abs == 0:
            continue
        
        offset = max_abs
        mask = (1 << (2 * max_abs + 1)) - 1
        
        dp = [0] * (len(U) + 1)
        dp[0] = 1 << offset
        
        for w in ws:
            for k in range(len(U), 0, -1):
                if dp[k-1] == 0:
                    continue
                if w >= 0:
                    shifted = dp[k-1] << w
                else:
                    shifted = dp[k-1] >> (-w)
                dp[k] |= (shifted & mask)
        
        if 0 <= offset + target_val <= 2 * max_abs:
            for k in range(1, len(U) + 1):
                if dp[k] & (1 << (offset + target_val)):
                    return True
    
    return False

def _quick_pair_check(x, S):
    """
    Quick check for 2-element subsets involving x.
    """
    n = len(S)
    for i in range(n):
        for j in range(i+1, n):
            if S[i] + S[j] == 2 * x:
                return True
    
    S_set = set(S)
    for a in S:
        b = 2 * a - x
        if b in S_set and b != a:
            return True
    
    return False

def _full_check(C):
    """
    Full verification: check if any element in C is the mean of a subset of 
    at least 2 other elements.
    """
    for i in range(len(C)):
        a = C[i]
        others = C[:i] + C[i+1:]
        if _check_x_is_mean_of_subset(a, others):
            return True
    return False

def _generate_sidon_seed():
    """
    Generate a Sidon sequence (B2 sequence) in [1, 512].
    A Sidon sequence has all pairwise sums distinct.
    We then filter it to remove any pair-mean violations.
    """
    # Greedy Sidon construction
    S = []
    sums = set()
    for x in range(1, 513):
        valid = True
        for a in S:
            s = a + x
            if s in sums:
                valid = False
                break
            sums.add(s)
        if valid:
            S.append(x)
    
    # Filter S to ensure no element is the mean of a pair of others
    # This is a "weakly sum-free" check for pairs
    valid_S = []
    S_set = set(S)
    for x in S:
        # Check if x is mean of a, b in S \ {x}
        # 2x = a + b
        is_mean = False
        for a in S:
            if a == x:
                continue
            b = 2 * x - a
            if b in S_set and b != x and b > a: # b > a to avoid double check? No, just check existence
                # Need to ensure b is in S and b != x
                if b in S_set and b != x:
                    is_mean = True
                    break
        if not is_mean:
            valid_S.append(x)
            
    # Also check if any a in valid_S is mean of pair in valid_S \ {a}
    # This is harder to do incrementally, so we just return valid_S and let the main loop refine
    # Actually, the main loop checks for violations.
    return valid_S

def _sa_count_violations(C):
    """
    Count the number of mean-violation constraints violated in set C.
    Returns (num_violations, list_of_violating_elements).
    """
    n = len(C)
    if n < 3:
        return 0, []
    
    violations = 0
    violating = []
    C_set = set(C)
    
    for i in range(n):
        a = C[i]
        others = [C[j] for j in range(n) if j != i]
        
        # Check if a is mean of any subset of others of size >= 2
        if _check_x_is_mean_of_subset(a, others):
            violations += 1
            violating.append(i)
    
    return violations, violating

def _sa_local_search(C, rng, max_iters=10000):
    """
    Simulated annealing to minimize violations.
    C is a list of selected elements.
    Returns improved set.
    """
    n_target = 48
    current = C[:]
    current_violations, _ = _sa_count_violations(current)
    
    best = current[:]
    best_violations = current_violations
    
    # Temperature schedule
    T0 = 100.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    T = T0
    iters = 0
    
    while T > T_min and iters < max_iters:
        if time.time() > _DEADLINE - 5:
            break
        
        iters += 1
        
        # Generate a move: swap one element in with one out, or just replace one
        move_type = rng.random()
        
        if move_type < 0.7:
            # Replace one element
            if len(current) == 0:
                break
            idx = rng.randint(0, len(current) - 1)
            old_val = current[idx]
            
            # Pick a new value not in current
            candidates = [x for x in range(1, 513) if x not in set(current)]
            if not candidates:
                break
            new_val = rng.choice(candidates)
            
            new_set = current[:]
            new_set[idx] = new_val
            
        elif move_type < 0.9:
            # Swap two elements
            if len(current) < 2:
                break
            i, j = rng.sample(range(len(current)), 2)
            new_set = current[:]
            new_set[i], new_set[j] = new_set[j], new_set[i]
            
            # But we need distinct values, so this only works if they're different
            if new_set[i] == new_set[j]:
                continue
        else:
            # Add or remove an element
            if len(current) < n_target:
                candidates = [x for x in range(1, 513) if x not in set(current)]
                if candidates:
                    new_set = current + [rng.choice(candidates)]
                else:
                    continue
            elif len(current) > 0:
                idx = rng.randint(0, len(current) - 1)
                new_set = current[:idx] + current[idx+1:]
            else:
                continue
        
        # Compute new violations
        new_violations, _ = _sa_count_violations(new_set)
        
        # Accept or reject
        delta = new_violations - current_violations
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            current = new_set
            current_violations = new_violations
            
            if current_violations < best_violations:
                best = current[:]
                best_violations = current_violations
                
                # If we found a valid solution, try to improve cardinality
                if best_violations == 0:
                    # Try to add more elements while maintaining validity
                    for _ in range(50):
                        if time.time() > _DEADLINE - 5:
                            break
                        candidates = [x for x in range(1, 513) if x not in set(best)]
                        if not candidates:
                            break
                        new_val = rng.choice(candidates)
                        test_set = best + [new_val]
                        if len(test_set) > n_target:
                            break
                        v, _ = _sa_count_violations(test_set)
                        if v == 0:
                            best = test_set
                            best_violations = 0
                        else:
                            break
        else:
            # Rejected, keep current
            pass
        
        T *= cooling_rate
    
    return best

def construct():
    start_time = time.time()
    best_set = []
    
    # Candidate pool
    candidates = list(range( 1, 513))
    
    # Generate diverse orders
    orders = []
    
    # 1. Sidon-based seed order
    # We want to prioritize elements that form a good Sidon set
    sidon_seed = _generate_sidon_seed()
    # Create an order that puts sidon_seed first, then the rest
    # We shuffle the sidon_seed a bit to allow different variations?
    # Or just use it as a priority list.
    rest = [x for x in candidates if x not in sidon_seed]
    orders.append(sidon_seed + rest)
    
    # 2. Structured orders
    orders.append(candidates[:])       # Increasing
    orders.append(candidates[::-1])    # Decreasing
    
    # 3. Spread out orders
    spread_order = []
    lo, hi = 1, 512
    while lo <= hi:
        spread_order.append(hi)
        if lo != hi:
            spread_order.append(lo)
        lo += 1
        hi -= 1
    orders.append(spread_order)
    
    # 4. Random orders with different seeds
    rng = random.Random(42)
    num_random_orders = 300
    
    for i in range(num_random_orders):
        if time.time() - start_time > _DEADLINE - 3:
            break
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        orders.append(rand_order)
    
    # 5. Biased random orders: weight towards extremes
    for i in range(100):
        if time.time() - start_time > _DEADLINE - 3:
            break
        biased = []
        for _ in range(512):
            r = rng.random()
            if r < 0.3:
                biased.append(rng.randint(1, 50))
            elif r < 0.6:
                biased.append(rng.randint(463, 512))
            else:
                biased.append(rng.randint(1, 512))
        seen = set()
        unique_biased = []
        for v in biased:
            if v not in seen:
                seen.add(v)
                unique_biased.append(v)
        for v in range(1, 513):
            if v not in seen:
                unique_biased.append(v)
        orders.append(unique_biased)
    
    # Phase 1: Greedy construction
    for order in orders:
        if time.time() - start_time > _DEADLINE - 10:
            break
            
        selected = []
        
        for x in order:
            if time.time() - start_time > _DEADLINE - 10:
                break
            
            if len(selected) >= 48:
                break
                
            if _quick_pair_check(x, selected):
                continue
            
            if _check_x_is_mean_of_subset(x, selected):
                continue
            
            if _check_existing_violation(x, selected):
                continue
            
            selected.append(x)
            
        if len(selected) > len(best_set):
            best_set = selected[:]
            if len(best_set) >= 48:
                break
    
    # Phase 2: Simulated annealing to improve
    if time.time() - start_time < _DEADLINE - 15:
        # Use best_set as starting point, or create a random one
        if len(best_set) >= 20:
            start_set = best_set[:]
        else:
            # Create a random starting set
            rng_sa = random.Random(123)
            start_set = rng_sa.sample(range(1, 513), min(48, len(range(1, 513))))
        
        # Run SA with multiple restarts
        sa_rng = random.Random(789)
        for restart in range(5):
            if time.time() - start_time > _DEADLINE - 10:
                break
            
            # Perturb the starting set for diversity
            if restart > 0:
                perturbed = start_set[:]
                for _ in range(min(10, len(perturbed))):
                    idx = sa_rng.randint(0, len(perturbed) - 1)
                    candidates = [x for x in range(1, 513) if x not in set(perturbed)]
                    if candidates:
                        perturbed[idx] = sa_rng.choice(candidates)
                sa_start = perturbed
            else:
                sa_start = start_set
            
            improved = _sa_local_search(sa_start, sa_rng, max_iters=20000)
            
            # Check if improved is valid
            if not _full_check(improved):
                if len(improved) > len(best_set):
                    best_set = improved[:]
                    if len(best_set) >= 48:
                        break
    
    if best_set:
        if not _full_check(best_set):
            return best_set
    
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
