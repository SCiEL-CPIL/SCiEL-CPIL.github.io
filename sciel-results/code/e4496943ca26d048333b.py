import time
import random
from fractions import Fraction

def construct():
    n = 256
    start_time = time.time()
    # Safety margin: The evaluator kills at 90s, but the task contract says 60s.
    # We aim to finish well within 60s. Let's use 55s as a hard deadline for search.
    deadline = start_time + 55.0
    
    # Precompute reciprocals for float sum
    recip = [0.0] * (n + 1)
    for i in range(1, n + 1):
        recip[i] = 1.0 / i

    # Precompute conflict pairs
    # conflict_pairs[x] contains pairs (y, z) such that {x, y, z} is a 3-AP.
    # If y and z are in the set, x cannot be added.
    conflict_pairs = [[] for _ in range(n + 1)]
    
    # Generate all 3-APs (a, b, c) with a < b < c and a + c = 2b
    for b in range(1, n + 1):
        # b is the middle term
        # a = b - d, c = b + d
        d = 1
        while b - d >= 1 and b + d <= n:
            a = b - d
            c = b + d
            # a conflicts with (b, c)
            conflict_pairs[a].append((b, c))
            # b conflicts with (a, c)
            conflict_pairs[b].append((a, c))
            # c conflicts with (a, b)
            conflict_pairs[c].append((a, b))
            d += 1

    def can_add(x, mask):
        """Check if x can be added to the set represented by mask without forming a 3-AP."""
        for y, z in conflict_pairs[x]:
            if ((mask >> y) & 1) and ((mask >> z) & 1):
                return False
        return True

    def get_sum_float(mask):
        """Compute sum of reciprocals using float for speed."""
        total = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            total += recip[idx]
            m ^= lsb
        return total

    def get_exact_sum(mask):
        """Compute exact sum using Fraction for high-precision comparison."""
        total = Fraction(0)
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            total += Fraction(1, idx)
            m ^= lsb
        return total

    def greedy_fill(mask, order=None):
        """Fill in elements greedily in the given order (or natural order 1..n)."""
        if order is None:
            order = list(range(1, n + 1))
        for x in order:
            if not ((mask >> x) & 1):
                if can_add(x, mask):
                    mask |= (1 << x)
        return mask

    def local_search(mask, use_exact=False, max_iterations=50):
        """
        Local search: try remove-and-refill and swap moves.
        Returns the improved mask.
        """
        improved = True
        current_sum = get_exact_sum(mask) if use_exact else get_sum_float(mask)
        iters = 0
        
        while improved and iters < max_iterations and time.time() < deadline - 3.0:
            improved = False
            iters += 1
            
            # 1. Try remove-and-refill
            # For each element in the set, remove it, then greedily fill the rest.
            # This can sometimes unlock better configurations.
            for rem in range(1, n + 1):
                if not ((mask >> rem) & 1):
                    continue
                
                trial_mask = mask & ~(1 << rem)
                # Greedy fill in natural order is a good heuristic for filling gaps
                trial_mask = greedy_fill(trial_mask)
                
                if use_exact:
                    new_sum = get_exact_sum(trial_mask)
                else:
                    new_sum = get_sum_float(trial_mask)
                
                # Use a small epsilon for float, exact comparison for Fraction
                if new_sum > current_sum + (Fraction(0, 1) if use_exact else 1e-12):
                    mask = trial_mask
                    current_sum = new_sum
                    improved = True
                    break
            
            # 2. Try swap: remove one element, add another
            if not improved:
                elements_in_mask = []
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    elements_in_mask.append(idx)
                    m ^= lsb
                
                for rem in elements_in_mask:
                    if time.time() > deadline - 3.0:
                        break
                    trial_mask = mask & ~(1 << rem)
                    # Try adding each non-present element
                    for add in range(1, n + 1):
                        if (trial_mask >> add) & 1:
                            continue
                        if can_add(add, trial_mask):
                            trial2 = trial_mask | (1 << add)
                            if use_exact:
                                new_sum = get_exact_sum(trial2)
                            else:
                                new_sum = get_sum_float(trial2)
                            
                            if new_sum > current_sum + (Fraction(0, 1) if use_exact else 1e-12):
                                mask = trial2
                                current_sum = new_sum
                                improved = True
                                break
                    if improved:
                        break
        
        return mask

    # 1. Initial Greedy (Natural Order)
    best_mask = greedy_fill(0)
    best_sum = get_sum_float(best_mask)
    
    # 2. Local Search on Initial
    improved_mask = local_search(best_mask)
    if get_sum_float(improved_mask) > best_sum:
        best_mask = improved_mask
        best_sum = get_sum_float(best_mask)
        
    # 3. Randomized Restarts with different orderings
    # This is the key improvement from Inspiration 1.
    random.seed(42)
    
    # Try a few full random restarts first to find a good basin
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
            
        order = list(range(1, n + 1))
        random.shuffle(order)
        m = greedy_fill(0, order)
        m = local_search(m)
        s = get_sum_float(m)
        if s > best_sum + 1e-12:
            best_sum = s
            best_mask = m

    # 4. Perturbation-based local search from the best solution
    # Remove a few random elements, refill, and local search.
    while time.time() < deadline - 10.0:
        perturbed_mask = best_mask
        
        # Remove 1-3 random elements to escape local optimum
        num_removals = random.randint(1, 3)
        elements_in_set = []
        m = perturbed_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            elements_in_set.append(idx)
            m ^= lsb
            
        if not elements_in_set:
            break
            
        for _ in range(num_removals):
            if not elements_in_set:
                break
            idx = random.randint(0, len(elements_in_set) - 1)
            rem_val = elements_in_set[idx]
            perturbed_mask &= ~(1 << rem_val)
            elements_in_set.pop(idx)
        
        # Refill with natural order (or random, but natural is often better for reciprocal weight)
        # Let's try natural order refill first, as smaller numbers have higher weight.
        perturbed_mask = greedy_fill(perturbed_mask)
        
        improved_mask = local_search(perturbed_mask)
        
        new_sum = get_sum_float(improved_mask)
        if new_sum > best_sum + 1e-12:
            best_sum = new_sum
            best_mask = improved_mask

    # 5. High-precision polish pass with exact arithmetic
    # This ensures we don't miss tiny improvements due to floating point errors
    if time.time() < deadline - 2.0:
        polished_mask = local_search(best_mask, use_exact=True, max_iterations=20)
        exact_best = get_exact_sum(best_mask)
        exact_polished = get_exact_sum(polished_mask)
        if exact_polished > exact_best:
            best_mask = polished_mask

    # Extract result
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx)
        m ^= lsb
        
    # Validation
    s_set = set(result)
    for i, a in enumerate(result):
        for b in result[i+1:]:
            if (a + b) % 2 == 0:
                mid = (a + b) // 2
                if mid != a and mid != b and mid in s_set:
                    raise AssertionError("3-AP found")
                    
    return result
