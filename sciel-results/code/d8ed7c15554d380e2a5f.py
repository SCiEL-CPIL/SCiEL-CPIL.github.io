import time
import random
from fractions import Fraction
from itertools import combinations

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute all 3-APs
    ap_triples = []
    for a in range(1, n + 1):
        for d in range(1, n - a + 1):
            b = a + d
            c = a + 2 * d
            if c <= n:
                ap_triples.append((a, b, c))

    # Precompute for each element x, the set of pairs (i, j) such that
    # x being present would create a 3-AP with i and j both present
    # i.e., we need to check if (i,j) are both in the set when trying to add x
    # For each x, collect all (i, j) pairs where {x, i, j} forms a 3-AP
    conflicts_with = [[] for _ in range(n + 1)]
    for (a, b, c) in ap_triples:
        # If we add a, need to check if b and c are both present
        conflicts_with[a].append((b, c))
        # If we add b, need to check if a and c are both present
        conflicts_with[b].append((a, c))
        # If we add c, need to check if a and b are both present
        conflicts_with[c].append((a, b))

    # Reciprocals
    recip = [0.0] * (n + 1)
    for i in range(1, n + 1):
        recip[i] = 1.0 / i

    def can_add(x, mask):
        """Check if x can be added to mask without creating a 3-AP."""
        for i, j in conflicts_with[x]:
            if ((mask >> i) & 1) and ((mask >> j) & 1):
                return False
        return True

    def get_sum_float(mask):
        total = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            total += recip[idx]
            m ^= lsb
        return total

    def get_sum_frac(mask):
        total = Fraction(0)
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            total += Fraction(1, idx)
            m ^= lsb
        return total

    def greedy_fill(mask):
        """Try to add elements 1,2,...,n in order if possible."""
        for x in range(1, n + 1):
            if not ((mask >> x) & 1):
                if can_add(x, mask):
                    mask |= (1 << x)
        return mask

    def greedy_fill_random(mask, rng):
        """Try to add elements in random order if possible."""
        order = list(range(1, n + 1))
        rng.shuffle(order)
        for x in order:
            if not ((mask >> x) & 1):
                if can_add(x, mask):
                    mask |= (1 << x)
        return mask

    def extract_elements(mask):
        """Return list of elements in mask."""
        elems = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            elems.append(idx)
            m ^= lsb
        return elems

    def local_search(mask, use_exact=False, max_iterations=100):
        """Local search via remove-and-refill and swaps."""
        improved = True
        iters = 0
        current_sum = get_sum_frac(mask) if use_exact else get_sum_float(mask)
        
        while improved and iters < max_iterations and time.time() < deadline - 5.0:
            improved = False
            iters += 1

            # Try remove-and-refill: remove element r, then greedily fill
            elements = extract_elements(mask)
            for r in elements:
                if time.time() > deadline - 5.0:
                    break
                trial = mask & ~(1 << r)
                trial = greedy_fill(trial)
                
                if use_exact:
                    new_sum = get_sum_frac(trial)
                    eps = Fraction(0, 1)
                else:
                    new_sum = get_sum_float(trial)
                    eps = 1e-15
                
                if new_sum > current_sum + eps:
                    mask = trial
                    current_sum = new_sum
                    improved = True
                    break

            # Try swap: remove r, add a (single element change)
            if not improved:
                elements = extract_elements(mask)
                for r in elements:
                    if time.time() > deadline - 5.0:
                        break
                    trial_base = mask & ~(1 << r)
                    for a in range(1, n + 1):
                        if (trial_base >> a) & 1:
                            continue
                        if not can_add(a, trial_base):
                            continue
                        trial = trial_base | (1 << a)
                        
                        if use_exact:
                            new_sum = get_sum_frac(trial)
                            eps = Fraction(0, 1)
                        else:
                            new_sum = get_sum_float(trial)
                            eps = 1e-15
                        
                        if new_sum > current_sum + eps:
                            mask = trial
                            current_sum = new_sum
                            improved = True
                            break
                    if improved:
                        break

            # Try 2-remove-2-add moves
            if not improved and len(elements) >= 2:
                found_swap = False
                for r1, r2 in combinations(elements, 2):
                    if time.time() > deadline - 5.0:
                        break
                    trial_base = mask & ~(1 << r1) & ~(1 << r2)
                    # Try adding pairs (a, b)
                    for a in range(1, n + 1):
                        if time.time() > deadline - 5.0:
                            break
                        if (trial_base >> a) & 1:
                            continue
                        if not can_add(a, trial_base):
                            continue
                        trial_with_a = trial_base | (1 << a)
                        for b in range(a + 1, n + 1):
                            if (trial_with_a >> b) & 1:
                                continue
                            if not can_add(b, trial_with_a):
                                continue
                            trial = trial_with_a | (1 << b)
                            
                            if use_exact:
                                new_sum = get_sum_frac(trial)
                                eps = Fraction(0, 1)
                            else:
                                new_sum = get_sum_float(trial)
                                eps = 1e-15
                            
                            if new_sum > current_sum + eps:
                                mask = trial
                                current_sum = new_sum
                                improved = True
                                found_swap = True
                                break
                        if found_swap:
                            break
                    if found_swap:
                        break

        return mask

    # Phase 1: Multiple randomized greedy initializations
    rng = random.Random(42)
    best_mask = 0
    best_sum = 0.0

    # Standard greedy
    m = greedy_fill(0)
    s = get_sum_float(m)
    if s > best_sum:
        best_sum = s
        best_mask = m

    # Multiple random-order greedy fills
    for _ in range(50):
        if time.time() > deadline - 60.0:
            break
        m = greedy_fill_random(0, rng)
        s = get_sum_float(m)
        if s > best_sum:
            best_sum = s
            best_mask = m

    # Phase 2: Local search on best
    improved_mask = local_search(best_mask, use_exact=False, max_iterations=200)
    imp_sum = get_sum_float(improved_mask)
    if imp_sum > best_sum + 1e-15:
        best_mask = improved_mask
        best_sum = imp_sum

    # Phase 3: Simulated annealing / randomized restarts
    best_frac = get_sum_frac(best_mask)

    while time.time() < deadline - 10.0:
        # Perturb: randomly remove 2-5 elements
        elems = extract_elements(best_mask)
        if len(elems) < 3:
            break
        k = rng.randint(2, min(5, len(elems)))
        removed = set(rng.sample(elems, k))
        perturbed = best_mask
        for r in removed:
            perturbed &= ~(1 << r)

        # Greedy refill with random order
        perturbed = greedy_fill_random(perturbed, rng)

        # Local search
        candidate = local_search(perturbed, use_exact=False, max_iterations=100)
        cand_sum = get_sum_float(candidate)

        if cand_sum > best_sum + 1e-15:
            best_sum = cand_sum
            best_mask = candidate
            best_frac = get_sum_frac(best_mask)

    # Phase 4: Final exact-arithmetic polish
    if time.time() < deadline - 3.0:
        polished = local_search(best_mask, use_exact=True, max_iterations=50)
        polished_frac = get_sum_frac(polished)
        if polished_frac > best_frac:
            best_mask = polished
            best_frac = polished_frac

    # Extract result
    result = extract_elements(best_mask)
    result.sort()

    # Validation
    s_set = set(result)
    for a in result:
        for b in result:
            if a >= b:
                continue
            c = 2 * b - a
            if c > b and c in s_set:
                raise AssertionError(f"3-AP found: {a}, {b}, {c}")
            # Also check a is middle
            if a < b < c:
                pass

    # Re-validate properly
    s_set = set(result)
    found_ap = False
    for a in range(1, n + 1):
        for d in range(1, n // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c <= n and a in s_set and b in s_set and c in s_set:
                found_ap = True
                break
        if found_ap:
            break
    if found_ap:
        raise AssertionError("3-AP found in result")

    if not result:
        result = [1]

    return result
