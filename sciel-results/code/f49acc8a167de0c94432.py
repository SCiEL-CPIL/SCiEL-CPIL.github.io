"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
# Prime exponent vectors over primes up to 128. A consecutive-block product is
# uniquely identified by the sum of the exponent vectors of its members, so
# distinctness of all block products == distinctness of these sums.
_PRIMES = []
for _n in range(2, 129):
    _d = _n
    _p = 2
    while _p * _p <= _d:
        if _d % _p == 0:
            _PRIMES.append(_p)
            while _d % _p == 0:
                _d //= _p
        _p += 1
    if _d > 1:
        _PRIMES.append(_d)
_PRIMES = sorted(set(_PRIMES))

_EXP = {}
for _x in range(2, 129):
    _d = _x
    _vec = []
    for _p in _PRIMES:
        if _p * _p > _d:
            break
        _c = 0
        while _d % _p == 0:
            _d //= _p
            _c += 1
        _vec.append(_c)
    if _d > 1:
        _vec.append(1)
    _EXP[_x] = tuple(_vec)


def _all_products(values):
    """Return set of all nonempty consecutive block products (integer products)."""
    n = len(values)
    prods = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= values[j]
            prods.add(prod)
    return prods


def _can_add(values, products, x):
    """Check if x can be appended to values (strictly increasing, no product collision)."""
    if values and x <= values[-1]:
        return False
    tails = [x]
    prod = x
    for a in reversed(values):
        prod *= a
        tails.append(prod)
    if products.intersection(tails):
        return False
    return True


def _greedy_pass(values, products, start_from=2):
    """Continue greedy from current state."""
    values = list(values)
    products = set(products)
    for x in range(start_from, 129):
        if _can_add(values, products, x):
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            values.append(x)
            products.update(tails)
    return values, products


def _local_search(values, products, deadline):
    """Local search with remove-reinsert, insert-new, swap, and replace moves."""
    values = list(values)
    best = len(values)
    best_values = list(values)
    
    def _is_valid(vals):
        if not vals:
            return True
        if vals[0] < 2 or vals[-1] > 128:
            return False
        for i in range(1, len(vals)):
            if vals[i] <= vals[i-1]:
                return False
        prods = _all_products(vals)
        return len(prods) == len(vals) * (len(vals) + 1) // 2
    
    for _ in range(6):  # Multiple passes
        improved = False
        n = len(values)
        
        # Remove-and-reinsert: drop one element, re-greedy the whole range.
        for i in range(n):
            if time.time() > deadline:
                break
            vals_without = values[:i] + values[i+1:]
            prods_without = _all_products(vals_without)
            new_vals, new_prods = _greedy_pass(vals_without, prods_without, start_from=2)
            if len(new_vals) > best:
                best = len(new_vals)
                best_values = list(new_vals)
                values = list(new_vals)
                n = len(values)
                improved = True
        
        # Insert-new: try inserting a fresh value at each gap.
        used_set = set(values)
        for i in range(len(values) + 1):
            if time.time() > deadline:
                break
            lo = (values[i-1] + 1) if i > 0 else 2
            hi = (values[i] - 1) if i < len(values) else 128
            for v in range(lo, hi + 1):
                if v in used_set:
                    continue
                test_vals = values[:i] + [v] + values[i:]
                test_prods = _all_products(test_vals)
                if len(test_prods) == len(test_vals) * (len(test_vals) + 1) // 2:
                    new_vals, new_prods = _greedy_pass(test_vals, test_prods, start_from=128)
                    if len(new_vals) > best:
                        best = len(new_vals)
                        best_values = list(new_vals)
                        values = list(new_vals)
                        used_set = set(values)
                        improved = True
        
        # Swap moves: try swapping adjacent or nearby elements
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, min(i + 4, n)):  # Swap with next 3 elements
                test_vals = values.copy()
                test_vals[i], test_vals[j] = test_vals[j], test_vals[i]
                # Check if still strictly increasing
                valid = True
                for k in range(1, len(test_vals)):
                    if test_vals[k] <= test_vals[k-1]:
                        valid = False
                        break
                if valid and _is_valid(test_vals):
                    new_vals, new_prods = _greedy_pass(test_vals, _all_products(test_vals), start_from=128)
                    if len(new_vals) > best:
                        best = len(new_vals)
                        best_values = list(new_vals)
                        values = list(new_vals)
                        n = len(values)
                        improved = True
        
        # Replace moves: try replacing one element with a different value
        for i in range(n):
            if time.time() > deadline:
                break
            lo = (values[i-1] + 1) if i > 0 else 2
            hi = (values[i+1] - 1) if i < n - 1 else 128
            for v in range(lo, hi + 1):
                if v == values[i]:
                    continue
                test_vals = values.copy()
                test_vals[i] = v
                if _is_valid(test_vals):
                    new_vals, new_prods = _greedy_pass(test_vals, _all_products(test_vals), start_from=128)
                    if len(new_vals) > best:
                        best = len(new_vals)
                        best_values = list(new_vals)
                        values = list(new_vals)
                        n = len(values)
                        improved = True
                        break
        
        if not improved:
            break
    
    return best_values


def construct():
    deadline = time.time() + 55.0  # Safety margin before the 60s kill

    # Strategy 1: Standard greedy
    values1, products1 = [], set()
    for x in range(2, 129):
        if _can_add(values1, products1, x):
            tails = [x]
            prod = x
            for a in reversed(values1):
                prod *= a
                tails.append(prod)
            values1.append(x)
            products1.update(tails)
    best_values = list(values1)
    best_len = len(values1)

    # Strategy 2: Greedy with different starting points (skip some small values)
    for skip_start in range(2, 10):
        if time.time() > deadline:
            break
        values_try, products_try = [], set()
        for x in range(2, 129):
            if x < skip_start and x != 2:
                continue
            if _can_add(values_try, products_try, x):
                tails = [x]
                prod = x
                for a in reversed(values_try):
                    prod *= a
                    tails.append(prod)
                values_try.append(x)
                products_try.update(tails)
        if len(values_try) > best_len:
            best_len = len(values_try)
            best_values = list(values_try)

    # Strategy 3: Randomized multi-start greedy with diverse strategies
    random.seed(42)
    for trial in range(40):
        if time.time() > deadline:
            break
        values_try, products_try = [], set()
        
        # Different random strategies
        if trial % 4 == 0:
            # Shuffle and sort
            candidates = list(range(2, 129))
            random.shuffle(candidates)
            candidates = sorted(candidates)
        elif trial % 4 == 1:
            # Random subset, then fill gaps greedily
            candidates = random.sample(range(2, 129), 60)
            candidates = sorted(candidates)
        elif trial % 4 == 2:
            # Weighted random: favor smaller numbers
            candidates = list(range(2, 129))
            weights = [1.0 / x for x in candidates]
            random.seed(trial)
            selected = []
            remaining = list(range(2, 129))
            for _ in range(65):
                if not remaining:
                    break
                total_w = sum(1.0 / x for x in remaining)
                r = random.random() * total_w
                cum = 0
                for idx, x in enumerate(remaining):
                    cum += 1.0 / x
                    if cum >= r:
                        selected.append(x)
                        remaining.pop(idx)
                        break
            candidates = sorted(selected)
        else:
            # Random start point, then greedy
            start = random.randint(2, 20)
            candidates = list(range(start, 129))
            random.shuffle(candidates[:10])  # Shuffle first 10
            candidates = sorted(candidates)
        
        for x in candidates:
            if _can_add(values_try, products_try, x):
                tails = [x]
                prod = x
                for a in reversed(values_try):
                    prod *= a
                    tails.append(prod)
                values_try.append(x)
                products_try.update(tails)
        if len(values_try) > best_len:
            best_len = len(values_try)
            best_values = list(values_try)

    # Apply local search to the best found
    if time.time() < deadline:
        products_best = _all_products(best_values)
        best_values = _local_search(best_values, products_best, deadline)
    
    # Final verification and cleanup
    # Ensure the result is valid
    final_prods = _all_products(best_values)
    if len(final_prods) != len(best_values) * (len(best_values) + 1) // 2:
        # Fallback to a known valid solution
        values_fallback, products_fallback = [], set()
        for x in range(2, 129):
            if _can_add(values_fallback, products_fallback, x):
                tails = [x]
                prod = x
                for a in reversed(values_fallback):
                    prod *= a
                    tails.append(prod)
                values_fallback.append(x)
                products_fallback.update(tails)
        best_values = values_fallback
    
    return best_values
# EVOLVE-BLOCK-END
