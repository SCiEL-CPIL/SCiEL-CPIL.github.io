"""Combined: fast greedy + insertion of excluded numbers + bounded swap local search."""
import time

def _all_products(values):
    prods = set()
    n = len(values)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= values[j]
            prods.add(p)
    return prods

def _can_append(values, products, x):
    """Check if x can be appended at the end (all blocks ending at x)."""
    prod = x
    if prod in products:
        return False
    for a in reversed(values):
        prod *= a
        if prod in products:
            return False
    return True

def _can_insert(values, products, x):
    """Check if x can be inserted at its sorted position in values."""
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    pos = lo
    if pos < len(values) and values[pos] == x:
        return False

    n = len(values)
    # Left suffix products: product of values[i..pos-1]
    left = [1] * pos
    p = 1
    for i in range(pos - 1, -1, -1):
        p *= values[i]
        left[i] = p
    # Right prefix products: product of values[pos..j]
    right = [1] * (n - pos)
    p = 1
    for j in range(pos, n):
        p *= values[j]
        right[j - pos] = p

    # All new block products that include x
    # [x]
    if x in products:
        return False
    # [values[i..pos-1], x] = left[i] * x
    # [x, values[pos..j]] = x * right[j]
    # [values[i..pos-1], x, values[pos..j]] = left[i] * x * right[j]
    for i in range(pos):
        lx = left[i] * x
        if lx in products:
            return False
        for j in range(n - pos):
            if lx * right[j] in products:
                return False
    for j in range(n - pos):
        if x * right[j] in products:
            return False
    return True

def _try_insert(values, products, x):
    """Insert x if possible; return (success, new_values, new_products)."""
    if not _can_insert(values, products, x):
        return False, values, products
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    pos = lo
    new_values = values[:pos] + [x] + values[pos:]
    new_products = _all_products(new_values)
    return True, new_values, new_products

def _try_swap(values, products, i, j, x):
    """Try swapping values[i], values[j] and inserting x at sorted position.
    Return (success, new_values, new_products) or (False, None, None)."""
    # Build candidate: remove values[i] and values[j], insert x, keep others sorted
    cand = values[:i] + values[i+1:j] + values[j+1:]
    # Insert x at sorted position
    lo, hi = 0, len(cand)
    while lo < hi:
        mid = (lo + hi) // 2
        if cand[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    pos = lo
    if pos < len(cand) and cand[pos] == x:
        return False, None, None
    new_values = cand[:pos] + [x] + cand[pos:]
    # Must be strictly increasing (it will be since sorted)
    # Check all products distinct
    new_products = _all_products(new_values)
    if len(new_products) != len(new_values) * (len(new_values) + 1) // 2:
        return False, None, None
    return True, new_values, new_products

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    # Phase 1: Greedy construction (fast)
    values, products = [], set()
    for x in range(2, 129):
        if time.time() > deadline:
            break
        prod = x
        ok = True
        if prod in products:
            ok = False
        else:
            for a in reversed(values):
                prod *= a
                if prod in products:
                    ok = False
                    break
        if ok:
            values.append(x)
            products.update([x])
            p = x
            for a in reversed(values[:-1]):
                p *= a
                products.add(p)

    base_len = len(values)

    # Phase 2: Try to insert excluded numbers (repeated passes)
    excluded = [x for x in range(2, 129) if x not in set(values)]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in excluded[:]:
            if time.time() >= deadline:
                break
            ok, new_values, new_products = _try_insert(values, products, x)
            if ok:
                values = new_values
                products = new_products
                excluded.remove(x)
                improved = True

    # Phase 3: Bounded swap-based local search
    # Try removing one element and inserting an excluded one, or swapping
    # We focus on: for each excluded x, try inserting it after removing one element
    # that conflicts, to see if we can grow the sequence.
    excluded = [x for x in range(2, 129) if x not in set(values)]
    if excluded and time.time() < deadline:
        # For each excluded x, try each possible removal of one element from values
        # then insert x. Accept if length grows or stays same but we find a better config.
        # Since we want max length, only accept if length grows.
        for x in excluded[:]:
            if time.time() >= deadline:
                break
            for r in range(len(values)):
                if time.time() >= deadline:
                    break
                # Remove values[r], insert x
                cand = values[:r] + values[r+1:]
                # Insert x at sorted position
                lo, hi = 0, len(cand)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if cand[mid] < x:
                        lo = mid + 1
                    else:
                        hi = mid
                pos = lo
                if pos < len(cand) and cand[pos] == x:
                    continue
                new_values = cand[:pos] + [x] + cand[pos:]
                new_products = _all_products(new_values)
                if len(new_products) == len(new_values) * (len(new_values) + 1) // 2:
                    # Valid, and length is same (removed 1, added 1)
                    # Accept if it helps future insertions? For now, accept to diversify.
                    # Actually, same length - only accept if it enables further growth.
                    # Keep it simple: accept if it's valid (may help later insertions).
                    values = new_values
                    products = new_products
                    excluded = [xx for xx in range(2, 129) if xx not in set(values)]
                    break

    return values
