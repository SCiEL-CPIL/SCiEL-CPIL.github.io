"""Combined program: structural greedy construction + aggressive local search refinement."""
import time
import random
import math

N = 256
DEADLINE = time.time() + 25.0  # safety margin before 30s kill

def _is_valid_addition(x, selected, sums):
    """Check if x can be added to the current Sidon set."""
    # Check 2x against existing sums
    if 2 * x in sums:
        return False
    # Check x+a against existing sums for all a in selected
    for a in selected:
        if (x + a) in sums:
            return False
    return True

def _greedy_construct(order):
    """Greedy: pick the first valid element from the given order."""
    selected = []
    sums = set()
    selected_set = set()
    
    for x in order:
        if time.time() > DEADLINE:
            break
        if x in selected_set:
            continue
        if _is_valid_addition(x, selected, sums):
            selected.append(x)
            selected_set.add(x)
            for a in selected[:-1]:
                sums.add(x + a)
            sums.add(2 * x)
    
    # Check maximality
    all_sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            all_sums.add(selected[i] + selected[j])
    
    is_maximal = True
    for y in range(1, N + 1):
        if y in selected_set:
            continue
        can_add = True
        for a in selected:
            if y + a in all_sums:
                can_add = False
                break
        if can_add and 2 * y in all_sums:
            can_add = False
        if can_add:
            is_maximal = False
            break
    
    if is_maximal:
        return selected
    return None

def _is_maximal_fast(S, sums):
    """Check maximality: every missing element must be blocked."""
    S_set = set(S)
    S_list = list(S)
    n = len(S_list)
    for x in range(1, N+1):
        if x in S_set:
            continue
        if 2*x in sums:
            continue
        # Check if any s in S has x+s in sums
        blocked = False
        for s in S_list:
            if x + s in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _get_sums(S):
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
    return sums

def _reduce(S, deadline=None):
    """Try to remove elements while maintaining maximality."""
    S = sorted(S)
    changed = True
    while changed:
        changed = False
        if deadline and time.time() > deadline:
            break
        for i in range(len(S)):
            if deadline and time.time() > deadline:
                break
            candidate = S[:i] + S[i+1:]
            if not candidate:
                continue
            sums = _get_sums(candidate)
            if _is_maximal_fast(candidate, sums):
                S = candidate
                changed = True
                break
    return S

def _local_search_swap(S, sums, deadline):
    """Try swap moves: remove one element, add another, check if still maximal and smaller or same size."""
    S = sorted(S)
    S_set = set(S)
    improved = False
    for i in range(len(S)):
        if time.time() > deadline:
            break
        # Try removing S[i]
        S_without = S[:i] + S[i+1:]
        if not S_without:
            continue
        sums_without = _get_sums(S_without)
        S_without_set = set(S_without)
        
        # Now try adding each missing element
        for x in range(1, N+1):
            if x in S_without_set:
                continue
            # Check if x can be added to S_without
            if not _is_valid_addition(x, S_without, sums_without):
                continue
            # S_without + [x] is a valid Sidon set of same size
            # Check if it's maximal
            new_S = S_without + [x]
            new_S.sort()
            new_sums = sums_without | set([x + a for a in S_without] + [2*x])
            if _is_maximal_fast(new_S, new_sums):
                return new_S
    return None

def _generate_orders(seed, count=50, n=N):
    """Generate diverse ordering strategies."""
    orders = []
    rng = random.Random(seed)
    
    # Pure random
    for _ in range(count // 2):
        order = list(range(1, n + 1))
        rng.shuffle(order)
        orders.append(order)
    
    # Arithmetic progressions with random fill
    for step in [2, 3, 4, 5, 7, 11, 13, 17, 19, 23]:
        if len(orders) >= count:
            break
        for base in range(1, min(step, 4) + 1):
            if len(orders) >= count:
                break
            order = [x for x in range(base, n + 1, step)]
            remaining = [x for x in range(1, n + 1) if x not in set(order)]
            rng.shuffle(remaining)
            order.extend(remaining)
            orders.append(order)
    
    # Quadratic residue-like orderings (known good for Sidon sets)
    if len(orders) < count:
        # Use x^2 mod p style ordering for small p
        for p in [17, 19, 23, 29, 31, 37, 41, 43]:
            if len(orders) >= count:
                break
            residues = set()
            for x in range(p):
                residues.add((x * x) % p)
            # Map residues to [1, n]
            order = []
            for r in sorted(residues):
                for k in range(1, n // p + 2):
                    val = r * p + k
                    if 1 <= val <= n:
                        order.append(val)
            remaining = [x for x in range(1, n + 1) if x not in set(order)]
            rng.shuffle(remaining)
            order.extend(remaining)
            orders.append(order)
    
    # Descending order
    if len(orders) < count:
        orders.append(list(range(n, 0, -1)))
    
    # Ascending order
    if len(orders) < count:
        orders.append(list(range(1, n + 1)))
    
    return orders[:count]

def _local_search(selected, n=N, deadline=None):
    """
    Local search: iteratively try removals and swaps to minimize size.
    """
    current = list(selected)
    rng = random.Random(12345)
    
    while time.time() < deadline - 0.5:
        improved = False
        
        # First, try all single removals
        smaller = _reduce(current, deadline)
        if len(smaller) < len(current):
            current = smaller
            improved = True
            continue
        
        # Try swaps
        sums = _get_sums(current)
        swapped = _local_search_swap(current, sums, deadline)
        if swapped is not None:
            current = swapped
            improved = True
            continue
        
        # No improvement found; break
        break
    
    return current

def construct():
    """Construct a small inclusion-maximal Sidon set in [1,256]."""
    start_time = time.time()
    best = None
    best_len = float('inf')
    
    # Multi-start with diverse initial orderings
    orders = _generate_orders(seed=42, count=50, n=N)
    
    for idx, order in enumerate(orders):
        if time.time() > DEADLINE - 3.0:
            break
        
        # Build initial maximal Sidon set
        maximal_set = _greedy_construct(order)
        
        if maximal_set is not None and len(maximal_set) < best_len:
            best_len = len(maximal_set)
            best = maximal_set[:]
        
        # Run local search from this starting point
        if time.time() < DEADLINE - 1.5:
            searched = _local_search(maximal_set, N, DEADLINE - 0.5)
            if len(searched) < best_len:
                best_len = len(searched)
                best = searched[:]
        
        # If we found a very small set, do extra polish
        if best_len <= 12 and time.time() < DEADLINE - 2.0:
            polished = _local_search(best, N, DEADLINE - 0.5)
            if len(polished) < best_len:
                best_len = len(polished)
                best = polished[:]
    
    # Final aggressive polish on the best solution
    if best is not None and time.time() < DEADLINE - 0.5:
        final = _local_search(best, N, DEADLINE - 0.1)
        if len(final) < len(best):
            best = final
    
    # Fallback
    if best is None or len(best) == 0:
        order = list(range(1, N + 1))
        random.shuffle(order)
        best = _greedy_construct(order)
        if best is None:
            # Absolute fallback: ascending greedy
            selected = []
            sums = set()
            for x in range(1, N + 1):
                if _is_valid_addition(x, selected, sums):
                    selected.append(x)
                    for a in selected[:-1]:
                        sums.add(x + a)
                    sums.add(2 * x)
            best = selected
    
    return best
