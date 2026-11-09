"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 256

# EVOLVE-BLOCK-START

def _build_sums(selected):
    """Build the set of all pairwise sums (including diagonals) for a Sidon set."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    return sums

def _can_add_fast(x, selected, sums):
    """Check if x can be added while maintaining Sidon property."""
    # Check x+x against existing sums
    if 2 * x in sums:
        return False
    # Check x+a against existing sums for all a in selected
    for a in selected:
        if (x + a) in sums:
            return False
    return True

def _is_maximal(selected, n=N):
    """Check if Sidon set is inclusion-maximal."""
    selected_set = set(selected)
    sums = _build_sums(selected)
    
    for x in range(1, n + 1):
        if x in selected_set:
            continue
        # Check if x can be added
        if 2 * x not in sums:
            conflict = False
            for a in selected:
                if (x + a) in sums:
                    conflict = True
                    break
            if not conflict:
                return False
    return True

def _greedy_maximal(order, n=N):
    """Build a maximal Sidon set by greedily adding elements in given order."""
    selected = []
    sums = set()
    
    for x in order:
        if x < 1 or x > n:
            continue
        if 2 * x not in sums:
            ok = True
            for a in selected:
                if (x + a) in sums:
                    ok = False
                    break
            if ok:
                selected.append(x)
                for a in selected:
                    sums.add(x + a)
                sums.add(2 * x)
    
    return selected, sums

def _try_all_removals(selected, n=N, deadline=None):
    """
    Try removing each element one at a time.
    Returns a smaller maximal set if found, else None.
    """
    for i in range(len(selected)):
        if deadline is not None and time.time() > deadline:
            break
        candidate = selected[:i] + selected[i+1:]
        if _is_maximal(candidate, n):
            return candidate
    return None

def _try_swaps(selected, n=N, deadline=None, rng=None):
    """
    Try swapping elements to find a smaller maximal set.
    Strategy: remove element i, try adding each possible replacement.
    """
    if rng is None:
        rng = random.Random()
    
    # Try each element as the one to remove
    for i in range(len(selected)):
        if deadline is not None and time.time() > deadline:
            break
        
        removed = selected[i]
        temp = selected[:i] + selected[i+1:]
        temp_sums = _build_sums(temp)
        temp_set = set(temp)
        
        # Find all elements that can be added to temp
        candidates = []
        for x in range(1, n + 1):
            if x in temp_set:
                continue
            if 2 * x not in temp_sums:
                ok = True
                for a in temp:
                    if (x + a) in temp_sums:
                        ok = False
                        break
                if ok:
                    candidates.append(x)
        
        # Try each candidate
        for x in candidates:
            if deadline is not None and time.time() > deadline:
                break
            candidate = temp + [x]
            if _is_maximal(candidate, n):
                return candidate
    
    return None

def _local_search(selected, n=N, deadline=None):
    """
    Local search: iteratively try removals and swaps to minimize size.
    """
    current = list(selected)
    rng = random.Random(12345)
    
    while time.time() < deadline - 0.5:
        improved = False
        
        # First, try all single removals
        smaller = _try_all_removals(current, n, deadline)
        if smaller is not None:
            current = smaller
            improved = True
            continue
        
        # Try swaps
        swapped = _try_swaps(current, n, deadline, rng)
        if swapped is not None:
            current = swapped
            improved = True
            continue
        
        # No improvement found; break
        break
    
    return current

def _generate_orders(seed, count=30, n=N):
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

def construct():
    """Construct a small inclusion-maximal Sidon set in [1,256]."""
    start_time = time.time()
    deadline = start_time + 27.0  # Safety margin before 30s kill
    
    best = None
    best_len = float('inf')
    
    # Multi-start with diverse initial orderings
    orders = _generate_orders(seed=42, count=40, n=N)
    
    for idx, order in enumerate(orders):
        if time.time() > deadline - 3.0:
            break
        
        # Build initial maximal Sidon set
        maximal_set, _ = _greedy_maximal(order, N)
        
        if len(maximal_set) < best_len:
            best_len = len(maximal_set)
            best = maximal_set[:]
        
        # Run local search from this starting point
        if time.time() < deadline - 1.5:
            searched = _local_search(maximal_set, N, deadline - 0.5)
            if len(searched) < best_len:
                best_len = len(searched)
                best = searched[:]
        
        # If we found a very small set, do extra polish
        if best_len <= 12 and time.time() < deadline - 2.0:
            polished = _local_search(best, N, deadline - 0.5)
            if len(polished) < best_len:
                best_len = len(polished)
                best = polished[:]
    
    # Final aggressive polish on the best solution
    if best is not None and time.time() < deadline - 0.5:
        final = _local_search(best, N, deadline - 0.1)
        if len(final) < len(best):
            best = final
    
    # Fallback
    if best is None or len(best) == 0:
        order = list(range(1, N + 1))
        random.shuffle(order)
        best, _ = _greedy_maximal(order, N)
    
    return best

# EVOLVE-BLOCK-END
