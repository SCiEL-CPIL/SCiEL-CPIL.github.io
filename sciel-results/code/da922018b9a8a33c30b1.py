import time
import random

N = 512
MAX = N

# Precompute for each x the list of (d) values for each of the 4 positions,
# so the inner loop is just iterating a precomputed tuple.
_pos_last   = [tuple(range(1, (x - 1) // 3 + 1)) for x in range(N + 1)]
_pos_first  = [tuple(range(1, (N - x) // 3 + 1)) for x in range(N + 1)]
_pos_second = [tuple(range(1, min(x - 1, (N - x) // 2) + 1)) for x in range(N + 1)]
_pos_third  = [tuple(range(1, min((x - 1) // 2, N - x) + 1)) for x in range(N + 1)]

def _can_add_bitset(x, bits):
    """Check if x can be added to the set represented by bits (int)."""
    for d in _pos_last[x]:
        if (bits >> (x - d)) & 1 and (bits >> (x - 2 * d)) & 1 and (bits >> (x - 3 * d)) & 1:
            return False
    for d in _pos_first[x]:
        if (bits >> (x + d)) & 1 and (bits >> (x + 2 * d)) & 1 and (bits >> (x + 3 * d)) & 1:
            return False
    for d in _pos_second[x]:
        if (bits >> (x - d)) & 1 and (bits >> (x + d)) & 1 and (bits >> (x + 2 * d)) & 1:
            return False
    for d in _pos_third[x]:
        if (bits >> (x - 2 * d)) & 1 and (bits >> (x - d)) & 1 and (bits >> (x + d)) & 1:
            return False
    return True

def _find_blockers(x, bits):
    """Find all elements that block x from being added."""
    blockers = set()
    for d in _pos_last[x]:
        a, b, c = x - d, x - 2 * d, x - 3 * d
        if ((bits >> a) & 1) and ((bits >> b) & 1) and ((bits >> c) & 1):
            blockers.add(a); blockers.add(b); blockers.add(c)
    for d in _pos_first[x]:
        a, b, c = x + d, x + 2 * d, x + 3 * d
        if ((bits >> a) & 1) and ((bits >> b) & 1) and ((bits >> c) & 1):
            blockers.add(a); blockers.add(b); blockers.add(c)
    for d in _pos_second[x]:
        a, b, c = x - d, x + d, x + 2 * d
        if ((bits >> a) & 1) and ((bits >> b) & 1) and ((bits >> c) & 1):
            blockers.add(a); blockers.add(b); blockers.add(c)
    for d in _pos_third[x]:
        a, b, c = x - 2 * d, x - d, x + d
        if ((bits >> a) & 1) and ((bits >> b) & 1) and ((bits >> c) & 1):
            blockers.add(a); blockers.add(b); blockers.add(c)
    return blockers

def _is_4ap_free_bits(bits):
    """Fast bitset-based check for 4-AP-free property."""
    for x in range(1, N + 1):
        if not ((bits >> x) & 1):
            continue
        for d in _pos_last[x]:
            if ((bits >> (x - d)) & 1) and ((bits >> (x - 2 * d)) & 1) and ((bits >> (x - 3 * d)) & 1):
                return False
        for d in _pos_first[x]:
            if ((bits >> (x + d)) & 1) and ((bits >> (x + 2 * d)) & 1) and ((bits >> (x + 3 * d)) & 1):
                return False
        for d in _pos_second[x]:
            if ((bits >> (x - d)) & 1) and ((bits >> (x + d)) & 1) and ((bits >> (x + 2 * d)) & 1):
                return False
        for d in _pos_third[x]:
            if ((bits >> (x - 2 * d)) & 1) and ((bits >> (x - d)) & 1) and ((bits >> (x + d)) & 1):
                return False
    return True

def _greedy_forward_bitset():
    """Fast forward greedy using bitset."""
    bits = 0
    for x in range(1, N + 1):
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits

def _greedy_reverse_order_bitset():
    """Reverse-order greedy using bitset."""
    bits = 0
    for x in range(N, 0, -1):
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits

def _cantor_set_bits():
    """Cantor-like construction: keep numbers whose ternary expansion has no '1' digits."""
    bits = 0
    for x in range(1, N + 1):
        t = x
        ok = True
        while t > 0:
            if t % 3 == 1:
                ok = False
                break
            t //= 3
        if ok:
            bits |= (1 << x)
    return bits

def _greedy_density_weighted_bitset():
    """Greedy that prioritizes elements with fewer conflicts."""
    scores = [0] * (N + 1)
    for x in range(1, N + 1):
        cnt = 0
        for d in _pos_last[x]: cnt += 1
        for d in _pos_first[x]: cnt += 1
        for d in _pos_second[x]: cnt += 1
        for d in _pos_third[x]: cnt += 1
        scores[x] = cnt
    
    order = sorted(range(1, N + 1), key=lambda x: (scores[x], x))
    bits = 0
    for x in order:
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits

def _greedy_random_order_bitset(rng):
    """Random-order greedy using bitset."""
    order = list(range(1, N + 1))
    rng.shuffle(order)
    bits = 0
    for x in order:
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits

def _greedy_random_reverse_bitset(rng):
    """Random reverse-order greedy using bitset."""
    order = list(range(N, 0, -1))
    rng.shuffle(order)
    bits = 0
    for x in order:
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits

def _bits_to_set(bits):
    """Convert bitset to set."""
    return {i for i in range(1, N + 1) if (bits >> i) & 1}

def _local_improve_bitset(bits, deadline, max_passes=10):
    """
    Fast local search using bitset.
    Includes a greedy refill step after swaps.
    """
    current_bits = bits
    current_size = bin(current_bits).count('1')
    
    for _pass in range(max_passes):
        if time.time() > deadline - 2:
            break
            
        improved = False
        
        # Pass 1: Add any available element
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if not ((current_bits >> x) & 1) and _can_add_bitset(x, current_bits):
                current_bits |= (1 << x)
                current_size += 1
                improved = True
        
        # Pass 2: Try swaps (remove b, add x)
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if (current_bits >> x) & 1:
                continue
            
            blockers = _find_blockers(x, current_bits)
            if not blockers:
                continue
                
            # Try each blocker as a swap candidate
            # Sort blockers by "importance" (e.g., how many other things they block?) 
            # For now, just iterate.
            for b in blockers:
                if time.time() > deadline - 2:
                    break
                # Try swap: remove b, add x
                candidate_bits = current_bits & ~(1 << b)
                if _can_add_bitset(x, candidate_bits):
                    candidate_bits |= (1 << x)
                    
                    # Greedy re-fill on candidate
                    changed = True
                    while changed and time.time() < deadline - 2:
                        changed = False
                        for y in range(1, N + 1):
                            if time.time() > deadline - 2:
                                break
                            if not ((candidate_bits >> y) & 1) and _can_add_bitset(y, candidate_bits):
                                candidate_bits |= (1 << y)
                                changed = True
                                
                    cand_size = bin(candidate_bits).count('1')
                    if cand_size > current_size:
                        current_bits = candidate_bits
                        current_size = cand_size
                        improved = True
                        break # Restart the x loop? Or just continue? 
                        # Breaking the inner loop and continuing the outer x loop is safer 
                        # to ensure we check other potential swaps with the new state.
                        # However, since we modified current_bits, the blockers for other x might change.
                        # It's better to restart the whole local search loop (break out to while) 
                        # or at least restart the x loop.
                        # Let's break out of the x loop and restart the pass.
                        break
            if improved:
                break # Restart the pass
        
        if not improved:
            break
            
    return current_bits

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin within 60s limit

    # 1) Deterministic seeds
    seeds = [
        _greedy_forward_bitset(),
        _greedy_reverse_order_bitset(),
        _cantor_set_bits(),
        _greedy_density_weighted_bitset()
    ]
    
    best_bits = 0
    best_size = 0
    for s in seeds:
        sz = bin(s).count('1')
        if sz > best_size:
            best_bits = s
            best_size = sz

    # 2) Initial Local Improvement on the best seed
    if time.time() < deadline - 10:
        improved = _local_improve_bitset(best_bits, deadline, max_passes=5)
        imp_size = bin(improved).count('1')
        if imp_size > best_size:
            best_bits = improved
            best_size = imp_size

    # 3) Iterated Local Search (ILS) + Randomized Restarts
    rng = random.Random(42)
    
    # We will alternate between ILS on the current best and pure random restarts
    while time.time() < deadline - 5:
        # Decide strategy: ILS (70%) or Random Restart (30%)
        if rng.random() < 0.7:
            # ILS: Perturb current best
            perturbed = best_bits
            # Randomly remove 3-10 elements
            num_remove = rng.randint(3, 10)
            elements = _bits_to_set(perturbed)
            to_remove = rng.sample(list(elements), min(num_remove, len(elements)))
            for elem in to_remove:
                perturbed &= ~(1 << elem)
            
            # Local Search on perturbed
            candidate = _local_improve_bitset(perturbed, deadline, max_passes=5)
            cand_size = bin(candidate).count('1')
            
            if cand_size > best_size:
                best_bits = candidate
                best_size = cand_size
        else:
            # Random Restart
            r = rng.random()
            if r < 0.5:
                bits = _greedy_random_order_bitset(rng)
            else:
                bits = _greedy_random_reverse_bitset(rng)
            
            # Quick local search on restart
            candidate = _local_improve_bitset(bits, deadline, max_passes=3)
            cand_size = bin(candidate).count('1')
            
            if cand_size > best_size:
                best_bits = candidate
                best_size = cand_size

    # 4) Final Polish
    if time.time() < deadline - 2:
        final_bits = _local_improve_bitset(best_bits, deadline, max_passes=10)
        final_size = bin(final_bits).count('1')
        if final_size > best_size:
            best_bits = final_bits
            best_size = final_size

    # 5) Convert to set for final verification and return
    best_set = _bits_to_set(best_bits)

    # 6) Final verification; fallback if invalid
    if not _is_4ap_free_bits(best_bits):
        # Fallback to a known valid simple construction
        best_set = _bits_to_set(_greedy_forward_bitset())

    return sorted(best_set)
