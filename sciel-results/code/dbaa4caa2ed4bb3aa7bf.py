"""Combined program: Cantor set seed + many randomized greedy restarts + fast swap local search with bitset verification."""
import time
import random
import itertools

N = 512
MAX = N

# Precompute for each x the list of (d) values for each of the 4 positions,
# so the inner loop is just iterating a precomputed tuple.
_pos_last   = [tuple(range(1, (x - 1) // 3 + 1)) for x in range(N + 1)]
_pos_first  = [tuple(range(1, (N - x) // 3 + 1)) for x in range(N + 1)]
_pos_second = [tuple(range(1, min(x - 1, (N - x) // 2) + 1)) for x in range(N + 1)]
_pos_third  = [tuple(range(1, min((x - 1) // 2, N - x) + 1)) for x in range(N + 1)]

# Precompute all 4-APs involving each x as a list of (a, b, c)
# where {x, a, b, c} is a 4-AP
_ap_triples = [None] * (N + 1)
for x in range(1, N + 1):
    triples = []
    for d in _pos_last[x]:
        a, b, c = x - 3*d, x - 2*d, x - d
        if a >= 1:
            triples.append((a, b, c))
    for d in _pos_first[x]:
        a, b, c = x + d, x + 2*d, x + 3*d
        if c <= N:
            triples.append((a, b, c))
    for d in _pos_second[x]:
        a, b, c = x - d, x + d, x + 2*d
        if a >= 1 and c <= N:
            triples.append((a, b, c))
    for d in _pos_third[x]:
        a, b, c = x - 2*d, x - d, x + d
        if a >= 1 and c <= N:
            triples.append((a, b, c))
    _ap_triples[x] = triples

def _can_add_bitset(x, bits):
    """Check if x can be added to the set represented by bits (int)."""
    for a, b, c in _ap_triples[x]:
        if (bits >> a) & 1 and (bits >> b) & 1 and (bits >> c) & 1:
            return False
    return True


def _is_4ap_free_bits(bits):
    """Fast bitset-based check for 4-AP-free property."""
    for x in range(1, N + 1):
        if not ((bits >> x) & 1):
            continue
        for a, b, c in _ap_triples[x]:
            if ((bits >> a) & 1) and ((bits >> b) & 1) and ((bits >> c) & 1):
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
    """Fast reverse-order greedy using bitset."""
    bits = 0
    for x in range(N, 0, -1):
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
    """Random-order greedy with reverse bias using bitset."""
    order = list(range(1, N + 1))
    rng.shuffle(order)
    # Bias towards larger numbers first
    order.sort(key=lambda x: -x if rng.random() < 0.7 else x)
    bits = 0
    for x in order:
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_density_weighted_bitset(rng):
    """
    Greedy that weights positions by their constraint density.
    Lower numbers have fewer potential APs (can't be last term of long APs),
    so they're "easier" to include. We bias towards including them.
    """
    # Compute constraint count for each position
    constraint_counts = []
    for x in range(1, N + 1):
        count = len(_pos_last[x]) + len(_pos_first[x]) + len(_pos_second[x]) + len(_pos_third[x])
        constraint_counts.append((x, count))
    
    # Sort by constraint count (ascending), with some randomness
    alpha = rng.uniform(0.3, 0.7)
    weighted_order = []
    for x, count in constraint_counts:
        # Lower constraint count -> higher priority
        weight = 1.0 / (1.0 + count * alpha)
        weighted_order.append((x, weight))
    
    # Sample from weighted order with some randomness
    order = []
    remaining = list(weighted_order)
    while remaining:
        if not remaining:
            break
        # Pick a random element with probability proportional to weight
        total_weight = sum(w for _, w in remaining)
        r = rng.random() * total_weight
        cum = 0
        for i, (x, w) in enumerate(remaining):
            cum += w
            if cum >= r:
                order.append(x)
                remaining.pop(i)
                break
    
    bits = 0
    for x in order:
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_hybrid_bitset(rng):
    """
    Hybrid greedy: start with low-constraint positions, then fill in randomly.
    """
    # Phase 1: Add low-constraint positions first
    constraint_counts = []
    for x in range(1, N + 1):
        count = len(_pos_last[x]) + len(_pos_first[x]) + len(_pos_second[x]) + len(_pos_third[x])
        constraint_counts.append((count, x))
    
    constraint_counts.sort()
    
    bits = 0
    added = set()
    
    # Add lowest-constraint positions first (with some randomization)
    for count, x in constraint_counts[:150]:
        if x not in added and _can_add_bitset(x, bits):
            bits |= (1 << x)
            added.add(x)
    
    # Phase 2: Fill in remaining positions randomly
    remaining = [x for x in range(1, N + 1) if x not in added]
    rng.shuffle(remaining)
    for x in remaining:
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
            added.add(x)
    
    return bits


def _bits_to_set(bits):
    """Convert bitset to set."""
    return {i for i in range(1, N + 1) if (bits >> i) & 1}


def _local_improve_bitset(bits, deadline):
    """Fast local search using bitset with improved swap strategy."""
    current_bits = bits
    current_size = bin(current_bits).count('1')
    
    # List of missing elements
    missing = [x for x in range(1, N + 1) if not ((current_bits >> x) & 1)]
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Pass 1: Add any available element
        new_missing = []
        for x in missing:
            if time.time() > deadline - 2:
                break
            if _can_add_bitset(x, current_bits):
                current_bits |= (1 << x)
                current_size += 1
                improved = True
            else:
                new_missing.append(x)
        missing = new_missing
        
        # Pass 2: Try swaps (remove b, add x) and 2-for-1 swaps
        if missing:
            # Shuffle missing to randomize order? Or just iterate.
            # We iterate all, but limit work per x.
            for x in missing:
                if time.time() > deadline - 2:
                    break
                # Find blocking triples
                blocking_triples = []
                for a, b, c in _ap_triples[x]:
                    if ((current_bits >> a) & 1) and ((current_bits >> b) & 1) and ((current_bits >> c) & 1):
                        blocking_triples.append((a, b, c))
                
                if not blocking_triples:
                    # Should have been caught in Pass 1, but just in case
                    current_bits |= (1 << x)
                    current_size += 1
                    improved = True
                    missing.remove(x)
                    continue
                
                # If too many blockers, skip
                if len(blocking_triples) > 4:
                    continue
                
                # Try removing 1 element
                # Union of all elements in blocking triples
                all_blockers = set()
                for t in blocking_triples:
                    all_blockers.update(t)
                
                # Try single removals
                for b in all_blockers:
                    if time.time() > deadline - 2:
                        break
                    candidate_bits = current_bits & ~(1 << b)
                    if _can_add_bitset(x, candidate_bits):
                        candidate_bits |= (1 << x)
                        # Quick refill
                        changed = True
                        while changed and time.time() < deadline - 2:
                            changed = False
                            for y in missing:
                                if time.time() > deadline - 2:
                                    break
                                if y == x: continue
                                if _can_add_bitset(y, candidate_bits):
                                    candidate_bits |= (1 << y)
                                    changed = True
                        cand_size = bin(candidate_bits).count('1')
                        if cand_size > current_size:
                            current_bits = candidate_bits
                            current_size = cand_size
                            improved = True
                            missing.remove(x)
                            # Update missing list for removed b? 
                            # b is now in missing
                            if b not in missing:
                                missing.append(b)
                            # Re-add any missing elements that were added during refill?
                            # This is complex. Simpler: rebuild missing if improved significantly?
                            # Or just let the next loop iteration handle it.
                            # For now, we just break and let the outer loop restart.
                            break
                if improved:
                    break
                
                # Try 2-for-1 swaps if single removal failed
                if len(blocking_triples) <= 3:
                    # Generate pairs of blockers
                    # To limit combinations, only consider pairs from distinct triples or small sets
                    # For simplicity, try all pairs from all_blockers if size is small
                    if len(all_blockers) <= 6:
                        blist = list(all_blockers)
                        for i in range(len(blist)):
                            if time.time() > deadline - 2:
                                break
                            for j in range(i + 1, len(blist)):
                                b1, b2 = blist[i], blist[j]
                                candidate_bits = current_bits & ~(1 << b1) & ~(1 << b2)
                                if _can_add_bitset(x, candidate_bits):
                                    candidate_bits |= (1 << x)
                                    # Quick refill
                                    changed = True
                                    while changed and time.time() < deadline - 2:
                                        changed = False
                                        for y in missing:
                                            if time.time() > deadline - 2:
                                                break
                                            if y == x: continue
                                            if _can_add_bitset(y, candidate_bits):
                                                candidate_bits |= (1 << y)
                                                changed = True
                                    cand_size = bin(candidate_bits).count('1')
                                    if cand_size > current_size:
                                        current_bits = candidate_bits
                                        current_size = cand_size
                                        improved = True
                                        missing.remove(x)
                                        if b1 not in missing:
                                            missing.append(b1)
                                        if b2 not in missing:
                                            missing.append(b2)
                                        break
                            if improved:
                                break
            if improved:
                # Rebuild missing list to be safe
                missing = [x for x in range(1, N + 1) if not ((current_bits >> x) & 1)]

    return current_bits


def _cantor_set_bits():
    """
    Generate a 4-AP-free set using the base-3 Cantor set construction.
    Numbers in [1, N] whose base-3 representation contains only 0s and 1s
    (no digit 2) form a 4-AP-free set.
    """
    bits = 0
    for x in range(1, N + 1):
        # Check if base-3 representation of x contains only 0s and 1s
        t = x
        valid = True
        while t > 0:
            if t % 3 == 2:
                valid = False
                break
            t //= 3
        if valid:
            bits |= (1 << x)
    return bits




def construct():
    deadline = time.time() + 55  # Safety margin within 60s limit

    # 1) Start with Cantor set (strong structured seed ~200 elements)
    cantor_bits = _cantor_set_bits()
    cantor_size = bin(cantor_bits).count('1')
    best_bits = cantor_bits
    best_size = cantor_size

    # 1b) Also try deterministic forward greedy
    fwd_bits = _greedy_forward_bitset()
    fwd_size = bin(fwd_bits).count('1')
    if fwd_size > best_size:
        best_bits = fwd_bits
        best_size = fwd_size

    # 1c) Deterministic reverse greedy
    rev_bits = _greedy_reverse_order_bitset()
    rev_size = bin(rev_bits).count('1')
    if rev_size > best_size:
        best_bits = rev_bits
        best_size = rev_size

    # 2) Local improvement on Cantor seed (this is the key phase)
    if time.time() < deadline - 15:
        cantor_improved = _local_improve_bitset(cantor_bits, deadline)
        cantor_imp_size = bin(cantor_improved).count('1')
        if cantor_imp_size > best_size:
            best_bits = cantor_improved
            best_size = cantor_imp_size

    # 3) Many randomized restarts with diverse strategies
    rng = random.Random(42)
    while time.time() < deadline - 10:
        r = rng.random()
        if r < 0.25:
            bits = _greedy_random_order_bitset(rng)
        elif r < 0.5:
            bits = _greedy_random_reverse_bitset(rng)
        elif r < 0.75:
            bits = _greedy_density_weighted_bitset(rng)
        else:
            bits = _greedy_hybrid_bitset(rng)
        size = bin(bits).count('1')
        if size > best_size:
            best_bits = bits
            best_size = size
            # Also try local improvement on this promising set
            if time.time() < deadline - 8:
                improved = _local_improve_bitset(bits, deadline - 5)
                imp_size = bin(improved).count('1')
                if imp_size > best_size:
                    best_bits = improved
                    best_size = imp_size

    # 4) Final local improvement on the best set found
    if time.time() < deadline - 5:
        improved_bits = _local_improve_bitset(best_bits, deadline)
        improved_size = bin(improved_bits).count('1')
        if improved_size > best_size:
            best_bits = improved_bits
            best_size = improved_size
    
    # 5) Try a few more local improvements with different random seeds
    if time.time() < deadline - 3:
        for seed in [123, 456, 789]:
            if time.time() > deadline - 3:
                break
            rng2 = random.Random(seed)
            bits = _greedy_density_weighted_bitset(rng2)
            size = bin(bits).count('1')
            if size >= best_size - 5:  # Close to best, worth improving
                improved = _local_improve_bitset(bits, deadline - 2)
                imp_size = bin(improved).count('1')
                if imp_size > best_size:
                    best_bits = improved
                    best_size = imp_size

    # 5) Convert to set for final verification and return
    best_set = _bits_to_set(best_bits)

    # 6) Final verification; fallback if invalid
    if not _is_4ap_free_bits(best_bits):
        # Fallback to Cantor set (always valid)
        best_set = _bits_to_set(_cantor_set_bits())

    return sorted(best_set)
