import time
import random

N = 512

# Precompute for each x the list of d values for each of the 4 positions in a 4-AP
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


def _get_blockers(x, bits):
    """Return set of elements in bits that block adding x."""
    blockers = set()
    for d in _pos_last[x]:
        a, b, c = x - d, x - 2 * d, x - 3 * d
        if ((bits >> a) & 1) and ((bits >> b) & 1) and ((bits >> c) & 1):
            blockers.update([a, b, c])
    for d in _pos_first[x]:
        a, b, c = x + d, x + 2 * d, x + 3 * d
        if ((bits >> a) & 1) and ((bits >> b) & 1) and ((bits >> c) & 1):
            blockers.update([a, b, c])
    for d in _pos_second[x]:
        a, b, c = x - d, x + d, x + 2 * d
        if ((bits >> a) & 1) and ((bits >> b) & 1) and ((bits >> c) & 1):
            blockers.update([a, b, c])
    for d in _pos_third[x]:
        a, b, c = x - 2 * d, x - d, x + d
        if ((bits >> a) & 1) and ((bits >> b) & 1) and ((bits >> c) & 1):
            blockers.update([a, b, c])
    return blockers


def _greedy_bitset(order, bits=0):
    """Greedy construction given an order of elements, starting from bits."""
    for x in order:
        if not ((bits >> x) & 1) and _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_forward():
    """Forward greedy using bitset."""
    bits = 0
    for x in range(1, N + 1):
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_reverse():
    """Reverse greedy using bitset."""
    bits = 0
    for x in range(N, 0, -1):
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_random(rng):
    """Random-order greedy using bitset."""
    order = list(range(1, N + 1))
    rng.shuffle(order)
    bits = 0
    for x in order:
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_density_weighted(rng):
    """Greedy that weights positions by their constraint density."""
    constraint_counts = []
    for x in range(1, N + 1):
        count = len(_pos_last[x]) + len(_pos_first[x]) + len(_pos_second[x]) + len(_pos_third[x])
        constraint_counts.append((x, count))
    
    alpha = rng.uniform(0.3, 0.7)
    weighted_order = []
    for x, count in constraint_counts:
        weight = 1.0 / (1.0 + count * alpha)
        weighted_order.append((x, weight))
    
    order = []
    remaining = list(weighted_order)
    while remaining:
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


def _solve_interval(lo, hi, deadline, rng, depth=0):
    """
    Solve for a 4-AP-free subset of [lo, hi] using divide-and-conquer.
    For small intervals, use greedy. For larger, split and merge.
    """
    if time.time() > deadline:
        return 0
    
    size = hi - lo + 1
    if size <= 32:
        # Small interval: just use greedy
        bits = 0
        for x in range(lo, hi + 1):
            if _can_add_bitset(x, bits):
                bits |= (1 << x)
        return bits
    
    # Divide and conquer
    mid = (lo + hi) // 2
    left_bits = _solve_interval(lo, mid, deadline, rng, depth + 1)
    if time.time() > deadline:
        return left_bits
    right_bits = _solve_interval(mid + 1, hi, deadline, rng, depth + 1)
    if time.time() > deadline:
        return right_bits
    
    # Merge: try to add right elements to left, then left to right
    merged = left_bits | right_bits
    
    # Try to add more elements from the full range
    for x in range(lo, hi + 1):
        if time.time() > deadline:
            break
        if not ((merged >> x) & 1) and _can_add_bitset(x, merged):
            merged |= (1 << x)
    
    return merged


def _local_search_bits(bits, deadline, rng):
    """
    Local search using bitset representation.
    Attempts to add elements, then performs swaps to escape local optima.
    """
    current = bits
    current_size = bin(current).count('1')
    all_elems = list(range(1, N + 1))

    while time.time() < deadline - 1:
        improved = False

        # 1. Try to add any element
        for x in all_elems:
            if time.time() > deadline - 1:
                break
            if not ((current >> x) & 1) and _can_add_bitset(x, current):
                current |= (1 << x)
                current_size += 1
                improved = True

        if improved:
            continue

        # 2. If no additions, try swaps
        attempts = 0
        max_attempts = 300

        while attempts < max_attempts and time.time() < deadline - 1:
            attempts += 1
            x = rng.choice(all_elems)
            if (current >> x) & 1:
                continue

            # Check if x is blocked
            blocking_elements = _get_blockers(x, current)

            if not blocking_elements:
                current |= (1 << x)
                current_size += 1
                improved = True
                break

            # Try to remove one of the blocking elements and add x
            rem = rng.choice(list(blocking_elements))

            # Check if removing rem allows adding x
            valid_swap = True
            for u, v, w in _get_blockers(x, current):
                if rem not in (u, v, w):
                    valid_swap = False
                    break

            if valid_swap:
                current &= ~(1 << rem)
                current |= (1 << x)
                improved = True
                break

        if not improved:
            # Stuck. Try a "kick": remove a few random elements and re-greedy
            if current_size > 5:
                for _ in range(3):
                    if time.time() > deadline - 1:
                        break
                    elem = rng.choice([e for e in all_elems if (current >> e) & 1])
                    current &= ~(1 << elem)
                current_size = bin(current).count('1')
                # Re-greedy fill
                for x in all_elems:
                    if time.time() > deadline - 1:
                        break
                    if not ((current >> x) & 1) and _can_add_bitset(x, current):
                        current |= (1 << x)
                current_size = bin(current).count('1')

    return current


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    rng = random.Random(42)

    # 1. Generate multiple seeds
    seeds = []

    # Forward greedy
    seeds.append(_greedy_forward())

    # Reverse greedy
    seeds.append(_greedy_reverse())

    # Random greedy (a few times)
    for _ in range(5):
        seeds.append(_greedy_random(rng))

    # Density-weighted greedy
    seeds.append(_greedy_density_weighted(rng))

    # Find best seed
    best = max(seeds, key=lambda b: bin(b).count('1'))
    best_size = bin(best).count('1')

    # 2. Try divide-and-conquer approach
    if time.time() < deadline - 15:
        dc_bits = _solve_interval(1, N, deadline - 10, rng)
        dc_size = bin(dc_bits).count('1')
        if dc_size > best_size:
            best = dc_bits
            best_size = dc_size

    # 3. Local search on best seed
    if time.time() < deadline - 10:
        improved = _local_search_bits(best, deadline - 5, rng)
        if bin(improved).count('1') > best_size:
            best = improved
            best_size = bin(best).count('1')

    # 4. Randomized restarts with local search
    restarts = 0
    while time.time() < deadline - 15 and restarts < 10:
        restarts += 1
        seed = _greedy_random(rng)

        if bin(seed).count('1') < best_size:
            if best_size > 1:
                elem = rng.choice([e for e in range(1, N + 1) if (best >> e) & 1])
                perturbed = best & ~(1 << elem)
                order = list(range(1, N + 1))
                rng.shuffle(order)
                seed = _greedy_bitset(order, perturbed)

        if bin(seed).count('1') > best_size:
            best = seed
            best_size = bin(best).count('1')

        if time.time() < deadline - 5:
            improved = _local_search_bits(seed, deadline - 2, rng)
            if bin(improved).count('1') > best_size:
                best = improved
                best_size = bin(best).count('1')

    # 5. Final local search on the global best
    if time.time() < deadline - 1:
        improved = _local_search_bits(best, deadline, rng)
        if bin(improved).count('1') > best_size:
            best = improved

    # 6. Verify
    for x in range(1, N + 1):
        if not ((best >> x) & 1):
            continue
        for d in _pos_last[x]:
            if ((best >> (x - d)) & 1) and ((best >> (x - 2 * d)) & 1) and ((best >> (x - 3 * d)) & 1):
                return sorted([i for i in range(1, N + 1) if (bin(_greedy_forward()).count('1') and ((bin(_greedy_forward()) >> i) & 1))])
        for d in _pos_first[x]:
            if ((best >> (x + d)) & 1) and ((best >> (x + 2 * d)) & 1) and ((best >> (x + 3 * d)) & 1):
                return sorted([i for i in range(1, N + 1) if ((bin(_greedy_forward()) >> i) & 1)])
        for d in _pos_second[x]:
            if ((best >> (x - d)) & 1) and ((best >> (x + d)) & 1) and ((best >> (x + 2 * d)) & 1):
                return sorted([i for i in range(1, N + 1) if ((bin(_greedy_forward()) >> i) & 1)])
        for d in _pos_third[x]:
            if ((best >> (x - 2 * d)) & 1) and ((best >> (x - d)) & 1) and ((best >> (x + d)) & 1):
                return sorted([i for i in range(1, N + 1) if ((bin(_greedy_forward()) >> i) & 1)])

    return sorted([i for i in range(1, N + 1) if (best >> i) & 1])


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
