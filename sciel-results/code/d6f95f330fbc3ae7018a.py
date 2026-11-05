"""Stronger algorithm: greedy seed + local search (add/remove) to maximize 3-APs
while maintaining 4-AP-free constraint. Uses bitmasks for fast 4-AP checks."""
import time
import random

N = 256
DEADLINE_FRACTION = 0.92  # safety margin against 90s kill


def _count_3aps(mask):
    """Count distinct 3-term APs (a, a+d, a+2d) with d>0 in the set represented by mask."""
    count = 0
    # iterate over differences
    for d in range(1, N // 2 + 1):
        # for each start a, check a, a+d, a+2d
        # use bit operations: for each a, check bits a, a+d, a+2d
        # faster: for each a where bit a is set, check a+d and a+2d
        # We'll iterate a from 0 to N-2d-1
        for a in range(0, N - 2 * d):
            if (mask >> a) & 1 and (mask >> (a + d)) & 1 and (mask >> (a + 2 * d)) & 1:
                count += 1
    return count


def _has_4ap(mask):
    """Check if mask contains a nonconstant 4-term AP."""
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            if ((mask >> a) & 1 and (mask >> (a + d)) & 1 and
                (mask >> (a + 2 * d)) & 1 and (mask >> (a + 3 * d)) & 1):
                return True
    return False


def _add_4ap_free(mask, x):
    """Check if adding x (0-indexed) to mask keeps it 4-AP-free."""
    bit = 1 << x
    if mask & bit:
        return False
    # x can be part of a 4-AP as position 0,1,2,3
    # position 0: x, x+d, x+2d, x+3d for d=1..(N-1-x)//3
    for d in range(1, (N - 1 - x) // 3 + 1):
        if d > 0 and x + 3 * d < N:
            if ((mask >> (x + d)) & 1 and (mask >> (x + 2 * d)) & 1 and
                (mask >> (x + 3 * d)) & 1):
                return False
    # position 1: x-d, x, x+d, x+2d
    for d in range(1, x + 1):
        if x + 2 * d < N:
            if ((mask >> (x - d)) & 1 and (mask >> (x + d)) & 1 and
                (mask >> (x + 2 * d)) & 1):
                return False
    # position 2: x-2d, x-d, x, x+d
    for d in range(1, (x + 1) // 2 + 1):
        if x + d < N:
            if ((mask >> (x - 2 * d)) & 1 and (mask >> (x - d)) & 1 and
                (mask >> (x + d)) & 1):
                return False
    # position 3: x-3d, x-2d, x-d, x
    for d in range(1, (x + 1) // 3 + 1):
        if ((mask >> (x - 3 * d)) & 1 and (mask >> (x - 2 * d)) & 1 and
            (mask >> (x - d)) & 1):
            return False
    return True


def _delta_3aps_add(mask, x):
    """Count how many 3-APs are created by adding x (0-indexed) to mask."""
    delta = 0
    # x as first element: x, x+d, x+2d
    for d in range(1, (N - 1 - x) // 2 + 1):
        if x + 2 * d < N and ((mask >> (x + d)) & 1 and (mask >> (x + 2 * d)) & 1):
            delta += 1
    # x as second element: x-d, x, x+d
    for d in range(1, x + 1):
        if x + d < N and ((mask >> (x - d)) & 1 and (mask >> (x + d)) & 1):
            delta += 1
    # x as third element: x-2d, x-d, x
    for d in range(1, (x + 1) // 2 + 1):
        if ((mask >> (x - 2 * d)) & 1 and (mask >> (x - d)) & 1):
            delta += 1
    return delta


def _delta_3aps_remove(mask, x):
    """Count how many 3-APs are destroyed by removing x (0-indexed) from mask."""
    delta = 0
    # x as first element
    for d in range(1, (N - 1 - x) // 2 + 1):
        if x + 2 * d < N and ((mask >> (x + d)) & 1 and (mask >> (x + 2 * d)) & 1):
            delta += 1
    # x as second element
    for d in range(1, x + 1):
        if x + d < N and ((mask >> (x - d)) & 1 and (mask >> (x + d)) & 1):
            delta += 1
    # x as third element
    for d in range(1, (x + 1) // 2 + 1):
        if ((mask >> (x - 2 * d)) & 1 and ((mask >> (x - d)) & 1)):
            delta += 1
    return delta


def _greedy_seed():
    """Build a greedy 4-AP-free set by adding elements that maximize 3-AP gain."""
    mask = 0
    for x in range(N):
        if _add_4ap_free(mask, x):
            mask |= (1 << x)
    return mask


def _local_search(mask, deadline):
    """Iterative local search: try adding or removing elements to improve 3-AP count."""
    best_mask = mask
    best_score = _count_3aps(mask)
    rng = random.Random(42)
    
    while time.time() < deadline:
        improved = False
        # Try all additions
        for x in range(N):
            if not (mask & (1 << x)):
                if _add_4ap_free(mask, x):
                    new_mask = mask | (1 << x)
                    new_score = best_score + _delta_3aps_add(mask, x)
                    if new_score > best_score:
                        best_mask = new_mask
                        best_score = new_score
                        improved = True
        mask = best_mask
        # Try all removals (only if removing increases score, which shouldn't happen
        # unless we're stuck, but let's try swaps)
        # Try swaps: remove one, add another
        if not improved:
            found_swap = False
            for x_remove in range(N):
                if mask & (1 << x_remove):
                    for x_add in range(N):
                        if not (mask & (1 << x_add)):
                            if _add_4ap_free(mask & ~(1 << x_remove), x_add):
                                new_mask = (mask & ~(1 << x_remove)) | (1 << x_add)
                                new_score = best_score - _delta_3aps_remove(mask, x_remove) + _delta_3aps_add(mask & ~(1 << x_remove), x_add)
                                if new_score > best_score:
                                    best_mask = new_mask
                                    best_score = new_score
                                    found_swap = True
                                    break
                if found_swap:
                    break
            mask = best_mask
            if not found_swap:
                # Try random restarts with perturbation
                if time.time() > deadline - 5:
                    break
                # Perturb: remove a few random elements, then re-greedy
                perturbed = mask
                for _ in range(rng.randint(1, 3)):
                    bits = [i for i in range(N) if (perturbed >> i) & 1]
                    if bits:
                        perturbed &= ~(1 << rng.choice(bits))
                # Re-add greedily
                for x in range(N):
                    if not (perturbed & (1 << x)) and _add_4ap_free(perturbed, x):
                        perturbed |= (1 << x)
                pscore = _count_3aps(perturbed)
                if pscore > best_score:
                    best_mask = perturbed
                    best_score = pscore
                    mask = best_mask
                    improved = True
                if not improved:
                    break
    
    return best_mask, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 90 * DEADLINE_FRACTION
    
    # Start with greedy seed
    mask = _greedy_seed()
    score = _count_3aps(mask)
    
    # Local search
    mask, score = _local_search(mask, deadline)
    
    # Convert to 1-indexed sorted list
    result = [i + 1 for i in range(N) if (mask >> i) & 1]
    
    if not result:
        result = [1]
    
    return sorted(result)


if __name__ == "__main__":
    result = construct()
    print(f"Cardinality: {len(result)}")
    print(f"Result: {result[:20]}...")
