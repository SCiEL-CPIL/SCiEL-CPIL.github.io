import time
import random

N = 512
DEADLINE = 85.0  # seconds, safety margin under 90s

def _initial_greedy():
    """Greedy increasing-order construction (same as seed) as a starting point."""
    present = [False] * (N + 1)
    size = 0
    for x in range(1, N + 1):
        ok = True
        # x as position 3 (last): need x-d, x-2d, x-3d in S
        for d in range(1, (x - 1) // 3 + 1):
            a = x - 3 * d
            if a >= 1 and present[a] and present[a + d] and present[a + 2 * d]:
                ok = False
                break
        if ok:
            # x as position 2: need x-d, x-2d and x+3d? no, x+3d > x not in S yet since increasing
            # In increasing greedy, only earlier elements exist, so x can only be the LAST term.
            present[x] = True
            size += 1
    return present, size

def _has_4ap_with(present, x):
    """Return True if adding x to the set (x not currently present) creates a 4-AP."""
    # x as last (pos3): a, a+d, a+2d, x  => a = x-3d
    for d in range(1, (x - 1) // 3 + 1):
        a = x - 3 * d
        if a >= 1 and present[a] and present[a + d] and present[a + 2 * d]:
            return True
    # x as first (pos0): x, x+d, x+2d, x+3d
    for d in range(1, (N - x) // 3 + 1):
        if present[x + d] and present[x + 2 * d] and present[x + 3 * d]:
            return True
    # x as pos1: a, x, x+d, x+2d  => a = x-d
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        a = x - d
        if a >= 1 and present[a] and present[x + d] and present[x + 2 * d]:
            return True
    # x as pos2: a, a+d, x, x+d  => a = x-2d
    for d in range(1, min((x - 1) // 2, N - x) + 1):
        a = x - 2 * d
        if a >= 1 and present[a] and present[a + d] and present[x + d]:
            return True
    return False

def _remove_4ap_count(present, x):
    """Count 4-APs in the set that include x (x is present)."""
    cnt = 0
    for d in range(1, (x - 1) // 3 + 1):
        a = x - 3 * d
        if a >= 1 and present[a] and present[a + d] and present[a + 2 * d]:
            cnt += 1
    for d in range(1, (N - x) // 3 + 1):
        if present[x + d] and present[x + 2 * d] and present[x + 3 * d]:
            cnt += 1
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        a = x - d
        if a >= 1 and present[a] and present[x + d] and present[x + 2 * d]:
            cnt += 1
    for d in range(1, min((x - 1) // 2, N - x) + 1):
        a = x - 2 * d
        if a >= 1 and present[a] and present[a + d] and present[x + d]:
            cnt += 1
    return cnt

def construct():
    start = time.time()
    random.seed(12345)

    best_present, best_size = _initial_greedy()
    present, size = best_present[:], best_size

    # Local search: try adds; on conflict try swaps to free space, keep best.
    missing = [x for x in range(1, N + 1) if not present[x]]
    random.shuffle(missing)

    iters = 0
    while time.time() - start < DEADLINE:
        # Try to add each missing element
        for x in missing:
            if present[x]:
                continue
            if not _has_4ap_with(present, x):
                present[x] = True
                size += 1
                missing.remove(x)
                if size > best_size:
                    best_size = size
                    best_present = present[:]
        if not missing:
            break
        # Swap phase: try to replace a present element with a missing one to enable future adds
        # Pick a random present element y with low AP involvement, remove it, then try to add a missing x.
        present_elems = [x for x in range(1, N + 1) if present[x]]
        if not present_elems:
            break
        y = random.choice(present_elems)
        present[y] = False
        size -= 1
        # Now try to add as many missing as possible
        progress = True
        while progress:
            progress = False
            for x in missing:
                if present[x]:
                    continue
                if not _has_4ap_with(present, x):
                    present[x] = True
                    size += 1
                    missing.remove(x)
                    progress = True
                    if size > best_size:
                        best_size = size
                        best_present = present[:]
        if size > best_size:
            best_size = size
            best_present = present[:]
        # If we didn't improve, consider restoring y if it helps? We keep the swapped state for further search.
        iters += 1
        if iters % 200 == 0 and time.time() - start > DEADLINE - 5:
            break

    # Final: ensure best is valid and return
    result = sorted(x for x in range(1, N + 1) if best_present[x])
    return result
