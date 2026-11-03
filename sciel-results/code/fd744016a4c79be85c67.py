"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def construct(n):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s limit

    # For small n, use exact known constructions
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Known shortest is 33
        return _construct_classic(4)

    # Use classic recursive construction as starting point
    word = _construct_classic(n)

    # Try to shorten using local search
    word = _local_search_shorten(word, n, deadline)

    return word

def _construct_classic(n):
    """Classic recursive construction of length 1! + 2! + ... + n!"""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_overlap = min(len(new), len(block))
            for k in range(max_overlap, -1, -1):
                if k == 0:
                    overlap = 0
                    break
                if new[len(new) - k:] == block[:k]:
                    overlap = k
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _get_all_windows(word, n):
    """Get all n-length windows as tuples"""
    windows = []
    for i in range(len(word) - n + 1):
        windows.append(tuple(int(c) for c in word[i:i+n]))
    return windows

def _count_covered_permutations(word, n):
    """Count how many distinct permutations of 1..n appear as windows"""
    windows = set()
    for i in range(len(word) - n + 1):
        window = tuple(int(c) for c in word[i:i+n])
        if len(set(window)) == n:
            windows.add(window)
    return len(windows)

def _is_valid_superpermutation(word, n):
    """Check if word contains all n! permutations of 1..n"""
    covered = _count_covered_permutations(word, n)
    return covered == int(_factorial(n))

def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def _local_search_shorten(word, n, deadline):
    """Shorten the word by greedily removing characters while keeping validity.

    Uses an incremental refcount over the n! permutations so each single-
    character removal trial costs O(n) instead of O(n*L).
    """
    # Work with a list of integers for speed
    w = [int(c) for c in word]
    L = len(w)
    target = int(_factorial(n))

    # Build the refcount of each permutation window.
    # A window is a permutation iff it contains all n distinct symbols.
    # We track, for every window, whether it is a permutation and its tuple.
    # refcount[t] = number of windows equal to permutation tuple t.
    refcount = {}
    # To test distinctness fast, use a bitmask over symbols 1..n.
    # symbol -> bit
    def is_perm_window(start, length):
        mask = 0
        for j in range(start, start + length):
            mask |= 1 << w[j]
        return mask == (1 << (n + 1)) - (1 << 1)  # bits 1..n set

    # Precompute which windows are permutations and store their tuple.
    # window_perm[i] = tuple of window starting at i if it is a permutation, else None
    window_perm = [None] * (L - n + 1)
    for i in range(L - n + 1):
        if is_perm_window(i, n):
            t = tuple(w[i:i + n])
            window_perm[i] = t
            refcount[t] = refcount.get(t, 0) + 1

    # Sanity: covered count
    covered = sum(1 for v in refcount.values() if v > 0)
    if covered < target:
        # Should not happen for a valid classic construction; return as-is.
        return "".join(map(str, w))

    def try_remove(pos):
        """Try removing position pos. Returns True if valid (and keeps it)."""
        nonlocal w, L, window_perm
        if pos <= 0 or pos >= L - 1:
            return False
        # Removing pos affects windows starting in [pos-n+1, pos].
        # For each such window, if it was a permutation, decrement its refcount.
        # If any refcount hits 0, invalid -> restore.
        lo = max(0, pos - n + 1)
        hi = min(pos, L - n)  # windows that include pos and are valid starts
        removed = []  # (window_start, tuple) to restore
        ok = True
        for i in range(lo, hi + 1):
            t = window_perm[i]
            if t is not None:
                removed.append((i, t))
                refcount[t] -= 1
                if refcount[t] == 0:
                    ok = False
                    break
        if not ok:
            # Restore the decrements we did
            for (i, t) in removed:
                refcount[t] += 1
            return False

        # Actually remove the character
        w.pop(pos)
        L -= 1

        # Rebuild window_perm for the affected region.
        # Windows starting in [lo, hi] after removal need recomputation.
        # Also, the set of valid window starts shrank by 1 (total L-n+1 now).
        # Recompute window_perm entries in the affected range.
        new_lo = max(0, lo)
        new_hi = min(hi, L - n)
        for i in range(new_lo, new_hi + 1):
            if is_perm_window(i, n):
                t = tuple(w[i:i + n])
                window_perm[i] = t
                refcount[t] = refcount.get(t, 0) + 1
            else:
                window_perm[i] = None
        # Entries beyond new_hi that were previously set but now out of range:
        # window_perm is indexed up to L-n; old indices above that are stale but
        # we only ever access i in [0, L-n], so fine.
        return True

    # Hill-climb: repeatedly scan for any removable position.
    # To be efficient, scan in a shuffled order and stop after a full pass
    # with no improvement, or on deadline.
    rng = random.Random(12345)
    improved_any = True
    passes = 0
    while improved_any and time.time() < deadline - 3.0:
        improved_any = False
        passes += 1
        # Build candidate positions (interior only; removing first/last char
        # cannot reduce length below needed and breaks the window set).
        positions = list(range(1, L - 1))
        rng.shuffle(positions)
        for pos in positions:
            if time.time() > deadline - 2.0:
                return "".join(map(str, w))
            # pos is an index in current w
            if pos <= 0 or pos >= L - 1:
                continue
            if try_remove(pos):
                improved_any = True
                # After removal, all indices after pos shifted left by 1.
                # Rebuild the position list to stay consistent.
                positions = list(range(1, L - 1))
                rng.shuffle(positions)
                break

    return "".join(map(str, w))

# EVOLVE-BLOCK-END
