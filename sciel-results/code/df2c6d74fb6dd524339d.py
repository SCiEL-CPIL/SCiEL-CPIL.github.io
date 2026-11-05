import time
import random
import math

def _classic_construct(n):
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
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word


def _init_counts(S, n):
    counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i + n])
        counts[p] = counts.get(p, 0) + 1
    return counts


def _is_valid_perm(S, i, n):
    # window S[i:i+n] must be a permutation of 1..n
    w = S[i:i + n]
    if len(w) != n:
        return False
    if len(set(w)) != n:
        return False
    return True


def _try_remove(S, counts, i, n):
    """Attempt to remove S[i]. Returns (success, affected_old, affected_new)."""
    L = len(S)
    # old windows containing i: j in [max(0,i-n+1), min(i, L-n)]
    start_old = max(0, i - n + 1)
    end_old = min(i, L - n)

    old_perms = []
    for j in range(start_old, end_old + 1):
        p = tuple(S[j:j + n])
        if counts.get(p, 0) == 1:
            return False, None, None
        old_perms.append(p)

    # new straddling windows: new string = S[:i]+S[i+1:], length L-1
    # new window at k straddles i if k < i <= k+n-1 => i-n+1 <= k <= i-1
    # and 0 <= k <= L-1-n
    start_new = max(0, i - n + 1)
    end_new = min(i - 1, L - 1 - n)

    new_perms = []
    for k in range(start_new, end_new + 1):
        # map new index -> S index
        chars = []
        for idx in range(k, k + n):
            s_idx = idx if idx < i else idx + 1
            chars.append(S[s_idx])
        new_perms.append(tuple(chars))

    return True, old_perms, new_perms


def _commit_remove(S, counts, i, old_perms, new_perms):
    for p in old_perms:
        counts[p] -= 1
        if counts[p] == 0:
            del counts[p]
    for p in new_perms:
        counts[p] = counts.get(p, 0) + 1
    S.pop(i)


def _greedy_remove_pass(S, counts, n, deadline, order='right'):
    """One full pass. Returns (removed_count, still_valid)."""
    removed = 0
    L = len(S)
    if order == 'right':
        rng = range(L - 1, -1, -1)
    else:
        rng = range(L)
    for i in rng:
        if time.time() > deadline:
            return removed, False
        if i >= len(S):
            continue
        ok, old_p, new_p = _try_remove(S, counts, i, n)
        if ok:
            _commit_remove(S, counts, i, old_p, new_p)
            removed += 1
    return removed, True


def _random_removal_search(S, counts, n, deadline, max_rounds=40):
    """Repeated random-order removal passes until stuck or deadline."""
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        # random permutation of indices
        idxs = list(range(len(S)))
        random.shuffle(idxs)
        removed = 0
        for i in idxs:
            if time.time() > deadline:
                break
            if i >= len(S):
                continue
            ok, old_p, new_p = _try_remove(S, counts, i, n)
            if ok:
                _commit_remove(S, counts, i, old_p, new_p)
                removed += 1
        if removed == 0:
            break
    return S, counts


def _swap_unlock(S, counts, n, deadline, max_attempts=300):
    """Try random swaps to unlock a removal. Returns (success, S, counts)."""
    L = len(S)
    if L < 2:
        return False, S, counts
    for _ in range(max_attempts):
        if time.time() > deadline:
            break
        a = random.randrange(L)
        b = random.randrange(L)
        if a == b or S[a] == S[b]:
            continue
        # swap
        S[a], S[b] = S[b], S[a]
        # recompute counts (cheap: n small, L ~ few thousand)
        counts = _init_counts(S, n)
        # try a few random removals
        improved = False
        for _ in range(50):
            if time.time() > deadline:
                break
            i = random.randrange(len(S))
            ok, old_p, new_p = _try_remove(S, counts, i, n)
            if ok:
                _commit_remove(S, counts, i, old_p, new_p)
                improved = True
        if improved:
            return True, S, counts
        # undo swap
        S[a], S[b] = S[b], S[a]
        counts = _init_counts(S, n)
    return False, S, counts


def _validate(S, n):
    seen = set()
    for i in range(len(S) - n + 1):
        w = tuple(S[i:i + n])
        if len(set(w)) == n:
            seen.add(w)
    return len(seen) == math.factorial(n)


def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin under 150s

    if n <= 1:
        return "1" if n == 1 else ""

    S = _classic_construct(n)
    best_S = S[:]
    best_len = len(S)

    if n >= 7:
        # Heavy optimization path
        counts = _init_counts(S, n)

        # Phase 1: right-to-left greedy
        _greedy_remove_pass(S, counts, n, deadline, order='right')
        if len(S) < best_len:
            best_S, best_len = S[:], len(S)

        # Phase 2: left-to-right greedy
        _greedy_remove_pass(S, counts, n, deadline, order='left')
        if len(S) < best_len:
            best_S, best_len = S[:], len(S)

        # Phase 3: random-order removals
        S, counts = _random_removal_search(S, counts, n, deadline, max_rounds=30)
        if len(S) < best_len:
            best_S, best_len = S[:], len(S)

        # Phase 4: swap-to-unlock (repeated)
        for _ in range(8):
            if time.time() > deadline:
                break
            ok, S, counts = _swap_unlock(S, counts, n, deadline, max_attempts=200)
            if ok:
                # follow up with greedy removals
                _greedy_remove_pass(S, counts, n, deadline, order='right')
                S, counts = _random_removal_search(S, counts, n, deadline, max_rounds=15)
                if len(S) < best_len:
                    best_S, best_len = S[:], len(S)
            else:
                # no improvement found; if stuck hard, break
                break

        # Phase 5: final polish - alternating passes
        for _ in range(6):
            if time.time() > deadline:
                break
            r1 = _greedy_remove_pass(S, counts, n, deadline, order='right')
            r2 = _greedy_remove_pass(S, counts, n, deadline, order='left')
            if len(S) < best_len:
                best_S, best_len = S[:], len(S)
            if r1[0] == 0 and r2[0] == 0:
                break

    # Final safety: validate best_S; if invalid, fall back to classic
    if not _validate(best_S, n):
        # try to validate current S
        if _validate(S, n):
            best_S = S[:]
        else:
            S = _classic_construct(n)
            best_S = S[:]

    return "".join(map(str, best_S))
