import time
import random
import itertools

def _classic_construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin for 150s limit

    if n == 1:
        return "1"

    # Standard recursive construction (guaranteed valid, length sum k!)
    base = _classic_construct(n)

    if n < 7:
        return "".join(map(str, base))

    # ---- Local search: multi-pass greedy removal of redundant characters ----
    def remove_pass(w, deadline):
        n = len(set(w))  # symbols are 1..n
        w = list(w)
        N = len(w)
        if N <= n:
            return w
        # count each permutation
        counts = {}
        for i in range(N - n + 1):
            p = tuple(w[i:i + n])
            counts[p] = counts.get(p, 0) + 1
        unique = set(p for p, c in counts.items() if c == 1)

        def try_remove(idx):
            # returns True if w[idx] was removed (and counts updated)
            lo = max(0, idx - n + 1)
            hi = min(idx, len(w) - n + 1)
            for s in range(lo, hi + 1):
                if tuple(w[s:s + n]) in unique:
                    return False
            # safe to remove w[idx]
            for s in range(lo, hi + 1):
                p = tuple(w[s:s + n])
                c = counts[p]
                if c == 1:
                    del counts[p]
                    unique.discard(p)
                elif c == 2:
                    counts[p] = 1
                    unique.add(p)
                else:
                    counts[p] = c - 1
            w.pop(idx)
            # new windows at boundary: starts in [idx-n+1, idx-1]
            for s in range(max(0, idx - n + 1), idx):
                if s < len(w) - n + 1:
                    p = tuple(w[s:s + n])
                    c = counts.get(p, 0)
                    if c == 0:
                        counts[p] = 1
                        unique.add(p)
                    elif c == 1:
                        counts[p] = 2
                        unique.discard(p)
                    else:
                        counts[p] = c + 1
            return True

        # Multiple passes: alternate forward/backward, plus a 2-removal pass
        passes = 0
        prev_len = len(w)
        while time.time() < deadline:
            passes += 1
            L = len(w)
            if L <= n:
                break
            # Pass A: forward greedy removal
            i = 0
            while i < L - n + 1:
                if time.time() > deadline:
                    return w
                if try_remove(i):
                    L = len(w)
                    # stay at same index (new char slid in)
                else:
                    i += 1
            # Pass B: backward greedy removal
            i = len(w) - 1
            while i >= 0:
                if time.time() > deadline:
                    return w
                if try_remove(i):
                    # stay at same index
                    i = min(i, len(w) - 1)
                else:
                    i -= 1
            # Check if we made progress
            if len(w) == prev_len:
                break
            prev_len = len(w)
        return w

    # Run removal pass on the base
    improved = remove_pass(base, deadline)
    if len(improved) < len(base):
        base = improved

    # Second removal pass on the result (catches new redundancies)
    if len(base) > n and time.time() < deadline - 2:
        improved2 = remove_pass(base, deadline)
        if len(improved2) < len(base):
            base = improved2

    # final safety verification
    try:
        allp = set(itertools.permutations(range(1, n + 1)))
        present = set()
        for i in range(len(base) - n + 1):
            present.add(tuple(base[i:i + n]))
        if not allp.issubset(present):
            base = _classic_construct(n)
    except Exception:
        base = _classic_construct(n)

    return "".join(map(str, base))
