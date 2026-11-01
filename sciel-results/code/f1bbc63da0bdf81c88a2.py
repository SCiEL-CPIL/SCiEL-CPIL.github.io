import time, random, sys
from collections import defaultdict

_DEADLINE = 150.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _build_greedy(n, seed=0):
    """Greedy longest-overlap superpermutation via hash-based suffix matching."""
    rng = random.Random(seed)
    # Start with a base permutation (identity), then repeatedly append the
    # permutation that gives the longest overlap with the current tail.
    # Represent permutations as tuples of ints 1..n.
    # We maintain the current word as a list of ints.
    # For overlap, we need: for candidate perm p (len n), find max k in [1, n-1]
    # such that word[-k:] == p[:k]. Then append p[k:].
    # To find all remaining perms quickly, we track which (n-1)-subsequences
    # have been seen as "tails" to know which perms are still needed.

    # Precompute all perms
    import itertools
    all_perms = list(itertools.permutations(range(1, n+1)))
    # We'll track which perms are still "needed". A perm p is "covered" if it
    # appears as a window. We start with word = identity perm, so that one is covered.
    # Greedy: at each step, among all not-yet-covered perms, pick the one with
    # longest overlap with current tail; append its non-overlapping suffix.
    # Then mark all newly-covered windows as covered.

    word = list(all_perms[0])
    covered = set()
    # Mark initial window
    covered.add(tuple(word[:n]))
    # To find overlaps efficiently, index current word's suffixes by their
    # (n-1)-prefix? Actually overlap of length k means word[-k:] == p[:k].
    # For a candidate p, the max k is the largest k where p[:k] == word[-k:].
    # We can precompute for the current tail: for each k in 1..n-1, the tuple
    # word[-k:]. Then for each candidate p, check p[:k] against these.
    # But checking all candidates each step is O(n! * n) which for 5040*6 is fine
    # per step, but 5040 steps * 5040 candidates = 25M * small = ok in Python?
    # 25M tuple comparisons of length up to 6 might be ~5-10s. Acceptable.
    # But we also need to recompute "covered" each step: after appending, new
    # windows appear. We can just recompute all windows in the new region.

    # Optimization: maintain a set of "active" candidates (not yet covered).
    # At each step, for each active candidate, compute overlap with tail.
    # Pick max overlap (tie-break: lexicographically smallest to be deterministic,
    # or random among ties for restarts).
    # After appending, remove any candidate that is now covered.

    active = set(all_perms)
    active.discard(tuple(word[:n]))

    # Precompute tail tuples for quick comparison
    def tail_tuples(w, n):
        # returns dict k -> tuple(w[-k:]) for k in 1..n-1
        d = {}
        for k in range(1, n):
            d[k] = tuple(w[-k:])
        return d

    step = 0
    while active:
        if _remaining() < 1.0:
            break
        step += 1
        tails = tail_tuples(word, n)
        best_p = None
        best_k = -1
        best_tie = None
        for p in active:
            # find max k where p[:k] == tails[k]
            # check from n-1 down to 1
            for k in range(n-1, 0, -1):
                if p[:k] == tails[k]:
                    if k > best_k:
                        best_k = k
                        best_p = p
                        best_tie = p
                        break
                    elif k == best_k:
                        # tie: pick random or lex
                        if rng.random() < 0.5:
                            best_tie = p
                    break
        if best_p is None:
            # Should not happen; fallback
            best_p = next(iter(active))
            best_k = 0
        # Append best_p[best_k:]
        word.extend(best_p[best_k:])
        active.discard(best_p)
        # Mark newly covered windows: the new windows are those ending in the
        # newly appended region. The new length is len(word). New windows start
        # at positions from (old_len - n + 1) to (new_len - n).
        # We don't track old_len easily here; just recompute all windows in the
        # last n appended chars region. Simpler: recompute all windows that end
        # in the new chars. The new chars are best_p[best_k:], length = n - best_k.
        # New windows: for each start s such that the window [s, s+n) includes
        # at least one new char, i.e., s > old_len - n.
        # Let's just recompute windows in the range [len(word)-n - (n-best_k), len(word)-n]
        new_len = len(word)
        start_from = new_len - n - (n - best_k) + 1  # first new window start
        if start_from < 0:
            start_from = 0
        for s in range(start_from, new_len - n + 1):
            w = tuple(word[s:s+n])
            if w in active:
                active.discard(w)
    return word

def _validate(word, n):
    """Check that all n! permutations appear as windows."""
    if len(word) < n:
        return False
    import itertools
    expected = set(itertools.permutations(range(1, n+1)))
    seen = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n:
            seen.add(w)
        if len(seen) == len(expected):
            return True
    return len(seen) == len(expected)

def construct(n):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 150.0

    if n <= 1:
        return "1"

    # For small n, we can use known optimal or near-optimal constructions.
    # For n=7, use greedy with a few restarts.
    best_word = None
    best_len = float('inf')

    # Try a few restarts with different random seeds for tie-breaking
    for seed in range(20):
        if _remaining() < 2.0:
            break
        word = _build_greedy(n, seed=seed)
        if len(word) < best_len:
            best_len = len(word)
            best_word = word
        # Early stop if we hit a known good length
        if n == 7 and best_len <= 5905:
            break

    if best_word is None:
        # Fallback: classic recursive construction
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
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        best_word = word

    return "".join(map(str, best_word))
