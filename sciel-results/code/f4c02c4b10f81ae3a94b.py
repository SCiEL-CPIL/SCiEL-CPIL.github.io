"""Ternary covering code K_3(8,1) - bitset/set-based local search."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0

    def elapsed():
        return time.time() - start_time

    def time_left():
        return deadline - time.time()

    # ---- Build the universe: all q^n words as integer labels ----
    total = q ** n
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    N = len(words)
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}

    # ---- For each word, its closed ball (itself + radius-1 neighbors) ----
    # For radius=1: a word covers itself and words differing in exactly one position
    covers = []
    for idx in range(N):
        w = words[idx]
        s = {idx}
        for pos in range(n):
            for alt in range(q):
                if alt == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = alt
                s.add(word_to_idx[tuple(nw)])
        covers.append(frozenset(s))

    # ---- Initial solution: tetracode direct sum (486 words) ----
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            init_codewords.append(tuple(h + list(tail)))

    selected = set(word_to_idx[tuple(cw)] for cw in init_codewords)

    # ---- Verify coverage; add missing if needed ----
    covered = set()
    for idx in selected:
        covered |= covers[idx]
    if len(covered) < N:
        # Greedy set cover for missing
        uncovered = set(range(N)) - covered
        available = set(range(N)) - selected
        while uncovered and available and time_left() > 2.0:
            best = None
            best_gain = 0
            # Sample if too many candidates
            cand = list(available)
            if len(cand) > 500:
                rng = np.random.default_rng()
                cand = list(rng.choice(len(cand), size=500, replace=False))
            for c in cand:
                gain = len(covers[c] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best = c
            if best is None or best_gain == 0:
                break
            selected.add(best)
            available.discard(best)
            covered |= covers[best]
            uncovered -= covers[best]
        # Force add remaining
        while uncovered and time_left() > 1.0:
            for c in covers:
                pass
            # Just add all that cover anything uncovered
            for c in range(N):
                if c not in selected and (covers[c] & uncovered):
                    selected.add(c)
                    covered |= covers[c]
                    uncovered -= covers[c]
            if not uncovered:
                break

    # ---- Local search: iterative removal + 1-swap ----
    def is_redundant(idx):
        """Check if removing idx still covers everything."""
        cov = covers[idx]
        # Check each word in cov is covered by another selected codeword
        for w in cov:
            found = False
            for s in selected:
                if s != idx and w in covers[s]:
                    found = True
                    break
            if not found:
                return False
        return True

    def is_redundant_fast(idx):
        """Faster redundancy check using set operations."""
        cov = covers[idx]
        # Union of covers of all other selected
        others_union = set()
        for s in selected:
            if s != idx:
                others_union |= covers[s]
        return cov.issubset(others_union)

    # Phase 1: Greedy removal (try each codeword, remove if redundant)
    improved = True
    while improved and time_left() > 5.0:
        improved = False
        for idx in list(selected):
            if time_left() < 5.0:
                break
            if is_redundant_fast(idx):
                selected.discard(idx)
                improved = True

    # Phase 2: 1-swap local search (remove one, add one if it helps)
    swap_iter = 0
    while time_left() > 5.0 and swap_iter < 200:
        swap_iter += 1
        found_swap = False
        for idx in list(selected):
            if time_left() < 5.0:
                break
            # Try removing idx
            selected.discard(idx)
            # Check what's now uncovered
            others_union = set()
            for s in selected:
                others_union |= covers[s]
            uncovered = set(range(N)) - others_union
            if not uncovered:
                # Pure removal worked
                found_swap = True
                break
            # Try to add one codeword to cover all uncovered
            best_add = None
            best_gain = 0
            for c in range(N):
                if c in selected:
                    continue
                gain = len(covers[c] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_add = c
            if best_add is not None and best_gain == len(uncovered):
                selected.add(best_add)
                found_swap = True
                break
            else:
                # Can't fix with one add, restore
                selected.add(idx)
        if not found_swap:
            # No more swaps in this pass; try a second pass
            if swap_iter > 100:
                break

    # Phase 3: Final greedy removal
    improved = True
    while improved and time_left() > 3.0:
        improved = False
        for idx in list(selected):
            if time_left() < 3.0:
                break
            if is_redundant_fast(idx):
                selected.discard(idx)
                improved = True

    # Final verification
    covered = set()
    for idx in selected:
        covered |= covers[idx]
    if len(covered) < N:
        # Emergency fix
        uncovered = set(range(N)) - covered
        for c in range(N):
            if c not in selected and (covers[c] & uncovered):
                selected.add(c)
                covered |= covers[c]
                uncovered -= covers[c]
            if not uncovered:
                break

    return [words[idx].tolist() for idx in sorted(selected)]
# EVOLVE-BLOCK-END
