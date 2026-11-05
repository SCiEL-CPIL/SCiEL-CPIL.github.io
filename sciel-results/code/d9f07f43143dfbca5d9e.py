"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random


def construct(q, n, radius):
    start = time.time()
    deadline = start + 115.0  # safety margin under 120s

    # ---- 1. Seed: direct sum of ternary tetracode (9 words len 4) with full tails ----
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    if time.time() > deadline:
        return code

    # ---- 2. Represent code as set of tuples for O(1) membership ----
    code_set = set(tuple(c) for c in code)

    def neighbors(c):
        """Yield all words within Hamming distance radius of c."""
        c = list(c)
        for _ in range(radius):
            for pos in range(n):
                old = c[pos]
                for val in range(q):
                    if val == old:
                        continue
                    c[pos] = val
                    yield tuple(c)
                c[pos] = old

    # Precompute coverage: for each codeword, the set of words it covers
    # We'll use a counter approach: count how many codewords cover each word.
    # Space is 3^8 = 6561, manageable.
    def build_coverage(code_list):
        """Return dict mapping word -> count of codewords covering it."""
        cov = {}
        for c in code_list:
            for w in neighbors(c):
                cov[w] = cov.get(w, 0) + 1
        return cov

    # ---- 3. Greedy removal: iteratively remove codewords that don't break coverage ----
    def try_remove(code_set, code_list):
        """Try to remove redundant codewords. Returns (new_code_set, new_code_list, removed_count)."""
        removed = 0
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Build coverage
            cov = {}
            for c in code_list:
                for w in neighbors(c):
                    cov[w] = cov.get(w, 0) + 1
            # Try removing each codeword
            for i, c in enumerate(list(code_list)):
                if time.time() > deadline - 5:
                    break
                # Check if removing c leaves all its covered words still covered
                safe = True
                for w in neighbors(c):
                    if cov.get(w, 0) <= 1:
                        safe = False
                        break
                if safe:
                    code_set.discard(tuple(c))
                    code_list.pop(i)
                    removed += 1
                    changed = True
                    break  # restart coverage computation
        return code_set, code_list, removed

    code_set, code, _ = try_remove(code_set, code)

    if time.time() > deadline - 10:
        return code

    # ---- 4. Local search: swap operations to reduce code size ----
    # For each codeword, try replacing it with a random word that covers more uncovered words
    # or try removing a pair and adding one word that covers both.
    rng = random.Random(42)

    def coverage_count(code_list):
        """Return the set of covered words."""
        covered = set()
        for c in code_list:
            for w in neighbors(c):
                covered.add(w)
        return covered

    def all_words():
        return set(itertools.product(range(q), repeat=n))

    full_space = all_words()

    # Current coverage
    covered = coverage_count(code)
    uncovered = full_space - covered
    # If uncovered is non-empty, we have an invalid code; fix it first
    if uncovered:
        # Add codewords to cover uncovered words
        for w in list(uncovered):
            # Find a codeword that covers w and maximizes new coverage
            best_c = None
            best_gain = -1
            # Try random candidates
            for _ in range(50):
                cand = tuple(rng.randint(0, q - 1) for _ in range(n))
                new_cov = set(neighbors(cand)) - covered
                if len(new_cov) > best_gain:
                    best_gain = len(new_cov)
                    best_c = cand
            if best_c is not None:
                code.append(list(best_c))
                code_set.add(best_c)
                covered |= set(neighbors(best_c))
                uncovered -= set(neighbors(best_c))
        if time.time() > deadline - 5:
            return code

    # Now code is valid. Try to shrink via swaps.
    def try_swap_remove(code, code_set, covered):
        """Try to remove a codeword and add a better one, or remove two and add one."""
        improved = False
        # Try single removal + replacement
        for i in range(len(code)):
            if time.time() > deadline - 5:
                break
            c = code[i]
            # Words only covered by c
            only_by_c = set()
            for w in neighbors(c):
                # Check if w is covered by another codeword
                is_covered_other = False
                for j, cj in enumerate(code):
                    if j == i:
                        continue
                    # Check if w is within radius of cj
                    d = sum(1 for a, b in zip(w, cj) if a != b)
                    if d <= radius:
                        is_covered_other = True
                        break
                if not is_covered_other:
                    only_by_c.add(w)
            if not only_by_c:
                # Can remove c entirely
                code.pop(i)
                code_set.discard(tuple(c))
                improved = True
                break
        return improved

    # More aggressive: try to find a codeword replacement that covers all of c's unique words
    # plus additional words, allowing removal of c.
    for iteration in range(50):
        if time.time() > deadline - 5:
            break
        improved = False
        # Build a map: word -> list of codeword indices covering it
        word_to_cws = {}
        for i, c in enumerate(code):
            for w in neighbors(c):
                word_to_cws.setdefault(w, []).append(i)

        # For each codeword, find words it uniquely covers
        for i in range(len(code)):
            if time.time() > deadline - 3:
                break
            c = code[i]
            unique_words = set()
            for w in neighbors(c):
                if len(word_to_cws.get(w, [])) == 1:
                    unique_words.add(w)
            if not unique_words:
                # Remove
                code.pop(i)
                code_set.discard(tuple(c))
                improved = True
                break
            # Try to find a replacement codeword that covers all unique_words
            # and is not already in code
            best_cand = None
            best_extra = -1
            for _ in range(30):
                cand = tuple(rng.randint(0, q - 1) for _ in range(n))
                if cand in code_set:
                    continue
                cand_neighbors = set(neighbors(cand))
                if unique_words.issubset(cand_neighbors):
                    extra = len(cand_neighbors - set(neighbors(c)))
                    if extra > best_extra:
                        best_extra = extra
                        best_cand = cand
            if best_cand is not None:
                code.pop(i)
                code_set.discard(tuple(c))
                code.append(list(best_cand))
                code_set.add(best_cand)
                improved = True
                break
        if not improved:
            # Try random perturbation: remove a random codeword, see if still valid,
            # if not, add back
            if len(code) > 100 and rng.random() < 0.3:
                idx = rng.randrange(len(code))
                c = code.pop(idx)
                code_set.discard(tuple(c))
                # Check validity
                covered_now = set()
                for cj in code:
                    for w in neighbors(cj):
                        covered_now.add(w)
                if full_space.issubset(covered_now):
                    improved = True
                else:
                    code.append(c)
                    code_set.add(tuple(c))
        if not improved and iteration > 10:
            break

    return code
# EVOLVE-BLOCK-END
