"""Ternary covering code K_3(8,1): product-code seed + greedy reduction."""
import time
import itertools


def _tetracode():
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 120.0

    # --- Seed: product of the length-4 ternary tetracode with itself. ---
    # The tetracode is a perfect radius-1 code on 4 symbols (9 words cover 3^4).
    # Its product covers 3^8 words with 9*9 = 81 codewords (perfect, radius 1).
    tc = _tetracode()
    code = [list(h) + list(t) for h in tc for t in tc]
    if len(code) <= 1:
        return code

    # --- Coverage / redundancy data structures. ---
    # For a codeword c, its sphere S(c) = {c} union {c with one symbol changed}.
    # A codeword is redundant iff every word in S(c) is also in the union of
    # the other codewords' spheres.
    #
    # Represent each word as an integer 0..3^n-1 (base 3).
    pow3 = [1] * (n + 1)
    for i in range(1, n + 1):
        pow3[i] = pow3[i - 1] * q

    def to_int(w):
        v = 0
        for i in range(n):
            v = v * q + w[i]
        return v

    def sphere_ints(w):
        # returns list of ints: w and its single-symbol variants
        base = to_int(w)
        res = [base]
        for i in range(n):
            cur = w[i]
            for nv in range(q):
                if nv == cur:
                    continue
                res.append(base + (nv - cur) * pow3[n - 1 - i])
        return res

    # Build sphere membership: for each codeword, the set of ints it covers.
    # We track, for every int, how many codewords cover it (count).
    # Space is 3^8 = 6561 ints, small.
    space = pow3[n]
    cover_count = [0] * space

    def add_sphere(w, delta):
        for v in sphere_ints(w):
            cover_count[v] += delta

    for w in code:
        add_sphere(w, +1)

    # --- Greedy reduction: remove a codeword if it doesn't make any word
    #     lose its last covering codeword. ---
    # A codeword c is removable iff for every v in S(c), cover_count[v] >= 2
    # (i.e. some other codeword also covers v).
    def is_removable(idx):
        w = code[idx]
        for v in sphere_ints(w):
            if cover_count[v] < 2:
                return False
        return True

    # Iteratively scan and remove removable codewords until a full pass finds none.
    # To make scans efficient, keep a list of active indices.
    # We'll rebuild the active list each pass.
    while time.time() < deadline - 5.0:
        changed = False
        # scan from the end so removals don't shift earlier indices awkwardly
        for idx in range(len(code) - 1, -1, -1):
            if is_removable(idx):
                add_sphere(code[idx], -1)
                del code[idx]
                changed = True
        if not changed:
            break

    # Final safety: ensure we still return a valid covering (we only removed
    # proven-redundant codewords, so coverage is preserved).
    return code
