"""Hypergraph Ramsey R(4,5;3) construction: maximise n with no red K4^3 and no blue K5^3."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin under 150s wall-clock

    def elapsed():
        return time.time() - t_start

    # Precompute all 4-subsets and 5-subsets of vertices for validation
    # and for fast checking.

    def validate(n, colors):
        """Return True if no red K4^3 and no blue K5^3."""
        m = len(colors)
        if m != n * (n - 1) * (n - 2) // 6:
            return False
        # Build a set of red triples as frozensets for fast lookup
        red_set = set()
        blue_set = set()
        idx = 0
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    if colors[idx] == 0:
                        red_set.add((a, b, c))
                    else:
                        blue_set.add((a, b, c))
                    idx += 1
        # Check red K4^3: any 4 vertices with all 4 triples red
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if ((a, b, c) in red_set and (a, b, d) in red_set and
                (a, c, d) in red_set and (b, c, d) in red_set):
                return False
        # Check blue K5^3: any 5 vertices with all 10 triples blue
        for quint in combinations(range(n), 5):
            ok = True
            for tri in combinations(quint, 3):
                if tri not in blue_set:
                    ok = False
                    break
            if ok:
                return False
        return True

    # Strategy:
    # 1. Try known small constructions / algebraic colorings for various n.
    # 2. Use local search (flip-based) starting from random or algebraic seeds
    #    to find valid colorings for larger n.
    # 3. Return the largest valid n found.

    best_n = 0
    best_colors = None

    def try_coloring(n, colors):
        nonlocal best_n, best_colors
        if n > best_n and validate(n, colors):
            best_n = n
            best_colors = list(colors)
            return True
        return False

    # --- Algebraic / structured colorings ---
    # Color triple (a,b,c) by (a+b+c) mod 3: red if 0, blue otherwise.
    # This gives n=9 valid (as in seed). Try other moduli and offsets.

    def algebraic_coloring(n, mod, red_residues):
        colors = []
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    s = (a + b + c) % mod
                    colors.append(0 if s in red_residues else 1)
        return colors

    # Try various algebraic colorings
    for n in range(9, 35):
        if elapsed() > deadline:
            break
        for mod in [2, 3, 4, 5, 6, 7]:
            for k in range(mod):
                # red if sum mod mod == k
                colors = algebraic_coloring(n, mod, {k})
                try_coloring(n, colors)
                # red if sum mod mod in {k, k+1} (for mod>=2)
                if mod >= 2:
                    colors2 = algebraic_coloring(n, mod, {k % mod, (k + 1) % mod})
                    try_coloring(n, colors2)
        # Also try product-based: red if (a*b*c) % mod == 0
        for mod in [2, 3, 4, 5]:
            colors = []
            for a in range(n):
                for b in range(a + 1, n):
                    for c in range(b + 1, n):
                        colors.append(0 if (a * b * c) % mod == 0 else 1)
            try_coloring(n, colors)

    # --- Local search / simulated annealing ---
    # Start from a random coloring and flip bits to eliminate violations.

    def local_search(n, max_time=30.0, seed=None):
        if seed is None:
            seed = int(time.time() * 1000) % (2**31)
        rng = np.random.RandomState(seed)

        total_triples = n * (n - 1) * (n - 2) // 6
        colors = rng.randint(0, 2, size=total_triples).tolist()

        # Precompute for each 4-subset the list of 4 triple indices
        # and for each 5-subset the list of 10 triple indices
        triple_index = {}
        idx = 0
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    triple_index[(a, b, c)] = idx
                    idx += 1

        red_k4_indices = []  # list of 4-tuples of triple indices for each 4-subset
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            red_k4_indices.append([
                triple_index[(a, b, c)],
                triple_index[(a, b, d)],
                triple_index[(a, c, d)],
                triple_index[(b, c, d)]
            ])

        blue_k5_indices = []  # list of 10-tuples of triple indices for each 5-subset
        for quint in combinations(range(n), 5):
            idxs = [triple_index[tri] for tri in combinations(quint, 3)]
            blue_k5_indices.append(idxs)

        def count_violations(colors):
            v = 0
            for idxs in red_k4_indices:
                if all(colors[i] == 0 for i in idxs):
                    v += 1
            for idxs in blue_k5_indices:
                if all(colors[i] == 1 for i in idxs):
                    v += 1
            return v

        v = count_violations(colors)
        if v == 0:
            return colors

        t = 2.0
        cooling = 0.9995
        min_t = 0.01
        t_end = time.time() + max_time

        while time.time() < t_end and v > 0:
            # Pick a random triple to flip
            i = rng.randint(0, total_triples)
            old = colors[i]
            colors[i] = 1 - old

            # Recompute violations (expensive but n<=34 so manageable)
            new_v = count_violations(colors)
            delta = new_v - v
            if delta <= 0 or rng.random() < np.exp(-delta / t):
                v = new_v
            else:
                colors[i] = old

            t = max(min_t, t * cooling)

        return colors

    # Try local search for increasing n
    for n in range(10, 35):
        if elapsed() > deadline - 10.0:
            break
        # Try a few seeds
        for attempt in range(3):
            if elapsed() > deadline - 5.0:
                break
            colors = local_search(n, max_time=min(20.0, max(5.0, deadline - elapsed() - 5.0)),
                                  seed=42 + attempt * 1000 + n)
            try_coloring(n, colors)
            if best_n >= n:
                break  # found valid for this n, move to next

    # Final validation of best result
    if best_n > 0 and best_colors is not None:
        if not validate(best_n, best_colors):
            # Fallback to the known valid n=9 algebraic
            n = 9
            best_colors = [0 if (a + b + c) % 3 == 0 else 1
                          for a, b, c in combinations(range(n), 3)]
            best_n = n

    if best_n == 0:
        n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1
                      for a, b, c in combinations(range(n), 3)]
        best_n = n

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
