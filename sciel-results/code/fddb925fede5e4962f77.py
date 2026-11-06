"""Rewritten program: NumPy-accelerated global greedy set cover with local-search polish."""
import time
import numpy as np

BASE = 51975


def _divisors():
    ds = []
    for d in range(3, BASE + 1, 2):
        if BASE % d == 0:
            ds.append(d)
    return ds


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin under 150s wall-clock
    divs = _divisors()

    uncovered = np.ones(BASE, dtype=np.uint8)
    classes = []
    used = set()

    # Global greedy: repeatedly pick (m, r) covering the most still-uncovered residues.
    while time.time() < deadline:
        best_gain = 0
        best_m = None
        best_r = None
        for m in divs:
            if m in used:
                continue
            # Count uncovered in each residue class mod m
            u = uncovered
            # Build counts per residue: sum of u[r::m]
            # Efficient: use reshape-free slicing
            counts = np.array([u[r::m].sum() for r in range(m)], dtype=np.int64)
            idx = int(np.argmax(counts))
            c = int(counts[idx])
            if c > best_gain:
                best_gain = c
                best_m = m
                best_r = idx
        if best_gain == 0:
            break
        classes.append([best_m, best_r])
        used.add(best_m)
        uncovered[best_r::best_m] = 0

    # Local-search polish: try to improve residue choice for each selected modulus.
    # Repeatedly scan for a swap that reduces uncovered count.
    polish_iters = 0
    while time.time() < deadline and polish_iters < 50:
        polish_iters += 1
        improved = False
        for i, (m, r) in enumerate(classes):
            # Try every alternative residue for this modulus
            u = uncovered.copy()
            # Remove current class
            u[r::m] = 1  # unmark (might already be 0 from other classes)
            # Actually we need to know which residues are covered ONLY by this class
            # Simpler: just try each r' and compute uncovered count
            current_uncovered = int(u.sum())
            for r2 in range(m):
                if r2 == r:
                    continue
                # Tentative: remove class (m, r), add (m, r2)
                u2 = uncovered.copy()
                u2[r::m] = 1  # remove old
                u2[r2::m] = 0  # add new
                new_uncovered = int(u2.sum())
                if new_uncovered < current_uncovered:
                    uncovered = u2
                    classes[i][1] = r2
                    improved = True
                    break
        if not improved:
            break

    return classes
