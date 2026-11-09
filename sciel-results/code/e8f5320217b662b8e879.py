# EVOLVE-BLOCK-START
import time
import numpy as np
import collections

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    total = q * q  # 2401
    deadline = time.time() + 600.0
    deadline_soft = deadline - 30.0

    # Field F_49 = F_7[alpha], alpha^2 = 3
    alpha2 = 3
    add = np.zeros((q, q), dtype=np.int64)
    mul = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        ai, bi = i // p, i % p
        for j in range(q):
            aj, bj = j // p, j % p
            add[i, j] = ((ai + aj) % p) * p + ((bi + bj) % p)
            mul[i, j] = ((ai * aj + bi * bj * alpha2) % p) * p + ((ai * bj + bi * aj) % p)

    # Precompute lines: lines[point_idx][dir_idx] = list of q point-indices
    # dir 0..q-1: direction (1, m) for m = 0..q-1
    # dir q: vertical (0, 1)
    lines = []
    for x in range(q):
        for y in range(q):
            point_idx = x * q + y
            point_lines = []
            for m in range(q):
                line_pts = []
                for t in range(q):
                    xt = add[x, t]
                    yt = add[y, mul[m, t]]
                    line_pts.append(xt * q + yt)
                point_lines.append(line_pts)
            # Vertical
            line_pts = []
            for t in range(q):
                yt = add[y, t]
                line_pts.append(x * q + yt)
            point_lines.append(line_pts)
            lines.append(point_lines)

    # State: in_N[i] = True means point i is in N.
    # Start with all in N.
    in_N = np.ones(total, dtype=bool)
    # good_count[x] = number of directions d for which the punctured line
    # through x in dir d is fully in N.
    good_count = np.full(total, q + 1, dtype=np.int32)
    # For each x and d, track whether the line through x in dir d is "good"
    # (all q-1 other points in N). Initially all good.
    # line_good[x, d] = True if punctured line through x in dir d is fully in N.
    line_good = np.ones((total, q + 1), dtype=bool)

    # When we remove point p from N (in_N[p] = False), we need to update:
    # For each direction d, for each x on the line through p in dir d (x != p):
    #   The line through x in dir d now contains p (excluded), so it's no longer good.
    #   line_good[x, d] = False, good_count[x] -= 1.
    #   If good_count[x] becomes 0, x is bad -> we can't remove p.
    #
    # To check if removing p is safe: simulate the removal and check if any x
    # has good_count[x] become 0.
    #
    # Actually, we can check: for each affected x, if good_count[x] == 1 and the
    # only good direction is d (where p is on the line), then removing p makes x bad.
    #
    # But we need to know WHICH direction is good for x. Let's track good_count
    # and also check line_good.

    # Greedy: try to remove points in random order.
    rng = np.random.default_rng(42)
    best_removed = 0
    best_in_N = None

    for trial in range(100):
        if time.time() > deadline_soft:
            break
        in_N = np.ones(total, dtype=bool)
        good_count = np.full(total, q + 1, dtype=np.int32)
        line_good = np.ones((total, q + 1), dtype=bool)

        order = rng.permutation(total)
        removed = 0
        for p in order:
            if time.time() > deadline_soft:
                break
            if not in_N[p]:
                continue
            # Check if removing p is safe
            safe = True
            # For each direction d, for each x on the line through p in dir d (x != p):
            #   if line_good[x, d] is True, then removing p will make it False.
            #   if good_count[x] == 1, then x becomes bad.
            for d in range(q + 1):
                for x in lines[p][d]:
                    if x == p:
                        continue
                    if line_good[x, d]:
                        if good_count[x] == 1:
                            safe = False
                            break
                if not safe:
                    break
            if safe:
                # Remove p from N
                in_N[p] = False
                removed += 1
                # Update line_good and good_count
                for d in range(q + 1):
                    for x in lines[p][d]:
                        if x == p:
                            continue
                        if line_good[x, d]:
                            line_good[x, d] = False
                            good_count[x] -= 1

        if removed > best_removed:
            best_removed = removed
            best_in_N = in_N.copy()

    # Build N from best_in_N
    if best_in_N is None:
        points = []
        for x in range(q):
            for y in range(q):
                xa, xb = x // p, x % p
                ya, yb = y // p, y % p
                points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)

    points = []
    for x in range(q):
        for y in range(q):
            idx = x * q + y
            if best_in_N[idx]:
                xa, xb = x // p, x % p
                ya, yb = y // p, y % p
                points.append([[xa, xb], [ya, yb]])

    if len(points) == 0:
        points = [[[0, 0], [0, 0]]]

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
