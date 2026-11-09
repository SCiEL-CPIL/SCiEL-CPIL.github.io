# EVOLVE-BLOCK-START
import numpy as np
import time


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.

    Uses the Keich binary construction for the base positions, then refines
    via a coordinate descent on the 1-D projection variables (x_base[i], y_base[j]).
    The union volume is separable: V = Vx * Vy, so we minimize Vx + Vy.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 600s kill

    # --- Initial Keich construction ---
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    x_base = keich.copy()
    y_base = keich.copy()

    # --- Helper: compute 1-D union length for a set of intervals ---
    def union_length_1d(base, N):
        """
        Given base positions base[0..N-1], each interval i is
        [base[i], base[i] + (i+1)/N] (at c=0) but sheared: at height c,
        interval i is [base[i] + c*i/N, base[i] + (i+1)/N + c*i/N].
        
        The union over c in [0,1] of these moving intervals: for each point x,
        it's covered if exists i, c in [0,1] such that
        base[i] + c*i/N <= x <= base[i] + (i+1)/N + c*i/N.
        
        This is equivalent to: the union of all points x that lie in at least one
        of the N "swept" regions. Each tube i sweeps the region
        [min_{c in [0,1]} (base[i] + c*i/N), max_{c in [0,1]} (base[i] + (i+1)/N + c*i/N)]
        which is [base[i], base[i] + (i+1)/N + i/N] if i >= 0, but more precisely:
        
        For tube i, the x-interval at height c is [base[i] + c*i/N, base[i] + (i+1)/N + c*i/N].
        As c ranges [0,1], the left endpoint goes from base[i] to base[i] + i/N,
        and the right endpoint goes from base[i] + (i+1)/N to base[i] + (i+1)/N + i/N.
        
        The union of these intervals over c is the interval
        [base[i], base[i] + (i+1)/N + i/N] = [base[i], base[i] + (2i+1)/N].
        
        Wait, that's not quite right. The union of a family of intervals that all
        contain a common point (if they do) is just the convex hull. But they may not
        all overlap. Let me think again.
        
        Actually, for fixed i, the set of (x, c) pairs is a parallelogram in the (x,c)
        plane. The projection onto x is the union of intervals
        [base[i] + c*i/N, base[i] + (i+1)/N + c*i/N] for c in [0,1].
        Since both endpoints are linear in c with the same slope i/N, the interval
        shifts uniformly. The union over c is simply
        [base[i], base[i] + (i+1)/N + i/N] = [base[i], base[i] + (2i+1)/N].
        
        Hmm, but that's the bounding box, not necessarily the union if the intervals
        don't all overlap. Actually, since the interval at c is [L(c), R(c)] where
        L(c) = base[i] + c*i/N and R(c) = L(c) + (i+1)/N, and L is continuous and
        monotone (for i >= 0), the union is [L(0), R(1)] = [base[i], base[i] + (i+1)/N + i/N].
        
        So the 1-D "shadow" of tube i is the interval [base[i], base[i] + (2i+1)/N].
        The 1-D union length is the length of the union of these N intervals.
        """
        intervals = []
        for i in range(N):
            left = base[i]
            right = base[i] + (2 * i + 1) / N
            intervals.append((left, right))
        # Merge intervals to compute union length
        intervals.sort()
        total = 0.0
        cur_left, cur_right = intervals[0]
        for l, r in intervals[1:]:
            if l <= cur_right:
                cur_right = max(cur_right, r)
            else:
                total += cur_right - cur_left
                cur_left, cur_right = l, r
        total += cur_right - cur_left
        return total

    # --- Objective: minimize Vx + Vy ---
    def objective(xb, yb):
        return union_length_1d(xb, N) + union_length_1d(yb, N)

    # --- Coordinate descent with adaptive step size ---
    best_obj = objective(x_base, y_base)
    step = 0.05
    no_improve_count = 0

    while time.time() < deadline:
        improved = False
        # Update x_base
        for i in range(N):
            old = x_base[i]
            for delta in [step, -step, step/2, -step/2, step/4, -step/4]:
                x_base[i] = old + delta
                obj = objective(x_base, y_base)
                if obj < best_obj - 1e-12:
                    best_obj = obj
                    improved = True
                    break
                x_base[i] = old
            if not improved:
                x_base[i] = old
        # Update y_base
        for j in range(N):
            old = y_base[j]
            for delta in [step, -step, step/2, -step/2, step/4, -step/4]:
                y_base[j] = old + delta
                obj = objective(x_base, y_base)
                if obj < best_obj - 1e-12:
                    best_obj = obj
                    improved = True
                    break
                y_base[j] = old
            if not improved:
                y_base[j] = old

        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            step *= 0.7
            if step < 1e-6:
                break

    # --- Final polish: finer local search ---
    if time.time() < deadline - 5.0:
        step = 1e-4
        for _ in range(20):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                old = x_base[i]
                for delta in [step, -step]:
                    x_base[i] = old + delta
                    obj = objective(x_base, y_base)
                    if obj < best_obj - 1e-14:
                        best_obj = obj
                        improved = True
                        break
                    x_base[i] = old
                if not improved:
                    x_base[i] = old
            for j in range(N):
                old = y_base[j]
                for delta in [step, -step]:
                    y_base[j] = old + delta
                    obj = objective(x_base, y_base)
                    if obj < best_obj - 1e-14:
                        best_obj = obj
                        improved = True
                        break
                    y_base[j] = old
                if not improved:
                    y_base[j] = old
            if not improved:
                break

    # --- Build final positions ---
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = x_base[i_flat]
    base_y = y_base[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
