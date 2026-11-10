# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    DEADLINE = 600
    p = 7
    k = 3
    q = 49
    total_points = q * q

    # Precompute field operations
    add_tbl = np.zeros((q, q), dtype=np.int64)
    sub_tbl = np.zeros((q, q), dtype=np.int64)
    mul_tbl = np.zeros((q, q), dtype=np.int64)
    inv_tbl = np.zeros(q, dtype=np.int64)

    for i in range(q):
        a0, a1 = i // p, i % p
        for j in range(q):
            b0, b1 = j // p, j % p
            add_tbl[i, j] = ((a0+b0)%p)*p + ((a1+b1)%p)
            sub_tbl[i, j] = ((a0-b0)%p)*p + ((a1-b1)%p)
            mul_tbl[i, j] = ((a0*b0+a1*b1*k)%p)*p + ((a0*b1+a1*b0)%p)
        if i != 0:
            denom = (a0*a0-a1*a1*k) % p
            if denom != 0:
                d_inv = pow(denom, p-2, p)
                inv_tbl[i] = (a0*d_inv%p)*p + ((-a1)%p*d_inv%p)

    # Unital membership
    U_arr = np.zeros(total_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2*xa + ya*ya - k*yb*yb) % p == 0:
                        U_arr[(xa*p+xb)*q + (ya*p+yb)] = True

    U_set = set(np.where(U_arr)[0].tolist())
    all_pts = set(range(total_points))
    outside_U = all_pts - U_set

    # Build lines
    lines = []
    for c in range(q):
        lines.append(np.array([c*q+y for y in range(q)], dtype=np.int64))
    for m in range(q):
        for b in range(q):
            lines.append(np.array([x*q + add_tbl[mul_tbl[m,x],b] for x in range(q)], dtype=np.int64))

    num_lines = len(lines)

    # Precompute U-points and outside-U points per line
    line_U_arrs = []
    line_out_arrs = []
    line_U_sets = []
    line_out_sets = []
    for L in lines:
        mask = U_arr[L]
        u_pts = L[mask]
        out_pts = L[~mask]
        line_U_arrs.append(u_pts)
        line_out_arrs.append(out_pts)
        line_U_sets.append(set(u_pts.tolist()))
        line_out_sets.append(set(out_pts.tolist()))

    # Multiple greedy attempts
    best_H = None
    best_size = float('inf')

    for attempt in range(15):
        if time.time() - start_time > DEADLINE - 120:
            break

        rng = random.Random(attempt * 7919 + 104729)
        uncovered = set(outside_U)
        H = set()
        H_arr = np.zeros(total_points, dtype=bool)
        uncovered_arr = np.zeros(total_points, dtype=bool)
        for pt in outside_U:
            uncovered_arr[pt] = True

        iteration = 0
        while uncovered and iteration < 80000:
            if time.time() - start_time > DEADLINE - 120:
                break

            best_idx = -1
            best_score = -1

            for i in range(num_lines):
                if not line_out_sets[i]:
                    continue
                out_arr = line_out_arrs[i]
                new_cov = int(np.sum(uncovered_arr[out_arr]))
                if new_cov == 0:
                    continue
                u_arr = line_U_arrs[i]
                new_cost = int(np.sum(~H_arr[u_arr]))

                score = new_cov / (new_cost + 1)
                if attempt > 0:
                    score += rng.random() * 0.003

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx == -1:
                break

            u_arr = line_U_arrs[best_idx]
            for pt in u_arr:
                if not H_arr[pt]:
                    H_arr[pt] = True
                    H.add(int(pt))

            out_arr = line_out_arrs[best_idx]
            for pt in out_arr:
                if uncovered_arr[pt]:
                    uncovered_arr[pt] = False
                    uncovered.remove(int(pt))

            iteration += 1

        if not uncovered and len(H) < best_size:
            best_size = len(H)
            best_H = set(H)

    if best_H is None:
        uncovered = set(outside_U)
        H = set()
        while uncovered:
            best_idx = -1
            best_score = -1
            for i in range(num_lines):
                if not line_out_sets[i]:
                    continue
                new_cov = len(line_out_sets[i] & uncovered)
                if new_cov == 0:
                    continue
                new_cost = len(line_U_sets[i] - H)
                score = new_cov / (new_cost + 1)
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx == -1:
                break
            H.update(line_U_sets[best_idx])
            uncovered -= line_out_sets[i]
        best_H = H

    C = U_set - best_H
    N = all_pts - C

    points = []
    for pt in sorted(N):
        x = pt // q
        y = pt % q
        xa, xb = x // p, x % p
        ya, yb = y // p, y % p
        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
