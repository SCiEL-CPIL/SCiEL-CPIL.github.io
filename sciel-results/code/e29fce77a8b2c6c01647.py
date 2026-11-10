# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using local search to maximize the complement.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401

    start_time = time.time()
    deadline = start_time + 580

    # Field: F_49 = F_7[alpha], alpha^2 = 3
    # Element (a, b) -> a + b*alpha, represented as int a*7+b
    def felem(a, b):
        return a * p + b

    def felem_inv(val):
        return (val // p, val % p)

    # Multiplication table
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ia, ib = felem_inv(i)
        for j in range(q):
            ja, jb = felem_inv(j)
            ra = (ia * ja + ib * jb * 3) % p
            rb = (ia * jb + ib * ja) % p
            mul_table[i, j] = felem(ra, rb)

    def point_to_idx(xa, xb, ya, yb):
        return xa * q * p + xb * q + ya * p + yb

    # 50 directions: v=(1,m) for m in F_49, plus v=(0,1)
    dir_list = []
    for ma in range(p):
        for mb in range(p):
            dir_list.append((1, felem(ma, mb)))
    dir_list.append((0, 1))
    n_dirs = len(dir_list)  # 50

    # Precompute line_id[d][x]: which line (0..48) point x is on in direction d
    line_id = np.zeros((n_dirs, total_points), dtype=np.int32)
    for d in range(n_dirs):
        vx, vy = dir_list[d]
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        x1 = felem(xa, xb)
                        x2 = felem(ya, yb)
                        x_idx = point_to_idx(xa, xb, ya, yb)
                        if vx == 0:
                            lid = x1
                        else:
                            m_x1 = mul_table[vy, x1]
                            lid = (x2 - m_x1) % q
                        line_id[d, x_idx] = lid

    # Precompute line_points[d][lid] = array of 49 point indices on line lid in direction d
    line_points = []
    for d in range(n_dirs):
        lines = {}
        for x in range(total_points):
            lid = line_id[d, x]
            if lid not in lines:
                lines[lid] = []
            lines[lid].append(x)
        lp = np.zeros((q, q), dtype=np.int32)
        for lid in range(q):
            if lid in lines:
                lp[lid, :] = lines[lid]
        line_points.append(lp)

    # Seed construction for initial S
    k_const = 3
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    S = np.zeros(total_points, dtype=np.bool_)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k_const) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        S[point_to_idx(xa, xb, ya, yb)] = True

    # Compute count[d][lid] and sat_count[x]
    def compute_state(S):
        count = np.zeros((n_dirs, q), dtype=np.int32)
        for x in range(total_points):
            if S[x]:
                for d in range(n_dirs):
                    count[d, line_id[d, x]] += 1
        sat_count = np.zeros(total_points, dtype=np.int32)
        for x in range(total_points):
            for d in range(n_dirs):
                lid = line_id[d, x]
                c = count[d, lid]
                if S[x]:
                    if c == 1:
                        sat_count[x] += 1
                else:
                    if c == 0:
                        sat_count[x] += 1
        return count, sat_count

    count, sat_count = compute_state(S)
    total_viol = int(np.sum(sat_count == 0))
    n_s = int(np.sum(S))
    print(f"Initial: |S|={n_s}, violations={total_viol}")

    # Greedy: try to add points
    candidates = [x for x in range(total_points) if not S[x]]
    random.seed(42)
    random.shuffle(candidates)

    for y in candidates:
        if time.time() - start_time > deadline - 60:
            break
        # Check if adding y is valid
        valid = True
        for d in range(n_dirs):
            lid = line_id[d, y]
            old_c = count[d, lid]
            line_pts = line_points[d][lid]
            for x in line_pts:
                if x == y:
                    continue
                if S[x]:
                    if old_c == 1 and sat_count[x] == 1:
                        valid = False
                        break
                else:
                    if old_c == 0 and sat_count[x] == 1:
                        valid = False
                        break
            if not valid:
                break
        if valid:
            S[y] = True
            n_s += 1
            for d in range(n_dirs):
                lid = line_id[d, y]
                old_c = count[d, lid]
                count[d, lid] = old_c + 1
                line_pts = line_points[d][lid]
                for x in line_pts:
                    if x == y:
                        continue
                    if S[x]:
                        if old_c == 1:
                            sat_count[x] -= 1
                    else:
                        if old_c == 0:
                            sat_count[x] -= 1
            total_viol = int(np.sum(sat_count == 0))

    print(f"After greedy: |S|={n_s}, violations={total_viol}")

    # SA phase: try swaps to increase |S| while maintaining validity
    best_S = S.copy()
    best_n = n_s

    T = 5.0
    cooling = 0.99995
    for step in range(300000):
        if time.time() - start_time > deadline - 20:
            break

        in_s = np.where(S)[0]
        not_in_s = np.where(~S)[0]
        if len(in_s) == 0 or len(not_in_s) == 0:
            break

        # Try a swap: remove y_r from S, add y_a to S
        y_r = in_s[random.randint(0, len(in_s) - 1)]
        y_a = not_in_s[random.randint(0, len(not_in_s) - 1)]
        if y_r == y_a:
            continue

        # Remove y_r
        S[y_r] = False
        for d in range(n_dirs):
            lid = line_id[d, y_r]
            old_c = count[d, lid]
            count[d, lid] = old_c - 1
            line_pts = line_points[d][lid]
            for x in line_pts:
                if x == y_r:
                    continue
                if S[x]:
                    if old_c - 1 == 1:
                        sat_count[x] += 1
                else:
                    if old_c - 1 == 0:
                        sat_count[x] += 1

        # Try to add y_a
        can_add = True
        for d in range(n_dirs):
            lid = line_id[d, y_a]
            old_c = count[d, lid]
            line_pts = line_points[d][lid]
            for x in line_pts:
                if x == y_a:
                    continue
                if S[x]:
                    if old_c == 1 and sat_count[x] == 1:
                        can_add = False
                        break
                else:
                    if old_c == 0 and sat_count[x] == 1:
                        can_add = False
                        break
            if not can_add:
                break

        if can_add:
            S[y_a] = True
            for d in range(n_dirs):
                lid = line_id[d, y_a]
                old_c = count[d, lid]
                count[d, lid] = old_c + 1
                line_pts = line_points[d][lid]
                for x in line_pts:
                    if x == y_a:
                        continue
                    if S[x]:
                        if old_c == 1:
                            sat_count[x] -= 1
                    else:
                        if old_c == 0:
                            sat_count[x] -= 1
            total_viol = int(np.sum(sat_count == 0))
            if total_viol == 0:
                best_S = S.copy()
                best_n = int(np.sum(S))
        else:
            # Revert: add y_r back
            S[y_r] = True
            for d in range(n_dirs):
                lid = line_id[d, y_r]
                old_c = count[d, lid]
                count[d, lid] = old_c + 1
                line_pts = line_points[d][lid]
                for x in line_pts:
                    if x == y_r:
                        continue
                    if S[x]:
                        if old_c == 1:
                            sat_count[x] -= 1
                    else:
                        if old_c == 0:
                            sat_count[x] -= 1

        T *= cooling

    S_final = best_S
    n_final = int(np.sum(S_final))
    print(f"Final: |S|={n_final}")

    # Build N = complement of S
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = point_to_idx(xa, xb, ya, yb)
                    if not S_final[idx]:
                        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
