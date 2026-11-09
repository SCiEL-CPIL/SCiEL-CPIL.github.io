import numpy as np
import time
import random


def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    n_points = q * q  # 2401

    # Directions: (1, m) for m in F_q, and (0, 1).
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 0), (0, 1)))
    n_dirs = len(directions)  # 50

    def f_mul(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)

    def f_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    field_elements = []
    for a in range(p):
        for b in range(p):
            field_elements.append((a, b))

    def get_coords(i):
        xa = i // 343
        xb = (i // 49) % 7
        ya = (i // 7) % 7
        yb = i % 7
        return xa, xb, ya, yb

    def get_index(xa, xb, ya, yb):
        return xa * 343 + xb * 49 + ya * 7 + yb

    # Precompute line_members: shape (n_points, n_dirs, q-1)
    line_members = np.zeros((n_points, n_dirs, q-1), dtype=np.int32)
    for i in range(n_points):
        xa, xb, ya, yb = get_coords(i)
        pt = ((xa, xb), (ya, yb))
        for d in range(n_dirs):
            v = directions[d]
            for t_idx in range(1, q):
                t = field_elements[t_idx]
                tv_x = f_mul(t, v[0])
                tv_y = f_mul(t, v[1])
                new_x = f_add(pt[0], tv_x)
                new_y = f_add(pt[1], tv_y)
                nxa, nxb = new_x
                nya, nyb = new_y
                j = get_index(nxa, nxb, nya, nyb)
                line_members[i, d, t_idx-1] = j

    # Initial S using unital construction
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    S_initial = np.zeros(n_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - 3 * yb * yb) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        idx = get_index(xa, xb, ya, yb)
                        S_initial[idx] = True

    start_time = time.time()
    deadline = start_time + 570

    def init_state(S_arr):
        is_valid = np.ones((n_points, n_dirs), dtype=bool)
        valid_dirs = np.zeros(n_points, dtype=np.int32)
        for x in range(n_points):
            for d in range(n_dirs):
                members = line_members[x, d, :]
                if np.any(S_arr[members]):
                    is_valid[x, d] = False
                else:
                    valid_dirs[x] += 1
        return is_valid, valid_dirs

    def run_greedy(order_seed):
        rng = random.Random(order_seed)
        S_arr = S_initial.copy()
        is_valid, valid_dirs = init_state(S_arr)
        candidates = list(range(n_points))
        rng.shuffle(candidates)
        count = int(np.sum(S_arr))
        for p_idx in candidates:
            if time.time() > deadline:
                break
            if S_arr[p_idx]:
                continue
            valid = True
            for d in range(n_dirs):
                members = line_members[p_idx, d, :]
                risky = np.sum((valid_dirs[members] == 1) & is_valid[members, d])
                if risky > 0:
                    valid = False
                    break
            if valid:
                S_arr[p_idx] = True
                count += 1
                for d in range(n_dirs):
                    members = line_members[p_idx, d, :]
                    was_valid = is_valid[members, d]
                    if np.any(was_valid):
                        is_valid[members[was_valid], d] = False
                        valid_dirs[members[was_valid]] -= 1
        return S_arr, count

    best_S = S_initial.copy()
    best_count = int(np.sum(S_initial))

    seed = 42
    while time.time() < deadline:
        S_arr, count = run_greedy(seed)
        if count > best_count:
            best_S = S_arr
            best_count = count
        seed += 1

    N_indices = np.where(~best_S)[0]
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for idx, i in enumerate(N_indices):
        xa, xb, ya, yb = get_coords(i)
        result[idx, 0, 0] = xa
        result[idx, 0, 1] = xb
        result[idx, 1, 0] = ya
        result[idx, 1, 1] = yb

    return result
