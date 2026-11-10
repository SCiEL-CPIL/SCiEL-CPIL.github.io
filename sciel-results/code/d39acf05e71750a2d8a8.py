# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 600s limit

    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    n = q * q  # 2401

    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 * a2 + k * b1 * b2) % p, (a1 * b2 + a2 * b1) % p)

    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p)
                  for ya in range(p) for yb in range(p)]

    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)

    slope_lines = {}
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            lines = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((ma, mb), (xa, xb))
                key = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines[key].append(i)
            slope_lines[(ma, mb)] = dict(lines)

    point_lines = [[] for _ in range(n)]
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        point_lines[i].append(('vert', (xa, xb), None))
        for ma in range(p):
            for mb in range(p):
                if ma == 0 and mb == 0:
                    continue
                mx = mul_f49((ma, mb), (xa, xb))
                key = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                point_lines[i].append(('slope', (ma, mb), key))

    def lkey_of(rep):
        if rep[0] == 'vert':
            return ('vert', rep[1])
        else:
            return ('slope', rep[1], rep[2])

    def line_pts_of(rep):
        if rep[0] == 'vert':
            return vert_lines[rep[1]]
        else:
            return slope_lines[rep[1]][rep[2]]

    def build_line_counts(in_N):
        line_not_in_N_count = {}
        for key, pts in vert_lines.items():
            line_not_in_N_count[('vert', key)] = 0
        for (ma, mb), lines in slope_lines.items():
            for key, pts in lines.items():
                line_not_in_N_count[('slope', (ma, mb), key)] = 0
        for i in range(n):
            if not in_N[i]:
                for rep in point_lines[i]:
                    line_not_in_N_count[lkey_of(rep)] += 1
        return line_not_in_N_count

    def compute_good_witness(in_N, line_not_in_N_count):
        point_good_witness = [0] * n
        for i in range(n):
            cnt = 0
            for rep in point_lines[i]:
                total_nin = line_not_in_N_count[lkey_of(rep)]
                nin_excl_i = total_nin - (0 if in_N[i] else 1)
                if nin_excl_i == 0:
                    cnt += 1
            point_good_witness[i] = cnt
        return point_good_witness

    def is_valid(in_N):
        # Every point (in N or not) must have a clean line through it.
        line_not_in_N_count = build_line_counts(in_N)
        for i in range(n):
            has_clean = False
            for rep in point_lines[i]:
                total_nin = line_not_in_N_count[lkey_of(rep)]
                nin_excl_i = total_nin - (0 if in_N[i] else 1)
                if nin_excl_i == 0:
                    has_clean = True
                    break
            if not has_clean:
                return False
        return True

    def greedy_extend(in_N, line_not_in_N_count, point_good_witness, rng, max_iter=400):
        order = list(range(n))
        rng.shuffle(order)
        for iteration in range(max_iter):
            if time.time() > deadline - 20:
                break
            found_removal = False
            for i in order:
                if time.time() > deadline - 20:
                    break
                if not in_N[i]:
                    continue
                can_remove = True
                for rep in point_lines[i]:
                    lkey = lkey_of(rep)
                    old_count = line_not_in_N_count[lkey]
                    new_count = old_count + 1
                    line_pts = line_pts_of(rep)
                    for j in line_pts:
                        if j == i:
                            continue
                        was_good = (old_count - (0 if in_N[j] else 1)) == 0
                        still_good = (new_count - (0 if in_N[j] else 1)) == 0
                        if was_good and not still_good:
                            if point_good_witness[j] <= 1:
                                can_remove = False
                                break
                    if not can_remove:
                        break
                if can_remove:
                    in_N[i] = False
                    found_removal = True
                    for rep in point_lines[i]:
                        line_not_in_N_count[lkey_of(rep)] += 1
                    for rep in point_lines[i]:
                        line_pts = line_pts_of(rep)
                        for j in line_pts:
                            if j == i:
                                continue
                            cnt = 0
                            for rep_j in point_lines[j]:
                                total_nin = line_not_in_N_count[lkey_of(rep_j)]
                                nin_excl_j = total_nin - (0 if in_N[j] else 1)
                                if nin_excl_j == 0:
                                    cnt += 1
                            point_good_witness[j] = cnt
                    break
            if not found_removal:
                rng.shuffle(order)
                if iteration > 5:
                    break

    def make_hyperplane_seed(rng):
        while True:
            c = [int(rng.integers(0, p)) for _ in range(4)]
            if any(cc != 0 for cc in c):
                break
        in_N = [True] * n
        for i, (xa, xb, ya, yb) in enumerate(all_points):
            val = (c[0] * xa + c[1] * xb + c[2] * ya + c[3] * yb) % p
            if val == 0:
                in_N[i] = False
        return in_N

    def make_unital_seed():
        in_N = [True] * n
        qr = set((u * u) % p for u in range(p))
        for i, (xa, xb, ya, yb) in enumerate(all_points):
            in_unital = (2 * xa + ya * ya - k * yb * yb) % p == 0
            if in_unital and ya in qr:
                in_N[i] = False
        return in_N

    def try_seed(in_N):
        if not is_valid(in_N):
            return None
        line_not_in_N_count = build_line_counts(in_N)
        point_good_witness = compute_good_witness(in_N, line_not_in_N_count)
        rng = np.random.default_rng(999)
        greedy_extend(in_N, line_not_in_N_count, point_good_witness, rng)
        if not is_valid(in_N):
            return None
        return sum(1 for v in in_N if v)

    # Try several hyperplane seeds + the unital seed, keep the smallest valid |N|
    best_size = n
    best_in_N = None

    seeds_to_try = []
    # A handful of random hyperplane seeds
    master_rng = np.random.default_rng(2024)
    for _ in range(12):
        s = int(master_rng.integers(0, 10**6))
        seeds_to_try.append(('hyper', s))
    seeds_to_try.append(('unital', None))

    for kind, s in seeds_to_try:
        if time.time() > deadline - 60:
            break
        if kind == 'hyper':
            rng = np.random.default_rng(s)
            in_N = make_hyperplane_seed(rng)
        else:
            in_N = make_unital_seed()

        sz = try_seed(in_N)
        if sz is not None and sz < best_size:
            best_size = sz
            best_in_N = in_N[:]

    if best_in_N is None:
        # Fallback: unital seed (known valid in parent)
        best_in_N = make_unital_seed()

    pts = [all_points[i] for i in range(n) if best_in_N[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
