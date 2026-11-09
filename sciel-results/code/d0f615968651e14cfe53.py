# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by excluding a set of points
    where no two excluded points share a line in any of the 50 directions.
    This guarantees the Nikodym property: for any point x and any excluded point e,
    x and e are on different lines in every direction, so removing e doesn't affect
    x's coverage by any direction that was previously valid.
    """
    p = 7
    q = p * p  # 49
    k_field = 3  # alpha^2 = 3

    # Represent elements of F_q as integers 0..48: (a, b) -> a + b*p
    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    # Precompute field operations
    mul_table = np.zeros((q, q), dtype=np.int32)
    for x in range(q):
        xa, xb = to_pair(x)
        for y in range(q):
            ya, yb = to_pair(y)
            real = (xa * ya + xb * yb * k_field) % p
            imag = (xa * yb + xb * ya) % p
            mul_table[x, y] = to_int(real, imag)

    add_table = np.zeros((q, q), dtype=np.int32)
    for x in range(q):
        xa, xb = to_pair(x)
        for y in range(q):
            ya, yb = to_pair(y)
            add_table[x, y] = to_int((xa + ya) % p, (xb + yb) % p)

    def f_add(x, y):
        return add_table[x, y]

    def f_mul(x, y):
        return mul_table[x, y]

    # Additive inverses
    neg_table = np.zeros(q, dtype=np.int32)
    for x in range(q):
        xa, xb = to_pair(x)
        neg_xa = (-xa) % p
        neg_xb = (-xb) % p
        neg_table[x] = to_int(neg_xa, neg_xb)

    def f_sub(x, y):
        return f_add(x, neg_table[y])

    # Multiplicative inverses (for nonzero elements)
    inv_table = np.full(q, -1, dtype=np.int32)
    for x in range(1, q):
        for y in range(1, q):
            if f_mul(x, y) == 1:
                inv_table[x] = y
                break

    # Directions: v = (1, m) for m in F_q, or v = (0, 1)
    directions = []
    for m in range(q):
        directions.append((1, m))
    directions.append((0, 1))
    n_dir = len(directions)  # 50

    total_points = q * q  # 2401

    # For each direction, precompute a "line ID" for each point
    # Two points are on the same line in direction d if they have the same line ID
    # For direction (1, m): line through (x0, y0) has equation y - y0 = m*(x - x0)
    #   => y - m*x = y0 - m*x0 (constant)
    # For direction (0, 1): vertical lines, identified by x coordinate

    # line_id[d_idx, p_idx] = identifier of the line through p_idx in direction d_idx
    # For non-vertical: line_id = y - m*x (in F_q)
    # For vertical: line_id = x

    line_id = np.zeros((n_dir, total_points), dtype=np.int32)
    for d_idx, (vx, vy) in enumerate(directions):
        for x_idx in range(q):
            for y_idx in range(q):
                p_idx = x_idx * q + y_idx
                if vx == 0:
                    # Vertical: line ID is x
                    line_id[d_idx, p_idx] = x_idx
                else:
                    # Non-vertical with vx=1: line ID = y - m*x where m = vy/vx = vy
                    m = vy  # since vx = 1
                    # line_id = y - m*x
                    mx = f_mul(m, x_idx)
                    line_id[d_idx, p_idx] = f_sub(y_idx, mx)

    # Greedily build excluded set: no two excluded points share a line in any direction
    excluded = set()
    excluded_points = []

    # For each direction, track which line IDs are already "used" by excluded points
    used_line_ids = [set() for _ in range(n_dir)]

    # Try to add points in a deterministic order
    for p_idx in range(total_points):
        can_add = True
        for d_idx in range(n_dir):
            lid = line_id[d_idx, p_idx]
            if lid in used_line_ids[d_idx]:
                can_add = False
                break
        if can_add:
            excluded.add(p_idx)
            excluded_points.append(p_idx)
            for d_idx in range(n_dir):
                used_line_ids[d_idx].add(line_id[d_idx, p_idx])

    # N = all points not in excluded
    in_N = np.ones(total_points, dtype=bool)
    for p_idx in excluded:
        in_N[p_idx] = False

    # Build result
    pts = []
    for i in range(total_points):
        if in_N[i]:
            x_idx = i // q
            y_idx = i % q
            xa, xb = to_pair(x_idx)
            ya, yb = to_pair(y_idx)
            pts.append([[xa, xb], [ya, yb]])

    return np.array(pts, dtype=np.int64)

# EVOLVE-BLOCK-END
