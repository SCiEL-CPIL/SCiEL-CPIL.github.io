# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Strategy:
    1. Start with a base Nikodym set derived from a Hermitian-unital-style exclusion.
    2. Iteratively remove points from N that are not essential for the Nikodym property.
    3. Use a count-based coverage check: for each point x and direction v, track how many
       points on line(x, v) \\ {x} are NOT in N. If count is 0, x is covered by v.
    4. Remove a point p from N only if no point loses all its covering directions.
    """
    deadline = time.time() + 600  # safety margin before 630s limit

    p = 7
    q = p * p  # 49
    k_field = 3  # alpha^2 = 3, a quadratic non-residue mod 7

    # Represent elements of F_q as integers 0..48: (a, b) -> a + b*p
    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    # Precompute field multiplication table for speed
    # mul_table[x][y] = x * y in F_q
    mul_table = np.zeros((q, q), dtype=np.int32)
    for x in range(q):
        xa, xb = to_pair(x)
        for y in range(q):
            ya, yb = to_pair(y)
            real = (xa * ya + xb * yb * k_field) % p
            imag = (xa * yb + xb * ya) % p
            mul_table[x, y] = to_int(real, imag)

    # Addition table
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

    # Directions: v = (1, m) for m in F_q, or v = (0, 1)
    directions = []
    for m in range(q):
        directions.append((1, m))
    directions.append((0, 1))
    n_dir = len(directions)  # 50

    # All points in F_q^2 as integer pairs (x_idx, y_idx) where x_idx, y_idx in 0..48
    # Point index: x_idx * q + y_idx
    total_points = q * q  # 2401
    point_index = {}
    for x_idx in range(q):
        for y_idx in range(q):
            point_index[(x_idx, y_idx)] = x_idx * q + y_idx

    # Base Nikodym set construction
    # Excluded: points (x, y) where Trace(x) + Norm(y) == 0 AND y is not "special"
    # Trace((a,b)) = 2a, Norm((a,b)) = a^2 - b^2*k
    # Special: y_a is a square mod p

    y_special = set()
    for u in range(p):
        sq = (u * u) % p
        for v in range(p):
            y_special.add(to_int(sq, v))

    in_N = np.ones(total_points, dtype=bool)

    for x_idx in range(q):
        xa, xb = to_pair(x_idx)
        trace_x = (2 * xa) % p
        for y_idx in range(q):
            ya, yb = to_pair(y_idx)
            norm_y = (ya * ya - yb * yb * k_field) % p
            if (trace_x + norm_y) % p == 0:
                if y_idx not in y_special:
                    in_N[x_idx * q + y_idx] = False

    # Precompute for each direction d and each point idx, the list of other point indices on the line
    # line[d][idx] = array of (q-1) point indices
    # Memory: 50 * 2401 * 48 * 4 bytes ≈ 23 MB, acceptable
    line_points = np.zeros((n_dir, total_points, q - 1), dtype=np.int32)

    for d_idx, (vx, vy) in enumerate(directions):
        for x_idx in range(q):
            for y_idx in range(q):
                p_idx = x_idx * q + y_idx
                for t in range(1, q):
                    # t * vx and t * vy
                    tvx = f_mul(t, vx)
                    tvy = f_mul(t, vy)
                    tx = f_add(x_idx, tvx)
                    ty = f_add(y_idx, tvy)
                    line_points[d_idx, p_idx, t - 1] = tx * q + ty

    # not_in_N_count[p_idx, d_idx] = number of points on line(p, v_d) \ {p} not in N
    not_in_N_count = np.zeros((total_points, n_dir), dtype=np.int32)

    # Compute initial counts
    for d_idx in range(n_dir):
        for p_idx in range(total_points):
            count = 0
            for other in line_points[d_idx, p_idx]:
                if not in_N[other]:
                    count += 1
            not_in_N_count[p_idx, d_idx] = count

    # cover_count[p_idx] = number of directions that cover p_idx (count == 0)
    cover_count = np.sum(not_in_N_count == 0, axis=1, dtype=np.int32)

    # Try to remove points from N
    # Sort points in N by cover_count descending (most redundant first)
    n_points_in_N = np.sum(in_N)
    points_in_N = np.where(in_N)[0]
    # Sort by cover_count descending
    sorted_indices = points_in_N[np.argsort(-cover_count[points_in_N])]

    removed = 0
    for p_idx in sorted_indices:
        if time.time() > deadline:
            break
        if not in_N[p_idx]:
            continue

        # Check if removing p_idx is valid
        # For each point x that has p_idx on its line in some direction, the count increases by 1
        # We need to check that no point x has all its covering directions broken

        # Find all (x, d) pairs where p_idx is on line(x, v_d) \ {x}
        # This is expensive to do naively. Instead, for each direction d,
        # find all x such that p_idx is on line(x, v_d) \ {x}

        # For a fixed direction d = (vx, vy), p_idx = (px, py) is on line(x, v_d) \ {x}
        # iff p_idx = x + t*v_d for some t != 0
        # i.e., x = p_idx - t*v_d for some t != 0
        # So for each direction d, the affected x's are: p_idx - t*v_d for t in 1..q-1

        # For each such x, check if removing p_idx from N would break x's coverage
        # x is covered by direction d if not_in_N_count[x, d] == 0
        # After removal, count becomes not_in_N_count[x, d] + 1 (since p_idx was in N)
        # So x loses direction d as a cover if not_in_N_count[x, d] == 0
        # x is still valid if it has at least one other covering direction

        valid_removal = True
        affected_pairs = []  # (x_idx, d_idx) pairs to update

        for d_idx in range(n_dir):
            vx, vy = directions[d_idx]
            for t in range(1, q):
                tvx = f_mul(t, vx)
                tvy = f_mul(t, vy)
                # x = p_idx - t*v_d
                px, py = to_pair(p_idx // q), to_pair(p_idx % q)
                # Actually p_idx = px_idx * q + py_idx
                px_idx = p_idx // q
                py_idx = p_idx % q
                tx = (px_idx - tvx) % q
                ty = (py_idx - tvy) % q
                x_idx = tx * q + ty

                # Check if p_idx is actually on line(x_idx, v_d) \ {x_idx}
                # i.e., x_idx != p_idx and p_idx is in line_points[d_idx, x_idx]
                if x_idx == p_idx:
                    continue

                # Check if not_in_N_count[x_idx, d_idx] == 0 (x is currently covered by d)
                if not_in_N_count[x_idx, d_idx] == 0:
                    # x would lose this covering direction
                    # Check if x has another covering direction
                    if cover_count[x_idx] <= 1:
                        # x would lose all covering directions -> invalid
                        valid_removal = False
                        break
                    else:
                        # x will lose this direction but has others
                        affected_pairs.append((x_idx, d_idx))
                else:
                    # x is not covered by d currently, but after removal count increases
                    # This doesn't affect coverage status (still not covered)
                    affected_pairs.append((x_idx, d_idx))

            if not valid_removal:
                break

        if not valid_removal:
            continue

        # Remove p_idx from N
        in_N[p_idx] = False
        removed += 1

        # Update counts and cover counts
        for x_idx, d_idx in affected_pairs:
            not_in_N_count[x_idx, d_idx] += 1
            if not_in_N_count[x_idx, d_idx] == 1:
                # This direction was a cover, now it's not
                cover_count[x_idx] -= 1

    # Build result array
    n_final = np.sum(in_N)
    result = np.zeros((n_final, 2, 2), dtype=np.int64)
    for i in range(total_points):
        if in_N[i]:
            x_idx = i // q
            y_idx = i % q
            xa, xb = to_pair(x_idx)
            ya, yb = to_pair(y_idx)
            # Find the row in result
            # Actually, let's collect points first
    # Collect points
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
