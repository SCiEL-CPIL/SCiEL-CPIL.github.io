# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    deadline = time.time() + 615
    p = 7
    q = p * p
    Q = q * q  # 2401 points

    # Precompute field tables (vectorized-friendly)
    to_a = np.zeros(q, dtype=np.int64)
    to_b = np.zeros(q, dtype=np.int64)
    for x in range(q):
        to_a[x] = x % p
        to_b[x] = (x // p) % p

    add_tab = (to_a[:, None] + to_a[None, :]) % p + p * ((to_b[:, None] + to_b[None, :]) % p)
    neg_tab = (p - to_a) % p + p * ((p - to_b) % p)
    sub_tab = add_tab[np.arange(q)[:, None], neg_tab]
    mul_tab = np.zeros((q, q), dtype=np.int64)
    for x in range(q):
        xa, xb = to_a[x], to_b[x]
        for y in range(q):
            ya, yb = to_a[y], to_b[y]
            real = (xa * ya + xb * yb * 3) % p
            imag = (xa * yb + xb * ya) % p
            mul_tab[x, y] = real + p * imag

    def f_add_vec(a, b):
        return add_tab[a, b]

    def f_sub_vec(a, b):
        return sub_tab[a, b]

    def f_mul_vec(a, b):
        return mul_tab[a, b]

    # Directions: (1, m) for m in F_49, plus (0, 1)
    dir_x = np.concatenate([np.ones(q, dtype=np.int64), np.zeros(1, dtype=np.int64)])
    dir_y = np.concatenate([np.arange(q, dtype=np.int64), np.ones(1, dtype=np.int64)])
    n_dir = q + 1

    # Point index mapping: point (x_idx, y_idx) -> linear index
    point_to_lin = np.arange(Q).reshape(q, q)  # [x_idx, y_idx] -> linear

    # Build "affects" structure: for each point p, which (x, d) pairs does it cover?
    # p = (px, py) lies on line through x=(x_idx, y_idx) in direction d with parameter t
    # if p = x + t*v_d, i.e., x = p - t*v_d
    # So for each p and each d, the x values are p - t*v_d for t in 1..q-1
    # We need: for each p, list of (x_lin, d) such that x = p - t*v_d

    # Build affects as: affects[p] = list of (x_lin, d)
    # For point p with coords (px, py):
    #   for each d: x = p - t*v_d = (px - t*vx, py - t*vy) for t in 1..q-1
    #   x_lin = point_to_lin[px - t*vx, py - t*vy]

    # Precompute line membership: for each point p, for each direction d, 
    # the set of points on the line through p in direction d
    # Actually we need the reverse: for each (x, d), which points are on that line?
    # Line through x in direction d: {x + t*v_d : t in F_49}
    # So point p is on line (x, d) iff p - x is in span{v_d}, i.e., p = x + t*v_d for some t
    # Equivalently: x = p - t*v_d for some t in F_49

    # For the batch removal, we need:
    # For each point p in N, and each direction d, does p help cover x?
    # p helps cover x in direction d if x is on the line through p in direction d AND x != p
    # i.e., x = p + t*v_d for some t in 1..q-1

    # Let's build: for each point p, for each direction d, the list of x that p covers
    # covers[p, d] = set of x such that x = p + t*v_d for t in 1..q-1

    # Actually for the validity check: point x is valid if for every direction d,
    # there exists some point p in N (p != x) on the line through x in direction d.
    # The line through x in direction d is {x + t*v_d : t in F_49}.
    # So we need: for each x and d, at least one of {x + t*v_d : t in 1..q-1} is in N.

    # Let's precompute for each (x, d): the list of other points on the line
    # line_points[x, d] = [x + t*v_d for t in 1..q-1] as linear indices

    line_points = np.zeros((Q, n_dir, q - 1), dtype=np.int32)
    for d in range(n_dir):
        vx, vy = dir_x[d], dir_y[d]
        for x_idx in range(q):
            for y_idx in range(q):
                x_lin = point_to_lin[x_idx, y_idx]
                for t in range(1, q):
                    tx = f_sub_vec(x_idx, f_mul_vec(t, vx))
                    ty = f_sub_vec(y_idx, f_mul_vec(t, vy))
                    line_points[x_lin, d, t - 1] = point_to_lin[tx, ty]

    # not_in_N_count[x, d] = number of points on line (x,d) that are NOT in N
    # Point x is valid iff for all d: not_in_N_count[x, d] < q-1
    # i.e., at least one point on the line (other than x) is in N

    def build_initial_exclusion(seed_offset=0):
        """Build initial set with some points excluded based on algebraic structure."""
        in_N = np.ones(Q, dtype=bool)
        
        # Exclude points where Tr(x) + N(y) = 0 and y is not a special value
        # Tr(x) = 2*xa (trace in F_7)
        # N(y) = ya^2 - 3*yb^2 (norm)
        # Special: y is a square in F_7 (real part is a square)
        
        squares_in_F7 = set()
        for u in range(p):
            squares_in_F7.add((u * u) % p)
        
        excluded = 0
        for x_idx in range(q):
            xa = to_a[x_idx]
            trace_x = (2 * xa) % p
            for y_idx in range(q):
                ya, yb = to_a[y_idx], to_b[y_idx]
                norm_y = (ya * ya - yb * yb * 3) % p
                if (trace_x + norm_y) % p == 0:
                    # Check if y is "special" (real part is a square in F_7)
                    if ya not in squares_in_F7:
                        pt = point_to_lin[x_idx, y_idx]
                        in_N[pt] = False
                        excluded += 1
        return in_N, excluded

    def compute_not_in_N_counts(in_N):
        """Compute for each (x, d) how many points on the line are not in N."""
        not_in_N_count = np.zeros((Q, n_dir), dtype=np.int32)
        not_in_N = ~in_N
        for d in range(n_dir):
            # line_points[:, d, :] has shape (Q, q-1)
            # For each x, count how many of line_points[x, d, :] are not in N
            lp = line_points[:, d, :]  # shape (Q, q-1)
            # Index into not_in_N
            not_in_N_mask = not_in_N[lp]  # shape (Q, q-1)
            not_in_N_count[:, d] = not_in_N_mask.sum(axis=1)
        return not_in_N_count

    def compute_cover_count(not_in_N_count):
        """For each point x, count how many directions d have not_in_N_count[x,d] == 0."""
        # A point is "critically covered" in direction d if not_in_N_count[x,d] == 0
        # cover_count[x] = number of directions where x has zero non-N points on its line
        return (not_in_N_count == 0).sum(axis=1)

    def is_removable(p_idx, in_N, not_in_N_count):
        """Check if removing p_idx from N keeps all points valid."""
        if not in_N[p_idx]:
            return False
        # For each direction d, p_idx is on the line through x in direction d
        # if x = p_idx - t*v_d for some t in 1..q-1
        # Removing p_idx increments not_in_N_count[x, d] for all such (x, d)
        # Point x becomes invalid if not_in_N_count[x, d] was 0 and becomes 1
        px, py = np.unravel_index(p_idx, (q, q))
        for d in range(n_dir):
            vx, vy = dir_x[d], dir_y[d]
            for t in range(1, q):
                tx = f_sub_vec(px, f_mul_vec(t, vx))
                ty = f_sub_vec(py, f_mul_vec(t, vy))
                x_lin = point_to_lin[tx, ty]
                if not_in_N_count[x_lin, d] == 0:
                    return False
        return True

    def batch_removal(in_N, not_in_N_count, max_rounds=50):
        """Remove all currently-removable points in each round, repeat."""
        changed = True
        rounds = 0
        while changed and rounds < max_rounds:
            if time.time() > deadline - 30:
                break
            changed = False
            rounds += 1
            # Find all removable points
            removable = []
            for p_idx in range(Q):
                if in_N[p_idx] and is_removable(p_idx, in_N, not_in_N_count):
                    removable.append(p_idx)
            
            if not removable:
                break
            
            # Remove all at once
            for p_idx in removable:
                in_N[p_idx] = False
                # Update not_in_N_count
                px, py = np.unravel_index(p_idx, (q, q))
                for d in range(n_dir):
                    vx, vy = dir_x[d], dir_y[d]
                    for t in range(1, q):
                        tx = f_sub_vec(px, f_mul_vec(t, vx))
                        ty = f_sub_vec(py, f_mul_vec(t, vy))
                        x_lin = point_to_lin[tx, ty]
                        not_in_N_count[x_lin, d] += 1
            changed = True
        
        return in_N, not_in_N_count, rounds

    def greedy_removal(in_N, not_in_N_count, order='cover_desc', max_rounds=200):
        """Greedy removal with specified ordering."""
        for _ in range(max_rounds):
            if time.time() > deadline - 30:
                break
            
            # Find removable points
            removable = []
            for p_idx in range(Q):
                if in_N[p_idx] and is_removable(p_idx, in_N, not_in_N_count):
                    removable.append(p_idx)
            
            if not removable:
                break
            
            # Order them
            if order == 'cover_desc':
                cc = compute_cover_count(not_in_N_count)
                removable.sort(key=lambda x: cc[x], reverse=True)
            elif order == 'random':
                np.random.shuffle(removable)
            elif order == 'cover_asc':
                cc = compute_cover_count(not_in_N_count)
                removable.sort(key=lambda x: cc[x])
            
            # Remove one at a time (first in order)
            p_idx = removable[0]
            in_N[p_idx] = False
            px, py = np.unravel_index(p_idx, (q, q))
            for d in range(n_dir):
                vx, vy = dir_x[d], dir_y[d]
                for t in range(1, q):
                    tx = f_sub_vec(px, f_mul_vec(t, vx))
                    ty = f_sub_vec(py, f_mul_vec(t, vy))
                    x_lin = point_to_lin[tx, ty]
                    not_in_N_count[x_lin, d] += 1
        
        return in_N, not_in_N_count

    def validate(in_N):
        """Check if in_N defines a valid Nikodym set."""
        not_in_N_count = compute_not_in_N_counts(in_N)
        # All points must have at least one point in N on each line
        # i.e., not_in_N_count[x, d] < q-1 for all x, d
        return np.all(not_in_N_count < q - 1)

    # Strategy: try multiple approaches, keep the best (smallest |N|)
    best_in_N = None
    best_size = Q + 1

    # Approach 1: Batch removal
    in_N1, excluded1 = build_initial_exclusion()
    nic1 = compute_not_in_N_counts(in_N1)
    if validate(in_N1):
        in_N1, nic1, _ = batch_removal(in_N1.copy(), nic1.copy(), max_rounds=30)
        if validate(in_N1):
            size1 = in_N1.sum()
            if size1 < best_size:
                best_in_N = in_N1.copy()
                best_size = size1

    if time.time() > deadline - 60:
        if best_in_N is None:
            best_in_N = build_initial_exclusion()[0]
        pts = np.array([[[int(to_a[x]), int(to_b[x])], [int(to_a[y]), int(to_b[y])]]
                       for x, y in zip(*np.where(best_in_N.reshape(q, q)))], dtype=np.int64)
        if len(pts) == 0:
            pts = np.array([[[0, 0], [0, 0]]], dtype=np.int64)
        return pts

    # Approach 2: Greedy with cover_desc
    in_N2, _ = build_initial_exclusion()
    nic2 = compute_not_in_N_counts(in_N2)
    if validate(in_N2):
        in_N2, nic2 = greedy_removal(in_N2.copy(), nic2.copy(), order='cover_desc', max_rounds=100)
        if validate(in_N2):
            size2 = in_N2.sum()
            if size2 < best_size:
                best_in_N = in_N2.copy()
                best_size = size2

    if time.time() > deadline - 60:
        if best_in_N is None:
            best_in_N = build_initial_exclusion()[0]
        pts = np.array([[[int(to_a[x]), int(to_b[x])], [int(to_a[y]), int(to_b[y])]]
                       for x, y in zip(*np.where(best_in_N.reshape(q, q)))], dtype=np.int64)
        if len(pts) == 0:
            pts = np.array([[[0, 0], [0, 0]]], dtype=np.int64)
        return pts

    # Approach 3: Random restarts with different initial exclusions
    for attempt in range(5):
        if time.time() > deadline - 60:
            break
        
        # Vary the exclusion: use different "special" sets
        in_N3 = np.ones(Q, dtype=bool)
        # Exclude based on a different algebraic condition
        for x_idx in range(q):
            xa = to_a[x_idx]
            for y_idx in range(q):
                ya, yb = to_a[y_idx], to_b[y_idx]
                # Different condition: use a linear form
                val = (xa * ya + xb * yb + attempt) % p  # xb not defined here, use to_b
                val = (to_a[x_idx] * to_a[y_idx] + to_b[x_idx] * to_b[y_idx] + attempt) % p
                if val == 0 and (x_idx + y_idx + attempt) % 3 == 0:
                    pt = point_to_lin[x_idx, y_idx]
                    in_N3[pt] = False
        
        nic3 = compute_not_in_N_counts(in_N3)
        if validate(in_N3):
            in_N3, nic3 = batch_removal(in_N3.copy(), nic3.copy(), max_rounds=20)
            if validate(in_N3):
                size3 = in_N3.sum()
                if size3 < best_size:
                    best_in_N = in_N3.copy()
                    best_size = size3

    # Approach 4: Start from full set, do greedy
    if time.time() < deadline - 90:
        in_N4 = np.ones(Q, dtype=bool)
        nic4 = compute_not_in_N_counts(in_N4)
        in_N4, nic4 = greedy_removal(in_N4, nic4, order='cover_desc', max_rounds=50)
        if validate(in_N4):
            size4 = in_N4.sum()
            if size4 < best_size:
                best_in_N = in_N4.copy()
                best_size = size4

    # Final validation
    if best_in_N is None:
        best_in_N = np.ones(Q, dtype=bool)
    
    if not validate(best_in_N):
        # Fallback: full set is always valid
        best_in_N = np.ones(Q, dtype=bool)

    # Convert to output format
    pts = []
    for x_idx in range(q):
        for y_idx in range(q):
            if best_in_N[point_to_lin[x_idx, y_idx]]:
                pts.append([[int(to_a[x_idx]), int(to_b[x_idx])],
                            [int(to_a[y_idx]), int(to_b[y_idx])]])

    if not pts:
        pts = [[[0, 0], [0, 0]]]

    return np.array(pts, dtype=np.int64)

# EVOLVE-BLOCK-END
