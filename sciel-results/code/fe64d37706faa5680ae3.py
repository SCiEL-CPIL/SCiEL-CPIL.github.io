# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by maximizing the complement set S 
    using a multi-start greedy algorithm.
    """
    p = 7
    start_time = time.time()
    deadline = 600.0  # seconds
    
    # Smallest quadratic non-residue mod p
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break

    # Precompute all points as indices 0..2400
    # Point index = ((xa * p + xb) * p + ya) * p + yb
    def point_index(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    # Precompute point coordinates
    points_coords = []
    for idx in range(p**4):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        points_coords.append((xa, xb, ya, yb))
        
    N = p**4

    # Precompute inv for all (c, d) != (0, 0)
    inv = {}
    for c in range(p):
        for d in range(p):
            if c == 0 and d == 0:
                continue
            det = (c * c - d * d * k) % p
            det_inv = pow(det, p - 2, p)  # Fermat's little theorem
            inv[(c, d)] = ((c * det_inv) % p, (-d * det_inv) % p)

    # Precompute dir_of_pair as a 2D array
    # dir_of_pair[i, j] is the direction index of the line passing through i and j.
    # Directions: 0 for vertical (0,1), 1 + m for (1,m) where m is encoded.
    dir_of_pair = np.full((N, N), -1, dtype=np.int8)

    for i in range(N):
        xa1, xb1, ya1, yb1 = points_coords[i]
        for j in range(N):
            if i == j:
                continue
            xa2, xb2, ya2, yb2 = points_coords[j]
            dx_a = (xa2 - xa1) % p
            dx_b = (xb2 - xb1) % p
            dy_a = (ya2 - ya1) % p
            dy_b = (yb2 - yb1) % p

            if dx_a == 0 and dx_b == 0:
                dir_of_pair[i, j] = 0  # vertical direction
            else:
                inv_a, inv_b = inv[(dx_a, dx_b)]
                ma = (dy_a * inv_a + dy_b * inv_b * k) % p
                mb = (dy_a * inv_b + dy_b * inv_a) % p
                dir_of_pair[i, j] = 1 + ma * p + mb

    # Initialize S = base_excluded
    y_special = set()
    for u in range(1, p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    base_S = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        idx = point_index(xa, xb, ya, yb)
                        base_S.add(idx)

    base_S_list = list(base_S)
    candidates = [idx for idx in range(N) if idx not in base_S]

    # Function to run one greedy pass
    def run_greedy(order):
        # Initialize count and good_dirs from base_S
        count = np.zeros((N, 50), dtype=np.int8)
        
        # For each q in base_S, for each p != q, d = dir_of_pair[p][q], count[p][d] += 1
        for q in base_S_list:
            d_col = dir_of_pair[:, q]
            # d_col[q] is -1, but we skip q in the loop or handle it
            # We need to increment count[p][d_col[p]] for p != q
            # Vectorized approach:
            valid = d_col >= 0
            ps = np.where(valid)[0]
            ds = d_col[ps]
            # Use np.add.at for unique indices if needed, but here (p, d) pairs are unique for a fixed q?
            # No, multiple p can have same d? No, for a fixed q and direction d, there is only one p on that line?
            # Wait. For a fixed q and direction d, the line is unique. The points on that line are q and 48 others.
            # So for a fixed q, and a fixed p, there is only one d.
            # But we are iterating over p. For a fixed q, each p has a specific d.
            # So we are incrementing count[p, d(p, q)] for each p.
            # This is a scatter operation.
            # np.add.at(count, (ps, ds), 1)
            np.add.at(count, (ps, ds), 1)

        good_dirs = np.sum(count == 0, axis=1)
        
        S_temp = set(base_S)
        
        for q in order:
            if q in S_temp:
                continue
            
            d_q = dir_of_pair[:, q].copy()
            d_q[q] = -1  # mark self
            
            # Get count[p][d_q[p]] for all p
            valid_mask = d_q >= 0
            # We need to check if any p has good_dirs[p] == 1 and count[p][d_q[p]] == 0
            # This means adding q would make good_dirs[p] == 0.
            
            # Efficient check:
            # Create an array of current counts for the specific directions
            # count_at_d[p] = count[p, d_q[p]]
            # We can do this with advanced indexing
            
            ps = np.where(valid_mask)[0]
            ds = d_q[ps]
            current_counts = count[ps, ds]
            
            # bad_mask for these ps: good_dirs[ps] == 1 AND current_counts == 0
            bad_mask = (good_dirs[ps] == 1) & (current_counts == 0)
            
            if np.any(bad_mask):
                continue  # Cannot add q
            
            # Safe to add q
            S_temp.add(q)
            
            # Update count and good_dirs
            # For each p in ps, d = ds
            # if count[p, d] == 0: good_dirs[p] -= 1
            # count[p, d] += 1
            
            zero_mask = current_counts == 0
            if np.any(zero_mask):
                good_dirs[ps[zero_mask]] -= 1
            
            count[ps, ds] += 1
        
        return S_temp

    best_S = base_S
    best_size = len(base_S)
    
    # Multi-start greedy
    # We run multiple iterations with different random orders
    # To speed up, we can also try to prioritize candidates that are "safe"
    
    num_starts = 0
    while time.time() - start_time < deadline - 10:
        # Randomize the order of candidates
        random.shuffle(candidates)
        
        S_current = run_greedy(candidates)
        
        if len(S_current) > best_size:
            best_S = S_current
            best_size = len(S_current)
            # Optional: break if we reached a theoretical max? No, we don't know it.
            # Just keep going.
        
        num_starts += 1
        # If we are running out of time, break
        if time.time() - start_time > deadline - 5:
            break

    # Build the Nikodym set (complement of S)
    points = []
    for idx in range(N):
        if idx not in best_S:
            xa, xb, ya, yb = points_coords[idx]
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
