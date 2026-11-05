import time
import random
import numpy as np

def objective_fast(A):
    """Vectorized objective computation using NumPy."""
    A = np.array(A, dtype=np.int64)
    n = len(A)
    # All pairs (i,j) with i <= j
    i_idx = np.repeat(np.arange(n), np.arange(n, 0, -1))
    j_idx = np.concatenate([np.arange(i, n) for i in range(n)])
    sums = A[i_idx] + A[j_idx]
    prods = A[i_idx] * A[j_idx]
    # Count unique values
    s_count = len(np.unique(sums))
    p_count = len(np.unique(prods))
    return max(s_count, p_count)

def objective(A):
    """Fallback objective (same as fast version)."""
    return objective_fast(A)

def objective_with_delta(A, changed_idx, old_val, new_val):
    """Compute objective change when replacing A[changed_idx] from old_val to new_val."""
    A = list(A)
    A[changed_idx] = new_val
    return objective(A)

def construct():
    start = time.time()
    deadline = start + 88.0  # safety margin under 90s
    
    random.seed(42)
    
    def valid_set(A):
        return len(set(A)) == 20 and all(1 <= x <= 10000 for x in A)
    
    # --- Lattice-based construction: minimize |A*A| via exponent structure ---
    # Represent elements as 2^i * 3^j * 5^k. Products become 2^(i1+i2) * 3^(j1+j2) * 5^(k1+k2).
    # Dense exponent sets lead to smaller product sets.
    
    def lattice_set_2d(i_max, j_max, subset_indices):
        """Generate a set from 2D exponent lattice points (2^i * 3^j)."""
        vals = []
        for (i, j) in subset_indices:
            v = (2 ** i) * (3 ** j)
            if v <= 10000:
                vals.append(v)
        return vals
    
    def lattice_set_3d(i_max, j_max, k_max, subset_indices):
        """Generate a set from 3D exponent lattice points (2^i * 3^j * 5^k)."""
        vals = []
        for (i, j, k) in subset_indices:
            v = (2 ** i) * (3 ** j) * (5 ** k)
            if v <= 10000:
                vals.append(v)
        return vals
    
    lattice_candidates = []
    
    # Dense 2D grid: i in 0..i_max, j in 0..j_max
    for i_max in range(0, 14):
        for j_max in range(0, 10):
            points = [(i, j) for i in range(i_max + 1) for j in range(j_max + 1)]
            if len(points) >= 20:
                points_sorted = sorted(points, key=lambda p: (2 ** p[0]) * (3 ** p[1]))
                subset = points_sorted[:20]
                vals = lattice_set_2d(i_max, j_max, subset)
                if len(set(vals)) == 20 and all(1 <= v <= 10000 for v in vals):
                    lattice_candidates.append(sorted(vals))
    
    # 3D lattice: 2^i * 3^j * 5^k with small exponents
    for i_max in range(0, 8):
        for j_max in range(0, 6):
            for k_max in range(0, 5):
                points = [(i, j, k) for i in range(i_max + 1) 
                         for j in range(j_max + 1) 
                         for k in range(k_max + 1)]
                if len(points) >= 20:
                    points_sorted = sorted(points, key=lambda p: (2 ** p[0]) * (3 ** p[1]) * (5 ** p[2]))
                    subset = points_sorted[:20]
                    vals = lattice_set_3d(i_max, j_max, k_max, subset)
                    if len(set(vals)) == 20 and all(1 <= v <= 10000 for v in vals):
                        lattice_candidates.append(sorted(vals))
    
    # Lines in 2D lattice
    for c in range(0, 20):
        points = []
        for j in range(0, 10):
            i = c - 2 * j
            if 0 <= i <= 14:
                points.append((i, j))
        if len(points) >= 20:
            points_sorted = sorted(points, key=lambda p: (2 ** p[0]) * (3 ** p[1]))
            subset = points_sorted[:20]
            vals = lattice_set_2d(14, 10, subset)
            if len(set(vals)) == 20 and all(1 <= v <= 10000 for v in vals):
                lattice_candidates.append(sorted(vals))
    
    for c in range(0, 25):
        points = []
        for i in range(0, 14):
            j = c - 2 * i
            if 0 <= j <= 10:
                points.append((i, j))
        if len(points) >= 20:
            points_sorted = sorted(points, key=lambda p: (2 ** p[0]) * (3 ** p[1]))
            subset = points_sorted[:20]
            vals = lattice_set_2d(14, 10, subset)
            if len(set(vals)) == 20 and all(1 <= v <= 10000 for v in vals):
                lattice_candidates.append(sorted(vals))
    
    # Mixed grid + offset
    base_grid = [(i, j) for i in range(0, 5) for j in range(0, 5)]
    for extra_i in range(5, 12):
        for extra_j in range(0, 8):
            subset = base_grid[:15] + [(extra_i, extra_j)]
            if len(subset) >= 20:
                subset = subset[:20]
                vals = lattice_set_2d(14, 10, subset)
                if len(set(vals)) == 20 and all(1 <= v <= 10000 for v in vals):
                    lattice_candidates.append(sorted(vals))
    
    # Deduplicate
    seen = set()
    unique_lattice = []
    for cand in lattice_candidates:
        key = tuple(cand)
        if key not in seen:
            seen.add(key)
            unique_lattice.append(cand)
    
    # Hand-crafted candidates
    hand_crafted = [
        list(range(1, 21)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 27, 28],
    ]
    
    # Geometric progressions with various bases
    for base in [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.5, 3.0]:
        A = [int(round(base ** i)) for i in range(20)]
        seen_g = set()
        uniq = []
        for x in A:
            if 1 <= x <= 10000 and x not in seen_g:
                seen_g.add(x)
                uniq.append(x)
        while len(uniq) < 20:
            uniq.append(uniq[-1] + 1)
        if len(uniq) == 20 and all(1 <= x <= 10000 for x in uniq):
            hand_crafted.append(uniq)
    
    # Arithmetic progressions with small common differences
    for start in [1, 2, 3]:
        for diff in [2, 3, 4, 5, 6]:
            A = [start + i * diff for i in range(20)]
            if all(1 <= x <= 10000 for x in A) and len(set(A)) == 20:
                hand_crafted.append(A)
    
    all_candidates = unique_lattice + hand_crafted
    
    # Evaluate all initial candidates
    best_set = None
    best_obj = float('inf')
    for cand in all_candidates:
        if not valid_set(cand):
            continue
        obj = objective(cand)
        if obj < best_obj:
            best_obj = obj
            best_set = cand[:]
    
    if best_set is None:
        best_set = list(range(1, 21))
        best_obj = objective(best_set)
    
    # --- Local search refinement ---
    def local_search(A_init, max_iters, mode="single", polish=False):
        nonlocal best_set, best_obj
        A = list(A_init)
        cur_obj = objective(A)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_set = A[:]
        
        it = 0
        stuck_count = 0
        while it < max_iters:
            if time.time() > deadline - (2.0 if polish else 0.5):
                return
            it += 1
            
            if mode == "single":
                # Replace one element
                i = random.randrange(20)
                x = A[i]
                cands = set()
                # Multiplicative neighbors (exploit lattice structure)
                for factor in [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
                    nv = int(round(x * factor))
                    if 1 <= nv <= 10000:
                        cands.add(nv)
                # Additive neighbors
                for d in [-3, -2, -1, 1, 2, 3, 5, 10, 20, 50, 100]:
                    nv = x + d
                    if 1 <= nv <= 10000:
                        cands.add(nv)
                # Random
                cands.add(random.randrange(1, 10001))
                # Lattice-inspired: multiply/divide by 2 or 3
                for div_mult in [2, 3, 5]:
                    cands.add(x * div_mult)
                    if x % div_mult == 0:
                        cands.add(x // div_mult)
                cands = [c for c in cands if c not in A and 1 <= c <= 10000]
                if not cands:
                    continue
                nv = random.choice(cands)
                old = A[i]
                A[i] = nv
                new_obj = objective(A)
                if new_obj < cur_obj:
                    cur_obj = new_obj
                    stuck_count = 0
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_set = A[:]
                elif new_obj == cur_obj and random.random() < 0.1:
                    # Allow neutral moves occasionally to escape plateaus
                    stuck_count = 0
                else:
                    A[i] = old
                    stuck_count += 1
                    
            elif mode == "swap":
                # Swap two elements
                i, j = random.sample(range(20), 2)
                A[i], A[j] = A[j], A[i]
                new_obj = objective(A)
                if new_obj < cur_obj:
                    cur_obj = new_obj
                    stuck_count = 0
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_set = A[:]
                elif new_obj == cur_obj and random.random() < 0.1:
                    stuck_count = 0
                else:
                    A[i], A[j] = A[j], A[i]
                    stuck_count += 1
                    
            elif mode == "perturb":
                # Perturb multiple elements
                num_perturb = random.randint(1, 3)
                indices = random.sample(range(20), num_perturb)
                old_vals = {i: A[i] for i in indices}
                for i in indices:
                    x = A[i]
                    cands = set()
                    for factor in [0.5, 0.7, 0.8, 1.2, 1.5, 2.0, 3.0]:
                        nv = int(round(x * factor))
                        if 1 <= nv <= 10000:
                            cands.add(nv)
                    cands.add(x + random.randint(-20, 20))
                    cands.add(random.randrange(1, 10001))
                    for div_mult in [2, 3]:
                        cands.add(x * div_mult)
                        if x % div_mult == 0:
                            cands.add(x // div_mult)
                    cands = [c for c in cands if c not in A and 1 <= c <= 10000]
                    if cands:
                        A[i] = random.choice(cands)
                new_obj = objective(A)
                if new_obj < cur_obj:
                    cur_obj = new_obj
                    stuck_count = 0
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_set = A[:]
                elif new_obj == cur_obj and random.random() < 0.1:
                    stuck_count = 0
                else:
                    for i in indices:
                        A[i] = old_vals[i]
                    stuck_count += 1
            
            # If stuck too long, try a larger perturbation
            if stuck_count > 50 and not polish:
                # Do a random restart of a few elements
                num_reset = random.randint(2, 5)
                reset_indices = random.sample(range(20), num_reset)
                for idx in reset_indices:
                    x = A[idx]
                    cands = set()
                    for factor in [0.5, 0.7, 1.3, 1.5, 2.0, 3.0]:
                        nv = int(round(x * factor))
                        if 1 <= nv <= 10000:
                            cands.add(nv)
                    cands.add(random.randrange(1, 10001))
                    cands = [c for c in cands if c not in A and 1 <= c <= 10000]
                    if cands:
                        A[idx] = random.choice(cands)
                new_obj = objective(A)
                if new_obj <= cur_obj:
                    cur_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_set = A[:]
                stuck_count = 0
        
        return
    
    # Run local search from best candidate
    local_search(best_set, 50000, mode="single")
    
    # Try local search from top lattice candidates
    top_lattice = []
    for cand in unique_lattice[:15]:
        if valid_set(cand):
            top_lattice.append(cand)
    
    for start_set in top_lattice:
        if time.time() > deadline - 20:
            break
        mode = random.choice(["single", "swap", "perturb"])
        local_search(start_set, 15000, mode=mode)
    
    # Random restarts with lattice-inspired generation
    for trial in range(40):
        if time.time() > deadline - 10:
            break
        # Generate lattice-inspired random set
        A = []
        for _ in range(20):
            # Mix 2D and 3D lattice points
            if random.random() < 0.6:
                i = random.randint(0, 13)
                j = random.randint(0, 9)
                v = (2 ** i) * (3 ** j)
            else:
                i = random.randint(0, 7)
                j = random.randint(0, 5)
                k = random.randint(0, 4)
                v = (2 ** i) * (3 ** j) * (5 ** k)
            if v <= 10000 and v not in A:
                A.append(v)
        while len(A) < 20:
            v = random.randrange(1, 10001)
            if v not in A:
                A.append(v)
        A = sorted(A[:20])
        if valid_set(A):
            local_search(A, 10000, mode="single")
    
    # Final high-precision polish pass with tighter tolerance
    if time.time() < deadline - 3:
        local_search(best_set, 50000, mode="single", polish=True)
    
    return best_set
