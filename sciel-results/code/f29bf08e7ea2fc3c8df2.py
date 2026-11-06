"""Sparse squares of twelve-term integer polynomials - Directed Collision Search."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    best_poly = None
    best_nz = float('inf')
    
    # Helper to convert (exponents, coeffs) to list representation
    def to_list(exps, coefs):
        max_deg = max(exps)
        poly = [0] * (max_deg + 1)
        for e, c in zip(exps, coefs):
            poly[e] = c
        return poly

    def validate(poly_coeff_list):
        if not poly_coeff_list or len(poly_coeff_list) > 41:
            return False
        if poly_coeff_list[-1] == 0:
            return False
        for c in poly_coeff_list:
            if not isinstance(c, int) or c < -64 or c > 64:
                return False
        if sum(1 for c in poly_coeff_list if c != 0) != 12:
            return False
        return True

    def square_nonzero(exps, coefs):
        """Fast exact square nonzero count."""
        c = np.asarray(coefs, dtype=np.int64)
        e = np.asarray(exps, dtype=np.int64)
        # Compute all pair sums and products
        # e_i + e_j, c_i * c_j
        # Use broadcasting
        degs = e[:, None] + e[None, :]
        vals = c[:, None] * c[None, :]
        # Bincount to sum coefficients for each degree
        # Max degree is 80 (40+40), so minlength 81
        result = np.bincount(degs.ravel(), weights=vals.ravel(), minlength=81)
        return int(np.count_nonzero(result))

    def update_best(exps, coefs):
        nonlocal best_poly, best_nz
        if len(set(exps)) != 12:
            return
        if any(e < 0 or e >= 41 for e in exps):
            return
        if any(c == 0 or abs(c) > 64 for c in coefs):
            return
        
        nz = square_nonzero(exps, coefs)
        if nz < best_nz:
            best_nz = nz
            best_poly = to_list(exps, coefs)

    def local_search(initial_exps, initial_coefs, max_iters=5000, t0=5.0, t_final=0.01, cooling=0.999):
        """
        Local search with directed collision moves.
        """
        nonlocal best_poly, best_nz
        
        cur_exps = sorted(initial_exps)
        cur_coefs = list(initial_coefs)
        cur_nz = square_nonzero(cur_exps, cur_coefs)
        
        if cur_nz < best_nz:
            best_nz = cur_nz
            best_poly = to_list(cur_exps, cur_coefs)

        best_in_run_exps = list(cur_exps)
        best_in_run_coefs = list(cur_coefs)
        best_in_run_nz = cur_nz

        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break

            t = t0 * (cooling ** it)
            if t < t_final:
                t = t_final

            new_exps = list(cur_exps)
            new_coefs = list(cur_coefs)
            
            r = random.random()
            
            # Move Types
            if r < 0.30:
                # Single exponent shift
                idx = random.randint(0, 11)
                shift = random.randint(-10, 10)
                new_e = new_exps[idx] + shift
                if 0 <= new_e < 41 and new_e not in new_exps:
                    new_exps[idx] = new_e
                    new_exps.sort()
                else:
                    continue
            elif r < 0.50:
                # Directed Collision Move:
                # Pick a random existing sum in the square (or a random target sum)
                # and try to align a new pair to it.
                # This creates a collision which *might* lead to cancellation if coeffs allow.
                # We pick an index i and try to set e_i such that e_i + e_j = target_sum
                # for a random j != i.
                i = random.randint(0, 11)
                j = random.randint(0, 11)
                if i == j: continue
                
                # Choose a target sum. 
                # Option A: Target an existing sum from current square to increase multiplicity.
                # Option B: Target a random sum in [0, 80].
                if random.random() < 0.5:
                    # Sample a random sum from the current support
                    # We need the support of the current square. 
                    # For speed, we just pick a random sum between min and max
                    min_sum = min(new_exps) + min(new_exps)
                    max_sum = max(new_exps) + max(new_exps)
                    target = random.randint(min_sum, max_sum)
                else:
                    # Random target
                    target = random.randint(0, 80)
                
                new_e = target - new_exps[j]
                if 0 <= new_e < 41 and new_e not in new_exps:
                    new_exps[i] = new_e
                    new_exps.sort()
                else:
                    continue
            elif r < 0.65:
                # Two exponent shift (coordinated)
                i, j = random.sample(range(12), 2)
                si = random.randint(-5, 5)
                sj = random.randint(-5, 5)
                ne_i = new_exps[i] + si
                ne_j = new_exps[j] + sj
                if (0 <= ne_i < 41 and 0 <= ne_j < 41 and 
                    ne_i not in new_exps and ne_j not in new_exps and ne_i != ne_j):
                    new_exps[i] = ne_i
                    new_exps[j] = ne_j
                    new_exps.sort()
                else:
                    continue
            elif r < 0.85:
                # Coefficient change
                idx = random.randint(0, 11)
                # Try small integers first
                new_val = random.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 12])
                if new_val != new_coefs[idx]:
                    new_coefs[idx] = new_val
            else:
                # Large jump: reset one exponent
                idx = random.randint(0, 11)
                new_e = random.randint(0, 40)
                if new_e not in new_exps:
                    new_exps[idx] = new_e
                    new_exps.sort()
                else:
                    continue

            # Validate
            if len(set(new_exps)) != 12 or any(e < 0 or e >= 41 for e in new_exps):
                continue
            if any(c == 0 or abs(c) > 64 for c in new_coefs):
                continue

            new_nz = square_nonzero(new_exps, new_coefs)
            delta = new_nz - cur_nz

            # Accept if better, or with probability exp(-delta/t)
            if delta < 0 or random.random() < np.exp(-delta / max(t, 1e-10)):
                cur_nz = new_nz
                cur_exps = list(new_exps)
                cur_coefs = list(new_coefs)

                if cur_nz < best_in_run_nz:
                    best_in_run_nz = cur_nz
                    best_in_run_exps = list(cur_exps)
                    best_in_run_coefs = list(cur_coefs)

                if cur_nz < best_nz:
                    best_nz = cur_nz
                    best_poly = to_list(cur_exps, cur_coefs)

        return best_in_run_exps, best_in_run_coefs, best_in_run_nz

    # Phase 1: Generate diverse starting points
    random.seed(42)
    starting_points = []

    # 1. Arithmetic Progressions with varying signs
    for d in [1, 2, 3, 4, 5]:
        max_a = max(0, 40 - 11 * d)
        for a in range(0, min(max_a + 1, 6)):
            exps = [a + i * d for i in range(12)]
            if exps[-1] < 41:
                # Alternating signs
                coefs = [1 if i % 2 == 0 else -1 for i in range(12)]
                starting_points.append((exps, coefs))
                # Block signs
                coefs2 = [1]*6 + [-1]*6
                starting_points.append((exps, coefs2))
                # Random signs
                coefs3 = [random.choice([-1, 1]) for _ in range(12)]
                starting_points.append((exps, coefs3))

    # 2. Two Clusters
    for d1 in [1, 2]:
        for d2 in [1, 2]:
            for gap in [5, 8, 10, 15]:
                for a1 in [0, 2, 4]:
                    a2 = a1 + gap
                    exp1 = [a1 + i * d1 for i in range(6)]
                    exp2 = [a2 + i * d2 for i in range(6)]
                    all_exps = sorted(exp1 + exp2)
                    if len(set(all_exps)) == 12 and all(e < 41 for e in all_exps):
                        coefs = [1]*6 + [-1]*6
                        starting_points.append((all_exps, coefs))
                        coefs_alt = [1, -1, 1, -1, 1, -1] * 2
                        starting_points.append((all_exps, coefs_alt))

    # 3. Symmetric around center
    for center in [10, 15, 20, 25]:
        for offsets in [[1,2,3,4,5,6], [1,2,3,4,6,8], [2,3,4,5,6,7], [1,3,5,7,9,11]]:
            exps = sorted([center - o for o in offsets] + [center + o for o in offsets])
            if all(0 <= e < 41 for e in exps) and len(set(exps)) == 12:
                coefs = [1]*6 + [-1]*6
                starting_points.append((exps, coefs))
                coefs2 = [1, -1, 1, -1, 1, -1] * 2
                starting_points.append((exps, coefs2))

    # 4. Random starts
    for _ in range(20):
        exps = sorted(random.sample(range(41), 12))
        coefs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        starting_points.append((exps, coefs))

    # Phase 2: Evaluate and Refine
    # Sort starting points by initial score to prioritize good structures?
    # Or just run SA on all.
    
    # First pass: quick evaluation
    scored_starts = []
    for exps, coefs in starting_points:
        nz = square_nonzero(exps, coefs)
        scored_starts.append((nz, exps, coefs))
        
    scored_starts.sort(key=lambda x: x[0])
    
    # Run SA on the best N starting points
    num_sa_runs = min(len(scored_starts), 40)
    time_per_run = (deadline - time.time() - 5) / max(num_sa_runs, 1)
    
    for i in range(num_sa_runs):
        if time.time() > deadline - 5:
            break
        
        nz, exps, coefs = scored_starts[i]
        # Estimate iterations
        # square_nonzero takes ~1-2ms. SA iter takes ~1ms + overhead.
        # Let's aim for 2000-5000 iters per run depending on time left.
        est_iters = int(time_per_run * 2000)
        est_iters = max(1000, min(est_iters, 10000))
        
        local_search(exps, coefs, max_iters=est_iters)

    # Phase 3: Continue from best found
    if best_poly is not None and time.time() < deadline - 8:
        best_exp = [i for i, c in enumerate(best_poly) if c != 0]
        best_coef = [c for c in best_poly if c != 0]
        
        # Multiple restarts from perturbed best
        remaining_time = deadline - time.time() - 5
        num_extra = max(1, int(remaining_time / 4.0))
        
        for restart in range(min(num_extra, 30)):
            if time.time() > deadline - 5:
                break
            
            pert_exp = list(best_exp)
            pert_coef = list(best_coef)
            
            # Perturb
            num_pert = random.randint(2, 4)
            for _ in range(num_pert):
                if random.random() < 0.6:
                    idx = random.randint(0, 11)
                    shift = random.randint(-8, 8)
                    new_e = pert_exp[idx] + shift
                    if 0 <= new_e < 41 and new_e not in pert_exp:
                        pert_exp[idx] = new_e
                        pert_exp.sort()
                else:
                    idx = random.randint(0, 11)
                    pert_coef[idx] = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
                    if pert_coef[idx] == 0:
                        pert_coef[idx] = 1
            
            if len(set(pert_exp)) == 12 and all(0 <= e < 41 for e in pert_exp) and \
               all(c != 0 and abs(c) <= 64 for c in pert_coef):
                remaining = deadline - time.time() - 5
                est_iters = max(1000, int(remaining * 1500))
                local_search(pert_exp, pert_coef, max_iters=min(est_iters, 8000))

    # Final fallback
    if best_poly is None:
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    if not validate(best_poly):
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    return best_poly

# EVOLVE-BLOCK-END
