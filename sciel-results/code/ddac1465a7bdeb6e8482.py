"""Sparse squares of twelve-term integer polynomials - collision-graph guided SA."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 85.0

    best_poly = None
    best_nz = float('inf')
    best_exps = None
    best_coefs = None

    def square_nonzero_fast(exponents, coeffs):
        c = np.asarray(coeffs, dtype=np.int64)
        e = np.asarray(exponents, dtype=np.int64)
        degs = e[:, None] + e[None, :]
        vals = c[:, None] * c[None, :]
        result = np.bincount(degs.ravel(), weights=vals.ravel(), minlength=82)
        return int(np.count_nonzero(result))

    def collision_count(exponents):
        """Count number of sum collisions (pairs of pairs with same sum)."""
        e = np.asarray(exponents, dtype=np.int64)
        degs = e[:, None] + e[None, :]
        # Only count upper triangle pairs (i<j) to avoid self-counting
        iu = np.triu_indices(12, k=1)
        pair_sums = degs[iu]
        # Count how many sums appear more than once
        unique, counts = np.unique(pair_sums, return_counts=True)
        # Each collision group of size k contributes C(k,2) colliding pairs
        collisions = int(np.sum(counts[counts > 1] * (counts[counts > 1] - 1) // 2))
        return collisions

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

    def update_best(exponents, coeffs, nz):
        nonlocal best_poly, best_nz, best_exps, best_coefs
        if nz < best_nz:
            best_nz = nz
            best_exps = list(exponents)
            best_coefs = list(coeffs)
            max_deg = max(exponents)
            poly = [0] * (max_deg + 1)
            for e, c in zip(exponents, coeffs):
                poly[e] = c
            best_poly = poly

    random.seed(42)

    # Phase 1: Structured seed generation
    seeds = []
    # AP structures
    for d in range(1, 8):
        max_a = max(0, 40 - 11 * d)
        for a in range(0, max_a + 1, 2):
            exps = [a + i * d for i in range(12)]
            if exps[-1] >= 41:
                continue
            for coefs in [[1]*12, [1 if i%2==0 else -1 for i in range(12)],
                         [1,1,-1,-1,1,1,-1,-1,1,1,-1,-1],
                         [random.choice([-2,-1,1,2]) for _ in range(12)]]:
                seeds.append((exps, coefs))
    # Two-AP
    for d1 in range(1, 4):
        for d2 in range(1, 4):
            for a1 in range(0, 10, 2):
                for gap in range(4, 16, 2):
                    a2 = a1 + gap
                    exps = sorted([a1 + i*d1 for i in range(6)] + [a2 + i*d2 for i in range(6)])
                    if len(set(exps)) != 12 or any(e >= 41 for e in exps):
                        continue
                    for s in [1, -1]:
                        coefs = [s]*6 + [s]*6
                        seeds.append((exps, coefs))
    # Symmetric
    for center in range(8, 33, 2):
        for offsets in [[1,2,3,4,5,6],[1,2,3,4,5,8],[1,2,4,5,7,9],[2,3,4,5,6,8]]:
            exps = sorted([center-o for o in offsets] + [center+o for o in offsets])
            if any(e < 0 or e >= 41 for e in exps) or len(set(exps)) != 12:
                continue
            for coefs in [[1]*6+[1]*6, [1]*6+[-1]*6, [1,-1,1,-1,1,-1]*2]:
                seeds.append((exps, coefs))

    # Evaluate seeds
    for exps, coefs in seeds:
        if time.time() > deadline - 50:
            break
        if len(set(exps)) != 12 or any(e < 0 or e >= 41 for e in exps):
            continue
        if any(c == 0 or abs(c) > 64 for c in coefs):
            continue
        nz = square_nonzero_fast(exps, coefs)
        update_best(exps, coefs, nz)

    # Phase 2: Collision-graph guided simulated annealing
    def simulated_annealing(init_exps, init_coefs, max_iters=5000):
        nonlocal best_poly, best_nz, best_exps, best_coefs
        cur_exps = sorted(init_exps)
        cur_coefs = list(init_coefs)
        cur_nz = square_nonzero_fast(cur_exps, cur_coefs)
        update_best(cur_exps, cur_coefs, cur_nz)

        T = 5.0
        T_min = 0.01
        cooling = 0.9995

        for it in range(max_iters):
            if time.time() > deadline - 3:
                break

            new_exps = list(cur_exps)
            new_coefs = list(cur_coefs)
            move_type = random.random()

            if move_type < 0.30:
                # Single exponent move
                idx = random.randint(0, 11)
                shift = random.randint(-5, 5)
                new_e = new_exps[idx] + shift
                if 0 <= new_e < 41 and new_e not in new_exps:
                    new_exps[idx] = new_e
                    new_exps.sort()
                else:
                    continue
            elif move_type < 0.50:
                # Two exponent move (structural swap)
                i, j = random.sample(range(12), 2)
                si = random.randint(-4, 4)
                sj = random.randint(-4, 4)
                ne_i = new_exps[i] + si
                ne_j = new_exps[j] + sj
                if (0 <= ne_i < 41 and 0 <= ne_j < 41 and
                    ne_i not in new_exps and ne_j not in new_exps and ne_i != ne_j):
                    new_exps[i] = ne_i
                    new_exps[j] = ne_j
                    new_exps.sort()
                else:
                    continue
            elif move_type < 0.60:
                # Three exponent move
                i, j, k = random.sample(range(12), 3)
                si = random.randint(-3, 3)
                sj = random.randint(-3, 3)
                sk = random.randint(-3, 3)
                ne_i = new_exps[i] + si
                ne_j = new_exps[j] + sj
                ne_k = new_exps[k] + sk
                if (0 <= ne_i < 41 and 0 <= ne_j < 41 and 0 <= ne_k < 41 and
                    ne_i not in new_exps and ne_j not in new_exps and ne_k not in new_exps and
                    ne_i != ne_j and ne_i != ne_k and ne_j != ne_k):
                    new_exps[i] = ne_i
                    new_exps[j] = ne_j
                    new_exps[k] = ne_k
                    new_exps.sort()
                else:
                    continue
            elif move_type < 0.75:
                # Coefficient modification
                idx = random.randint(0, 11)
                new_val = random.choice([-8,-6,-4,-3,-2,-1,1,2,3,4,5,6,8,12])
                if new_val != new_coefs[idx] and abs(new_val) <= 64:
                    new_coefs[idx] = new_val
                else:
                    continue
            elif move_type < 0.85:
                # Two coefficient modification
                i, j = random.sample(range(12), 2)
                new_coefs[i] = random.choice([-4,-3,-2,-1,1,2,3,4,5])
                new_coefs[j] = random.choice([-4,-3,-2,-1,1,2,3,4,5])
            else:
                # Collision-guided move: try to create a new sum collision
                # Pick two pairs and try to align their sums
                i, j = random.sample(range(12), 2)
                k, l = random.sample([x for x in range(12) if x not in (i,j)], 2)
                target_sum = new_exps[i] + new_exps[j]
                # Try to move k or l so that new_exps[k]+new_exps[l] == target_sum
                delta_k = target_sum - new_exps[l] - new_exps[k]
                delta_l = target_sum - new_exps[i] - new_exps[j]  # already matches
                # Move k to hit target
                new_k = new_exps[k] + delta_k
                if 0 <= new_k < 41 and new_k not in new_exps and new_k != new_exps[l]:
                    new_exps[k] = new_k
                    new_exps.sort()
                else:
                    # Try moving l
                    new_l = target_sum - new_exps[k]
                    if 0 <= new_l < 41 and new_l not in new_exps and new_l != new_exps[k]:
                        new_exps[l] = new_l
                        new_exps.sort()
                    else:
                        continue

            if len(set(new_exps)) != 12 or any(e < 0 or e >= 41 for e in new_exps):
                continue
            if any(c == 0 or abs(c) > 64 for c in new_coefs):
                continue

            new_nz = square_nonzero_fast(new_exps, new_coefs)
            delta = new_nz - cur_nz

            if delta <= 0 or random.random() < np.exp(-delta / max(T, T_min)):
                cur_nz = new_nz
                cur_exps = list(new_exps)
                cur_coefs = list(new_coefs)
                update_best(cur_exps, cur_coefs, cur_nz)

            T = max(T * cooling, T_min)

        return cur_exps, cur_coefs, cur_nz

    # Run SA from multiple seeds
    sa_budget = 15
    for i in range(sa_budget):
        if time.time() > deadline - 5:
            break

        if i < 5 and best_exps is not None:
            # Perturb best solution
            exps = list(best_exps)
            coefs = list(best_coefs)
            for _ in range(random.randint(1, 3)):
                if random.random() < 0.5:
                    idx = random.randint(0, 11)
                    shift = random.randint(-3, 3)
                    new_e = exps[idx] + shift
                    if 0 <= new_e < 41 and new_e not in exps:
                        exps[idx] = new_e
                        exps.sort()
                else:
                    idx = random.randint(0, 11)
                    coefs[idx] = random.choice([-4,-3,-2,-1,1,2,3,4,5,6])
                    if coefs[idx] == 0:
                        coefs[idx] = 1
            if len(set(exps)) == 12 and all(0 <= e < 41 for e in exps) and \
               all(c != 0 and abs(c) <= 64 for c in coefs):
                simulated_annealing(exps, coefs, max_iters=4000)
        else:
            # Random structured start
            rtype = random.random()
            if rtype < 0.4:
                d = random.randint(1, 6)
                max_a = max(0, 40 - 11 * d)
                a = random.randint(0, max_a)
                exps = [a + i * d for i in range(12)]
                if exps[-1] >= 41:
                    continue
            elif rtype < 0.7:
                d1 = random.randint(1, 3)
                d2 = random.randint(1, 3)
                a1 = random.randint(0, 8)
                gap = random.randint(4, 14)
                a2 = a1 + gap
                exps = sorted([a1 + i*d1 for i in range(6)] + [a2 + i*d2 for i in range(6)])
                if len(set(exps)) != 12 or any(e >= 41 for e in exps):
                    continue
            else:
                exps = sorted(random.sample(range(41), 12))
            coefs = [random.choice([-3,-2,-1,1,2,3]) for _ in range(12)]
            simulated_annealing(exps, coefs, max_iters=3000)

    # Phase 3: Intensive SA on best solution with longer runs
    if best_exps is not None and time.time() < deadline - 10:
        # Final intensive SA
        simulated_annealing(best_exps, best_coefs, max_iters=8000)

    # Final fallback
    if best_poly is None:
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    if not validate(best_poly):
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    return best_poly

# EVOLVE-BLOCK-END
