import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _square_nonzero_count(exponents, coefficients):
    """Compute P(x)^2 from sparse representation and count nonzero coefficients."""
    n = len(exponents)
    square = defaultdict(int)
    for i in range(n):
        for j in range(i, n):
            e = exponents[i] + exponents[j]
            c = coefficients[i] * coefficients[j]
            if i != j:
                c *= 2
            square[e] += c
    return sum(1 for c in square.values() if c != 0)


def _build_coeff_list(exponents, coefficients, max_degree=40):
    """Build a coefficient list in ascending degree from sparse representation."""
    max_exp = max(exponents)
    if max_exp > max_degree:
        return None
    result = [0] * (max_exp + 1)
    for e, c in zip(exponents, coefficients):
        result[e] = c
    nonzero = sum(1 for x in result if x != 0)
    if nonzero != 12:
        return None
    if result[-1] == 0:
        return None
    if any(abs(x) > 64 for x in result):
        return None
    if len(result) > 41:
        return None
    return result


def _poly_square_sparse(sparse):
    """Square a sparse polynomial and return sparse result."""
    n = len(sparse)
    result = defaultdict(int)
    for i in range(n):
        for j in range(i, n):
            e = sparse[i][0] + sparse[j][0]
            c = sparse[i][1] * sparse[j][1]
            if i != j:
                c *= 2
            result[e] += c
    return [(e, c) for e, c in result.items() if c != 0]


def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Fallback
    fallback = [1] * 12
    fallback_score = 21

    best_score = fallback_score
    best_candidate = fallback

    def additive_energy(exps):
        n = len(exps)
        sums = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                s = exps[i] + exps[j]
                sums[s].append((i, j))
        count = 0
        for s, pairs in sums.items():
            if len(pairs) > 1:
                count += len(pairs) * (len(pairs) - 1) // 2
        return count

    # Generate candidate exponent sets with high additive energy
    candidate_exps = []

    # Arithmetic progressions
    for start in range(0, 10):
        for step in range(1, 8):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                candidate_exps.append(exps)

    # Two interleaved arithmetic progressions
    for s1 in range(0, 5):
        for s2 in range(5, 15):
            for step in range(1, 5):
                exps1 = [s1 + i * step for i in range(6)]
                exps2 = [s2 + i * step for i in range(6)]
                exps = sorted(exps1 + exps2)
                if len(set(exps)) == 12 and max(exps) <= 40:
                    candidate_exps.append(exps)

    # Three interleaved
    for s1 in range(0, 4):
        for s2 in range(4, 12):
            for s3 in range(12, 25):
                for step in range(1, 4):
                    exps1 = [s1 + i * step for i in range(4)]
                    exps2 = [s2 + i * step for i in range(4)]
                    exps3 = [s3 + i * step for i in range(4)]
                    exps = sorted(exps1 + exps2 + exps3)
                    if len(set(exps)) == 12 and max(exps) <= 40:
                        candidate_exps.append(exps)

    # Four interleaved groups of 3
    for s1 in range(0, 3):
        for s2 in range(3, 10):
            for s3 in range(10, 20):
                for s4 in range(20, 30):
                    for step in range(1, 3):
                        exps1 = [s1 + i * step for i in range(3)]
                        exps2 = [s2 + i * step for i in range(3)]
                        exps3 = [s3 + i * step for i in range(3)]
                        exps4 = [s4 + i * step for i in range(3)]
                        exps = sorted(exps1 + exps2 + exps3 + exps4)
                        if len(set(exps)) == 12 and max(exps) <= 40:
                            candidate_exps.append(exps)

    # Quadratic-like patterns
    for a in range(1, 4):
        for b in range(0, 10):
            for c in range(0, 20):
                exps = [a * i * i + b * i + c for i in range(12)]
                if len(set(exps)) == 12 and max(exps) <= 40 and min(exps) >= 0:
                    candidate_exps.append(sorted(exps))

    # Mixed: 2 clusters of 6 with different steps
    for s1 in range(0, 5):
        for step1 in range(1, 4):
            for s2 in range(8, 20):
                for step2 in range(1, 4):
                    exps1 = [s1 + i * step1 for i in range(6)]
                    exps2 = [s2 + i * step2 for i in range(6)]
                    exps = sorted(exps1 + exps2)
                    if len(set(exps)) == 12 and max(exps) <= 40:
                        candidate_exps.append(exps)

    # Symmetric patterns: pairs summing to constant
    for total in range(12, 41):
        for i in range(6):
            for j in range(i + 1, 6):
                # Try symmetric structure around total/2
                half = total // 2
                exps = sorted([0, 1, 2, 3, total-3, total-2, total-1, total,
                              half-1, half, half+1, half+2])
                if len(set(exps)) == 12 and max(exps) <= 40 and min(exps) >= 0:
                    candidate_exps.append(sorted(exps))

    # Deduplicate
    seen_exps = set()
    unique_exps = []
    for exps in candidate_exps:
        key = tuple(exps)
        if key not in seen_exps:
            seen_exps.add(key)
            unique_exps.append(exps)

    def refined_score(exps):
        ae = additive_energy(exps)
        sums = defaultdict(int)
        for i in range(12):
            for j in range(i + 1, 12):
                sums[exps[i] + exps[j]] += 1
        max_collision = max(sums.values()) if sums else 0
        degree_penalty = max(0, max(exps) - 30)
        return ae + max_collision * 2 - degree_penalty

    scored_exps = []
    for exps in unique_exps:
        rs = refined_score(exps)
        scored_exps.append((rs, exps))
    scored_exps.sort(key=lambda x: -x[0])

    def optimize_signs(exps, max_iter=50):
        signs = [1] * 12
        current_score = _square_nonzero_count(exps, signs)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            for i in range(12):
                signs[i] = -signs[i]
                new_score = _square_nonzero_count(exps, signs)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    signs[i] = -signs[i]
            for i in range(12):
                for j in range(i + 1, 12):
                    signs[i] = -signs[i]
                    signs[j] = -signs[j]
                    new_score = _square_nonzero_count(exps, signs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        signs[i] = -signs[i]
                        signs[j] = -signs[j]
            if not improved:
                break
        return current_score, signs

    def optimize_coeffs(exps, max_iter=40):
        coeffs = [1] * 12
        current_score = _square_nonzero_count(exps, coeffs)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            for i in range(12):
                old_val = coeffs[i]
                for new_val in [-3, -2, -1, 1, 2, 3]:
                    if new_val == old_val:
                        continue
                    coeffs[i] = new_val
                    new_score = _square_nonzero_count(exps, coeffs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        coeffs[i] = old_val
                        break
            if not improved:
                break
        return current_score, coeffs

    # Try top exponent sets
    for ae, exps in scored_exps[:500]:
        if time.time() > deadline:
            break
        
        score, signs = optimize_signs(exps, max_iter=30)
        
        if score < best_score:
            best_score = score
            candidate = _build_coeff_list(exps, signs)
            if candidate is not None:
                best_candidate = candidate
        
        if time.time() < deadline - 5:
            score2, coeffs = optimize_coeffs(exps, max_iter=20)
            
            if score2 < best_score:
                best_score = score2
                candidate = _build_coeff_list(exps, coeffs)
                if candidate is not None:
                    best_candidate = candidate

    # Factorization approach: P = A * B where A, B are 6-term
    if time.time() < deadline - 15:
        def gen_6term(max_exp=20):
            polys = []
            for start in range(0, 8):
                for step in range(1, 6):
                    exps = [start + i * step for i in range(6)]
                    if max(exps) > max_exp:
                        continue
                    for signs in itertools.product([-1, 1], repeat=6):
                        coeffs = list(signs)
                        sparse = [(e, c) for e, c in zip(exps, coeffs)]
                        polys.append(sparse)
            for base_exps in [[0, 1, 2, 4, 8, 16], [0, 1, 3, 7, 15, 31]]:
                if max(base_exps) > max_exp:
                    continue
                for signs in itertools.product([-1, 1], repeat=6):
                    coeffs = list(signs)
                    sparse = [(e, c) for e, c in zip(base_exps, coeffs)]
                    polys.append(sparse)
            return polys

        poly_library = gen_6term(max_exp=20)
        
        seen = set()
        unique_polys = []
        for p in poly_library:
            key = tuple(sorted(p))
            if key not in seen:
                seen.add(key)
                unique_polys.append(p)
        
        if len(unique_polys) > 200:
            import random
            random.seed(123)
            unique_polys = random.sample(unique_polys, 200)
        
        for i in range(len(unique_polys)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(unique_polys)):
                if time.time() > deadline:
                    break
                A = unique_polys[i]
                B = unique_polys[j]
                # P = A * B
                result = defaultdict(int)
                for ea, ca in A:
                    for eb, cb in B:
                        result[ea + eb] += ca * cb
                P_sparse = [(e, c) for e, c in result.items() if c != 0]
                
                if len(P_sparse) != 12:
                    continue
                if P_sparse[-1][0] > 40:
                    continue
                
                square_sparse = _poly_square_sparse(P_sparse)
                score = len(square_sparse)
                
                if score < best_score:
                    best_score = score
                    exps = [e for e, c in P_sparse]
                    coeffs = [c for e, c in P_sparse]
                    candidate = _build_coeff_list(exps, coeffs)
                    if candidate is not None:
                        best_candidate = candidate

    if best_score < fallback_score:
        return best_candidate
    
    return fallback

# EVOLVE-BLOCK-END
