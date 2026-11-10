import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    best_result = [1] * 12
    best_nonzero = 144

    def count_nonzero_square(coeffs, exponents):
        square = defaultdict(int)
        n = len(coeffs)
        for i in range(n):
            for j in range(i, n):
                deg = exponents[i] + exponents[j]
                if i == j:
                    square[deg] += coeffs[i] * coeffs[j]
                else:
                    square[deg] += 2 * coeffs[i] * coeffs[j]
        return sum(1 for v in square.values() if v != 0)

    def is_valid_result(coeffs, exponents):
        if len(coeffs) != 12:
            return False
        if any(c == 0 for c in coeffs):
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        if any(e < 0 for e in exponents):
            return False
        max_deg = max(exponents)
        if 2 * max_deg + 1 > 41:
            return False
        square = defaultdict(int)
        n = len(coeffs)
        for i in range(n):
            for j in range(i, n):
                deg = exponents[i] + exponents[j]
                if i == j:
                    square[deg] += coeffs[i] * coeffs[j]
                else:
                    square[deg] += 2 * coeffs[i] * coeffs[j]
        if square[max_deg * 2] == 0:
            return False
        return True

    def coeffs_from_state(state):
        coeffs = [0] * (2 * max(state[1]) + 1)
        for c, e in zip(state[0], state[1]):
            coeffs[e] = c
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs

    def build_collision_map(exponents):
        collision = defaultdict(list)
        n = len(exponents)
        for i in range(n):
            for j in range(i, n):
                deg = exponents[i] + exponents[j]
                if i == j:
                    collision[deg].append((i, j, 1))
                else:
                    collision[deg].append((i, j, 2))
        return collision

    def compute_square_nonzero(collision, coeffs):
        square = defaultdict(int)
        for deg, terms in collision.items():
            s = 0
            for i, j, mult in terms:
                s += mult * coeffs[i] * coeffs[j]
            if s != 0:
                square[deg] = s
        return len(square)

    # Promising exponent patterns with high collision density
    exponent_patterns = [
        list(range(12)),
        list(range(0, 24, 2)),
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 20],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 3, 5, 6, 8, 10, 11, 13, 15, 17, 19],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ]
    
    seen = set()
    unique_patterns = []
    for p in exponent_patterns:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique_patterns.append(p)
    exponent_patterns = unique_patterns

    # For each pattern, try structured coefficient searches
    for exponents in exponent_patterns:
        if time.time() > deadline:
            break
        
        max_deg = max(exponents)
        if 2 * max_deg + 1 > 41:
            continue
            
        collision = build_collision_map(exponents)
        
        # Try various coefficient patterns
        # Pattern 1: Alternating signs
        for base in [1, 2, 3]:
            for sign_start in [-1, 1]:
                coeffs = [sign_start * base * ((-1) ** i) for i in range(12)]
                if is_valid_result(coeffs, exponents):
                    nz = compute_square_nonzero(collision, coeffs)
                    if nz < best_nonzero:
                        best_nonzero = nz
                        result = coeffs_from_state((coeffs, exponents))
                        if len(result) <= 41 and result[-1] != 0:
                            best_result = result
        
        # Pattern 2: Periodic coefficients
        for period in [2, 3, 4, 6]:
            for p_vals in itertools.product([-3, -2, -1, 1, 2, 3], repeat=period):
                if 0 in p_vals:
                    continue
                coeffs = [p_vals[i % period] for i in range(12)]
                if is_valid_result(coeffs, exponents):
                    nz = compute_square_nonzero(collision, coeffs)
                    if nz < best_nonzero:
                        best_nonzero = nz
                        result = coeffs_from_state((coeffs, exponents))
                        if len(result) <= 41 and result[-1] != 0:
                            best_result = result
        
        # Pattern 3: Symmetric coefficients c_i = c_{11-i} or c_i = -c_{11-i}
        for sym_type in ['sym', 'antisym']:
            for first_half in itertools.product([-4, -3, -2, -1, 1, 2, 3, 4], repeat=6):
                if 0 in first_half:
                    continue
                coeffs = list(first_half)
                for i in range(5, -1, -1):
                    if sym_type == 'sym':
                        coeffs.append(first_half[i])
                    else:
                        coeffs.append(-first_half[i])
                if is_valid_result(coeffs, exponents):
                    nz = compute_square_nonzero(collision, coeffs)
                    if nz < best_nonzero:
                        best_nonzero = nz
                        result = coeffs_from_state((coeffs, exponents))
                        if len(result) <= 41 and result[-1] != 0:
                            best_result = result
        
        # Pattern 4: Two-block structure
        for k in range(1, 10):
            for a0 in range(-3, 4):
                if a0 == 0: continue
                for a1 in range(-3, 4):
                    if a1 == 0: continue
                    for a2 in range(-3, 4):
                        if a2 == 0: continue
                        for a3 in range(-3, 4):
                            if a3 == 0: continue
                            for a4 in range(-3, 4):
                                if a4 == 0: continue
                                for a5 in range(-3, 4):
                                    if a5 == 0: continue
                                    A = [a0, a1, a2, a3, a4, a5]
                                    if k > 5:
                                        coeffs = A + A
                                        exponents2 = list(range(6)) + list(range(k, k + 6))
                                        if max(exponents2) * 2 + 1 > 41:
                                            continue
                                        if is_valid_result(coeffs, exponents2):
                                            col2 = build_collision_map(exponents2)
                                            nz = compute_square_nonzero(col2, coeffs)
                                            if nz < best_nonzero:
                                                best_nonzero = nz
                                                result = coeffs_from_state((coeffs, exponents2))
                                                if len(result) <= 41 and result[-1] != 0:
                                                    best_result = result
                                    if time.time() > deadline:
                                        break
                                if time.time() > deadline:
                                    break
                            if time.time() > deadline:
                                break
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    # Final validation
    if len(best_result) > 41 or best_result[-1] == 0:
        best_result = [1] * 12
    nz_count = sum(1 for c in best_result if c != 0)
    if nz_count != 12:
        best_result = [1] * 12
    if any(c < -64 or c > 64 for c in best_result):
        best_result = [1] * 12
    
    return best_result
