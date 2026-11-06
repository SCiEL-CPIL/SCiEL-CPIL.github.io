import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    def eval_poly(exponents, coeffs):
        """Validate and return nonzero count in square, or infinity if invalid."""
        if len(set(exponents)) != 12:
            return float('inf')
        if any(e < 0 or e > 40 for e in exponents):
            return float('inf')
        if any(c == 0 for c in coeffs):
            return float('inf')
        if any(abs(c) > 64 for c in coeffs):
            return float('inf')
        # Check max degree of square <= 40
        max_e = max(exponents)
        if 2 * max_e > 40:
            return float('inf')
        # Build square coefficient dict
        n = len(exponents)
        sq = {}
        for i in range(n):
            for j in range(i, n):
                e = exponents[i] + exponents[j]
                c = coeffs[i] * coeffs[j]
                if i != j:
                    c *= 2
                sq[e] = sq.get(e, 0) + c
        nonzero = sum(1 for v in sq.values() if v != 0)
        return nonzero
    
    # Initial solution
    best_exponents = list(range(12))
    best_coeffs = [1] * 12
    best_score = eval_poly(best_exponents, best_coeffs)
    
    # Try some hand-crafted structured starts
    candidates = []
    
    # Start 1: consecutive
    candidates.append((list(range(12)), [1]*12))
    
    # Start 2: even exponents
    candidates.append((list(range(0, 24, 2)), [1]*12))
    
    # Start 3: alternating signs on even exponents
    candidates.append((list(range(0, 24, 2)), [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]))
    
    # Start 4: random samples
    for _ in range(50):
        exps = sorted(random.sample(range(0, 21), 12))  # max exp <= 20 so 2*max <= 40
        coefs = [random.choice([-1, 1]) for _ in range(12)]
        candidates.append((exps, coefs))
    
    # Evaluate all candidates
    for exps, coefs in candidates:
        s = eval_poly(exps, coefs)
        if s < best_score:
            best_score = s
            best_exponents = exps[:]
            best_coeffs = coefs[:]
    
    # Simulated annealing
    current_exponents = best_exponents[:]
    current_coeffs = best_coeffs[:]
    current_score = best_score
    
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Change one exponent
            i = random.randint(0, 11)
            old_e = current_exponents[i]
            new_e = random.randint(0, 20)
            if new_e != old_e and new_e not in current_exponents:
                current_exponents[i] = new_e
                current_exponents.sort()
                new_score = eval_poly(current_exponents, current_coeffs)
                if new_score < current_score or random.random() < math.exp((current_score - new_score) / T):
                    if new_score < best_score:
                        best_score = new_score
                        best_exponents = current_exponents[:]
                        best_coeffs = current_coeffs[:]
                    current_score = new_score
                else:
                    current_exponents[i] = old_e
                    current_exponents.sort()
        elif move_type < 0.7:
            # Change one coefficient sign
            i = random.randint(0, 11)
            current_coeffs[i] *= -1
            new_score = eval_poly(current_exponents, current_coeffs)
            if new_score < current_score or random.random() < math.exp((current_score - new_score) / T):
                if new_score < best_score:
                    best_score = new_score
                    best_exponents = current_exponents[:]
                    best_coeffs = current_coeffs[:]
                current_score = new_score
            else:
                current_coeffs[i] *= -1
        elif move_type < 0.85:
            # Swap two exponents
            i, j = random.sample(range(12), 2)
            old_ei, old_ej = current_exponents[i], current_exponents[j]
            current_exponents[i], current_exponents[j] = old_ej, old_ei
            new_score = eval_poly(current_exponents, current_coeffs)
            if new_score < current_score or random.random() < math.exp((current_score - new_score) / T):
                if new_score < best_score:
                    best_score = new_score
                    best_exponents = current_exponents[:]
                    best_coeffs = current_coeffs[:]
                current_score = new_score
            else:
                current_exponents[i], current_exponents[j] = old_ei, old_ej
        else:
            # Change coefficient magnitude
            i = random.randint(0, 11)
            old_c = current_coeffs[i]
            new_c = random.choice([1, 2, 3, 4, 5, 8, 16, 32]) * (1 if old_c > 0 else -1)
            if abs(new_c) <= 64 and new_c != 0:
                current_coeffs[i] = new_c
                new_score = eval_poly(current_exponents, current_coeffs)
                if new_score < current_score or random.random() < math.exp((current_score - new_score) / T):
                    if new_score < best_score:
                        best_score = new_score
                        best_exponents = current_exponents[:]
                        best_coeffs = current_coeffs[:]
                    current_score = new_score
                else:
                    current_coeffs[i] = old_c
        
        T *= cooling
    
    # Build result from best solution
    exps = best_exponents
    coefs = best_coeffs
    
    # Create coefficient list
    max_deg = max(exps)
    result = [0] * (max_deg + 1)
    for e, c in zip(exps, coefs):
        result[e] = c
    
    # Trim trailing zeros (but ensure nonzero final coefficient)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    # Validate
    assert len(result) <= 41
    assert result[-1] != 0
    assert all(-64 <= x <= 64 for x in result)
    assert sum(1 for x in result if x != 0) == 12
    
    return result


# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Polynomial: {result}")
    print(f"Degree: {len(result)-1}")
    print(f"Nonzero terms: {sum(1 for x in result if x != 0)}")
    # Compute square
    n = len(result)
    sq = [0] * (2*n - 1)
    for i in range(n):
        for j in range(i, n):
            e = i + j
            c = result[i] * result[j]
            if i != j:
                c *= 2
            sq[e] += c
    nz = sum(1 for x in sq if x != 0)
    print(f"Square nonzero terms: {nz}")
