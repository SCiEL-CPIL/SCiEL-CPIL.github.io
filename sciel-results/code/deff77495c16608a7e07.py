import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    def count_nonzero_square(coeffs):
        """Count nonzero terms in the square of polynomial with given coeffs."""
        n = len(coeffs)
        square = [0] * (2 * n - 1)
        for i in range(n):
            if coeffs[i] == 0:
                continue
            for j in range(i, n):
                if coeffs[j] == 0:
                    continue
                val = coeffs[i] * coeffs[j]
                square[i + j] += val
                if i != j:
                    square[i + j] += val
        return sum(1 for x in square if x != 0)
    
    def make_poly(exponents, coefficients, max_deg=40):
        """Create coefficient list from exponents and coefficients."""
        if max(exponents) > max_deg:
            return None
        coeffs = [0] * (max_deg + 1)
        for e, c in zip(exponents, coefficients):
            coeffs[e] = c
        return coeffs
    
    def square_nonzero_count(exponents, coefficients, max_deg=40):
        """Count nonzero terms in square, handling sparse representation."""
        if max(exponents) > max_deg:
            return 999
        coeffs = [0] * (max_deg + 1)
        for e, c in zip(exponents, coefficients):
            coeffs[e] = c
        return count_nonzero_square(coeffs)
    
    # Start with a reasonable initial solution
    # Try arithmetic progression with alternating signs
    best_score = 999
    best_exponents = None
    best_coefficients = None
    
    # Initial: try some structured patterns
    # Pattern 1: exponents 0,1,2,...,11 with alternating signs
    for k in range(1, 4):
        exponents = list(range(0, 12 * k, k))
        if max(exponents) > 40:
            continue
        for sign_pattern in [
            [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
            [1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2],
        ]:
            if len(sign_pattern) != 12:
                continue
            score = square_nonzero_count(exponents, sign_pattern)
            if score < best_score:
                best_score = score
                best_exponents = exponents[:]
                best_coefficients = sign_pattern[:]
    
    # Local search: try to improve by changing exponents or coefficients
    random.seed(42)
    
    def local_search(exponents, coefficients, max_iter=5000):
        nonlocal best_score, best_exponents, best_coefficients
        current_score = square_nonzero_count(exponents, coefficients)
        current_exponents = exponents[:]
        current_coefficients = coefficients[:]
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if current_score == best_score:
                best_exponents = current_exponents[:]
                best_coefficients = current_coefficients[:]
            
            # Random perturbation
            if random.random() < 0.5:
                # Change an exponent
                idx = random.randint(0, 11)
                old_e = current_exponents[idx]
                new_e = random.randint(0, 40)
                if new_e not in current_exponents:
                    current_exponents[idx] = new_e
                    current_exponents.sort()
                    new_score = square_nonzero_count(current_exponents, current_coefficients)
                    if new_score <= current_score + random.randint(-1, 2):
                        current_score = new_score
                    else:
                        current_exponents[idx] = old_e
                        current_exponents.sort()
            else:
                # Change a coefficient
                idx = random.randint(0, 11)
                old_c = current_coefficients[idx]
                new_c = random.randint(-64, 64)
                if new_c != 0:
                    current_coefficients[idx] = new_c
                    new_score = square_nonzero_count(current_exponents, current_coefficients)
                    if new_score <= current_score + random.randint(-1, 2):
                        current_score = new_score
                    else:
                        current_coefficients[idx] = old_c
        
        return current_score, current_exponents, current_coefficients
    
    # Run local search from multiple starting points
    for restart in range(20):
        if time.time() > deadline:
            break
        
        if restart == 0:
            exponents = best_exponents[:]
            coefficients = best_coefficients[:]
        else:
            # Random restart with some structure
            if random.random() < 0.5:
                k = random.randint(1, 3)
                exponents = list(range(0, 12 * k, k))
                if max(exponents) > 40:
                    exponents = [random.randint(0, 40) for _ in range(12)]
                    exponents.sort()
                    # Ensure all unique
                    while len(set(exponents)) < 12:
                        exponents[random.randint(0, 11)] = random.randint(0, 40)
                        exponents.sort()
                coefficients = [random.choice([-1, 1, -2, 2]) for _ in range(12)]
            else:
                exponents = [random.randint(0, 40) for _ in range(12)]
                exponents.sort()
                while len(set(exponents)) < 12:
                    exponents[random.randint(0, 11)] = random.randint(0, 40)
                    exponents.sort()
                coefficients = [random.choice([-1, 1]) for _ in range(12)]
        
        score, exps, coeffs = local_search(exponents, coefficients, max_iter=3000)
        if score < best_score:
            best_score = score
            best_exponents = exps[:]
            best_coefficients = coeffs[:]
    
    # Final polish: try to improve coefficients for the best exponents
    if time.time() < deadline - 5:
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            idx = random.randint(0, 11)
            old_c = best_coefficients[idx]
            new_c = random.randint(-64, 64)
            if new_c != 0:
                best_coefficients[idx] = new_c
                score = square_nonzero_count(best_exponents, best_coefficients)
                if score > best_score:
                    best_coefficients[idx] = old_c
    
    # Build final coefficient list
    max_deg = max(best_exponents)
    result = [0] * (max_deg + 1)
    for e, c in zip(best_exponents, best_coefficients):
        result[e] = c
    
    # Ensure length <= 41 and nonzero final coefficient
    if len(result) > 41:
        result = result[:41]
    if result[-1] == 0:
        # Find last nonzero
        for i in range(len(result) - 1, -1, -1):
            if result[i] != 0:
                result = result[:i+1]
                break
    
    # Verify exactly 12 nonzero terms
    nz = sum(1 for x in result if x != 0)
    assert nz == 12, f"Expected 12 nonzero terms, got {nz}"
    assert len(result) <= 41
    assert result[-1] != 0
    assert all(-64 <= x <= 64 for x in result)
    
    return result
