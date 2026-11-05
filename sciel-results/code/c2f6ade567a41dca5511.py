"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _square_nonzero_count(coeffs):
    """Compute number of nonzero coefficients in the square of a sparse polynomial."""
    if not coeffs:
        return 0
    square = {}
    keys = list(coeffs.keys())
    vals = list(coeffs.values())
    n = len(keys)
    for i in range(n):
        for j in range(i, n):
            exp = keys[i] + keys[j]
            if i == j:
                val = vals[i] * vals[j]
            else:
                val = 2 * vals[i] * vals[j]
            square[exp] = square.get(exp, 0) + val
    return sum(1 for v in square.values() if v != 0)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    best_nz = 144
    best_coeffs = None
    
    random.seed(12345)
    
    # Strategy 1: Arithmetic progression exponents with optimized coefficients
    # For exponents {0, d, 2d, ..., 11d}, try various coefficient patterns
    for d in [1, 2, 3]:
        if time.time() > deadline:
            break
        max_exp = 11 * d
        if max_exp > 40:
            continue
        
        # Try all sign patterns (2^12 = 4096)
        for mask in range(4096):
            if time.time() > deadline:
                break
            coeffs = {}
            for i in range(12):
                s = 1 if (mask >> i) & 1 else -1
                coeffs[i * d] = s
            nz = _square_nonzero_count(coeffs)
            if nz < best_nz:
                best_nz = nz
                best_coeffs = dict(coeffs)
        
        # Try small integer coefficients [-3..3] excluding 0 via local search
        if time.time() > deadline:
            continue
        
        # Initialize with best sign pattern
        best_local = dict(best_coeffs) if best_coeffs else {i*d: 1 for i in range(12)}
        
        for trial in range(50000):
            if time.time() > deadline:
                break
            # Random coefficient perturbation
            coeffs = dict(best_local)
            exps = list(coeffs.keys())
            idx = random.randint(0, 11)
            e = exps[idx]
            old_val = coeffs[e]
            # Try changing to a different small integer
            candidates = [v for v in range(-64, 65) if v != 0 and v != old_val]
            # Bias toward small values
            small_cands = [v for v in candidates if abs(v) <= 8]
            if small_cands:
                new_val = random.choice(small_cands)
            else:
                new_val = random.choice(candidates)
            coeffs[e] = new_val
            
            if any(abs(v) > 64 for v in coeffs.values()):
                continue
            if len(coeffs) != 12:
                continue
            
            nz = _square_nonzero_count(coeffs)
            if nz < best_nz:
                best_nz = nz
                best_coeffs = dict(coeffs)
                best_local = dict(coeffs)
    
    if time.time() > deadline:
        pass
    else:
        # Strategy 2: Structured exponent sets - pairs that create cancellation
        # Try exponents where many pairs sum to the same value
        # e.g., symmetric sets around a center
        
        for trial in range(100000):
            if time.time() > deadline:
                break
            
            # Generate structured exponent set
            # Try: base + arithmetic progression with some perturbations
            base = random.randint(0, 5)
            d = random.choice([1, 2, 3])
            exponents = set()
            for i in range(12):
                e = base + i * d
                if e <= 40:
                    exponents.add(e)
                else:
                    # Wrap or adjust
                    e = base + i * d - 11 * d  # shift back
                    if e >= 0:
                        exponents.add(e)
                    else:
                        exponents.add(0)
            
            if len(exponents) < 12:
                # Fill with random
                while len(exponents) < 12:
                    exponents.add(random.randint(0, 40))
            
            exponents = sorted(list(exponents))[:12]
            if len(set(exponents)) != 12:
                continue
            
            # Random small integer coefficients
            coeffs = {}
            for e in exponents:
                coeffs[e] = random.choice([1, -1, 2, -2, 3, -3])
            
            nz = _square_nonzero_count(coeffs)
            if nz < best_nz:
                best_nz = nz
                best_coeffs = dict(coeffs)
    
    if time.time() > deadline:
        pass
    else:
        # Strategy 3: Hill climbing on the best found
        if best_coeffs is not None:
            for trial in range(200000):
                if time.time() > deadline:
                    break
                
                coeffs = dict(best_coeffs)
                keys = list(coeffs.keys())
                
                # Random perturbation: change one coefficient value
                idx = random.randint(0, 11)
                e = keys[idx]
                old_val = coeffs[e]
                
                # Try nearby values
                delta = random.choice([-2, -1, 1, 2])
                new_val = old_val + delta
                if new_val == 0:
                    continue
                if abs(new_val) > 64:
                    continue
                
                coeffs[e] = new_val
                
                nz = _square_nonzero_count(coeffs)
                if nz < best_nz:
                    best_nz = nz
                    best_coeffs = dict(coeffs)
                
                # Occasionally try exponent perturbation
                if random.random() < 0.1:
                    coeffs = dict(best_coeffs)
                    idx = random.randint(0, 11)
                    e = keys[idx]
                    new_e = e + random.choice([-1, 1])
                    if new_e < 0 or new_e > 40:
                        continue
                    if new_e in coeffs:
                        continue
                    val = coeffs[e]
                    del coeffs[e]
                    coeffs[new_e] = val
                    
                    if len(coeffs) != 12:
                        continue
                    
                    nz = _square_nonzero_count(coeffs)
                    if nz < best_nz:
                        best_nz = nz
                        best_coeffs = dict(coeffs)
    
    # Convert to coefficient list
    if best_coeffs is None:
        return [1] * 12
    
    max_exp = max(best_coeffs.keys())
    result = [0] * (max_exp + 1)
    for exp, val in best_coeffs.items():
        result[exp] = val
    
    # Ensure length <= 41
    if len(result) > 41:
        result = result[:41]
    
    # Ensure nonzero final coefficient
    if result[-1] == 0:
        while result and result[-1] == 0:
            result.pop()
        if not result:
            result = [1] * 12
    
    return result
# EVOLVE-BLOCK-END
