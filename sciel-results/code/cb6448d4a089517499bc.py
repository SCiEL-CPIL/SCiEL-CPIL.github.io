import time
import random
import math

def _count_sq_nonzero(coeff_list):
    """Count nonzero terms in the square of polynomial given as coefficient list."""
    n = len(coeff_list)
    # Determine max degree
    max_deg = 0
    for i, c in enumerate(coeff_list):
        if c != 0:
            max_deg = i
    # Square has degree 2 * max_deg
    sq = [0] * (2 * max_deg + 1)
    
    # Extract nonzero terms (exponent, coeff)
    terms = [(i, c) for i, c in enumerate(coeff_list) if c != 0]
    
    # Convolve
    for i, ci in terms:
        for j, cj in terms:
            sq[i + j] += ci * cj
            
    return sum(1 for c in sq if c != 0)

def _validate(coeff_list):
    """Check if coefficient list is valid. Returns True if valid."""
    if len(coeff_list) > 41:
        return False
    if len(coeff_list) < 1:
        return False
    if coeff_list[-1] == 0:
        return False
    nz = sum(1 for c in coeff_list if c != 0)
    if nz != 12:
        return False
    for c in coeff_list:
        if c < -64 or c > 64:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_sq = 999
    best_coeffs = None
    
    def _time_check():
        return time.time() > deadline

    # Generate candidate exponent sets
    # Focus on Arithmetic Progressions (AP) as they maximize collisions
    exp_sets = []
    
    # APs with small step d
    for d in range(1, 6):
        for a in range(0, 41 - 11 * d + 1):
            exps = tuple(a + i * d for i in range(12))
            exp_sets.append(exps)
            
    # Add some random sets to explore non-AP structures
    random.seed(42)
    for _ in range(50):
        exps = tuple(sorted(random.sample(range(0, 41), 12)))
        exp_sets.append(exps)
        
    # Remove duplicates
    exp_sets = list(set(exp_sets))
    
    # For each exponent set, optimize coefficients
    # We use a local search / simulated annealing approach on the coefficients
    
    # Pre-calculate the "collision map" for each exp set to speed up evaluation?
    # No, direct calculation is fast enough (144 mults).
    
    for exps in exp_sets:
        if _time_check():
            break
            
        # Initialize coefficients
        # Try starting with all 1s, then perturb
        current_coefs = [1] * 12
        current_sq = 999
        
        # Calculate initial square
        coeff_list = [0] * (max(exps) + 1)
        for i, e in enumerate(exps):
            coeff_list[e] = current_coefs[i]
        current_sq = _count_sq_nonzero(coeff_list)
        
        if current_sq < best_sq:
            best_sq = current_sq
            best_coeffs = list(coeff_list)
            
        # Local search
        T = 1.0
        no_improve = 0
        max_iter = 2000 # Limit iterations per exp set
        
        for _ in range(max_iter):
            if _time_check():
                break
                
            # Pick random index to mutate
            idx = random.randint(0, 11)
            old_val = current_coefs[idx]
            
            # Generate new value
            # Prefer small integers for cancellation
            new_val = random.choice([-3, -2, -1, 1, 2, 3])
            if new_val == old_val:
                continue
                
            current_coefs[idx] = new_val
            
            # Evaluate
            coeff_list = [0] * (max(exps) + 1)
            for i, e in enumerate(exps):
                coeff_list[e] = current_coefs[i]
            new_sq = _count_sq_nonzero(coeff_list)
            
            delta = new_sq - current_sq
            
            if delta < 0:
                # Accept
                current_sq = new_sq
                no_improve = 0
                if new_sq < best_sq:
                    best_sq = new_sq
                    best_coeffs = list(coeff_list)
            else:
                # Reject or accept with probability
                if random.random() < math.exp(-delta / T):
                    current_sq = new_sq
                    no_improve += 1
                else:
                    # Revert
                    current_coefs[idx] = old_val
                    no_improve += 1
                    
            if no_improve > 100:
                # Restart with new random coefs or lower T
                T *= 0.9
                if T < 0.1:
                    break
                # Random restart
                current_coefs = [random.choice([-1, 1]) for _ in range(12)]
                coeff_list = [0] * (max(exps) + 1)
                for i, e in enumerate(exps):
                    coeff_list[e] = current_coefs[i]
                current_sq = _count_sq_nonzero(coeff_list)
                no_improve = 0
                if current_sq < best_sq:
                    best_sq = current_sq
                    best_coeffs = list(coeff_list)

    if best_coeffs is not None:
        return best_coeffs
    
    # Fallback
    return [1] * 12
