"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _square_nonzero_count(coeff_list):
    """
    Computes the number of non-zero coefficients in the square of the polynomial.
    coeff_list: list of integers representing coefficients from degree 0 to n-1.
    Returns: (count_nonzero, square_coeffs_list)
    """
    n = len(coeff_list)
    if n == 0:
        return 0, []
    
    # Max degree of square is 2*(n-1)
    max_deg = 2 * n - 1
    sq = [0] * (max_deg + 1)
    
    # Optimization: only iterate over non-zero terms in the original polynomial
    non_zero_indices = [i for i, c in enumerate(coeff_list) if c != 0]
    
    for i in non_zero_indices:
        ci = coeff_list[i]
        for j in non_zero_indices:
            cj = coeff_list[j]
            sq[i + j] += ci * cj
            
    count = sum(1 for v in sq if v != 0)
    return count, sq

def construct():
    # Deadline guard: 90s limit, use 85s to be safe
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial best: the seed program's output
    # [1, 1, ..., 1] (12 ones)
    # Square has terms from x^0 to x^22.
    # Coeffs are 1, 2, 3, ..., 12, 11, ..., 1. All non-zero.
    # Count = 23.
    best_poly = [1] * 12
    best_count, _ = _square_nonzero_count(best_poly)
    
    # If we can't beat 23, we return the seed.
    # But we want to search for better.
    
    # Search Strategy: Randomized Local Search
    # State: list of (exponent, coefficient) for 12 terms.
    # To simplify mutation, we represent the polynomial as a list of length L (e.g., 41)
    # with exactly 12 non-zero entries.
    
    MAX_DEG = 40 # length 41
    NUM_TERMS = 12
    
    def random_poly():
        """Generate a random sparse polynomial with 12 non-zero terms."""
        # Choose 12 distinct exponents from [0, MAX_DEG]
        exponents = random.sample(range(MAX_DEG + 1), NUM_TERMS)
        # Assign random coefficients in [-64, 64] \ {0}
        coeffs = {}
        for e in exponents:
            c = random.randint(1, 64)
            if random.random() < 0.5:
                c = -c
            coeffs[e] = c
        # Convert to list
        poly = [0] * (MAX_DEG + 1)
        for e, c in coeffs.items():
            poly[e] = c
        return poly

    def mutate(poly):
        """
        Create a mutated version of the polynomial.
        Strategy: Pick one term, either change its coefficient, move it, or swap it.
        Must maintain exactly 12 non-zero terms.
        """
        new_poly = poly[:]
        
        # Identify non-zero indices
        nz_indices = [i for i, c in enumerate(new_poly) if c != 0]
        
        if not nz_indices:
            return new_poly
            
        # Pick a random term to modify
        idx = random.choice(nz_indices)
        old_coeff = new_poly[idx]
        
        mutation_type = random.random()
        
        if mutation_type < 0.4:
            # Change coefficient value (keep index)
            # Try small changes or random
            if random.random() < 0.5:
                # Random small change
                delta = random.randint(-5, 5)
                new_c = old_coeff + delta
            else:
                # Random new value
                new_c = random.randint(1, 64)
                if random.random() < 0.5:
                    new_c = -new_c
            
            # Ensure non-zero
            if new_c == 0:
                new_c = 1
                
            # Ensure within bounds [-64, 64]
            if new_c > 64:
                new_c = 64
            elif new_c < -64:
                new_c = -64
                
            new_poly[idx] = new_c
            
        elif mutation_type < 0.7:
            # Move the term to a new random position
            # Pick a new index that is currently 0
            zero_indices = [i for i in range(MAX_DEG + 1) if new_poly[i] == 0]
            if zero_indices:
                new_idx = random.choice(zero_indices)
                new_poly[idx] = 0
                new_poly[new_idx] = old_coeff
            else:
                # No space, just change coeff slightly
                new_poly[idx] = old_coeff + random.randint(-1, 1)
                if new_poly[idx] == 0:
                    new_poly[idx] = 1
                if new_poly[idx] > 64: new_poly[idx] = 64
                if new_poly[idx] < -64: new_poly[idx] = -64
                
        else:
            # Swap with another term's coefficient (if signs/values are useful)
            # Or just randomize the sign
            if random.random() < 0.5:
                new_poly[idx] = -old_coeff
            else:
                # Randomize value
                new_c = random.randint(1, 64)
                if random.random() < 0.5:
                    new_c = -new_c
                new_poly[idx] = new_c

        # Validate: exactly 12 non-zeros?
        # If we moved to a zero index, count stays 12.
        # If we changed value, count stays 12 (unless it became 0, handled above).
        # Let's double check to be safe.
        current_nz = sum(1 for c in new_poly if c != 0)
        if current_nz != NUM_TERMS:
            # Fallback: revert to original if invalid
            return poly
            
        return new_poly

    # Main Loop
    current_poly = random_poly()
    current_count, _ = _square_nonzero_count(current_poly)
    
    # If the random start is already better, update best
    if current_count < best_count:
        best_count = current_count
        best_poly = current_poly[:]
        
    # Local Search
    # We will run multiple restarts if time permits.
    # Each restart runs a hill climbing loop.
    
    restarts = 0
    while time.time() < deadline:
        # If we've found a very good score (e.g., < 15), we might want to keep polishing
        # But for now, standard hill climbing with random restarts.
        
        # Run a hill climbing sequence from current_poly
        # Limit steps per restart to avoid getting stuck in a local minimum for too long
        # or to allow for restarts.
        steps_in_restart = 0
        max_steps_per_restart = 5000 # Adjust based on speed
        
        while time.time() < deadline and steps_in_restart < max_steps_per_restart:
            neighbor = mutate(current_poly)
            n_count, _ = _square_nonzero_count(neighbor)
            
            if n_count < current_count:
                current_poly = neighbor
                current_count = n_count
                steps_in_restart = 0 # Reset counter on improvement
                
                if current_count < best_count:
                    best_count = current_count
                    best_poly = current_poly[:]
                    
                    # If we hit a theoretical lower bound or very low value, 
                    # we can tighten the search or stop.
                    # Minimum possible for 12 terms?
                    # If all cross terms cancel, we have 12 squares.
                    # But squares are positive. So at least 12 non-zeros?
                    # Actually, if P = sum c_i x^e_i, P^2 = sum c_i^2 x^2e_i + 2 sum_{i<j} c_i c_j x^(e_i+e_j).
                    # If all e_i are distinct and sums e_i+e_j are distinct and not equal to any 2e_k,
                    # and no cancellations happen, count is 12 + 12*11/2 = 78.
                    # We want CANCELLATIONS.
                    # So we want sums to collide and coefficients to cancel.
                    # This is like finding a set of numbers with many additive relations.
                    # Best case: P is a product of polynomials with few terms?
                    # No, P must have exactly 12 terms.
                    
            else:
                steps_in_restart += 1
                
        # Restart with a new random polynomial
        restarts += 1
        current_poly = random_poly()
        current_count, _ = _square_nonzero_count(current_poly)
        if current_count < best_count:
            best_count = current_count
            best_poly = current_poly[:]
            
    # Return the best polynomial found
    # Ensure format: list of integers, length <= 41, last coeff non-zero
    # Our poly is always length 41.
    # We need to trim trailing zeros? 
    # "nonzero final coefficient" means the last element of the returned list must be non-zero.
    # If our best_poly has trailing zeros, we should trim them to satisfy "nonzero final coefficient" 
    # OR just ensure the list ends with a non-zero.
    # The requirement says: "Return coefficient list in ascending degree, length<=41, nonzero final coefficient".
    # This implies if the polynomial is 1 + x^10, we should return [1, 0, ..., 0, 1] (length 11) 
    # or is it allowed to be longer? "length<=41".
    # Usually "coefficient list" implies the list represents the polynomial exactly, so no trailing zeros.
    
    # Trim trailing zeros
    while len(best_poly) > 1 and best_poly[-1] == 0:
        best_poly.pop()
        
    # Ensure it's not empty and last is non-zero (it is, by construction of non-zero terms)
    # But what if the highest degree term in our 12 terms was 0? 
    # Our representation keeps 12 non-zeros. So the highest non-zero index is < 41.
    # After trimming, the last element is the highest non-zero coefficient, which is non-zero.
    
    return best_poly
# EVOLVE-BLOCK-END
