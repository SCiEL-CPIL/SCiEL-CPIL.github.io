import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill
    
    # Fast cost function using NumPy convolution
    def count_nonzero_square(coeffs):
        # coeffs is a numpy array
        # Square has degree 2*(n-1)
        square_coeffs = np.convolve(coeffs, coeffs)
        return np.count_nonzero(square_coeffs)
    
    # We need a polynomial with exactly 12 nonzero terms.
    # The problem statement says: "Return coefficient list in ascending degree, length<=41, nonzero final coefficient, entries integers in [-64,64], exactly 12 nonzero terms."
    # This implies the INPUT polynomial has exactly 12 nonzero terms.
    # The output is the coefficient list of the INPUT polynomial.
    
    # Strategy:
    # 1. Define a support of 12 indices.
    # 2. Assign integer coefficients to these indices.
    # 3. Optimize the coefficients to minimize the number of non-zero terms in the square.
    
    # To speed up search, we can fix the support to be the first 12 degrees (0..11) initially,
    # or allow the support to shift. Shifting the support doesn't change the number of non-zero terms in the square,
    # only their positions. So we can fix the support to 0..11 for simplicity, 
    # unless we want to explore non-contiguous supports.
    # Non-contiguous supports (e.g., gaps) can help avoid overlaps in the square.
    # Example: Support {0, 2, 4, ..., 22} (even degrees only). Square support will be even degrees 0..44.
    # If we use {0, 1, 2, ..., 11}, square support is 0..22.
    
    # Let's allow the support to be a subset of 0..40.
    # However, searching over all subsets of 40 is huge.
    # Let's stick to contiguous supports first, then try structured non-contiguous ones.
    
    # Actually, the prompt says "exactly 12 nonzero terms". It does NOT say the degrees must be 0..11.
    # But "nonzero final coefficient" means the highest degree term must be non-zero.
    
    # Let's represent the polynomial by:
    # 1. A sorted list of 12 indices (degrees) where coefficients are non-zero.
    # 2. A list of 12 integer coefficients for those indices.
    
    # To make it easier, let's first search over coefficients for a fixed support 0..11.
    # Then, if we have time, we can try moving the support.
    
    # Simulated Annealing Parameters
    best_coeffs = None
    best_score = float('inf')
    
    # Initialize with a random polynomial
    # Coefficients in [-64, 64], non-zero.
    def random_poly():
        # Support 0..11
        coeffs = np.array([random.randint(1, 64) * random.choice([1, -1]) for _ in range(12)], dtype=np.int64)
        return coeffs

    current_coeffs = random_poly()
    current_score = count_nonzero_square(current_coeffs)
    
    if current_score < best_score:
        best_score = current_score
        best_coeffs = current_coeffs.copy()
    
    T = 100.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
        
        # Generate a neighbor: change one coefficient
        i = random.randint(0, 11)
        old_val = current_coeffs[i]
        
        # Propose a new value
        # Bias towards small values for cancellation? Or random?
        # Let's try a mix.
        if random.random() < 0.5:
            new_val = random.randint(1, 64) * random.choice([1, -1])
        else:
            # Small step
            new_val = old_val + random.randint(-10, 10)
            if new_val == 0:
                new_val = 1 if random.random() < 0.5 else -1
            if abs(new_val) > 64:
                new_val = 64 if new_val > 0 else -64
        
        current_coeffs[i] = new_val
        new_score = count_nonzero_square(current_coeffs)
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_coeffs = current_coeffs.copy()
        else:
            # Reject
            current_coeffs[i] = old_val
            
        T *= cooling_rate
        
        # Occasionally, restart or jump to a different structure?
        # For now, stick to 0..11 support.
        
        # Check time
        if time.time() > deadline:
            break
    
    # If we have time, try a few different supports
    # Example: Support {0, 2, 4, ..., 22}
    # This effectively scales the polynomial. P(x^2). Square is P(x^2)^2.
    # The number of non-zero terms is the same as P(x)^2.
    # So the support structure doesn't change the count, just the positions.
    # Therefore, optimizing coefficients on 0..11 is sufficient to find the minimal count.
    
    # Convert numpy array to list of ints
    result = best_coeffs.tolist()
    
    # Ensure constraints:
    # 1. Length 12 (<= 41) - Yes
    # 2. Nonzero final coeff - Yes, we ensured non-zero
    # 3. Entries in [-64, 64] - Yes
    # 4. Exactly 12 nonzero terms - Yes
    
    return result

# EVOLVE-BLOCK-END
