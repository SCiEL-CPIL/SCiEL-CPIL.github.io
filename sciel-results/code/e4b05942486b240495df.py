import numpy as np
import time
import random

def evaluate(coeffs):
    """
    Evaluates the normalized sup norm of the polynomial P(z) = sum c_k z^k.
    coeffs: list of 24 integers in {1, -1}
    """
    # Use a sufficient number of points for accurate sup norm estimation
    # The evaluator uses dense sampling and golden-section refinement.
    # For local search, we need a fast but accurate enough proxy.
    N = 1024 # Power of 2 for FFT speed, 1024 points is usually enough for local search
    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    z = np.exp(1j * thetas)
    
    # Compute P(z) using Horner's method or direct summation
    # Direct summation is O(N * degree) which is 1024 * 24 ~ 24576 ops, very fast
    P = np.zeros(N, dtype=complex)
    zk = np.ones(N, dtype=complex)
    for k, c in enumerate(coeffs):
        P += c * zk
        zk *= z
        
    mags = np.abs(P)
    max_mag = np.max(mags)
    return max_mag / np.sqrt(24)

def local_search(coeffs):
    """
    Performs local search by flipping single bits until no improvement is found.
    """
    current_coeffs = list(coeffs)
    current_val = evaluate(current_coeffs)
    improved = True
    start_time = time.time()
    
    while improved and (time.time() - start_time) < 2: # Limit local search time per restart
        improved = False
        # Shuffle order of bits to avoid bias
        indices = list(range(24))
        random.shuffle(indices)
        for k in indices:
            new_coeffs = current_coeffs[:]
            new_coeffs[k] *= -1
            new_val = evaluate(new_coeffs)
            if new_val < current_val - 1e-12:
                current_coeffs = new_coeffs
                current_val = new_val
                improved = True
                
    return current_coeffs, current_val

def construct():
    """
    Returns the 24 coefficients of the polynomial P(z) = sum c_k z^k 
    that minimizes the sup norm on the unit circle.
    """
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin
    
    best_coeffs = None
    best_val = float('inf')
    
    # Try a few hardcoded known good polynomials if available
    # The optimal value is around 1.2583758
    # One known optimal polynomial (from Odlyzko's tables or similar sources)
    # Let's try to find one via search.
    
    # Start with random restarts
    restart_count = 0
    while time.time() < deadline:
        # Generate random coefficients
        rand_coeffs = [random.choice([1, -1]) for _ in range(24)]
        
        # Local search
        c, v = local_search(rand_coeffs)
        
        if v < best_val:
            best_val = v
            best_coeffs = c
            
            # If we reach the known optimum, we can stop early
            if best_val <= 1.2583758 + 1e-6:
                break
        
        restart_count += 1
        # To prevent infinite loop if time is very short, ensure we do at least a few
        if restart_count > 100 and time.time() > start_time + 10:
             # If we've done many restarts and still no improvement, maybe increase N?
             # But for now, just keep going.
             pass

    return best_coeffs

