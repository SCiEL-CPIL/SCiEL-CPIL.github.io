import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Grid for evaluation
    # The evaluator uses dense sampling and golden-section refinement.
    # A grid of 2048 points is a good compromise for speed vs accuracy in search.
    N = 2048
    thetas = 2 * np.pi * np.arange(N) / N
    z = np.exp(1j * thetas)
    
    # Precompute powers matrix: powers[k, j] = z_j^k
    # Shape: (24, N)
    powers = np.ones((24, N), dtype=complex)
    for k in range(1, 24):
        powers[k] = powers[k-1] * z
        
    def get_score(coeffs):
        """
        Compute the max norm of the polynomial on the grid.
        coeffs: array of shape (24,)
        Returns: max absolute value
        """
        # coeffs @ powers
        # coeffs is (24,), powers is (24, N) -> result is (N,)
        P_vals = np.dot(coeffs, powers)
        return np.max(np.abs(P_vals))

    # Initial candidate: Random or GRS
    # GRS (Golay-Rudin-Shapiro) is a good starting point.
    # p = [1], q = [1]
    # while len(p) < 24: p, q = p+q, p+[-c for c in q]
    # But random restarts in SA might find better. Let's start with a random one 
    # or GRS. GRS is known to be good.
    
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    current_coeffs = np.array(p[:24], dtype=np.float64)
    
    current_score = get_score(current_coeffs)
    best_coeffs = current_coeffs.copy()
    best_score = current_score
    
    # Simulated Annealing
    # Parameters
    T_start = 1.0
    T_end = 1e-4
    alpha = 0.999  # Cooling rate
    num_iterations = 100000  # Adjust based on time
    
    # We need to be careful with time.
    # Each iteration involves:
    # 1. Flip a random bit (or few bits)
    # 2. Evaluate new score
    # 3. Decide accept/reject
    
    # To speed up, we can vectorize the flip? No, flipping is O(1) on the array.
    # Evaluation is O(N) via dot product.
    # 100,000 iterations * 2048 complex ops = 200M ops. NumPy can do this fast.
    
    # Let's run SA
    T = T_start
    rng = np.random.default_rng(42)
    
    # We will track time and stop if we hit deadline
    # We can also do multiple restarts if time permits
    
    # First, run SA from GRS
    for i in range(num_iterations):
        if time.time() > deadline - 5.0:
            break
            
        # Flip a random bit
        idx = rng.integers(0, 24)
        new_coeffs = current_coeffs.copy()
        new_coeffs[idx] *= -1
        
        new_score = get_score(new_coeffs)
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_coeffs = new_coeffs
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_coeffs.copy()
                
        T *= alpha
        
    # Local Search Polish
    # Strict hill climbing with 1-bit flips
    improved = True
    while improved:
        improved = False
        for i in range(24):
            new_coeffs = best_coeffs.copy()
            new_coeffs[i] *= -1
            new_score = get_score(new_coeffs)
            if new_score < best_score:
                best_coeffs = new_coeffs
                best_score = new_score
                improved = True
                
    # 2-bit flip search to escape shallow local minima
    improved = True
    while improved:
        improved = False
        for i in range(24):
            for j in range(i+1, 24):
                new_coeffs = best_coeffs.copy()
                new_coeffs[i] *= -1
                new_coeffs[j] *= -1
                new_score = get_score(new_coeffs)
                if new_score < best_score:
                    best_coeffs = new_coeffs
                    best_score = new_score
                    improved = True
                    
    # Check if we have time for more restarts
    # The known optimum is 1.2583758.
    # best_score is the max norm.
    # Ratio = best_score / sqrt(24)
    
    target_ratio = 1.2583758
    current_ratio = best_score / np.sqrt(24)
    
    if current_ratio > target_ratio + 1e-5:
        # Try a few more random restarts with SA
        num_restarts = 5
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Random start
            current_coeffs = rng.choice([-1, 1], size=24).astype(np.float64)
            current_score = get_score(current_coeffs)
            
            # Short SA
            T = 0.5
            for i in range(20000):
                if time.time() > deadline - 5.0:
                    break
                idx = rng.integers(0, 24)
                new_coeffs = current_coeffs.copy()
                new_coeffs[idx] *= -1
                new_score = get_score(new_coeffs)
                delta = new_score - current_score
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_coeffs = new_coeffs
                    current_score = new_score
                T *= 0.9995
                
            # Polish
            improved = True
            while improved:
                improved = False
                for i in range(24):
                    new_coeffs = current_coeffs.copy()
                    new_coeffs[i] *= -1
                    new_score = get_score(new_coeffs)
                    if new_score < best_score:
                        best_coeffs = new_coeffs
                        best_score = new_score
                        improved = True
            # 2-bit polish
            improved = True
            while improved:
                improved = False
                for i in range(24):
                    for j in range(i+1, 24):
                        new_coeffs = best_coeffs.copy()
                        new_coeffs[i] *= -1
                        new_coeffs[j] *= -1
                        new_score = get_score(new_coeffs)
                        if new_score < best_score:
                            best_coeffs = new_coeffs
                            best_score = new_score
                            improved = True

    # Final check: If we are very close to the target, we might need a higher precision check
    # but the evaluator does the refinement.
    # We just return the best coefficients found.
    
    return [int(c) for c in best_coeffs]
