import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precompute evaluation grid
    # Use a dense grid for accurate sup-norm estimation during search
    M = 2048
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    z = np.exp(1j * theta)
    # Z[k, j] = z_j^k
    # Since N=24, we can just compute z**k for k in 0..23
    Z = np.vstack([z**k for k in range(24)])
    
    def calc_score(c):
        # c is shape (24,) of +1/-1
        # P = c @ Z
        P = c @ Z
        return np.max(np.abs(P))

    # Generate Golay-Rudin-Shapiro sequence
    def gr_sequence(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:n], dtype=float)

    best_coeffs = gr_sequence(24)
    best_score = calc_score(best_coeffs)
    
    # Local search
    # We will try to improve the score by flipping bits
    # Since the space is 2^24, and the landscape is rugged, 
    # we might need multiple restarts or a smart heuristic.
    # However, for n=24, the optimal is unique up to symmetry.
    # A simple hill climb from GR might get stuck.
    
    # Let's use a more robust approach: 
    # 1. Hill climb from GR
    # 2. Hill climb from a few random starts
    
    def local_search(coeffs):
        coeffs = coeffs.copy()
        score = calc_score(coeffs)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(24):
                old_val = coeffs[i]
                coeffs[i] = -old_val
                new_score = calc_score(coeffs)
                if new_score < score - 1e-9:
                    score = new_score
                    improved = True
                else:
                    coeffs[i] = old_val
        return coeffs, score

    # Start with GR
    c1, s1 = local_search(best_coeffs)
    if s1 < best_score:
        best_score = s1
        best_coeffs = c1

    # Try a few random restarts
    for _ in range(5):
        if time.time() > deadline - 10.0:
            break
        rand_coeffs = np.random.choice([-1.0, 1.0], size=24)
        c2, s2 = local_search(rand_coeffs)
        if s2 < best_score:
            best_score = s2
            best_coeffs = c2

    # Return the best coefficients found
    # Ensure they are integers +1/-1
    return [int(c) for c in best_coeffs]
