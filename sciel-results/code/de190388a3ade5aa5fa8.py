import numpy as np
import time

def construct():
    start_time = time.time()
    limit = 150.0
    safety_margin = 5.0
    
    N = 24
    K = 4096  # Sampling points
    
    # Precompute exponentials E[k, j] = exp(1j * k * theta_j)
    thetas = np.linspace(0, 2 * np.pi, K, endpoint=False)
    # E has shape (N, K)
    # E[k, j] = exp(1j * k * theta_j)
    # We can compute this as outer product of k and theta
    k_indices = np.arange(N)
    E = np.exp(1j * np.outer(k_indices, thetas))
    
    # Function to compute sup norm from coefficient vector c
    def get_sup_norm(c):
        # c is shape (N,)
        # P_j = sum c_k E[k, j]
        P = np.dot(c, E)
        # Find max
        m = np.max(np.abs(P))
        return m, P
        
    def refine_sup_norm(c, P_approx, m_approx):
        # Refine the peak location
        # Find index of max
        j_max = np.argmax(np.abs(P_approx))
        # Window around j_max
        # Use golden section or minimize_scalar
        # For speed, we can just return the sampled max if we are just comparing
        # But for the final result, we need high accuracy.
        # The task says "accuracy about 1e-9".
        # However, for search, we just need to compare relative values.
        # We will do a high-precision refinement only at the end.
        pass

    # Initial candidate: Golay-Rudin-Shapiro
    # p, q = [1], [1]
    # while len(p) < N:
    #     p, q = p + q, p + [-c for c in q]
    # c_init = np.array(p[:N]) * 2 - 1  # Convert 0/1 to -1/1? 
    # GR sequence is usually defined with +1/-1.
    # Let's generate it properly.
    p, q = np.array([1]), np.array([1])
    while len(p) < N:
        p = np.concatenate([p, q])
        q = np.concatenate([p, -q]) # Wait, standard recursion:
        # P -> (P, Q), Q -> (P, -Q)
        # Let's re-verify.
        # Golay pair (A, B). Next pair (A+B, A-B).
        # If we start with (1, 1).
        # (1,1) -> (2, 0) ? No.
        # Rudin-Shapiro: 
        # p_0 = 1
        # p_{n+1} = p_n, (-1)^{n+1} p_n ... 
        # Let's use the standard construction for length 2^m.
        # m=1: [1, 1]
        # m=2: [1, 1, 1, -1]
        # m=3: [1, 1, 1, -1, 1, 1, -1, -1]
        # m=4: length 16.
        # We need length 24. We can take the first 24 of length 32.
        
    p = np.array([1])
    q = np.array([1])
    while len(p) < 32:
        p_new = np.concatenate([p, q])
        q_new = np.concatenate([p, -q])
        p, q = p_new, q_new
        
    c_init = p[:N]
    c_init = c_init.astype(np.float64)
    
    best_c = c_init.copy()
    best_m, best_P = get_sup_norm(best_c)
    best_score = best_m / np.sqrt(N)
    
    # Local search function
    def local_search(c):
        m, P = get_sup_norm(c)
        c = c.copy()
        # Iterate
        for _ in range(100): # Max iterations
            # Check all flips
            # P_trial[k, j] = P[j] - 2 * c[k] * E[k, j]
            # c[k] is +1 or -1.
            # If c[k] is 1, new is -1, diff is -2. P_new = P - 2 E[k]
            # If c[k] is -1, new is 1, diff is +2. P_new = P + 2 E[k]
            # So P_new = P - 2 * c[k] * E[k]
            
            # Vectorized flip check
            # c is shape (N,). E is (N, K).
            # We want to compute M_k = max_j |P - 2 c_k E[k, j]| for all k.
            
            # Create P_trials of shape (N, K)
            # P_trials[k, :] = P - 2 * c[k] * E[k, :]
            # Broadcasting: c[:, None] * E
            P_trials = P[None, :] - 2 * c[:, None] * E
            M_trials = np.max(np.abs(P_trials), axis=1)
            
            k_best = np.argmin(M_trials)
            M_best_trial = M_trials[k_best]
            
            if M_best_trial < m - 1e-10:
                # Update
                c[k_best] = -c[k_best]
                m = M_best_trial
                P = P_trials[k_best]
            else:
                break
        return c, m, P

    # Run local search from GR
    c_curr, m_curr, P_curr = local_search(c_init)
    if m_curr < best_m:
        best_m = m_curr
        best_c = c_curr.copy()
        best_P = P_curr.copy()
        best_score = best_m / np.sqrt(N)
        
    # Random restarts
    np.random.seed(42)
    while time.time() < limit - safety_margin:
        # Generate random c
        c_rand = np.random.choice([-1, 1], size=N)
        c_rand, m_rand, P_rand = local_search(c_rand)
        
        if m_rand < best_m - 1e-10:
            best_m = m_rand
            best_c = c_rand.copy()
            best_P = P_rand.copy()
            best_score = best_m / np.sqrt(N)
            
    # Final high-precision refinement
    # We need to report the score with high accuracy.
    # The task says "score is ... computed by dense sampling and golden-section refinement".
    # We should mimic this for the final returned value? 
    # No, the evaluator computes the score. We just return the coefficients.
    # But we want to be sure we have the best coefficients.
    # The local search uses K=4096 sampling. This is quite dense.
    # The error in max estimation is small.
    
    return best_c.astype(int).tolist()
