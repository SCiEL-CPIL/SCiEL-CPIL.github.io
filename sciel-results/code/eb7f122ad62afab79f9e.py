import numpy as np
import time

def _eval_norm(coeffs):
    """
    Computes the sup norm of the polynomial P(z) = sum c_k z^k for |z|=1.
    Uses dense sampling and golden section refinement.
    """
    n = len(coeffs)
    coeffs = np.array(coeffs)
    
    # Dense sampling
    N = 4096
    theta = 2 * np.pi * np.arange(N) / N
    z = np.exp(1j * theta)
    
    # Evaluate P(z) at all sample points
    # P[j] = sum_k coeffs[k] * z[j]**k
    # Using matrix multiplication: coeffs (n,) @ powers (n, N) -> (N,)
    # powers[k, j] = z[j]**k
    k = np.arange(n)
    # z shape (N,), k shape (n,)
    # We want z[j]**k for all j, k
    # Broadcasting: z[None, :] ** k[:, None] -> (n, N)
    powers = z[None, :] ** k[:, None]
    P = coeffs @ powers
    
    # Find max |P|
    abs_P = np.abs(P)
    max_idx = np.argmax(abs_P)
    max_val = abs_P[max_idx]
    
    # Refine around max_idx using golden section search
    # Window size
    w = 10
    start_idx = (max_idx - w) % N
    end_idx = (max_idx + w) % N
    
    # We need to handle wrap-around
    if start_idx > end_idx:
        # Split into two intervals
        # Interval 1: start_idx to N-1
        # Interval 2: 0 to end_idx
        # For simplicity, just search in a linearized window if possible
        # Or just do a fine grid search in the window
        pass
        
    # Simple fine grid search in the window
    # Map indices to theta
    # theta_i = 2 * pi * i / N
    # We want to search theta in [theta_{max-w}, theta_{max+w}]
    # Handle wrap around
    theta_start = 2 * np.pi * start_idx / N
    theta_end = 2 * np.pi * end_idx / N
    
    if theta_start > theta_end:
        # Wrap around
        # Search [theta_start, 2pi] and [0, theta_end]
        # Just do a fine grid over the whole circle if wrap is an issue? 
        # No, just shift the window
        # Actually, since N=4096, the window is small.
        # Let's just create a fine grid of 100 points in the window
        # If wrap, we have two segments.
        
        # Segment 1
        theta1 = np.linspace(theta_start, 2*np.pi, 50)
        z1 = np.exp(1j * theta1)
        P1 = coeffs @ (z1[None, :] ** k[:, None])
        max1 = np.max(np.abs(P1))
        
        # Segment 2
        theta2 = np.linspace(0, theta_end, 50)
        z2 = np.exp(1j * theta2)
        P2 = coeffs @ (z2[None, :] ** k[:, None])
        max2 = np.max(np.abs(P2))
        
        max_val = max(max1, max2)
    else:
        theta_fine = np.linspace(theta_start, theta_end, 100)
        z_fine = np.exp(1j * theta_fine)
        P_fine = coeffs @ (z_fine[None, :] ** k[:, None])
        max_val = np.max(np.abs(P_fine))
        
    return max_val

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n = 24
    # Initialize with Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:n])
    best_score = _eval_norm(best_coeffs)
    
    # Local search
    # Precompute powers for the sample points for fast evaluation
    N = 4096
    theta = 2 * np.pi * np.arange(N) / N
    z = np.exp(1j * theta)
    k = np.arange(n)
    powers = z[None, :] ** k[:, None]
    
    current_coeffs = best_coeffs.copy()
    current_score = best_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1-flip search
        best_move = None
        best_move_score = current_score
        for i in range(n):
            new_coeffs = current_coeffs.copy()
            new_coeffs[i] *= -1
            # Fast evaluation
            # P_new = P_old - 2*c_i*z^i
            # We can compute P_old once
            P_old = current_coeffs @ powers
            # P_new = P_old - 2*current_coeffs[i]*z**i
            # z**i is powers[i, :]
            P_new = P_old - 2 * current_coeffs[i] * powers[i, :]
            score = np.max(np.abs(P_new))
            if score < best_move_score - 1e-12:
                best_move_score = score
                best_move = i
                
        if best_move is not None:
            current_coeffs[best_move] *= -1
            current_score = best_move_score
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_coeffs.copy()
            improved = True
            continue
        
        # 2-flip search
        best_move = None
        best_move_score = current_score
        P_old = current_coeffs @ powers
        for i in range(n):
            for j in range(i+1, n):
                # Flip i and j
                P_new = P_old - 2 * current_coeffs[i] * powers[i, :] - 2 * current_coeffs[j] * powers[j, :]
                score = np.max(np.abs(P_new))
                if score < best_move_score - 1e-12:
                    best_move_score = score
                    best_move = (i, j)
                    
        if best_move is not None:
            i, j = best_move
            current_coeffs[i] *= -1
            current_coeffs[j] *= -1
            current_score = best_move_score
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_coeffs.copy()
            improved = True
            continue
            
        # If stuck, try a random perturbation (e.g., flip 3 random bits)
        # and continue local search
        if time.time() > deadline - 10:
            break
            
        # Random perturbation
        for _ in range(10):
            idxs = np.random.choice(n, 3, replace=False)
            perturbed = current_coeffs.copy()
            perturbed[idxs] *= -1
            score = _eval_norm(perturbed)
            if score < current_score:
                current_coeffs = perturbed
                current_score = score
                if current_score < best_score:
                    best_score = current_score
                    best_coeffs = current_coeffs.copy()
                improved = True
                break
        # If no perturbation helped, we might be in a local minimum.
        # Try a larger perturbation or restart?
        # For now, just break if we can't improve.
        if not improved:
            # Try a few more random restarts
            for _ in range(5):
                random_coeffs = np.random.choice([-1, 1], n)
                score = _eval_norm(random_coeffs)
                if score < best_score:
                    best_score = score
                    best_coeffs = random_coeffs.copy()
                    current_coeffs = random_coeffs.copy()
                    current_score = score
                    improved = True
                    break
            if not improved:
                break

    return best_coeffs.tolist()
