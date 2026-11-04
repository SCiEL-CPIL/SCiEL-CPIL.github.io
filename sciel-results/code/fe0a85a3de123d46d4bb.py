import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook fallback (Rank 18)
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(nm)]
    V_fb = [[0] * rank_schoolbook for _ in range(mp)]
    W_fb = [[0] * rank_schoolbook for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_fb, "V": V_fb, "W": W_fb}

    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin

    best_result = fallback
    best_rank = rank_schoolbook

    # Try to find lower rank decompositions
    # We search from 17 down to 10
    for target_rank in range(17, 9, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
                
            np.random.seed(1000 + trial)
            # Initialize with small random values
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1

            # Simple iterative refinement (Gradient Descent with adaptive step)
            # We want to minimize || T - sum(U_r V_r W_r) ||^2
            lr = 0.05
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Compute residual
                T_current = np.einsum('ar,br,cr->abc', U, V, W)
                residual = T - T_current
                loss = np.sum(residual**2)
                
                if loss < 1e-6:
                    break
                    
                # Gradients
                # dL/dU = 2 * residual * (V outer W) -> einsum
                grad_U = np.einsum('abc,br,cr->ar', residual, V, W)
                grad_V = np.einsum('abc,ar,cr->br', residual, U, W)
                grad_W = np.einsum('abc,ar,br->cr', residual, U, V)
                
                # Adaptive step size to avoid divergence
                # Scale gradients by a small factor
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Optional: Normalize columns to prevent drift?
                # For tensor rank, scale invariance exists.
                # U[:, r] * alpha, V[:, r] / alpha, W[:, r] (or similar)
                # Let's keep it simple.

            # Final check
            if it >= 0: # if loop ran
                T_current = np.einsum('ar,br,cr->abc', U, V, W)
                loss = np.sum((T - T_current)**2)
                
                if loss < 1e-4:
                    # Found a valid low-rank decomposition
                    # Convert to lists
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    
                    # Check if entries are close to half-integers?
                    # The problem requires integer/half-integer entries.
                    # If the found solution is not close to half-integers, it might be invalid for the strict check.
                    # However, the prompt says "integer/half-integer entries that exactly decompose".
                    # If my float solution is 1.234, it's not valid.
                    # I should try to round to nearest 0.5 and re-verify.
                    
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    T_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                    loss_rounded = np.sum((T - T_rounded)**2)
                    
                    if loss_rounded < 1e-6:
                        U_list = [[float(x) for x in row] for row in U_rounded]
                        V_list = [[float(x) for x in row] for row in V_rounded]
                        W_list = [[float(x) for x in row] for row in W_rounded]
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = target_rank
                        found = True
                        break
                    else:
                        # If rounding failed, maybe the real solution is valid but not half-integer?
                        # The contract says "integer/half-integer entries".
                        # So I must return half-integers.
                        # If I can't find a half-integer solution, I must stick to higher rank.
                        pass

        if found:
            break

    return best_result
