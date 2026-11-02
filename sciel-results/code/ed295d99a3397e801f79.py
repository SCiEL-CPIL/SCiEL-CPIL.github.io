import numpy as np
import time
import sys

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # Dimensions of factors
    dim_U = n * m      # 4
    dim_V = m * p      # 26
    dim_W = p * n      # 26
    
    # Construct the target tensor T of shape (dim_U, dim_V, dim_W)
    # T[a, b, c] = 1 if a,b,c correspond to A[i,j], B[j,k], C[i,k]
    # Convention:
    # A[i,j] -> a = i*m + j
    # B[j,k] -> b = j*p + k
    # C[i,k] -> c = k*n + i  (as per W convention in prompt)
    
    T = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # We want to find rank R such that T = sum_r U[:,r] outer V[:,r] outer W[:,r]
    # We will try to find the minimum R.
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_rank = 46  # Fallback to current known solution
    best_U = None
    best_V = None
    best_W = None
    
    # Helper to check if a decomposition is valid and returns integer/half-integer
    def check_solution(U, V, W, T, tol=1e-9):
        # Reconstruct T from U, V, W
        # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # This is a tensor contraction.
        # Efficiently:
        # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check error
        error = np.linalg.norm(T_recon - T)
        if error > tol:
            return False
        
        # Check if entries are integer or half-integer
        # Multiply by 2 and check if close to integer
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        
        if not np.allclose(U2, np.round(U2), atol=1e-7): return False
        if not np.allclose(V2, np.round(V2), atol=1e-7): return False
        if not np.allclose(W2, np.round(W2), atol=1e-7): return False
        
        # Snap to half-integers
        U_final = np.round(U2) / 2.0
        V_final = np.round(V2) / 2.0
        W_final = np.round(W2) / 2.0
        
        # Final exact check with snapped values
        T_final = np.einsum('ar,br,cr->abc', U_final, V_final, W_final)
        if not np.allclose(T_final, T, atol=1e-9):
            return False
            
        return True

    def try_rank(R, max_iters=5000, lr=0.01):
        # Initialize
        np.random.seed(42)
        U = np.random.randn(dim_U, R) * 0.1
        V = np.random.randn(dim_V, R) * 0.1
        W = np.random.randn(dim_W, R) * 0.1
        
        # Gradient Descent
        # Loss = || T_recon - T ||^2
        # dL/dU = 2 * (T_recon - T) contracted with V and W
        # dU[a,r] = 2 * sum_{b,c} (T_recon[a,b,c] - T[a,b,c]) V[b,r] W[c,r]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return False, U, V, W
            
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_recon - T
            
            # Gradients
            # dU[a,r] = 2 * sum_{b,c} diff[a,b,c] V[b,r] W[c,r]
            dU = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
            dV = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
            dW = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Early stopping if loss is very small
            loss = np.linalg.norm(diff)
            if loss < 1e-5:
                break
        
        # Check solution
        if check_solution(U, V, W, T):
            return True, U, V, W
        return False, U, V, W

    # Search for minimal rank
    # Theoretical lower bound is higher than simple counting.
    # Let's try from 20 up to 45.
    # If we find a solution, we return it.
    
    # To save time, we can try a few specific ranks first.
    # But linear search is safer.
    
    for R in range(20, 46):
        if time.time() > deadline:
            break
            
        success, U, V, W = try_rank(R)
        if success:
            # Convert to list of lists for output
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}

    # If no better rank found, return the hardcoded rank 46 solution
    # Re-implement the hardcoded solution from the prompt to ensure correctness
    rank = 46
    nU = n * m   # 4
    nV = m * p   # 26
    nW = p * n   # 26

    U = [[0] * rank for _ in range(nU)]
    V = [[0] * rank for _ in range(nV)]
    W = [[0] * rank for _ in range(nW)]

    col = 0

    def strassen_terms(k0, k1):
        terms = []
        terms.append(({0: 1, 3: 1}, {k0: 1, 13 + k1: 1}, {2 * k0: 1, 2 * k1 + 1: 1}))
        terms.append(({2: 1, 3: 1}, {k0: 1}, {2 * k0 + 1: 1, 2 * k1 + 1: -1}))
        terms.append(({0: 1}, {k1: 1, 13 + k1: -1}, {2 * k1: 1, 2 * k1 + 1: 1}))
        terms.append(({3: 1}, {13 + k0: 1, k0: -1}, {2 * k0: 1, 2 * k0 + 1: 1}))
        terms.append(({0: 1, 1: 1}, {13 + k1: 1}, {2 * k0: -1, 2 * k1: 1}))
        terms.append(({2: 1, 0: -1}, {k0: 1, k1: 1}, {2 * k1 + 1: 1}))
        terms.append(({1: 1, 3: -1}, {13 + k0: 1, 13 + k1: 1}, {2 * k0: 1}))
        return terms

    for blk in range(6):
        k0 = 2 * blk
        k1 = 2 * blk + 1
        for u_t, v_t, w_t in strassen_terms(k0, k1):
            for idx, val in u_t.items():
                U[idx][col] += val
            for idx, val in v_t.items():
                V[idx][col] += val
            for idx, val in w_t.items():
                W[idx][col] += val
            col += 1

    k = 12
    U[0][col] = 1
    V[k][col] = 1
    W[2 * k][col] = 1
    col += 1

    U[1][col] = 1
    V[13 + k][col] = 1
    W[2 * k][col] = 1
    col += 1

    U[2][col] = 1
    V[k][col] = 1
    W[2 * k + 1][col] = 1
    col += 1

    U[3][col] = 1
    V[13 + k][col] = 1
    W[2 * k + 1][col] = 1
    col += 1

    return {"U": U, "V": V, "W": W}
