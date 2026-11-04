import numpy as np
import time
import sys

def construct():
    n, m, p = 2, 3, 7
    N_U = n * m  # 6
    N_V = m * p  # 21
    N_W = p * n  # 14

    start_time = time.time()
    deadline = 145.0  # 5s safety margin from 150s limit

    # Build the target tensor
    T_target = np.zeros((N_U, N_V, N_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0

    def verify(U, V, W):
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_calc, T_target, atol=1e-9)

    # Schoolbook rank: 42
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((N_U, rank))
        V = np.zeros((N_V, rank))
        W = np.zeros((N_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Known rank-39 decomposition
    def get_rank39():
        # 2x2x2 Strassen rank 7
        def strassen_2x2x2():
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            U[0, 0] = 1; U[3, 0] = 1
            V[0, 0] = 1; V[3, 0] = 1
            W[0, 0] = 1; W[3, 0] = 1
            U[2, 1] = 1; U[3, 1] = 1
            V[0, 1] = 1
            W[2, 1] = 1; W[3, 1] = -1
            U[0, 2] = 1
            V[1, 2] = 1; V[3, 2] = -1
            W[1, 2] = 1; W[3, 2] = 1
            U[3, 3] = 1
            V[2, 3] = 1; V[0, 3] = -1
            W[0, 3] = 1; W[2, 3] = 1
            U[0, 4] = 1; U[1, 4] = 1
            V[3, 4] = 1
            W[0, 4] = -1; W[1, 4] = 1
            U[2, 5] = 1; U[0, 5] = -1
            V[0, 5] = 1; V[1, 5] = 1
            W[3, 5] = 1
            U[1, 6] = 1; U[3, 6] = -1
            V[2, 6] = 1; V[3, 6] = 1
            W[0, 6] = 1
            return U, V, W

        def rank212():
            U = np.zeros((2, 4))
            V = np.zeros((2, 4))
            W = np.zeros((4, 4))
            U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
            U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
            U[1, 2] = 1; V[0, 2] = 1; W[2, 2] = 1
            U[1, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
            return U, V, W

        def rank231():
            U = np.zeros((6, 6))
            V = np.zeros((3, 6))
            W = np.zeros((2, 6))
            col = 0
            for i in range(2):
                for j in range(3):
                    U[i*3+j, col] = 1
                    V[j, col] = 1
                    W[i, col] = 1
                    col += 1
            return U, V, W

        def rank232():
            U = np.zeros((6, 11))
            V = np.zeros((9, 11))
            W = np.zeros((4, 11))
            Us, Vs, Ws = strassen_2x2x2()
            U212, V212, W212 = rank212()
            U[0, 0:7] = Us[0, :]
            U[1, 0:7] = Us[1, :]
            U[3, 0:7] = Us[2, :]
            U[4, 0:7] = Us[3, :]
            V[0, 0:7] = Vs[0, :]
            V[1, 0:7] = Vs[1, :]
            V[3, 0:7] = Vs[2, :]
            V[4, 0:7] = Vs[3, :]
            W[0, 0:7] = Ws[0, :]
            W[1, 0:7] = Ws[1, :]
            W[2, 0:7] = Ws[2, :]
            W[3, 0:7] = Ws[3, :]
            U[2, 7:11] = U212[0, :]
            U[5, 7:11] = U212[1, :]
            V[6, 7:11] = V212[0, :]
            V[7, 7:11] = V212[1, :]
            W[0, 7:11] = W212[0, :]
            W[1, 7:11] = W212[1, :]
            W[2, 7:11] = W212[2, :]
            W[3, 7:11] = W212[3, :]
            return U, V, W

        R = 39
        U_full = np.zeros((N_U, R))
        V_full = np.zeros((N_V, R))
        W_full = np.zeros((N_W, R))
        
        # Three 2x3x2 blocks + one 2x3x1 block
        for block_idx in range(3):
            U232, V232, W232 = rank232()
            U_full[:, block_idx*11:(block_idx+1)*11] = U232
            for j in range(3):
                for k in range(2):
                    V_full[j*7 + block_idx*2 + k, block_idx*11:(block_idx+1)*11] = V232[j*2+k, :]
            for k in range(2):
                for i in range(2):
                    W_full[(block_idx*2+k)*2+i, block_idx*11:(block_idx+1)*11] = W232[k*2+i, :]
        
        U231, V231, W231 = rank231()
        U_full[:, 33:39] = U231
        for j in range(3):
            V_full[j*7+6, 33:39] = V231[j, :]
        for i in range(2):
            W_full[6*2+i, 33:39] = W231[i, :]
        
        return U_full, V_full, W_full

    # ALS-based rank minimization
    def als_decompose(target, n_factors, n_cols, n_rows, rank, max_iter=500, tol=1e-10, init=None):
        """
        target: (n_factors, n_cols, n_rows) tensor
        Returns: list of factor matrices
        """
        if init is not None:
            factors = [f.copy() for f in init]
        else:
            factors = []
            for d in range(n_factors):
                factors.append(np.random.randn(factors_shapes[d], rank) * 0.1)
        
        for it in range(max_iter):
            for d in range(n_factors):
                # Contract target with all other factors to get a matrix
                # For factor d, we need to solve: target * (product of other factors) = factor_d
                # Unfold target along dimension d
                # target is (a, b, c), factors are U(a,r), V(b,r), W(c,r)
                # For U: sum_{b,c} target[a,b,c] V[b,r'] W[c,r'] = U[a,r] for each r
                # This is a linear system: U = (V^T W) contracted with target
                
                if d == 0:
                    # Update U: U[a, r] = sum_{b,c} target[a,b,c] V[b,r'] W[c,r']
                    # = (target * (V ⊗ W)) but we need to do it properly
                    # target shape (a, b, c), V shape (b, r), W shape (c, r)
                    # We want: for each r, U[:, r] = sum_{b,c} target[:, b, c] V[b, r] W[c, r]
                    # This is: U = einsum('abc,br,cr->ar', target, V, W)
                    U_new = np.einsum('abc,br,cr->ar', target, factors[1], factors[2])
                    factors[0] = U_new
                elif d == 1:
                    V_new = np.einsum('abc,ar,cr->br', target, factors[0], factors[2])
                    factors[1] = V_new
                else:
                    W_new = np.einsum('abc,ar,br->cr', target, factors[0], factors[1])
                    factors[2] = W_new
            
            # Check convergence
            if it > 0:
                # Compute residual
                T_calc = np.einsum('ar,br,cr->abc', factors[0], factors[1], factors[2])
                residual = np.linalg.norm(T_calc - target)
                if residual < tol:
                    break
        
        return factors

    # Try to find a lower-rank decomposition using ALS
    best_result = None
    best_rank = 42  # schoolbook

    # First, verify the rank-39 decomposition
    U39, V39, W39 = get_rank39()
    if verify(U39, V39, W39):
        best_result = (U39, V39, W39)
        best_rank = 39
    
    # Try ALS with various ranks below best_rank
    # The tensor has 6*21*14 = 1764 entries, rank r means 6r + 21r + 14r = 41r parameters
    # For exact decomposition, we need the tensor to lie in the rank-r variety
    
    # Try ranks from best_rank-1 down to some minimum
    # Use the rank-39 as initialization for ALS
    
    # Try rank 38, 37, 36, ...
    for test_rank in range(best_rank - 1, 30, -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple initializations
        found = False
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            
            # Initialize with random perturbation of a rank-39-like structure
            # Or use random initialization
            U_init = np.random.randn(N_U, test_rank) * 0.5
            V_init = np.random.randn(N_V, test_rank) * 0.5
            W_init = np.random.randn(N_W, test_rank) * 0.5
            
            # Run ALS
            factors = als_decompose(T_target, 3, N_U, N_V, N_W, test_rank, 
                                     max_iter=200, tol=1e-12,
                                     init=[U_init, V_init, W_init])
            
            U_test, V_test, W_test = factors
            
            # Verify
            if verify(U_test, V_test, W_test):
                best_result = (U_test, V_test, W_test)
                best_rank = test_rank
                found = True
                break
        
        if found:
            # Try to go even lower
            continue
    
    # Return the best result
    if best_result is not None:
        U, V, W = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook
    U_s, V_s, W_s = get_schoolbook()
    return {
        "U": U_s.tolist(),
        "V": V_s.tolist(),
        "W": W_s.tolist()
    }
