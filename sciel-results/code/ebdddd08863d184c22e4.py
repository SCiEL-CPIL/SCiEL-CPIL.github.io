import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Verify a decomposition
    def verify(U, V, W, r, tol=1e-10):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    # Rank-55 decomposition: j=0 separately + Strassen for j=1,2
    def rank55_decomposition():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Part A: j=0 (first column of A, first row of B)
        for i in range(n):
            for k in range(p):
                U[i*m + 0, col] = 1
                V[0*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
        
        # Part B: j=1,2 with Strassen on 5 blocks of 2x2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[u_21, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1
            W[w_11, col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1
            V[v_22, col] += -1
            W[w_01, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1
            V[v_11, col] += -1
            W[w_00, col] += 1
            W[w_10, col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[u_11, col] += 1
            U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1
            W[w_01, col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1
            U[u_11, col] += -1
            V[v_11, col] += 1
            V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1
            U[u_22, col] += -1
            V[v_21, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
        
        return U, V, W, rank
    
    U_best, V_best, W_best, rank_best = rank55_decomposition()
    err = verify(U_best, V_best, W_best, rank_best)
    
    # Peeling algorithm: iteratively find rank-1 components with sparse V
    def peel_decomposition(T_tensor, max_rank=54):
        """
        Peeling algorithm that tries to find rank-1 components with sparse V factors.
        Returns (U, V, W, rank) if successful, else None.
        """
        T_remaining = T_tensor.copy()
        U_components = []
        V_components = []
        W_components = []
        
        # For our tensor T[i,j,k]=1, try to find rank-1 components where
        # V has non-zeros at positions (j1, k1) and (j2, k2) with j1 != j2.
        # 
        # The rank-1 component contributes:
        #   U[i] * V[j1*p+k1] * W[k1*n+i] to T[i, j1, k1]
        #   U[i] * V[j2*p+k2] * W[k2*n+i] to T[i, j2, k2]
        #
        # For this to match T=1, we need:
        #   U[i] * V[j1*p+k1] * W[k1*n+i] = 1 for all i
        #   U[i] * V[j2*p+k2] * W[k2*n+i] = 1 for all i
        #
        # This implies:
        #   W[k1*n+i] / W[k2*n+i] = V[j2*p+k2] / V[j1*p+k1] = constant
        #
        # Let's parameterize: set V[j1*p+k1] = 1, V[j2*p+k2] = r
        # Then W[k1*n+i] = r * W[k2*n+i]
        # And U[i] = 1 / (W[k1*n+i]) = 1 / (r * W[k2*n+i])
        #
        # So we need to choose W such that W[k1*n+i] = r * W[k2*n+i] for all i,
        # and then U[i] = 1 / W[k1*n+i]
        #
        # The remaining tensor T - U*V*W should have entries:
        #   T[i, j1, k1] - U[i]*V[j1*p+k1]*W[k1*n+i] = 1 - 1 = 0
        #   T[i, j2, k2] - U[i]*V[j2*p+k2]*W[k2*n+i] = 1 - 1 = 0
        #   T[i, j, k] - 0 = 1 for other (j,k)
        #
        # So the remaining tensor has zeros at (j1,k1) and (j2,k2) for all i,
        # and ones elsewhere.
        #
        # This remaining tensor can be decomposed if we can cover all the 1s
        # with rank-1 components that don't touch the zero positions.
        
        # Let's try: for each pair (j1,k1), (j2,k2) with j1 != j2,
        # and for each r in a set of values, try to peel.
        
        # Actually, let's try a simpler approach: just try to reduce the rank
        # by finding a good decomposition directly.
        
        # For now, return None to use the ALS approach.
        return None
    
    # Try peeling
    peeled = peel_decomposition(T, max_rank=54)
    if peeled is not None:
        U_p, V_p, W_p, rank_p = peeled
        if rank_p < rank_best:
            U_best, V_best, W_best, rank_best = U_p, V_p, W_p, rank_p
    
    # Try to reduce rank using ALS with improved strategy
    if time.time() < deadline - 10:
        # Try ranks from 54 down to some minimum
        for r in range(rank_best - 1, max(nm, mp, pn) - 1, -1):
            if time.time() > deadline - 5:
                break
            
            found = False
            # Increase restarts for higher ranks, decrease for lower
            max_restarts = 30 if r >= 52 else 20
            
            for trial in range(max_restarts):
                if time.time() > deadline - 5:
                    break
                
                np.random.seed(trial * 1000 + r)
                
                # Multiple initialization strategies
                if trial % 3 == 0:
                    # Initialize near the rank-55 solution with small perturbation
                    if r <= rank_best:
                        U_c = U_best[:, :r].copy() + 0.05 * np.random.randn(nm, r)
                        V_c = V_best[:, :r].copy() + 0.05 * np.random.randn(mp, r)
                        W_c = W_best[:, :r].copy() + 0.05 * np.random.randn(pn, r)
                    else:
                        U_c = np.random.randn(nm, r)
                        V_c = np.random.randn(mp, r)
                        W_c = np.random.randn(pn, r)
                elif trial % 3 == 1:
                    # Random initialization
                    U_c = np.random.randn(nm, r)
                    V_c = np.random.randn(mp, r)
                    W_c = np.random.randn(pn, r)
                else:
                    # Sparse initialization: try to make V sparse
                    U_c = np.random.randn(nm, r)
                    V_c = np.random.randn(mp, r)
                    W_c = np.random.randn(pn, r)
                    # Make some V entries zero
                    for col in range(r):
                        # Keep only 2-3 non-zero entries in V[:, col]
                        num_nz = np.random.randint(2, 4)
                        keep = np.random.choice(mp, size=num_nz, replace=False)
                        mask = np.ones(mp, dtype=bool)
                        mask[keep] = False
                        V_c[:, col][mask] = 0
                
                # Alternating least squares with more iterations
                for it in range(25):
                    if time.time() > deadline - 5:
                        break
                    
                    # Solve for U
                    M = np.einsum('jr,kr->jkr', V_c, W_c).reshape(mp * pn, r)
                    try:
                        U_c = T.reshape(nm, mp * pn) @ np.linalg.pinv(M.T)
                    except:
                        break
                    
                    # Solve for V
                    N = np.einsum('ir,kr->ikr', U_c, W_c).reshape(nm * pn, r)
                    try:
                        V_c = T.reshape(mp, nm * pn) @ np.linalg.pinv(N.T)
                    except:
                        break
                    
                    # Solve for W
                    P = np.einsum('ir,jr->ijr', U_c, V_c).reshape(nm * mp, r)
                    try:
                        W_c = T.reshape(pn, nm * mp) @ np.linalg.pinv(P.T)
                    except:
                        break
                    
                    # Check convergence early
                    if it > 5:
                        T_check = np.einsum('ir,jr,kr->ijk', U_c, V_c, W_c)
                        error = np.max(np.abs(T_check - T))
                        if error < 1e-10:
                            break
                
                # Check convergence
                T_check = np.einsum('ir,jr,kr->ijk', U_c, V_c, W_c)
                error = np.max(np.abs(T_check - T))
                
                if error < 1e-8:
                    # Try to round to half-integers
                    U_r = np.round(U_c * 2) / 2
                    V_r = np.round(V_c * 2) / 2
                    W_r = np.round(W_c * 2) / 2
                    
                    T_check2 = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                    if np.max(np.abs(T_check2 - T)) < 1e-10:
                        U_best, V_best, W_best, rank_best = U_r, V_r, W_r, r
                        found = True
                        break
                    else:
                        # Use unrounded if close enough
                        if error < 1e-10:
                            U_best, V_best, W_best, rank_best = U_c, V_c, W_c, r
                            found = True
                            break
            
            if found:
                break
    
    # Final verification
    final_err = verify(U_best, V_best, W_best, rank_best)
    if final_err > 1e-8:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        U_best, V_best, W_best = U, V, W
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
