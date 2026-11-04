import numpy as np
import time
import sys

def construct():
    n, m, p = 3, 4, 6
    deadline = time.time() + 110
    
    # Tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def verify(U, V, W, atol=1e-9):
        # T_computed[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_comp, T, atol=atol)

    def round_half(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r, V_r, W_r

    def round_int(U, V, W):
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        return U_r, V_r, W_r

    def try_round_verify(U, V, W):
        # Try integer
        Ui, Vi, Wi = round_int(U, V, W)
        if verify(Ui, Vi, Wi):
            return Ui, Vi, Wi
        # Try half-integer
        Uh, Vh, Wh = round_half(U, V, W)
        if verify(Uh, Vh, Wh):
            return Uh, Vh, Wh
        return None

    # Schoolbook decomposition
    rank_sch = n * m * p
    U_sch = np.zeros((n*m, rank_sch))
    V_sch = np.zeros((m*p, rank_sch))
    W_sch = np.zeros((p*n, rank_sch))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*m*p + j*p + k
                U_sch[i*m+j, r] = 1.0
                V_sch[j*p+k, r] = 1.0
                W_sch[k*n+i, r] = 1.0

    best_U, best_V, best_W = U_sch, V_sch, W_sch
    best_rank = rank_sch

    # Matricizations
    T_A = T.reshape(n*m, m*p*p*n)
    T_B = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
    T_C = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)

    def als_decompose(U, V, W, max_iter=50, tol=1e-10):
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            # Update U
            MW = np.einsum('br,cr->bcr', V, W).reshape(m*p*p*n, W.shape[1])
            G = MW.T @ MW
            try:
                G_inv = np.linalg.inv(G + tol * np.eye(W.shape[1]))
                U_new = T_A @ MW @ G_inv
            except:
                break
            # Update V
            UW = np.einsum('ar,cr->acr', U_new, W).reshape(n*m*p*n, W.shape[1])
            G = UW.T @ UW
            try:
                G_inv = np.linalg.inv(G + tol * np.eye(W.shape[1]))
                V_new = T_B @ UW @ G_inv
            except:
                break
            # Update W
            UV = np.einsum('ar,br->abr', U_new, V_new).reshape(n*m*m*p, W.shape[1])
            G = UV.T @ UV
            try:
                G_inv = np.linalg.inv(G + tol * np.eye(W.shape[1]))
                W_new = T_C @ UV @ G_inv
            except:
                break
            
            U, V, W = U_new, V_new, W_new
            if np.linalg.norm(U_new - U) < 1e-12:
                break
        return U, V, W

    # Search for lower ranks
    # Start from 72 down to 60 (likely range for improvement)
    # Use block-structured initialization
    for target_rank in range(rank_sch - 1, 55, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Multiple restarts
        for restart in range(8):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(123 + restart)
            
            # Initialization: Mix schoolbook columns
            # Select target_rank columns from schoolbook
            # But to allow rank reduction, we need to mix them.
            # Create a random orthogonal matrix Q of size (rank_sch, target_rank)? No, that's projection.
            # Better: Take random linear combinations of the schoolbook basis vectors.
            
            # Let's try: U_init = U_sch @ R where R is (rank_sch, target_rank) random
            # This forces the new solution to be in the span of the schoolbook basis,
            # which is guaranteed to contain the solution, but might not find the minimal rank
            # if the minimal rank solution requires directions outside the span?
            # Actually, the minimal rank solution IS in the span of the tensor's support.
            # The schoolbook basis spans the space of all valid decompositions? No.
            # But any valid decomposition can be represented.
            
            # Let's use a different init: 
            # Split m=4 into two blocks of 2.
            # Block 1: i in 0..2, j in 0..1, k in 0..5 -> 3*2*6 = 36 terms
            # Block 2: i in 0..2, j in 2..3, k in 0..5 -> 3*2*6 = 36 terms
            # If target_rank < 72, we need sharing.
            
            # Random Gaussian is often better for finding global minima in non-convex problems
            # than structured ones if the structure is too restrictive.
            # Let's use Gaussian but scaled.
            
            U_init = np.random.randn(n*m, target_rank)
            V_init = np.random.randn(m*p, target_rank)
            W_init = np.random.randn(p*n, target_rank)
            
            # Scale to match schoolbook magnitude
            U_init /= np.linalg.norm(U_init, axis=0)
            V_init /= np.linalg.norm(V_init, axis=0)
            W_init /= np.linalg.norm(W_init, axis=0)
            
            U_sol, V_sol, W_sol = als_decompose(U_init, V_init, W_init, max_iter=80)
            
            # Verify continuous
            if verify(U_sol, V_sol, W_sol, atol=1e-8):
                rounded = try_round_verify(U_sol, V_sol, W_sol)
                if rounded:
                    best_U, best_V, best_W = rounded
                    best_rank = target_rank
                    found = True
                    break
                
                # If continuous works but rounding fails, try to polish
                # and round again
                U_pol, V_pol, W_pol = als_decompose(U_sol, V_sol, W_sol, max_iter=100, tol=1e-12)
                rounded = try_round_verify(U_pol, V_pol, W_pol)
                if rounded:
                    best_U, best_V, best_W = rounded
                    best_rank = target_rank
                    found = True
                    break

        if found:
            break

    # If no improvement, return schoolbook
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
