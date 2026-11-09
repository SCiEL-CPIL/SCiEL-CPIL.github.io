"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 105
    
    nm, mp, pn = n*m, m*p, p*n
    target = np.ones((n, m, p))
    
    def compute_tensor(U, V, W):
        T = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    T[i, j, :] += uc[i, j] * vc[j, :] * wc[:, i]
        return T
    
    def verify(U, V, W, tol=1e-9):
        T = compute_tensor(U, V, W)
        return np.allclose(T, 1.0, atol=tol)
    
    def round_mat(M, denom=20):
        return [[float(Fraction(x).limit_denominator(denom)) for x in row] for row in M]
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    def als_solve(r, seed, max_iters=500):
        """Alternating Least Squares for CP decomposition."""
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, r)
        V = rng.randn(mp, r)
        W = rng.randn(pn, r)
        
        # Precompute index mappings
        # For U: equation (i,j,k) involves U[i*m+j, :]
        # For V: equation (i,j,k) involves V[j*p+k, :]
        # For W: equation (i,j,k) involves W[k*n+i, :]
        
        for it in range(max_iters):
            # Solve for U given V, W
            # For each (i,j): sum_c U[i*m+j,c] * (sum_k V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
            # This is: for each (i,j), we need U[i*m+j,:] such that
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # Let A[i,j,k,c] = V[j*p+k,c] * W[k*n+i,c]
            # Then sum_c U[i*m+j,c] * A[i,j,k,c] = 1 for all k
            # This is a linear system: for each (i,j), solve A_ij @ U[i*m+j,:] = ones(p)
            
            for i in range(n):
                for j in range(m):
                    # A is p x r
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve A @ x = ones(p)
                    try:
                        x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except:
                        x = np.zeros(r)
                    U[i*m+j, :] = x
            
            # Solve for V given U, W
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        x = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    except:
                        x = np.zeros(r)
                    V[j*p+k, :] = x
            
            # Solve for W given U, V
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        x = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    except:
                        x = np.zeros(r)
                    W[k*n+i, :] = x
            
            # Check convergence
            T = compute_tensor(U, V, W)
            loss = np.sum((T - target)**2)
            if loss < 1e-12:
                break
        
        T = compute_tensor(U, V, W)
        loss = np.sum((T - target)**2)
        if loss < 1e-8:
            for denom in [10, 20, 50, 100]:
                Ur = np.array(round_mat(U, denom))
                Vr = np.array(round_mat(V, denom))
                Wr = np.array(round_mat(W, denom))
                if verify(Ur, Vr, Wr, tol=1e-10):
                    return Ur.tolist(), Vr.tolist(), Wr.tolist()
        return None
    
    # Try ranks from high to low
    for r in range(rank_sb - 1, max(rank_sb - 20, 1), -1):
        if time.time() > deadline:
            break
        
        found = False
        for seed in [r, r+100, r+200, r+300]:
            if time.time() > deadline:
                break
            result = als_solve(r, seed, max_iters=300)
            if result is not None:
                U_l, V_l, W_l = result
                best = {"U": U_l, "V": V_l, "W": W_l}
                best_rank = r
                found = True
                break
        if found:
            break
    
    return best
# EVOLVE-BLOCK-END
