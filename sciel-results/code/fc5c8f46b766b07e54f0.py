"""Rewritten program for matrix multiplication tensor decomposition (3,3,5)."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    """Construct a rank-minimal tensor decomposition for (3,3,5) matrix multiplication.
    
    Returns dict with keys 'U', 'V', 'W' where:
    - U has shape (n*m, rank) = (9, rank)
    - V has shape (m*p, rank) = (15, rank)  
    - W has shape (p*n, rank) = (15, rank)
    
    The decomposition satisfies: T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    where T[i,j,k] = 1 iff i = a*m+b, j = b*p+c, k = c*n+a for some a,b,c.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    I, J, K = n * m, m * p, p * n  # 9, 15, 15
    
    def build_tensor():
        """Build the (n,m,p) matrix multiplication tensor."""
        T = np.zeros((I, J, K))
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    i = a * m + b
                    j = b * p + c
                    k = c * n + a
                    T[i, j, k] = 1
        return T
    
    T = build_tensor()
    
    def verify(U, V, W, atol=1e-10):
        """Verify that U, V, W form a valid decomposition."""
        rank = U.shape[1]
        recon = np.zeros((I, J, K))
        for c in range(rank):
            recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.allclose(recon, T, atol=atol)
    
    def to_rational_matrix(M, tol=1e-8):
        """Convert a float matrix to rational (Fraction) matrix, preferring small denominators."""
        rows, cols = M.shape
        R = [[Fraction(0) for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                v = M[i, j]
                if abs(v) < tol:
                    R[i][j] = Fraction(0)
                else:
                    # Try denominators 1, 2, 3, 4, 5, 10, 20, 50, 100
                    found = False
                    for den in [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000]:
                        num = round(v * den)
                        if abs(v - num / den) < tol:
                            R[i][j] = Fraction(num, den)
                            found = True
                            break
                    if not found:
                        R[i][j] = Fraction(v).limit_denominator(10000)
        return R
    
    def schoolbook():
        """Schoolbook decomposition (rank = n*m*p = 45)."""
        rank = n * m * p
        U = [[0] * rank for _ in range(I)]
        V = [[0] * rank for _ in range(J)]
        W = [[0] * rank for _ in range(K)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def als_decompose(r, max_iter=200, n_restarts=10, seed=42, init_type='random'):
        """Optimized Alternating Least Squares for tensor rank-r decomposition.
        
        Args:
            r: target rank
            max_iter: max iterations per restart
            n_restarts: number of random restarts
            seed: random seed
            init_type: 'random', 'schoolbook_subset', 'sparse'
            
        Returns:
            Tuple (U, V, W) if success, else None
        """
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        # Precompute non-zero structure for faster updates
        i_to_jk = {i: [] for i in range(I)}
        j_to_ik = {j: [] for j in range(J)}
        k_to_ij = {k: [] for k in range(K)}
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    i = a * m + b
                    j = b * p + c
                    k = c * n + a
                    i_to_jk[i].append((j, k))
                    j_to_ik[j].append((i, k))
                    k_to_ij[k].append((i, j))
                    
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            rng = np.random.RandomState(seed + restart)
            
            if init_type == 'random':
                U = rng.randn(I, r)
                V = rng.randn(J, r)
                W = rng.randn(K, r)
            elif init_type == 'schoolbook_subset':
                # Initialize with first r columns of schoolbook
                U = np.zeros((I, r))
                V = np.zeros((J, r))
                W = np.zeros((K, r))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < r:
                                U[i * m + j][col] = 1
                                V[j * p + k][col] = 1
                                W[k * n + i][col] = 1
                                col += 1
                # Add small noise to break symmetry
                U += rng.randn(I, r) * 0.01
                V += rng.randn(J, r) * 0.01
                W += rng.randn(K, r) * 0.01
            elif init_type == 'sparse':
                U = np.zeros((I, r))
                V = np.zeros((J, r))
                W = np.zeros((K, r))
                for c in range(r):
                    for _ in range(3):
                        U[rng.randint(I), c] = rng.choice([-1, 1])
                        V[rng.randint(J), c] = rng.choice([-1, 1])
                        W[rng.randint(K), c] = rng.choice([-1, 1])
            else:
                U = rng.randn(I, r)
                V = rng.randn(J, r)
                W = rng.randn(K, r)
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return best_U, best_V, best_W
                
                # Update U
                for i in range(I):
                    num = np.zeros(r)
                    den = np.zeros(r)
                    for j, k in i_to_jk[i]:
                        vw = V[j, :] * W[k, :]
                        num += vw
                        den += vw * vw
                    mask = den > 1e-12
                    U[i, mask] = num[mask] / den[mask]
                    U[i, ~mask] = 0
                
                # Update V
                for j in range(J):
                    num = np.zeros(r)
                    den = np.zeros(r)
                    for i, k in j_to_ik[j]:
                        uw = U[i, :] * W[k, :]
                        num += uw
                        den += uw * uw
                    mask = den > 1e-12
                    V[j, mask] = num[mask] / den[mask]
                    V[j, ~mask] = 0
                
                # Update W
                for k in range(K):
                    num = np.zeros(r)
                    den = np.zeros(r)
                    for i, j in k_to_ij[k]:
                        uv = U[i, :] * V[j, :]
                        num += uv
                        den += uv * uv
                    mask = den > 1e-12
                    W[k, mask] = num[mask] / den[mask]
                    W[k, ~mask] = 0
                
                # Check convergence
                recon = np.zeros((I, J, K))
                for c in range(r):
                    recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                err = np.linalg.norm(recon - T)
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-10:
                    return U, V, W
        
        return best_U, best_V, best_W
    
    # Main search logic
    # Try ranks from 44 down to 35.
    # We prioritize lower ranks (closer to 45) as they are more likely to exist and be found.
    for r in range(44, 34, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Try different initialization types
        init_types = ['schoolbook_subset', 'random', 'sparse']
        
        for init_type in init_types:
            if time.time() > deadline - 5:
                break
                
            # Increase restarts for ranks close to 45
            n_restarts = 10 if r >= 42 else 5
            max_iter = 100 if r >= 42 else 50
            
            result = als_decompose(r, max_iter=max_iter, n_restarts=n_restarts, 
                                   seed=r, init_type=init_type)
            if result is not None:
                U_res, V_res, W_res = result
                recon = np.zeros((I, J, K))
                for c in range(r):
                    recon += np.einsum('i,j,k->ijk', U_res[:, c], V_res[:, c], W_res[:, c])
                err = np.linalg.norm(recon - T)
                if err < 1e-8:
                    # Attempt rationalization
                    U_r = to_rational_matrix(U_res)
                    V_r = to_rational_matrix(V_res)
                    W_r = to_rational_matrix(W_res)
                    
                    if verify(U_r, V_r, W_r):
                        return {"U": U_r, "V": V_r, "W": W_r}
                    
                    # Fallback: check if close to half-integers
                    is_rational = True
                    for mat in [U_res, V_res, W_res]:
                        for row in mat:
                            for val in row:
                                rounded = round(val * 2) / 2
                                if abs(val - rounded) > 1e-6:
                                    is_rational = False
                                    break
                            if not is_rational:
                                break
                        if not is_rational:
                            break
                    
                    if is_rational:
                        U_exact = [[round(val * 2) / 2 for val in row] for row in U_res]
                        V_exact = [[round(val * 2) / 2 for val in row] for row in V_res]
                        W_exact = [[round(val * 2) / 2 for val in row] for row in W_res]
                        if verify(U_exact, V_exact, W_exact):
                            return {"U": U_exact, "V": V_exact, "W": W_exact}
                    found = True
                    break
        
        if found:
            continue
    
    # Fallback: schoolbook rank 45
    return schoolbook()

# EVOLVE-BLOCK-END
