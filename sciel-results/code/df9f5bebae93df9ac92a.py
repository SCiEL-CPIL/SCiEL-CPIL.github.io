"""Program for 3x3x4 matrix multiplication tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify that sum_k u_k ⊗ v_k ⊗ w_k equals the matmul tensor T."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Build tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a,b,c] = 1 if i=i', j=j', k=k' else 0
    # i = a//m, j = a%m
    # j' = b//p, k = b%p
    # k' = c//n, i' = c%n
    # T=1 iff i==i' and j==j' and k==k'
    
    # Build the decomposition tensor D
    D = np.zeros((n*m, m*p, p*n), dtype=float)
    for k in range(rank):
        u = np.array([U[i][k] for i in range(n*m)], dtype=float)
        v = np.array([V[j][k] for j in range(m*p)], dtype=float)
        w = np.array([W[ell][k] for ell in range(p*n)], dtype=float)
        D += np.einsum('a,b,c->abc', u, v, w)
    
    # Build target T
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    return np.allclose(D, T, atol=1e-9)

def _check_half_integer(U, V, W):
    """Check if all entries are integer or half-integer."""
    for mat in (U, V, W):
        for row in mat:
            for x in row:
                if not isinstance(x, (int, float, Fraction)):
                    return False
                if isinstance(x, Fraction):
                    # check denominator divides 2
                    if x.denominator not in (1, 2):
                        return False
                else:
                    xf = float(x)
                    if xf != xf:  # NaN
                        return False
                    if abs(xf * 2 - round(xf * 2)) > 1e-9:
                        return False
    return True

def _try_rank_search(deadline):
    """Try to find a decomposition with rank < 36.
    Strategy: fix U and V to small random sparse patterns, solve for W linearly.
    The matmul tensor is T = sum_k u_k ⊗ v_k ⊗ w_k.
    If U and V are fixed, for each output index c, we have:
      T[:, :, c] = sum_k (u_k ⊗ v_k) * w_k[c]
    This is a linear system in w_k[c].
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    target_rank = 35  # try one below schoolbook
    
    # Build target tensor T as a list of matrices T_c of shape (nm, mp)
    T_c = [np.zeros((nm, mp)) for c in range(pn)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_c[c][a, b] = 1.0
    
    # For a given U (nm x r) and V (mp x r), the system for W (pn x r) is:
    # For each c: sum_k W[c,k] * (U[:,k] ⊗ V[:,k]) = T_c
    # i.e., sum_k W[c,k] * vec(U[:,k] * V[:,k]) = vec(T_c)
    # This is a linear system: A * W_c = b_c where A is (nm*mp) x r
    # A[a*mp+b, k] = U[a,k] * V[b,k]
    # b_c[a*mp+b] = T_c[a,b]
    
    import random
    random.seed(42)
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Generate random sparse U and V with entries in {-1, 0, 1}
        r = target_rank
        U = np.zeros((nm, r), dtype=float)
        V = np.zeros((mp, r), dtype=float)
        
        for k in range(r):
            # Each column of U: sparse with ~2 non-zeros
            nz_u = random.randint(1, 3)
            idx_u = random.sample(range(nm), nz_u)
            for idx in idx_u:
                U[idx, k] = random.choice([-1, 1])
            
            nz_v = random.randint(1, 3)
            idx_v = random.sample(range(mp), nz_v)
            for idx in idx_v:
                V[idx, k] = random.choice([-1, 1])
        
        # Build A matrix: (nm*mp) x r
        # The r-th column of A corresponds to vec(U[:, r] V[:, r]^T)
        # This is the Kronecker product of the column vectors
        A = np.zeros((nm * mp, r), dtype=float)
        for k in range(r):
            A[:, k] = np.kron(U[:, k], V[:, k])
        
        # Solve for W: for each c, A * W[c,:] = vec(T_c)
        # Use least squares
        try:
            W = np.zeros((pn, r), dtype=float)
            for c in range(pn):
                b = T_c[c].flatten()
                sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                W[c, :] = sol
        except np.linalg.LinAlgError:
            continue
        
        # Check if W is close to half-integer
        W_rounded = np.round(W * 2) / 2.0
        if not np.allclose(W, W_rounded, atol=1e-6):
            continue
        
        # Check if U and V are close to half-integer (they should be since we used ±1,0)
        # Verify the decomposition
        U_list = [[float(U[i][k]) for k in range(r)] for i in range(nm)]
        V_list = [[float(V[i][k]) for k in range(r)] for i in range(mp)]
        W_list = [[float(W_rounded[i][k]) for k in range(r)] for i in range(pn)]
        
        if _verify(U_list, V_list, W_list):
            if _check_half_integer(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    deadline = time.time() + 120  # safety margin before 150s kill
    
    # Try to find a better decomposition
    result = _try_rank_search(deadline - 5)
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return _schoolbook()
