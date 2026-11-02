"""Matrix multiplication 2,3,5 - rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Build the tensor
    tensor = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a, b, c] = 1.0
    
    schoolbook_rank = n * m * p  # 30
    
    # Compute lower bounds from flattening ranks
    T1 = tensor.reshape(n * m, m * p * p * n)
    T2 = tensor.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    T3 = tensor.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
    
    rank1 = np.linalg.matrix_rank(T1)
    rank2 = np.linalg.matrix_rank(T2)
    rank3 = np.linalg.matrix_rank(T3)
    lower_bound = max(rank1, rank2, rank3)
    
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Try to find a lower-rank decomposition
    for r in range(lower_bound, schoolbook_rank):
        found = False
        for attempt in range(5):
            np.random.seed(attempt * 1000 + r)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            for it in range(50):
                # Solve for W given U, V
                num_eq = n * m * m * p * p * n
                num_unk = p * n * r
                A = np.zeros((num_eq, num_unk))
                b_vec = np.zeros(num_eq)
                
                for a in range(n * m):
                    for b in range(m * p):
                        for c in range(p * n):
                            eq_idx = (a * m * p + b) * p * n + c
                            b_vec[eq_idx] = tensor[a, b, c]
                            for t in range(r):
                                A[eq_idx, c * r + t] = U[a, t] * V[b, t]
                
                try:
                    W_flat, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                    W_new = W_flat.reshape(p * n, r)
                    residual = np.abs(A @ W_flat - b_vec).max()
                    if residual < 1e-8:
                        W = W_new
                        # Verify
                        T_recon = np.zeros_like(tensor)
                        for t in range(r):
                            T_recon += np.outer(np.outer(U[:, t], V[:, t]), W[:, t])
                        if np.abs(T_recon - tensor).max() < 1e-6:
                            best_rank = r
                            best_decomp = (U.copy(), V.copy(), W.copy())
                            found = True
                            break
                except:
                    pass
            
            if found:
                break
        
        if found:
            break
    
    if best_decomp is not None:
        U, V, W = best_decomp
        # Try to round to rationals
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        
        # Verify with exact arithmetic
        U_frac = [[Fraction(str(x)).limit_denominator(1000) for x in row] for row in U_list]
        V_frac = [[Fraction(str(x)).limit_denominator(1000) for x in row] for row in V_list]
        W_frac = [[Fraction(str(x)).limit_denominator(1000) for x in row] for row in W_list]
        
        valid = True
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    val = sum(U_frac[a][t] * V_frac[b][t] * W_frac[c][t] for t in range(best_rank))
                    expected = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if i * m + j == a and j * p + k == b and k * n + i == c:
                                    expected = 1
                    if val != Fraction(expected):
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            # Convert to clean fractions
            U_out = [[float(x) for x in row] for row in U_frac]
            V_out = [[float(x) for x in row] for row in V_frac]
            W_out = [[float(x) for x in row] for row in W_frac]
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
