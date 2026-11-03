"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    Each term is a tuple (U_coeffs, V_coeffs, W_coeffs) where:
      U_coeffs: dict {(i, j): coeff} for the 2x2 left matrix
      V_coeffs: dict {(j, k): coeff} for the 2x2 right matrix
      W_coeffs: dict {(i, k): coeff} for the 2x2 output
    """
    terms = []
    
    # M1 = (P00+P11)*(Q00+Q11)
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    
    # M2 = (P10+P11)*Q00
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    
    # M3 = P00*(Q01-Q11)
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    
    # M4 = P11*(Q10-Q00)
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    
    # M5 = (P00+P01)*Q11
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    
    # M6 = (P10-P00)*(Q00+Q01)
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    
    # M7 = (P01-P11)*(Q10+Q11)
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    
    return terms

def _strassen_block_decomposition():
    """
    Construct a rank-42 decomposition using Strassen's algorithm.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    strassen_terms = _strassen_2x2x2_terms()
    
    col = 0
    for a_block in range(2):  # 0: A cols 0,1; 1: A cols 2,3
        for b_col_block in range(3):  # B col blocks: 0,1,2 (each width 2)
            for t in range(7):  # 7 Strassen terms
                U_coeffs, V_coeffs, W_coeffs = strassen_terms[t]
                
                # Update U
                for (i_loc, j_loc), coeff in U_coeffs.items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V
                for (j_loc, k_loc), coeff in V_coeffs.items():
                    j = a_block * 2 + j_loc
                    k = b_col_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W
                for (i_loc, k_loc), coeff in W_coeffs.items():
                    i = i_loc
                    k = b_col_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _naive_decomposition():
    """
    Construct a rank-48 naive decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}

def _verify_tensor_support(result, n=2, m=4, p=6):
    """
    Verify that the reconstructed tensor has non-zero entries ONLY where the target tensor is non-zero.
    Target tensor T[u,v,w] is 1 if u=(i,j), v=(j',k), w=(k',i') with j=j', k=k', i=i'.
    """
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    
    # Check all non-zero entries of the reconstructed tensor
    # Reconstructed T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
    # We don't want to iterate over all u,v,w (8*24*12 = 2304) and all r (42).
    # Instead, we can check if any combination produces a non-zero where it shouldn't.
    
    # Target support:
    # u = i*m + j
    # v = j'*p + k
    # w = k'*n + i'
    # Valid if j==j' and k==k' and i==i'
    
    # If T[u,v,w] != 0, then there must exist r such that U[u,r]!=0, V[v,r]!=0, W[w,r]!=0.
    # And for that r, the indices must match the target support condition?
    # No, the sum might cancel out.
    # But if we use integer coefficients, cancellation is possible.
    # However, the Strassen decomposition is exact.
    
    # A safer check:
    # For each r, the support of U_r, V_r, W_r must be "compatible".
    # Actually, the simplest check is to reconstruct the tensor and compare with the target.
    
    T_rec = np.zeros((n*m, m*p, p*n), dtype=int)
    for r in range(rank):
        u_col = np.array(U[:, r])
        v_col = np.array(V[:, r])
        w_col = np.array(W[:, r])
        # T_rec += outer(outer(u_col, v_col), w_col)
        # This is slow in pure python loops.
        # But 42 * (8*24*12) operations is small.
        
        for u in range(n*m):
            if u_col[u] != 0:
                for v in range(m*p):
                    if v_col[v] != 0:
                        for w in range(p*n):
                            if w_col[w] != 0:
                                T_rec[u, v, w] += u_col[u] * v_col[v] * w_col[w]
    
    # Check support
    for u in range(n*m):
        i = u // m
        j = u % m
        for v in range(m*p):
            jp = v // p
            k = v % p
            if j != jp:
                # If j != jp, T_rec[u,v,w] must be 0 for all w
                if np.any(T_rec[u, v, :] != 0):
                    return False
            else:
                for w in range(p*n):
                    kp = w // n
                    ip = w % n
                    if k != kp or i != ip:
                        if T_rec[u, v, w] != 0:
                            return False
                    else:
                        # Should be 1
                        if T_rec[u, v, w] != 1:
                            return False
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition.
    """
    # Try Strassen (Rank 42)
    try:
        strassen = _strassen_block_decomposition()
        if _verify_tensor_support(strassen):
            return strassen
    except Exception:
        pass
        
    # Fallback to Naive (Rank 48)
    naive = _naive_decomposition()
    if _verify_tensor_support(naive):
        return naive
        
    # If even naive fails (should not), return naive anyway
    return naive

# EVOLVE-BLOCK-END
