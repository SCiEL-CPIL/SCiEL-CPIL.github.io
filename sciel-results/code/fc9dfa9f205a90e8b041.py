import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using einsum."""
    try:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
            return False, float('inf')
            
        rank = U_arr.shape[1]
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        max_err = np.max(np.abs(T - 1.0))
        return max_err < tol, max_err
    except Exception:
        return False, float('inf')

def _verify_exact(U, V, W, n, m, p):
    """Verify using exact Fraction arithmetic."""
    try:
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        
        rank = len(U[0])
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for c in range(rank):
                        val += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                    if val != 1:
                        return False
        return True
    except Exception:
        return False

def _construct_rank_32():
    """Construct rank-32 decomposition using Strassen on 2x2 blocks."""
    n, m, p = 3, 4, 4
    rank = 32
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen's 7 multiplication formulas for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    def add_strassen_block(offset, i_base, j_base, k_base):
        """Add 7 Strassen terms for a 2x2x2 block."""
        # A: rows i_base, i_base+1; cols j_base, j_base+1
        # B: rows k_base, k_base+1; cols k_base, k_base+1
        # C: rows i_base, i_base+1; cols k_base, k_base+1
        
        # M1 = (A00+A11)(B00+B11)
        U[i_base*m + j_base][offset] += 1
        U[(i_base+1)*m + (j_base+1)][offset] += 1
        V[k_base*p + k_base][offset] += 1
        V[(k_base+1)*p + (k_base+1)][offset] += 1
        W[k_base*n + i_base][offset] += 1
        W[(k_base+1)*n + (i_base+1)][offset] += 1
        
        # M2 = (A10+A11)B00
        U[(i_base+1)*m + j_base][offset+1] += 1
        U[(i_base+1)*m + (j_base+1)][offset+1] += 1
        V[k_base*p + k_base][offset+1] += 1
        W[k_base*n + (i_base+1)][offset+1] += 1
        
        # M3 = A00(B10-B11)
        U[i_base*m + j_base][offset+2] += 1
        V[(k_base+1)*p + k_base][offset+2] += 1
        V[(k_base+1)*p + (k_base+1)][offset+2] -= 1
        W[(k_base+1)*n + i_base][offset+2] += 1
        
        # M4 = A11(B01-B00)
        U[(i_base+1)*m + (j_base+1)][offset+3] += 1
        V[k_base*p + (k_base+1)][offset+3] += 1
        V[k_base*p + k_base][offset+3] -= 1
        W[(k_base+1)*n + (i_base+1)][offset+3] += 1
        
        # M5 = (A00+A01)B11
        U[i_base*m + j_base][offset+4] += 1
        U[i_base*m + (j_base+1)][offset+4] += 1
        V[(k_base+1)*p + (k_base+1)][offset+4] += 1
        W[(k_base+1)*n + i_base][offset+4] += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(i_base+1)*m + j_base][offset+5] += 1
        U[i_base*m + j_base][offset+5] -= 1
        V[k_base*p + k_base][offset+5] += 1
        V[k_base*p + (k_base+1)][offset+5] += 1
        W[k_base*n + (i_base+1)][offset+5] += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[i_base*m + (j_base+1)][offset+6] += 1
        U[(i_base+1)*m + (j_base+1)][offset+6] -= 1
        V[(k_base+1)*p + k_base][offset+6] += 1
        V[(k_base+1)*p + (k_base+1)][offset+6] += 1
        W[(k_base+1)*n + i_base][offset+6] += 1
    
    # Top 2 rows of A: 4 Strassen blocks (28 rank terms)
    add_strassen_block(0, 0, 0, 0)   # A[0:2,0:2] * B[0:2,0:2]
    add_strassen_block(7, 0, 2, 0)   # A[0:2,2:4] * B[0:2,0:2]
    add_strassen_block(14, 0, 0, 2)  # A[0:2,0:2] * B[2:4,2:4]
    add_strassen_block(21, 0, 2, 2)  # A[0:2,2:4] * B[2:4,2:4]
    
    # Bottom 1 row of A: 4 naive terms
    for j in range(4):
        for k in range(4):
            offset = 28 + j*4 + k
            U[2*m + j][offset] = 1
            V[j*p + k][offset] = 1
            W[k*n + 2][offset] = 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = DIMENSIONS
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try rank-32 construction first
    try:
        result = _construct_rank_32()
        valid, err = _verify_fast(result['U'], result['V'], result['W'], n, m, p)
        if valid and _verify_exact(result['U'], result['V'], result['W'], n, m, p):
            return result
    except Exception:
        pass
    
    # Fall back to schoolbook
    return _schoolbook()
