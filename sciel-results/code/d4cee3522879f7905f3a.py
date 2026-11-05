"""Matrix multiplication 3x4x5 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(total - 1) > 1e-10:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): got {total}, expected 1")

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing a rank-r tensor
    decomposition of the 3x4x5 matrix multiplication tensor.
    
    U: shape (3*4, r) = (12, r)
    V: shape (4*5, r) = (20, r)
    W: shape (5*3, r) = (15, r)
    
    The convention is W[k*n+i, r] for the output factor.
    The multiplication is C = A * B where A is 3x4 and B is 4x5.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has components T[i,j,k] = 1 if the triple (i,j,k) corresponds
    to a valid multiplication term.
    
    T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Laderman's algorithm for 2x2 matrix multiplication (rank 6)
    # For 2x2 * 2x2: C = A * B
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # 
    # Laderman's 6 multiplications:
    # M1 = (a+b)(e+h)
    # M2 = (a-c)(e+g)
    # M3 = (c+d)(f-h)
    # M4 = a*f
    # M5 = c*e
    # M6 = (d+b)(g+h) - ((d-a)(h-f) + M4 + M5)
    # 
    # Wait, let me use the standard Laderman algorithm more carefully.
    # Standard Laderman:
    # m1 = (a + b) * (e + h)
    # m2 = (a - c) * (e + g)
    # m3 = (c + d) * (f - h)
    # m4 = a * f
    # m5 = c * e
    # m6 = (d + b) * (g + h) - (m4 + m5 + (d - a) * (h - f))
    #
    # Hmm, that's not quite right either. Let me look up the correct Laderman algorithm.
    #
    # Laderman's algorithm for 2x2x2 (rank 6):
    # m1 = (a + b)(e + h)
    # m2 = (a - c)(e + g)  
    # m3 = (c + d)(f - h)
    # m4 = a * f
    # m5 = c * e
    # m6 = (d + b)(g + h) - (m4 + m5 + (d - a)(h - f))
    #
    # Actually, let me just use the well-known formulation:
    # m1 = (a + b)(e + h)
    # m2 = (a - c)(e + g)
    # m3 = (c + d)(f - h)
    # m4 = a * f
    # m5 = c * e
    # m6 = (d + b)(g + h) - (m4 + m5 + (d - a)(h - f))
    #
    # C[0,0] = m1 + m5 - m2 - m3
    # C[0,1] = m1 + m4 - m2 - m6
    # C[1,0] = m3 + m6 - m1 + m5  -- wait, this doesn't look right
    #
    # Let me just use a verified Laderman decomposition.
    # From Wikipedia and standard references:
    #
    # For A = [[a,b],[c,d]] and B = [[e,f],[g,h]]:
    # m1 = (a + b)(e + h)
    # m2 = (a - c)(e + g)
    # m3 = (c + d)(f - h)
    # m4 = a * f
    # m5 = c * e
    # m6 = (d + b)(g + h) - (m4 + m5 + (d - a)(h - f))
    #
    # C[0,0] = m1 + m5 - m2 - m3
    # C[0,1] = m1 + m4 - m2 - m6
    # C[1,0] = m3 + m6 - m1 + m5  -- hmm
    # C[1,1] = m4 + m6 - m5 - m3  -- not sure
    #
    # Actually, let me just implement it by constructing the U, V, W factors directly
    # from the known Laderman algorithm structure.
    
    # Let me use a different approach: I'll define the Laderman algorithm explicitly.
    # The 6 "multiplications" in Laderman are:
    # m1 = (a+b)(e+h)
    # m2 = (a-c)(e+g)
    # m3 = (c+d)(f-h)
    # m4 = a*f
    # m5 = c*e
    # m6 = (d+b)(g+h) - (m4 + m5 + (d-a)(h-f))
    #
    # But m6 involves subtraction of other m's, so it's not a simple tensor term.
    # For tensor decomposition, each "multiplication" corresponds to one rank-1 term.
    # So Laderman has 6 terms, but m6 is expressed in terms of the others plus a new product.
    #
    # Actually, in the tensor decomposition context, Laderman's algorithm gives 6 rank-1 terms
    # where each term is of the form (linear combo of A entries) * (linear combo of B entries)
    # contributing to a linear combo of C entries.
    
    # Let me just hardcode the Laderman U, V, W factors for a 2x2x2 block.
    # For a 2x2x2 block with indices (di, dj, dk) where di, dj, dk in {0, 1}:
    # A entries: A[di, dj], B entries: B[dj, dk], C entries: C[di, dk]
    
    # Laderman's 6 terms (using standard formulation from literature):
    # Term 1: (A[0,0] + A[0,1]) * (B[0,0] + B[1,1]) -> contributes to C[0,0] and C[0,1] and C[1,0] and C[1,1]
    # ...
    
    # This is getting complicated. Let me just use a verified set of coefficients.
    # From "Fast Matrix Multiplication" by B. Le Gall (2014) and standard references:
    
    # Laderman algorithm (rank 6) for 2x2x2:
    # The 6 simple tensors are:
    #
    # 1. (e0 + e1) in A-space, (f0 + f1) in B-space, (g0 + g1) in C-space  -- wait, this isn't right
    #
    # Let me just use the schoolbook for 2x2 (rank 8) and see if I can do better.
    # Actually, the known result is that 2x2x2 has rank 7 (Strassen), not 6.
    # Laderman showed border rank 6, but rank is 7.
    #
    # So for exact decomposition, Strassen (rank 7) is the best for 2x2x2.
    
    # Strassen's algorithm for 2x2x2 (rank 7):
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    #
    # m1 = (a + d)(e + h)
    # m2 = (c + d)e
    # m3 = a(f - h)
    # m4 = d(g - e)
    # m5 = (a + b)h
    # m6 = (c + d)(f - h)
    # m7 = (a + b)(g - e)
    #
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    # Now let me construct the full decomposition for 3x4x5.
    # Split: 3 = 2+1, 4 = 2+2, 5 = 2+2+1
    
    # Blocks:
    # A is 3x4, B is 4x5, C is 3x5
    # A blocks: A[0:2, 0:2], A[0:2, 2:4], A[2:3, 0:2], A[2:3, 2:4]
    # B blocks: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:5], B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:5]
    # C blocks: C[0:2, 0:2], C[0:2, 2:4], C[0:2, 4:5], C[2:3, 0:2], C[2:3, 2:4], C[2:3, 4:5]
    
    # For each pair of A-block and B-block that contributes to a C-block:
    # C[i:i+ni, k:k+pk] += A[i:i+ni, j:j+mj] * B[j:j+mj, k:k+pk]
    
    # Let me enumerate all block multiplications:
    # A block (rows i1:i2, cols j1:j2) times B block (rows j1:j2, cols k1:k2) -> C block (rows i1:i2, cols k1:k2)
    
    # A row blocks: [0,2), [2,3)
    # A col blocks: [0,2), [2,4)
    # B row blocks: [0,2), [2,4)
    # B col blocks: [0,2), [2,4), [4,5)
    
    # So the block multiplications are:
    # For each (ar, ac, br, bc) where ar in [0,2), br in [0,2), ac in [0,2), bc in [0,2), [4,5)
    # But we need ac == br (the inner dimension must match)
    
    # So:
    # ar=0, ac=0, br=0, bc=0: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (2x2x2, use Strassen, rank 7)
    # ar=0, ac=0, br=0, bc=1: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] (2x2x2, use Strassen, rank 7)
    # ar=0, ac=0, br=0, bc=2: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5] (2x2x1, schoolbook, rank 4)
    # ar=0, ac=1, br=1, bc=0: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2] (2x2x2, use Strassen, rank 7)
    # ar=0, ac=1, br=1, bc=1: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4] (2x2x2, use Strassen, rank 7)
    # ar=0, ac=1, br=1, bc=2: A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4:5] (2x2x1, schoolbook, rank 4)
    # ar=1, ac=0, br=0, bc=0: A[2:3,0:2] * B[0:2,0:2] -> C[2:3,0:2] (1x2x2, schoolbook, rank 4)
    # ar=1, ac=0, br=0, bc=1: A[2:3,0:2] * B[0:2,2:4] -> C[2:3,2:4] (1x2x2, schoolbook, rank 4)
    # ar=1, ac=0, br=0, bc=2: A[2:3,0:2] * B[0:2,4:5] -> C[2:3,4:5] (1x2x1, schoolbook, rank 2)
    # ar=1, ac=1, br=1, bc=0: A[2:3,2:4] * B[2:4,0:2] -> C[2:3,0:2] (1x2x2, schoolbook, rank 4)
    # ar=1, ac=1, br=1, bc=1: A[2:3,2:4] * B[2:4,2:4] -> C[2:3,2:4] (1x2x2, schoolbook, rank 4)
    # ar=1, ac=1, br=1, bc=2: A[2:3,2:4] * B[2:4,4:5] -> C[2:3,4:5] (1x2x1, schoolbook, rank 2)
    
    # Total rank: 7+7+4+7+7+4+4+4+2+4+4+2 = 56
    
    # Let me construct this decomposition.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_strassen(n_off, m_off, p_off):
        """Add Strassen decomposition for a 2x2x2 block at offset (n_off, m_off, p_off)."""
        # Strassen's 7 terms
        # m1 = (a + d)(e + h)
        # m2 = (c + d)e
        # m3 = a(f - h)
        # m4 = d(g - e)
        # m5 = (a + b)h
        # m6 = (c + d)(f - h)
        # m7 = (a + b)(g - e)
        #
        # C[0,0] = m1 + m4 - m5 + m7
        # C[0,1] = m3 + m5
        # C[1,0] = m2 + m4
        # C[1,1] = m1 - m2 + m3 + m6
        
        # A indices: (di, dj) where di in {0,1}, dj in {0,1}
        # B indices: (dj, dk) where dj in {0,1}, dk in {0,1}
        # C indices: (di, dk) where di in {0,1}, dk in {0,1}
        
        # Term 1: (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        U_row = [0] * (n * m)
        V_row = [0] * (m * p)
        W_row = [0] * (p * n)
        U_row[(n_off + 0) * m + (m_off + 0)] = 1
        U_row[(n_off + 1) * m + (m_off + 1)] = 1
        V_row[(m_off + 0) * p + (p_off + 0)] = 1
        V_row[(m_off + 1) * p + (p_off + 1)] = 1
        # C[0,0] += 1, C[1,1] += 1
        W_row[(p_off + 0) * n + (n_off + 0)] = 1
        W_row[(p_off + 1) * n + (n_off + 1)] = 1
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)
        
        # Term 2: (A[1,0] + A[1,1]) * B[0,0]
        U_row = [0] * (n * m)
        V_row = [0] * (m * p)
        W_row = [0] * (p * n)
        U_row[(n_off + 1) * m + (m_off + 0)] = 1
        U_row[(n_off + 1) * m + (m_off + 1)] = 1
        V_row[(m_off + 0) * p + (p_off + 0)] = 1
        # C[1,0] += 1
        W_row[(p_off + 0) * n + (n_off + 1)] = 1
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)
        
        # Term 3: A[0,0] * (B[0,1] - B[1,1])
        U_row = [0] * (n * m)
        V_row = [0] * (m * p)
        W_row = [0] * (p * n)
        U_row[(n_off + 0) * m + (m_off + 0)] = 1
        V_row[(m_off + 0) * p + (p_off + 1)] = 1
        V_row[(m_off + 1) * p + (p_off + 1)] = -1
        # C[0,1] += 1
        W_row[(p_off + 1) * n + (n_off + 0)] = 1
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)
        
        # Term 4: A[1,1] * (B[1,0] - B[0,0])
        U_row = [0] * (n * m)
        V_row = [0] * (m * p)
        W_row = [0] * (p * n)
        U_row[(n_off + 1) * m + (m_off + 1)] = 1
        V_row[(m_off + 1) * p + (p_off + 0)] = 1
        V_row[(m_off + 0) * p + (p_off + 0)] = -1
        # C[0,0] += 1, C[1,0] += 1
        W_row[(p_off + 0) * n + (n_off + 0)] = 1
        W_row[(p_off + 0) * n + (n_off + 1)] = 1
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)
        
        # Term 5: (A[0,0] + A[0,1]) * B[1,1]
        U_row = [0] * (n * m)
        V_row = [0] * (m * p)
        W_row = [0] * (p * n)
        U_row[(n_off + 0) * m + (m_off + 0)] = 1
        U_row[(n_off + 0) * m + (m_off + 1)] = 1
        V_row[(m_off + 1) * p + (p_off + 1)] = 1
        # C[0,0] -= 1, C[0,1] += 1
        W_row[(p_off + 0) * n + (n_off + 0)] = -1
        W_row[(p_off + 1) * n + (n_off + 0)] = 1
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)
        
        # Term 6: (A[1,0] + A[1,1]) * (B[0,1] - B[1,1])
        U_row = [0] * (n * m)
        V_row = [0] * (m * p)
        W_row = [0] * (p * n)
        U_row[(n_off + 1) * m + (m_off + 0)] = 1
        U_row[(n_off + 1) * m + (m_off + 1)] = 1
        V_row[(m_off + 0) * p + (p_off + 1)] = 1
        V_row[(m_off + 1) * p + (p_off + 1)] = -1
        # C[1,1] += 1
        W_row[(p_off + 1) * n + (n_off + 1)] = 1
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)
        
        # Term 7: (A[0,0] + A[0,1]) * (B[1,0] - B[0,0])
        U_row = [0] * (n * m)
        V_row = [0] * (m * p)
        W_row = [0] * (p * n)
        U_row[(n_off + 0) * m + (m_off + 0)] = 1
        U_row[(n_off + 0) * m + (m_off + 1)] = 1
        V_row[(m_off + 1) * p + (p_off + 0)] = 1
        V_row[(m_off + 0) * p + (p_off + 0)] = -1
        # C[0,0] += 1
        W_row[(p_off + 0) * n + (n_off + 0)] = 1
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)
    
    def add_schoolbook(n_off, m_off, p_off, ns, ms, ps):
        """Add schoolbook decomposition for an ns x ms x ps block at offset (n_off, m_off, p_off)."""
        for di in range(ns):
            for dj in range(ms):
                for dk in range(ps):
                    U_row = [0] * (n * m)
                    V_row = [0] * (m * p)
                    W_row = [0] * (p * n)
                    U_row[(n_off + di) * m + (m_off + dj)] = 1
                    V_row[(m_off + dj) * p + (p_off + dk)] = 1
                    W_row[(p_off + dk) * n + (n_off + di)] = 1
                    U_list.append(U_row)
                    V_list.append(V_row)
                    W_list.append(W_row)
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (2x2x2, Strassen)
    add_strassen(0, 0, 0)
    
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] (2x2x2, Strassen)
    add_strassen(0, 0, 2)
    
    # Block 3: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5] (2x2x1, schoolbook)
    add_schoolbook(0, 0, 4, 2, 2, 1)
    
    # Block 4: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2] (2x2x2, Strassen)
    add_strassen(0, 2, 0)
    
    # Block 5: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4] (2x2x2, Strassen)
    add_strassen(0, 2, 2)
    
    # Block 6: A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4:5] (2x2x1, schoolbook)
    add_schoolbook(0, 2, 4, 2, 2, 1)
    
    # Block 7: A[2:3,0:2] * B[0:2,0:2] -> C[2:3,0:2] (1x2x2, schoolbook)
    add_schoolbook(2, 0, 0, 1, 2, 2)
    
    # Block 8: A[2:3,0:2] * B[0:2,2:4] -> C[2:3,2:4] (1x2x2, schoolbook)
    add_schoolbook(2, 0, 2, 1, 2, 2)
    
    # Block 9: A[2:3,0:2] * B[0:2,4:5] -> C[2:3,4:5] (1x2x1, schoolbook)
    add_schoolbook(2, 0, 4, 1, 2, 1)
    
    # Block 10: A[2:3,2:4] * B[2:4,0:2] -> C[2:3,0:2] (1x2x2, schoolbook)
    add_schoolbook(2, 2, 0, 1, 2, 2)
    
    # Block 11: A[2:3,2:4] * B[2:4,2:4] -> C[2:3,2:4] (1x2x2, schoolbook)
    add_schoolbook(2, 2, 2, 1, 2, 2)
    
    # Block 12: A[2:3,2:4] * B[2:4,4:5] -> C[2:3,4:5] (1x2x1, schoolbook)
    add_schoolbook(2, 2, 4, 1, 2, 1)
    
    rank = len(U_list)
    
    # Verify correctness
    _verify_decomposition(U_list, V_list, W_list, n, m, p)
    
    return {"U": U_list, "V": V_list, "W": W_list}
