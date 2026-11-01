"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    size_a = n * m  # 4
    size_b = m * p  # 28
    size_c = p * n  # 28
    
    # Build the true tensor T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T_true = np.zeros((size_a, size_b, size_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1.0

    def verify(U, V, W, atol=1e-9):
        """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T_true[a,b,c]."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
        return np.allclose(T_recon, T_true, atol=atol)

    def schoolbook():
        """Return the standard rank-56 schoolbook decomposition."""
        rank = n * m * p
        U = np.zeros((size_a, rank), dtype=int)
        V = np.zeros((size_b, rank), dtype=int)
        W = np.zeros((size_c, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def try_strassen_block_decomposition():
        """
        Try a Strassen-like block decomposition.
        
        For 2x2x2, Strassen reduces rank from 8 to 7 using:
          M1 = (A11+A22)(B11+B22)
          M2 = (A21+A22)B11
          M3 = A11(B21-B22)
          M4 = A22(B12-B11)
          M5 = (A11+A12)B22
          M6 = (A21-A11)(B11+B12)
          M7 = (A12-A22)(B21+B22)
          
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        
        For 2x2x14, we can partition the 14 columns into blocks and apply
        Strassen to each pair of blocks, then combine.
        
        Strategy: Group the 14 columns into 7 pairs: (0,1), (2,3), (4,5), ..., (12,13)
        For each pair, use Strassen-like structure but adapted.
        
        Actually, a better approach: treat the 2x14 matrix as 2x(2*7).
        We can apply a 2x2 Strassen per pair of column-blocks of size 1.
        
        Let me think differently. The tensor is C_{i,k} = sum_j A_{i,j} B_{j,k}
        where i in {0,1}, j in {0,1}, k in {0,...,13}.
        
        Using Strassen for the 2x2 multiplication per k:
        For each k, we need A_{0,k} = A00*B0k + A01*B1k
                 A_{1,k} = A10*B0k + A11*B1k
        
        Strassen combines two such multiplications. But here each "multiplication"
        is just a 2x1 (outer product) for a fixed k.
        
        Alternative: Use the 7-term Strassen to handle 7 pairs of k values at once.
        Partition k into two sets of 7: K1 = {0,...,6}, K2 = {7,...,13}.
        
        For K1 and K2, we have two "vectors" B0 = [B_{0,0},...,B_{0,6}] and B1 = [B_{1,0},...,B_{1,6}]
        and similarly for K2.
        
        Actually, let me try a direct approach:
        Use rank 7 per 2-column block, giving 7 * 7 = 49 total rank.
        
        For columns 2l and 2l+1 (l = 0,...,6):
        We need to compute:
          C_{0,2l}   = A00*B0,2l + A01*B1,2l
          C_{0,2l+1} = A00*B0,2l+1 + A01*B1,2l+1
          C_{1,2l}   = A10*B0,2l + A11*B1,2l
          C_{1,2l+1} = A10*B0,2l+1 + A11*B1,2l+1
        
        This is a 2x2 matrix times a 2x2 matrix (where the 2x2 B has columns [B_{0,2l}, B_{0,2l+1}] and [B_{1,2l}, B_{1,2l+1}]).
        
        Strassen for this 2x2x2 block uses 7 terms:
          M1 = (A00 + A11) * (B0,2l + B1,2l)    -> contributes to C_{0,2l} and C_{1,2l+1}
          M2 = (A10 + A11) * B0,2l               -> contributes to C_{1,2l}
          M3 = A00 * (B1,2l - B1,2l+1)           -> contributes to C_{0,2l+1}
          M4 = A11 * (B0,2l+1 - B0,2l)           -> contributes to C_{1,2l+1}
          M5 = (A00 + A01) * B1,2l+1             -> contributes to C_{0,2l+1}
          M6 = (A10 - A00) * (B0,2l + B0,2l+1)   -> contributes to C_{1,2l}
          M7 = (A01 - A11) * (B1,2l + B1,2l+1)   -> contributes to C_{0,2l}
          
        Wait, let me be more careful. The standard Strassen for C = A*B where A,B are 2x2:
          A = [[a,b],[c,d]], B = [[e,f],[g,h]]
          C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
          
          M1 = (a+d)(e+h)
          M2 = (c+d)e
          M3 = a(f-g)
          M4 = d(h-f)
          M5 = (a+b)h
          M6 = (c-a)(e+f)
          M7 = (b-d)(g+h)
          
          C11 = M1 + M4 - M5 + M7
          C12 = M3 + M5
          C21 = M2 + M4
          C22 = M1 - M2 + M3 + M6
        
        In our case, for block l (columns 2l, 2l+1):
          A = [[A00, A01],[A10, A11]]
          B = [[B0,2l, B0,2l+1],[B1,2l, B1,2l+1]]
          
          e = B0,2l, f = B0,2l+1, g = B1,2l, h = B1,2l+1
          
          M1 = (A00+A11)(B0,2l + B1,2l+1)
          M2 = (A10+A11) * B0,2l
          M3 = A00 * (B0,2l+1 - B1,2l)
          M4 = A11 * (B1,2l+1 - B0,2l+1)
          M5 = (A00+A01) * B1,2l+1
          M6 = (A10-A00) * (B0,2l + B0,2l+1)
          M7 = (A01-A11) * (B1,2l + B1,2l+1)
          
          C_{0,2l}   = M1 + M4 - M5 + M7
          C_{0,2l+1} = M3 + M5
          C_{1,2l}   = M2 + M4
          C_{1,2l+1} = M1 - M2 + M3 + M6
          
        Now, each M term is a product of a linear combination of A's and a linear combination of B's.
        For the tensor decomposition, each rank-1 term corresponds to one M.
        
        For M1: U[a,r] * V[b,r] * W[c,r]
          - A part: (A00 + A11) means U[0,0]=1, U[1,0]=1, U[2,r]=0, U[3,r]=0
            (a=0 -> A00, a=1 -> A01, a=2 -> A10, a=3 -> A11)
          - B part: (B0,2l + B1,2l+1) means V[2l,r]=1, V[14+2l+1,r]=1, others=0
            (b=j*p+k, so j=0,k=2l -> b=2l; j=1,k=2l+1 -> b=14+2l+1)
          - C part: M1 contributes to C_{0,2l} with +1, C_{1,2l+1} with +1
            c = k*n+i, so C_{0,2l}: k=2l,i=0 -> c=2l; C_{1,2l+1}: k=2l+1,i=1 -> c=2(2l+1)+1=4l+3
        
        Let me code this up for rank 49 (7 blocks * 7 terms per block).
        """
        rank = 49
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)
        
        for l in range(7):
            col_start = 7 * l
            col_end = col_start + 7
            
            # M1: (A00+A11)(B0,2l + B1,2l+1)
            # U: A00=1, A01=0, A10=0, A11=1
            U[0, col_start:col_start+1] = 1
            U[1, col_start:col_start+1] = 0
            U[2, col_start:col_start+1] = 0
            U[3, col_start:col_start+1] = 1
            # V: B0,2l=1, B1,2l+1=1 (b=2l for B0,2l; b=14+2l+1 for B1,2l+1)
            V[2 * l, col_start:col_start+1] = 1
            V[14 + 2 * l + 1, col_start:col_start+1] = 1
            # W: C_{0,2l} +1, C_{1,2l+1} +1
            # c for C_{0,2l}: k=2l, i=0 -> c=2l*2+0 = 4l
            # c for C_{1,2l+1}: k=2l+1, i=1 -> c=(2l+1)*2+1 = 4l+3
            W[4 * l, col_start:col_start+1] = 1
            W[4 * l + 3, col_start:col_start+1] = 1
            
            # M2: (A10+A11) * B0,2l
            U[0, col_start+1:col_start+2] = 0
            U[1, col_start+1:col_start+2] = 0
            U[2, col_start+1:col_start+2] = 1
            U[3, col_start+1:col_start+2] = 1
            V[2 * l, col_start+1:col_start+2] = 1
            W[4 * l + 2, col_start+1:col_start+2] = 1  # C_{1,2l}: k=2l, i=1 -> c=4l+2
            
            # M3: A00 * (B0,2l+1 - B1,2l)
            U[0, col_start+2:col_start+3] = 1
            U[1, col_start+2:col_start+3] = 0
            U[2, col_start+2:col_start+3] = 0
            U[3, col_start+2:col_start+3] = 0
            V[2 * l + 1, col_start+2:col_start+3] = 1    # B0,2l+1
            V[14 + 2 * l, col_start+2:col_start+3] = -1  # -B1,2l
            W[4 * l + 1, col_start+2:col_start+3] = 1    # C_{0,2l+1}: k=2l+1, i=0 -> c=4l+2
            
            # M4: A11 * (B1,2l+1 - B0,2l+1)
            U[0, col_start+3:col_start+4] = 0
            U[1, col_start+3:col_start+4] = 0
            U[2, col_start+3:col_start+4] = 0
            U[3, col_start+3:col_start+4] = 1
            V[14 + 2 * l + 1, col_start+3:col_start+4] = 1   # B1,2l+1
            V[2 * l + 1, col_start+3:col_start+4] = -1        # -B0,2l+1
            W[4 * l, col_start+3:col_start+4] = 1             # C_{0,2l}: +1
            W[4 * l + 3, col_start+3:col_start+4] = 1         # C_{1,2l+1}: +1
            
            # M5: (A00+A01) * B1,2l+1
            U[0, col_start+4:col_start+5] = 1
            U[1, col_start+4:col_start+5] = 1
            U[2, col_start+4:col_start+5] = 0
            U[3, col_start+4:col_start+5] = 0
            V[14 + 2 * l + 1, col_start+4:col_start+5] = 1
            W[4 * l, col_start+4:col_start+5] = -1            # C_{0,2l}: -1
            W[4 * l + 1, col_start+4:col_start+5] = 1         # C_{0,2l+1}: +1
            
            # M6: (A10-A00) * (B0,2l + B0,2l+1)
            U[0, col_start+5:col_start+6] = -1
            U[1, col_start+5:col_start+6] = 0
            U[2, col_start+5:col_start+6] = 1
            U[3, col_start+5:col_start+6] = 0
            V[2 * l, col_start+5:col_start+6] = 1
            V[2 * l + 1, col_start+5:col_start+6] = 1
            W[4 * l + 2, col_start+5:col_start+6] = 1         # C_{1,2l}: +1
            
            # M7: (A01-A11) * (B1,2l + B1,2l+1)
            U[0, col_start+6:col_start+7] = 0
            U[1, col_start+6:col_start+7] = 1
            U[2, col_start+6:col_start+7] = 0
            U[3, col_start+6:col_start+7] = -1
            V[14 + 2 * l, col_start+6:col_start+7] = 1
            V[14 + 2 * l + 1, col_start+6:col_start+7] = 1
            W[4 * l, col_start+6:col_start+7] = 1             # C_{0,2l}: +1
            W[4 * l + 3, col_start+6:col_start+7] = 0         # Not contributing
        
        return U, V, W

    # First try the constructive rank-49 Strassen decomposition
    U, V, W = try_strassen_block_decomposition()
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If that didn't work, check with looser tolerance
    if verify(U, V, W, atol=1e-6):
        # Try to clean up
        U_clean = np.round(U * 2) / 2
        V_clean = np.round(V * 2) / 2
        W_clean = np.round(W * 2) / 2
        if verify(U_clean, V_clean, W_clean):
            return {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 56
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
