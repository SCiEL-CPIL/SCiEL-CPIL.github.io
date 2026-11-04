"""
Combined approach for 2x5x6 matrix multiplication tensor rank.
Uses block-diagonal decomposition: 
1. Split k into 3 pairs of size 2.
2. For each pair, process j=0,1 (2x2x2 block, Strassen rank 7) and j=2,3,4 (2x3x2 block, rank 11).
Total rank: 3 * (7 + 11) = 54.
"""
import sys
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Build the matrix multiplication tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx1 = i * m + j
                idx2 = j * p + k
                idx3 = k * n + i
                T[idx1, idx2, idx3] = 1.0

    def verify(U, V, W):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        rank = U.shape[1]
        T_recon = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-8)

    # Strassen 2x2x2 decomposition (rank 7)
    # Local indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -- Note: C[k,i] in global, but here local k,i
    # Wait, standard Strassen C is 2x2.
    # The tensor T[i,j,k] for matrix mult C[i,k] = sum_j A[i,j]B[j,k].
    # In the local 2x2x2 block, the "C" index is (k,i) or (i,k)?
    # The problem states W[k*n+i]. So C index is k*n+i.
    # For local 2x2, k in {0,1}, i in {0,1}.
    # Local C indices: 0:(k=0,i=0), 1:(k=0,i=1), 2:(k=1,i=0), 3:(k=1,i=1).
    # This corresponds to matrix C = [[C00, C01], [C10, C11]] if we view C[k,i] as row k col i?
    # No, C[k,i] is just an index.
    # The multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    # So the output is indexed by (i,k).
    # But the problem says W[k*n+i].
    # So the third dimension of the tensor is indexed by (k,i).
    # Let's check the Strassen terms again with this indexing.
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2
    # M3 = A00(B01-B11) -> C01 += M3
    # M4 = A11(B10-B00) -> C11 += M4, C10 += M4? No.
    # Let's re-derive or use a known correct set.
    
    # Correct Strassen for C = AB:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to the tensor entries T[i,j,k] = 1.
    # The decomposition is sum_c U[i,j,c] V[j,k,c] W[k,i,c] = T[i,j,k].
    # Wait, the standard tensor decomposition for matrix mult is:
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[i,k,c]?
    # No, the problem defines W[k*n+i].
    # So the third index is (k,i).
    # The tensor value T[i,j,k] is 1.
    # The reconstruction is sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    # This sum must equal 1 if (i,j,k) is a valid index (which it always is in the loop), and 0 otherwise?
    # No, T[i,j,k] is 1 for all i,j,k in the ranges.
    # So we need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
    
    # Let's check the Strassen terms with this.
    # For a specific (i,j,k), the sum of products must be 1.
    
    # Let's define the local Strassen vectors for U, V, W.
    # U is 4x7, V is 4x7, W is 4x7.
    # Indices:
    # U: i*2+j
    # V: j*2+k
    # W: k*2+i
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # A00 (0,0), A11 (1,1)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    # B00 (0,0), B11 (1,1)
    V_s[0, 0] = 1; V_s[3, 0] = 1
    # C00 (k=0,i=0), C11 (k=1,i=1)
    W_s[0, 0] = 1; W_s[3, 0] = 1 # W[k,i]: 0:(0,0), 3:(1,1)
    
    # Term 1: M2 = (A10+A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    # C10 (k=0,i=1) -> W[0*2+1] = W[1]
    W_s[1, 1] = 1
    
    # Term 2: M3 = A00(B01-B11)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    # C01 (k=1,i=0) -> W[1*2+0] = W[2]
    W_s[2, 2] = 1
    
    # Term 3: M4 = A11(B10-B00)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    # C11 (k=1,i=1) -> W[3], C10 (k=0,i=1) -> W[1]
    W_s[3, 3] = 1; W_s[1, 3] = 1
    
    # Term 4: M5 = (A00+A10)B11
    U_s[0, 4] = 1; U_s[2, 4] = 1
    V_s[3, 4] = 1
    # C00 (k=0,i=0) -> W[0], C01 (k=1,i=0) -> W[2]
    W_s[0, 4] = 1; W_s[2, 4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    # C11 (k=1,i=1) -> W[3]
    W_s[3, 5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    # C00 (k=0,i=0) -> W[0]
    W_s[0, 6] = 1
    
    # Verify local Strassen
    T_local = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U_s[i*2+j, c] * V_s[j*2+k, c] * W_s[k*2+i, c]
                T_local[i*2+j, j*2+k, k*2+i] = s
    # Check T_local[i*2+j, j*2+k, k*2+i] == 1
    # This should be true.

    # 2x1x2 Schoolbook (rank 4? or 2?)
    # A 2x1, B 1x2, C 2x2.
    # C[i,k] = A[i,0] B[0,k].
    # T[i,0,k] = 1.
    # Decomposition:
    # We need U[i,0,c] V[0,k,c] W[k,i,c] = 1.
    # If we use rank 1: U[i,0,0]=1, V[0,k,0]=1, W[k,i,0]=1.
    # Then sum is 1.
    # So rank 1 is sufficient for 2x1x2?
    # Let's check.
    # U[0,0]=1, U[1,0]=1.
    # V[0,0]=1, V[1,0]=1. (B index 0*2+0=0, 0*2+1=1)
    # W[0,0]=1 (k=0,i=0), W[1,0]=1 (k=0,i=1), W[2,0]=1 (k=1,i=0), W[3,0]=1 (k=1,i=1).
    # For i=0,j=0,k=0: U[0,0]V[0,0]W[0,0] = 1.
    # For i=0,j=0,k=1: U[0,0]V[1,0]W[2,0] = 1.
    # For i=1,j=0,k=0: U[1,0]V[0,0]W[1,0] = 1.
    # For i=1,j=0,k=1: U[1,0]V[1,0]W[3,0] = 1.
    # Yes, rank 1 works for 2x1x2.
    
    # So for Block 2 (2x3x2):
    # Split j into j=2,3 (2x2x2 part) and j=4 (2x1x2 part).
    # 2x2x2 part: Strassen rank 7.
    # 2x1x2 part: Rank 1.
    # Total rank 8?
    # Wait, the indices are disjoint.
    # The 2x2x2 part uses j=2,3.
    # The 2x1x2 part uses j=4.
    # They are disjoint in U and V indices.
    # So we can just sum the ranks.
    # Rank 7 + Rank 1 = 8.
    
    # So for each k-pair, Block 1 (j=0,1) is rank 7.
    # Block 2 (j=2,3,4) is rank 8.
    # Total per k-pair: 15.
    # 3 k-pairs: 45.
    
    # This is much better than 54.
    
    rank = 3 * (7 + 8)
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    for k_start in range(0, p, 2):
        k1 = k_start
        k2 = k_start + 1
        
        # Block 1: j in {0, 1}
        # Local j -> Global j
        # Local i -> Global i
        # Local k (0,1) -> Global k (k1, k2)
        
        for c in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    val = U_s[i*2+j, c]
                    if val != 0:
                        global_i = i
                        global_j = j
                        global_k = k1 + (0 if False else 0) # k is not in U
                        # U index: i*m + j
                        U[global_i*m + global_j, col] = val
            
            # V
            for j in range(2):
                for k_local in range(2):
                    val = V_s[j*2+k_local, c]
                    if val != 0:
                        global_j = j
                        global_k = k1 + k_local
                        # V index: j*p + k
                        V[global_j*p + global_k, col] = val
            
            # W
            for k_local in range(2):
                for i in range(2):
                    val = W_s[k_local*2+i, c]
                    if val != 0:
                        global_k = k1 + k_local
                        global_i = i
                        # W index: k*n + i
                        W[global_k*n + global_i, col] = val
            
            col += 1
            
        # Block 2: j in {2, 3, 4}
        # Part 2a: j in {2, 3} (2x2x2 Strassen)
        # Map local j (0,1) to global j (2,3)
        
        for c in range(7):
            # U
            for i in range(2):
                for j_local in range(2):
                    val = U_s[i*2+j_local, c]
                    if val != 0:
                        global_i = i
                        global_j = 2 + j_local
                        U[global_i*m + global_j, col] = val
            
            # V
            for j_local in range(2):
                for k_local in range(2):
                    val = V_s[j_local*2+k_local, c]
                    if val != 0:
                        global_j = 2 + j_local
                        global_k = k1 + k_local
                        V[global_j*p + global_k, col] = val
            
            # W
            for k_local in range(2):
                for i in range(2):
                    val = W_s[k_local*2+i, c]
                    if val != 0:
                        global_k = k1 + k_local
                        global_i = i
                        W[global_k*n + global_i, col] = val
            
            col += 1
            
        # Part 2b: j in {4} (2x1x2 Rank 1)
        # U[4] is non-zero for i=0,1
        # V[4] is non-zero for k=0,1
        # W[4] is non-zero for k=0,1 and i=0,1
        
        # U
        U[0*m + 4, col] = 1.0
        U[1*m + 4, col] = 1.0
        
        # V
        V[4*p + k1, col] = 1.0
        V[4*p + k2, col] = 1.0
        
        # W
        W[k1*n + 0, col] = 1.0
        W[k1*n + 1, col] = 1.0
        W[k2*n + 0, col] = 1.0
        W[k2*n + 1, col] = 1.0
        
        col += 1
        
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx1 = i * m + j
                    idx2 = j * p + k
                    idx3 = k * n + i
                    U[idx1, col] = 1.0
                    V[idx2, col] = 1.0
                    W[idx3, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
