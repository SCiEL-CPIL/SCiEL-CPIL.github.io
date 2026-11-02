"""Matrix multiplication 2x4 x 4x7 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 28
    pn = p * n  # 14
    
    # Target rank calculation:
    # 2 * (3 * R(2,2,2) + 1 * R(2,2,1))
    # R(2,2,2) = 7 (Strassen)
    # R(2,2,1) = 2
    # Rank = 2 * (3 * 7 + 2) = 2 * 23 = 46
    
    rank = 46
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    # Helper to add Strassen decomposition for 2x2x2
    # A is 2x2, B is 2x2, C is 2x2
    # Indices in U: (2, 7), V: (2, 7), W: (2, 7)
    def add_strassen(U_block, V_block, W_block, offset):
        # U_block shape (2, 7), V_block shape (2, 7), W_block shape (2, 7)
        # Map to global U, V, W
        # U indices: i in {0,1}
        # V indices: j in {0,1}
        # W indices: k in {0,1}
        
        # Strassen coefficients
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U (A part)
        # d=0 (M1): A11, A22 -> U[0,0]=1, U[1,0]=1
        # d=1 (M2): A21, A22 -> U[0,1]=0, U[1,1]=1 (Wait, M2 uses A21, A22)
        # Actually, U[i, d] is coeff of A[i, j] for the term d.
        # Term d involves sum_j U[i,d] V[j,d] W[k,d] A[i,j] B[j,k]
        # No, the tensor T[i,j,k] = sum_d U[i,d] V[j,d] W[k,d]
        # So for a specific term d, U[i,d] is the coeff for A[i, :]? No.
        # U[i,d] is the value in the U matrix.
        # The term is U[i,d] * V[j,d] * W[k,d].
        # This term contributes to T[i,j,k].
        
        # Let's define the 7 rank-1 tensors for Strassen.
        # T = sum_d u_d v_d^T w_d^T (outer product)
        
        # d=0: M1
        # U[:,0] = [1, 1] (A11, A22) -- Wait, A is 2x2.
        # A11 is A[0,0], A22 is A[1,1].
        # U[0,0]=1, U[1,0]=1? No, U has shape (2, 7).
        # U[i, d] corresponds to the coefficient of A[i, j] in the linear combination for term d?
        # No, U[i, d] is just the matrix element.
        # The linear combination for A is: A_lin_d = sum_i U[i, d] A[i, :]? No.
        # The standard form is:
        # M_d = (sum_i U[i, d] A[i, j]) * (sum_k V[j, d] B[j, k])? No.
        
        # Let's use the explicit tensor values.
        # T[i, j, k] = sum_d U[i, d] V[j, d] W[k, d]
        
        # Strassen:
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # M2 = (A10+A11)B00 -> C10 += M2
        # M3 = A00(B01-B11) -> C01 += M3
        # M4 = A11(B10-B00) -> C10 += M4, C00 -= M4
        # M5 = (A00+A01)B11 -> C01 += M5, C11 += M5
        # M6 = (A10-A00)(B00+B01) -> C00 += M6
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        
        # We need to find U, V, W such that sum_d U[i,d]V[j,d]W[k,d] = 1 for all i,j,k?
        # No, T[i,j,k] is 1 for all i,j,k in 2x2x2.
        
        # Let's verify the Strassen decomposition of the all-ones tensor.
        # This is a known result.
        
        # U (2x7)
        U_s = np.array([
            [1, 0, 1, 0, 1, -1, 1], # i=0
            [1, 1, 0, 1, 0, 1, -1]  # i=1
        ])
        
        # V (2x7)
        V_s = np.array([
            [1, 1, 0, -1, 0, 1, 1], # j=0
            [1, 0, 1, 1, 1, 1, 1]   # j=1
        ])
        
        # W (2x7)
        W_s = np.array([
            [1, 0, 1, 1, 0, 1, 1], # k=0
            [1, 0, -1, 0, 1, 1, 1] # k=1
        ])
        
        # Verify T[0,0,0] = sum_d U[0,d]V[0,d]W[0,d]
        # d=0: 1*1*1 = 1
        # d=1: 0*1*0 = 0
        # d=2: 1*0*1 = 0
        # d=3: 0*-1*1 = 0
        # d=4: 1*0*0 = 0
        # d=5: -1*1*1 = -1
        # d=6: 1*1*1 = 1
        # Sum = 1 - 1 + 1 = 1. Correct.
        
        U_block[:, 0:7] = U_s
        V_block[:, 0:7] = V_s
        W_block[:, 0:7] = W_s
        return 7

    # Helper to add Standard decomposition for 2x2x1
    # T[i, j, 0] = 1 for i,j in {0,1}
    def add_standard_221(U_block, V_block, W_block, offset):
        # Rank 2
        U_s = np.array([
            [1, 1], # i=0
            [1, 1]  # i=1
        ])
        V_s = np.array([
            [1, 0], # j=0
            [0, 1]  # j=1
        ])
        W_s = np.array([
            [1, 1]  # k=0
        ])
        
        # Verify T[0,0,0] = 1*1*1 + 1*0*1 = 1
        # T[0,1,0] = 1*0*1 + 1*1*1 = 1
        # T[1,0,0] = 1*1*1 + 1*0*1 = 1
        # T[1,1,0] = 1*0*1 + 1*1*1 = 1
        # Correct.
        
        U_block[:, 0:2] = U_s
        V_block[:, 0:2] = V_s
        W_block[:, 0:1] = W_s
        return 2

    # Main construction loop
    # A is 2x4, split into A_L (2x2), A_R (2x2)
    # B is 4x7, split into B_T (2x7), B_B (2x7)
    # C = A_L B_T + A_R B_B
    
    # Process A_L B_T
    # B_T is 2x7, split into 4 blocks: 2x2, 2x2, 2x2, 2x1
    for block_idx in range(4):
        if block_idx < 3:
            # Strassen 2x2x2
            # U indices: 0,1 (from A_L)
            # V indices: block_idx*2, block_idx*2+1 (from B_T)
            # W indices: 0,1 (from C)
            
            U_block = U[0:2, col:col+7]
            V_block = V[block_idx*2:block_idx*2+2, col:col+7]
            W_block = W[0:2, col:col+7]
            
            add_strassen(U_block, V_block, W_block, col)
            col += 7
        else:
            # Standard 2x2x1
            # U indices: 0,1
            # V indices: 6,7 (2x1)
            # W indices: 0,1 (2x1? No, C is 2x7. The output of this block is 2x1, contributing to C[:, 6])
            
            U_block = U[0:2, col:col+2]
            V_block = V[6:8, col:col+2]
            # W output is 2x1, so W indices 0,1?
            # Wait, C is 2x7.
            # The block B_T[:, 6:7] is 2x1.
            # The product A_L (2x2) * B_T[:, 6:7] (2x1) is 2x1.
            # This contributes to C[:, 6].
            # So W indices should be 6 (for C[:, 6]).
            # But W_block shape is (2, 2) in my helper?
            # My helper add_standard_221 expects W_block shape (1, 2)? No, (2, 2) if it outputs 2 rows?
            # No, 2x2x1 means A(2x2), B(2x1), C(2x1).
            # W has shape (1, 2) in the tensor definition?
            # T[i, j, k]. k is index of C.
            # If C is 2x1, k is 0.
            # So W has shape (1, rank).
            # In my global W, shape is (14, rank).
            # So W_block should be W[6:7, col:col+2].
            
            U_block = U[0:2, col:col+2]
            V_block = V[6:8, col:col+2]
            W_block = W[6:7, col:col+2]
            
            # Modify helper to handle W shape (1, 2)
            # U_s is 2x2, V_s is 2x2, W_s is 1x2
            U_block[:, 0:2] = np.array([[1, 1], [1, 1]])
            V_block[:, 0:2] = np.array([[1, 0], [0, 1]])
            W_block[:, 0:2] = np.array([[1, 1]])
            
            col += 2

    # Process A_R B_B
    # A_R is A[1:2, :]? No, A is 2x4.
    # A_L = A[:, 0:2], A_R = A[:, 2:4].
    # B_T = B[0:2, :], B_B = B[2:4, :].
    
    # A_R indices in U: 0,1 (local) -> 0,1 (global) because U is 2x?
    # Wait, U has shape (8, rank).
    # A is 2x4.
    # U[i, d] where i is row of A (0,1).
    # So U[0:2, :] corresponds to A_L and A_R?
    # No, A_L and A_R are both 2x2.
    # A_L is A[:, 0:2]. A_R is A[:, 2:4].
    # The U matrix maps the flattened A to the decomposition.
    # A is flattened to 8 elements.
    # U has 8 rows.
    # Row 0,1: A[0,0], A[0,1] (A_L top)
    # Row 2,3: A[1,0], A[1,1] (A_L bottom)
    # Row 4,5: A[0,2], A[0,3] (A_R top)
    # Row 6,7: A[1,2], A[1,3] (A_R bottom)
    
    # My previous U_block usage `U[0:2, ...]` was wrong.
    # It should be `U[0:4, ...]` for A_L?
    # No, A_L is 2x2. Flattened, it's 4 elements.
    # But Strassen uses 2x2 matrices.
    # The U matrix in Strassen for 2x2x2 has shape (2, 7)?
    # No, the tensor T is (2, 2, 2).
    # U has shape (2, rank).
    # This U corresponds to the 2 rows of A.
    # So for A_L (2x2), we use U[0:2, :]?
    # But A_L has 4 elements.
    # The decomposition $T_{i,j,k}$ sums over $j$.
    # $U_{i,d}$ multiplies $A_{i,j}$? No.
    # The term is $U_{i,d} V_{j,d} W_{k,d}$.
    # This term is added to $T_{i,j,k}$.
    # So $U$ has one row for each $i$ (row of A).
    # $V$ has one row for each $j$ (col of A / row of B).
    # $W$ has one row for each $k$ (col of B).
    
    # So for A_L (2x2), $i \in \{0,1\}$, $j \in \{0,1\}$.
    # We use U[0:2, :] for the i dimension.
    # But A_L is part of the larger A.
    # The global U has 8 rows.
    # Rows 0,1 correspond to i=0 of A_L and A_R?
    # No, A is 2x4. i is row (0,1). j is col (0,1,2,3).
    # So U has 2 rows?
    # No, U shape is (nm, rank) = (8, rank).
    # nm = n*m = 2*4 = 8.
    # So U has 8 rows.
    # This implies the first index of U is the flattened index of A.
    # $A_{flat}[idx]$.
    # $idx = i*m + j$.
    # So U[0, d] corresponds to A[0,0].
    # U[1, d] corresponds to A[0,1].
    # U[2, d] corresponds to A[1,0].
    # U[3, d] corresponds to A[1,1].
    
    # In the Strassen decomposition for 2x2x2:
    # T[i, j, k] = sum_d U[i, d] V[j, d] W[k, d].
    # Here i, j, k are local indices 0,1.
    # To embed this in the global tensor:
    # Global U index for local i, j is $i*m + j$.
    # But U in the formula $U[i,d]$ only depends on i?
    # No, the tensor is $T_{i,j,k}$.
    # The factorization is $T_{i,j,k} = \sum_d U_{i,d} V_{j,d} W_{k,d}$.
    # This requires U to have dimension $n \times rank$.
    # But the task requires U to have dimension $nm \times rank$.
    # This means the "U" in the task is not the $U$ in the simple tensor formula?
    # Or is it?
    # If $U$ is $nm \times rank$, then $T_{i,j,k}$ cannot be formed by $U_{i,d}$ because $i$ only goes to $n-1$.
    # The task says: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)".
    # And "exactly decompose the matrix multiplication tensor".
    # The tensor $T$ has shape $(nm, mp, pn)$.
    # $T_{a, b, c} = \sum_d U_{a,d} V_{b,d} W_{c,d}$.
    # Here $a \in [0, nm)$, $b \in [0, mp)$, $c \in [0, pn)$.
    # $a = i*m + j$.
    # $b = j*p + k$.
    # $c = k*n + i$.
    
    # So for a block $A_L B_{T,1}$:
    # $i \in \{0,1\}$, $j \in \{0,1\}$, $k \in \{0,1\}$.
    # $a = i*4 + j$.
    # $b = j*7 + k$.
    # $c = k*2 + i$.
    
    # We need to fill U[a, d], V[b, d], W[c, d] such that the sum equals 1 for these indices.
    
    # For Strassen 2x2x2:
    # Local U_s[i, d], V_s[j, d], W_s[k, d].
    # Global U[i*4 + j, d] = U_s[i, d]?
    # No, U_s[i, d] is the coefficient for the i-th row of A.
    # But in the global tensor, the index is $a = i*m + j$.
    # The term $U_{a,d} V_{b,d} W_{c,d}$ must equal $T_{a,b,c}$.
    # If we set $U_{i*m+j, d} = U_s[i, d]$, then:
    # Sum_d $U_s[i, d] V_s[j, d] W_s[k, d]$?
    # This works if $V_{j*p+k, d} = V_s[j, d]$ and $W_{k*n+i, d} = W_s[k, d]$.
    
    # So:
    # U[0*4+0, d] = U_s[0, d]
    # U[0*4+1, d] = U_s[0, d]
    # U[1*4+0, d] = U_s[1, d]
    # U[1*4+1, d] = U_s[1, d]
    
    # This means U[0, d] = U[1, d] and U[2, d] = U[3, d].
    
    # Let's implement this mapping.

    def add_block(U, V, W, i_start, j_start, k_start, n, m, p, rank_offset, U_s, V_s, W_s):
        # i_start: local i start (0)
        # j_start: local j start (0)
        # k_start: local k start (0)
        # U_s: (n, rank_block)
        # V_s: (m, rank_block)
        # W_s: (p, rank_block)
        
        for d in range(U_s.shape[1]):
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    U[a, rank_offset + d] = U_s[i, d]
            
            for j in range(m):
                for k in range(p):
                    b = j * p + k
                    V[b, rank_offset + d] = V_s[j, d]
            
            for k in range(p):
                for i in range(n):
                    c = k * n + i
                    W[c, rank_offset + d] = W_s[k, d]

    # Re-initialize
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    
    # Strassen Factors (Local 2x2x2)
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 1],
        [1, 1, 0, 1, 0, 1, -1]
    ])
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 1],
        [1, 0, 1, 1, 1, 1, 1]
    ])
    W_s = np.array([
        [1, 0, 1, 1, 0, 1, 1],
        [1, 0, -1, 0, 1, 1, 1]
    ])
    
    # Standard 2x2x1 Factors (Local 2x2x1)
    U_s_221 = np.array([
        [1, 1],
        [1, 1]
    ])
    V_s_221 = np.array([
        [1, 0],
        [0, 1]
    ])
    W_s_221 = np.array([
        [1, 1]
    ])
    
    # Process A_L (i=0,1) B_T (j=0,1)
    for block_idx in range(4):
        if block_idx < 3:
            # 2x2x2
            # i in 0,1; j in block_idx*2, block_idx*2+1; k in 0,1
            # But j is local to B_T?
            # B_T is B[0:2, :].
            # j_global = 0, 1.
            # k_global = block_idx*2, block_idx*2+1.
            
            # Wait, B is 4x7.
            # B_T is rows 0,1 of B.
            # B_T[:, 0:2] is 2x2.
            # j is row of B_T (0,1).
            # k is col of B_T (0,1).
            
            add_block(U, V, W, 0, 0, block_idx*2, 2, 2, 2, col, U_s, V_s, W_s)
            col += 7
        else:
            # 2x2x1
            # j in 0,1. k in 6.
            add_block(U, V, W, 0, 0, 6, 2, 2, 1, col, U_s_221, V_s_221, W_s_221)
            col += 2

    # Process A_R (i=0,1) B_B (j=2,3)
    # A_R is A[:, 2:4].
    # i is still 0,1.
    # j is 2,3 (rows of B_B).
    # k is 0..6.
    
    for block_idx in range(4):
        if block_idx < 3:
            # 2x2x2
            # i in 0,1; j in 2,3; k in block_idx*2, block_idx*2+1
            add_block(U, V, W, 0, 2, block_idx*2, 2, 2, 2, col, U_s, V_s, W_s)
            col += 7
        else:
            # 2x2x1
            # j in 2,3; k in 6
            add_block(U, V, W, 0, 2, 6, 2, 2, 1, col, U_s_221, V_s_221, W_s_221)
            col += 2

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
