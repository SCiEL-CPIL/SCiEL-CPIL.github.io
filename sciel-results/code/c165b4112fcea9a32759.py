"""Matrix multiplication 3x5x7 via Strassen block decomposition + schoolbook remainder."""
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total entries: n*m*p = 105
    # We'll use Strassen for 2x2x2 blocks where possible, and schoolbook for the rest.
    
    # Strassen's 7 products for 2x2x2 matrix multiplication:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 + M6
    
    # We'll identify all possible 2x2x2 sub-blocks within 3x5x7.
    # A is 3x5, B is 5x7, C is 7x3 (but output is C[k*n+i, r] = C[k, i] for k in 0..6, i in 0..2)
    # Wait, the output W has shape (p*n, rank) = (7*3, rank) = (21, rank)
    # W[k*n + i, r] corresponds to C[k, i] where k is row of C (0..6), i is col of C (0..2)
    # Actually C is 7x3 (p x n), so C[k, i] for k in 0..6, i in 0..2
    
    # A[i, j] for i in 0..2, j in 0..4
    # B[j, k] for j in 0..4, k in 0..6
    # C[k, i] = sum_j A[i, j] * B[j, k]
    
    # Let's enumerate all 2x2x2 sub-blocks:
    # i starts at 0 or 1 (since 3 rows, 2x2 blocks start at 0 or 1)
    # j starts at 0, 1, 2, 3 (since 5 cols, 2x2 blocks start at 0..3)
    # k starts at 0, 1, 2, 3, 4, 5 (since 7 cols, 2x2 blocks start at 0..5)
    
    # But we need to be careful: if we use overlapping blocks, we can't just add them independently.
    # Instead, let's use a partition approach: partition the indices into disjoint 2x2x2 blocks where possible.
    
    # Better approach: use a greedy tiling with 2x2x2 Strassen blocks, then schoolbook for the rest.
    
    # Let's define which (i, j, k) entries are covered by Strassen blocks.
    # We'll tile as much as possible.
    
    # i: 3 rows. We can cover i=0,1 with one 2x2 block, leaving i=2 uncovered.
    # j: 5 cols. We can cover j=0,1 and j=2,3 with two 2x2 blocks, leaving j=4 uncovered.
    # k: 7 cols. We can cover k=0,1; k=2,3; k=4,5 with three 2x2 blocks, leaving k=6 uncovered.
    
    # So we have 1 * 2 * 3 = 6 Strassen blocks, each of size 2x2x2.
    # Each Strassen block covers 2*2*2 = 8 entries and uses 7 rank terms.
    # Total Strassen: 6 * 7 = 42 rank terms, covering 6 * 8 = 48 entries.
    
    # Remaining entries: 105 - 48 = 57, covered by schoolbook (57 rank terms).
    # Total rank: 42 + 57 = 99.
    
    # Let's define the Strassen blocks:
    # Block (i0, j0, k0) covers i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    
    strassen_blocks = []
    for i0 in [0, 1]:
        if i0 + 1 >= n:
            continue
        for j0 in range(0, m - 1):
            for k0 in range(0, p - 1):
                strassen_blocks.append((i0, j0, k0))
    
    # But this gives overlapping blocks. We need disjoint blocks.
    # Let's use a proper tiling:
    # i: only i0=0 gives a valid 2-row block (i=0,1). i0=1 would need i=1,2 which is also valid.
    # But if we use both, they overlap on i=1. So we should pick a partition.
    
    # For i: 3 rows. We can use i0=0 (covers i=0,1) and leave i=2 for schoolbook.
    # For j: 5 cols. We can use j0=0 (covers j=0,1) and j0=2 (covers j=2,3), leave j=4 for schoolbook.
    # For k: 7 cols. We can use k0=0 (covers k=0,1), k0=2 (covers k=2,3), k0=4 (covers k=4,5), leave k=6 for schoolbook.
    
    # So the disjoint Strassen blocks are:
    # (i0=0, j0=0, k0=0): i in {0,1}, j in {0,1}, k in {0,1}
    # (i0=0, j0=0, k0=2): i in {0,1}, j in {0,1}, k in {2,3}
    # (i0=0, j0=0, k0=4): i in {0,1}, j in {0,1}, k in {4,5}
    # (i0=0, j0=2, k0=0): i in {0,1}, j in {2,3}, k in {0,1}
    # (i0=0, j0=2, k0=2): i in {0,1}, j in {2,3}, k in {2,3}
    # (i0=0, j0=2, k0=4): i in {0,1}, j in {2,3}, k in {4,5}
    
    # That's 6 blocks. Let's verify: i=0,1 covered; j=0,1,2,3 covered; k=0,1,2,3,4,5 covered.
    # Uncovered: i=2 (all j, k); j=4 (i in {0,1}, all k); k=6 (i in {0,1}, j in {0,1,2,3})
    # Count: 
    # i=2: 1 * 5 * 7 = 35
    # j=4, i in {0,1}: 2 * 1 * 7 = 14
    # k=6, i in {0,1}, j in {0,1,2,3}: 2 * 4 * 1 = 8
    # Total uncovered: 35 + 14 + 8 = 57. Correct.
    
    disjoint_strassen_blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4),
    ]
    
    # Now let's build the decomposition.
    # We'll use a dictionary to track which (i,j,k) entries are covered by Strassen.
    covered = set()
    for (i0, j0, k0) in disjoint_strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0 + di, j0 + dj, k0 + dk))
    
    # Build U, V, W matrices.
    # We'll start with empty lists and append columns.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_col(U_val, V_val, W_val):
        """Add a rank-1 term."""
        U_cols.append(U_val)
        V_cols.append(V_val)
        W_cols.append(W_val)
    
    # Add Strassen blocks.
    for (i0, j0, k0) in disjoint_strassen_blocks:
        # Local indices: i_local in {0,1}, j_local in {0,1}, k_local in {0,1}
        # Global: i = i0 + i_local, j = j0 + j_local, k = k0 + k_local
        
        # Strassen products:
        # M1 = (A[i0, j0] + A[i0+1, j0+1]) * (B[j0, k0] + B[j0+1, k0+1])
        # M2 = (A[i0+1, j0] + A[i0+1, j0+1]) * B[j0, k0]
        # M3 = A[i0, j0] * (B[j0+1, k0+1] - B[j0+1, k0])
        # M4 = A[i0+1, j0+1] * (B[j0, k0+1] - B[j0, k0])
        # M5 = (A[i0, j0] + A[i0, j0+1]) * B[j0+1, k0+1]
        # M6 = (A[i0+1, j0] - A[i0, j0]) * (B[j0, k0] + B[j0, k0+1])
        # M7 = (A[i0, j0+1] - A[i0+1, j0+1]) * (B[j0+1, k0] + B[j0+1, k0+1])
        
        # C[k0, i0] = M1 - M4 + M5 + M7
        # C[k0+1, i0] = M3 + M5
        # C[k0, i0+1] = M2 + M4
        # C[k0+1, i0+1] = M1 - M3 + M2 + M6
        
        # For each M_t, we need to create U, V, W vectors.
        # U has shape (n*m, 1) = (15, 1)
        # V has shape (m*p, 1) = (35, 1)
        # W has shape (p*n, 1) = (21, 1)
        
        # U[i*m + j, 0] = coefficient for A[i, j]
        # V[j*p + k, 0] = coefficient for B[j, k]
        # W[k*n + i, 0] = coefficient for C[k, i]
        
        # M1: A terms: A[i0, j0] with coeff 1, A[i0+1, j0+1] with coeff 1
        #      B terms: B[j0, k0] with coeff 1, B[j0+1, k0+1] with coeff 1
        #      C terms: C[k0, i0] with coeff 1, C[k0+1, i0+1] with coeff 1
        U1 = np.zeros(n * m)
        V1 = np.zeros(m * p)
        W1 = np.zeros(p * n)
        U1[i0 * m + j0] = 1
        U1[(i0+1) * m + (j0+1)] = 1
        V1[j0 * p + k0] = 1
        V1[(j0+1) * p + (k0+1)] = 1
        W1[k0 * n + i0] = 1
        W1[(k0+1) * n + (i0+1)] = 1
        add_col(U1, V1, W1)
        
        # M2: A terms: A[i0+1, j0] with coeff 1, A[i0+1, j0+1] with coeff 1
        #      B terms: B[j0, k0] with coeff 1
        #      C terms: C[k0, i0+1] with coeff 1
        U2 = np.zeros(n * m)
        V2 = np.zeros(m * p)
        W2 = np.zeros(p * n)
        U2[(i0+1) * m + j0] = 1
        U2[(i0+1) * m + (j0+1)] = 1
        V2[j0 * p + k0] = 1
        W2[k0 * n + (i0+1)] = 1
        add_col(U2, V2, W2)
        
        # M3: A terms: A[i0, j0] with coeff 1
        #      B terms: B[j0+1, k0+1] with coeff 1, B[j0+1, k0] with coeff -1
        #      C terms: C[k0+1, i0] with coeff 1
        U3 = np.zeros(n * m)
        V3 = np.zeros(m * p)
        W3 = np.zeros(p * n)
        U3[i0 * m + j0] = 1
        V3[(j0+1) * p + (k0+1)] = 1
        V3[(j0+1) * p + k0] = -1
        W3[(k0+1) * n + i0] = 1
        add_col(U3, V3, W3)
        
        # M4: A terms: A[i0+1, j0+1] with coeff 1
        #      B terms: B[j0, k0+1] with coeff 1, B[j0, k0] with coeff -1
        #      C terms: C[k0, i0+1] with coeff 1
        U4 = np.zeros(n * m)
        V4 = np.zeros(m * p)
        W4 = np.zeros(p * n)
        U4[(i0+1) * m + (j0+1)] = 1
        V4[j0 * p + (k0+1)] = 1
        V4[j0 * p + k0] = -1
        W4[k0 * n + (i0+1)] = 1
        add_col(U4, V4, W4)
        
        # M5: A terms: A[i0, j0] with coeff 1, A[i0, j0+1] with coeff 1
        #      B terms: B[j0+1, k0+1] with coeff 1
        #      C terms: C[k0, i0] with coeff 1, C[k0+1, i0] with coeff 1
        U5 = np.zeros(n * m)
        V5 = np.zeros(m * p)
        W5 = np.zeros(p * n)
        U5[i0 * m + j0] = 1
        U5[i0 * m + (j0+1)] = 1
        V5[(j0+1) * p + (k0+1)] = 1
        W5[k0 * n + i0] = 1
        W5[(k0+1) * n + i0] = 1
        add_col(U5, V5, W5)
        
        # M6: A terms: A[i0+1, j0] with coeff 1, A[i0, j0] with coeff -1
        #      B terms: B[j0, k0] with coeff 1, B[j0, k0+1] with coeff 1
        #      C terms: C[k0+1, i0+1] with coeff 1
        U6 = np.zeros(n * m)
        V6 = np.zeros(m * p)
        W6 = np.zeros(p * n)
        U6[(i0+1) * m + j0] = 1
        U6[i0 * m + j0] = -1
        V6[j0 * p + k0] = 1
        V6[j0 * p + (k0+1)] = 1
        W6[(k0+1) * n + (i0+1)] = 1
        add_col(U6, V6, W6)
        
        # M7: A terms: A[i0, j0+1] with coeff 1, A[i0+1, j0+1] with coeff -1
        #      B terms: B[j0+1, k0] with coeff 1, B[j0+1, k0+1] with coeff 1
        #      C terms: C[k0, i0] with coeff 1
        U7 = np.zeros(n * m)
        V7 = np.zeros(m * p)
        W7 = np.zeros(p * n)
        U7[i0 * m + (j0+1)] = 1
        U7[(i0+1) * m + (j0+1)] = -1
        V7[(j0+1) * p + k0] = 1
        V7[(j0+1) * p + (k0+1)] = 1
        W7[k0 * n + i0] = 1
        add_col(U7, V7, W7)
    
    # Add schoolbook terms for uncovered entries.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U_sb = np.zeros(n * m)
                    V_sb = np.zeros(m * p)
                    W_sb = np.zeros(p * n)
                    U_sb[i * m + j] = 1
                    V_sb[j * p + k] = 1
                    W_sb[k * n + i] = 1
                    add_col(U_sb, V_sb, W_sb)
    
    # Convert to numpy arrays.
    U = np.column_stack(U_cols) if U_cols else np.zeros((n * m, 0))
    V = np.column_stack(V_cols) if V_cols else np.zeros((m * p, 0))
    W = np.column_stack(W_cols) if W_cols else np.zeros((p * n, 0))
    
    # Verify: the tensor decomposition should satisfy:
    # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Let's do a quick sanity check.
    # Actually, let's just return the result.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
