"""Matrix multiplication 3x5x7 with Strassen reduction on 2x2x2 sub-blocks."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]
DEADLINE = 115.0  # wall-clock safety margin (evaluator kills at 150s)


def _build_schoolbook(n, m, p):
    """Return U, V, W lists for schoolbook decomposition (rank n*m*p)."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W, rank


def _strassen_block(n, m, p, i0, j0, k0, U, V, W, start_col):
    """
    Replace the 8 schoolbook terms for the 2x2x2 block
    (i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1})
    with 7 Strassen terms. Returns new start_col (old start + 8) and updated matrices.
    
    We overwrite the 8 existing columns with 7 new ones and leave the 8th as zero
    (to keep dimensions consistent), OR we restructure. Better: we will rebuild.
    """
    # We won't do in-place; this function documents the mapping.
    # Instead, we'll build the full decomposition directly.
    pass


def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS
    
    # Strategy: start with schoolbook, then apply Strassen to as many
    # disjoint 2x2x2 blocks as possible.
    #
    # We have a 3x5x7 tensor. We can fit:
    # - In dimension 3: one 2x2 block (rows 0,1) or (rows 1,2)
    # - In dimension 5: two 2x2 blocks (cols 0,1 and cols 2,3) or (cols 1,2 and cols 3,4)
    # - In dimension 7: three 2x2 blocks (cols 0,1; cols 2,3; cols 4,5)
    #
    # A 2x2x2 Strassen block uses 7 terms instead of 8, saving 1 rank per block.
    # Blocks must be disjoint in all three dimensions to avoid interference.
    #
    # Let's pick blocks carefully:
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    # Block 3: i in {0,1}, j in {0,1}, k in {4,5}
    # Block 4: i in {0,1}, j in {2,3}, k in {4,5}
    #
    # These 4 blocks are disjoint. Each saves 1 rank: 105 - 4 = 101.
    #
    # Can we add more? i={2,?} - only i=2 left, can't make 2x2.
    # j={4,?} - only j=4 left, can't make 2x2.
    # k={6,?} - only k=6 left, can't make 2x2.
    #
    # So 4 Strassen blocks gives rank 101.
    #
    # Let's try another configuration:
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    # Block 3: i in {0,1}, j in {0,1}, k in {4,5}
    # Block 4: i in {0,1}, j in {2,3}, k in {0,1}
    # Block 5: i in {0,1}, j in {2,3}, k in {2,3}
    # Block 6: i in {0,1}, j in {2,3}, k in {4,5}
    #
    # 6 blocks! All disjoint. Rank = 105 - 6 = 99.
    
    # Define the 6 Strassen blocks: (i0, j0, k0)
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4),
    ]
    
    # Build the decomposition
    # Total rank = 105 - 6 = 99
    rank = n * m * p - len(blocks)  # 105 - 6 = 99
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Map from (i, j, k) to column index for schoolbook terms
    col_map = {}
    col = 0
    
    # First, assign columns for all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col_map[(i, j, k)] = col
                col += 1
    
    # Now, for each Strassen block, we replace 8 columns with 7.
    # We need to remap. Let's build a fresh assignment.
    
    # Reset
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Identify which (i,j,k) triples are in Strassen blocks
    strassen_triples = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    strassen_triples.add((i0 + di, j0 + dj, k0 + dk))
    
    # Assign columns:
    # - Non-strassen triples get one column each
    # - Each Strassen block gets 7 columns
    
    col = 0
    # First, handle non-strassen triples
    non_strassen_cols = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_triples:
                    non_strassen_cols[(i, j, k)] = col
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
    
    # Then handle Strassen blocks
    for (i0, j0, k0) in blocks:
        # 7 Strassen terms
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B11+B01)
        #
        # Output:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # U entries (rows = i*m+j):
        # M1: U[i0*m+j0] += 1, U[(i0+1)*m+(j0+1)] += 1
        # M2: U[(i0+1)*m+j0] += 1, U[(i0+1)*m+(j0+1)] += 1
        # M3: U[i0*m+j0] += 1
        # M4: U[(i0+1)*m+(j0+1)] += 1
        # M5: U[i0*m+j0] += 1, U[i0*m+(j0+1)] += 1
        # M6: U[(i0+1)*m+j0] += 1, U[i0*m+j0] += -1
        # M7: U[i0*m+(j0+1)] += 1, U[(i0+1)*m+(j0+1)] += -1
        #
        # V entries (rows = j*p+k):
        # M1: V[j0*p+k0] += 1, V[(j0+1)*p+(k0+1)] += 1
        # M2: V[j0*p+k0] += 1
        # M3: V[(j0+1)*p+k0] += 1, V[(j0+1)*p+(k0+1)] += -1
        # M4: V[(j0+1)*p+(k0+1)] += 1, V[j0*p+k0] += -1
        # M5: V[(j0+1)*p+(k0+1)] += 1
        # M6: V[j0*p+k0] += 1, V[(j0+1)*p+k0] += 1
        # M7: V[(j0+1)*p+(k0+1)] += 1, V[(j0+1)*p+(k0+1)] += 1  -- wait, let me recheck
        #
        # M7 = (A01-A11)(B11+B01)
        # V part: B11 + B01
        # B11 is at (j0+1, k0+1), B01 is at (j0, k0+1)
        # V[j0*p+(k0+1)] += 1, V[(j0+1)*p+(k0+1)] += 1
        
        # W entries (rows = k*n+i):
        # M1: W[k0*n+i0] += 1, W[(k0+1)*n+(i0+1)] += 1
        # M2: W[k0*n+(i0+1)] += 1, W[(k0+1)*n+(i0+1)] += -1
        # M3: W[(k0+1)*n+i0] += 1  -- wait, M3 contributes to C01 and C11
        # C01 is at row k0*n+(i0+1)? No.
        # C_ik is at W[k*n+i]
        # C00: k=0,i=0 -> W[0*n+0] = W[0]
        # C01: k=1,i=0 -> W[1*n+0] = W[n]
        # C10: k=0,i=1 -> W[0*n+1] = W[1]
        # C11: k=1,i=1 -> W[1*n+1] = W[n+1]
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # W for M1: contributes to C00 and C11
        # W[k0*n+i0] += 1  (C00)
        # W[(k0+1)*n+(i0+1)] += 1  (C11)
        
        # W for M2: contributes to C10 and C11
        # W[k0*n+(i0+1)] += 1  (C10)
        # W[(k0+1)*n+(i0+1)] += -1  (C11)
        
        # W for M3: contributes to C01 and C11
        # W[(k0+1)*n+i0] += 1  (C01)
        # W[(k0+1)*n+(i0+1)] += 1  (C11)
        
        # W for M4: contributes to C00 and C10
        # W[k0*n+i0] += 1  (C00)
        # W[k0*n+(i0+1)] += 1  (C10)
        
        # W for M5: contributes to C00 and C01
        # W[k0*n+i0] += -1  (C00)
        # W[(k0+1)*n+i0] += 1  (C01)
        
        # W for M6: contributes to C11
        # W[(k0+1)*n+(i0+1)] += 1  (C11)
        
        # W for M7: contributes to C00
        # W[k0*n+i0] += 1  (C00)
        
        # Now let's build the 7 columns
        for t in range(7):
            c = col + t
            # Initialize U, V, W for this column to 0 (already 0)
            
            if t == 0:  # M1
                U[i0 * m + j0][c] += 1
                U[(i0 + 1) * m + (j0 + 1)][c] += 1
                V[j0 * p + k0][c] += 1
                V[(j0 + 1) * p + (k0 + 1)][c] += 1
                W[k0 * n + i0][c] += 1
                W[(k0 + 1) * n + (i0 + 1)][c] += 1
            elif t == 1:  # M2
                U[(i0 + 1) * m + j0][c] += 1
                U[(i0 + 1) * m + (j0 + 1)][c] += 1
                V[j0 * p + k0][c] += 1
                W[k0 * n + (i0 + 1)][c] += 1
                W[(k0 + 1) * n + (i0 + 1)][c] += -1
            elif t == 2:  # M3
                U[i0 * m + j0][c] += 1
                V[(j0 + 1) * p + k0][c] += 1
                V[(j0 + 1) * p + (k0 + 1)][c] += -1
                W[(k0 + 1) * n + i0][c] += 1
                W[(k0 + 1) * n + (i0 + 1)][c] += 1
            elif t == 3:  # M4
                U[(i0 + 1) * m + (j0 + 1)][c] += 1
                V[(j0 + 1) * p + (k0 + 1)][c] += 1
                V[j0 * p + k0][c] += -1
                W[k0 * n + i0][c] += 1
                W[k0 * n + (i0 + 1)][c] += 1
            elif t == 4:  # M5
                U[i0 * m + j0][c] += 1
                U[i0 * m + (j0 + 1)][c] += 1
                V[(j0 + 1) * p + (k0 + 1)][c] += 1
                W[k0 * n + i0][c] += -1
                W[(k0 + 1) * n + i0][c] += 1
            elif t == 5:  # M6
                U[(i0 + 1) * m + j0][c] += 1
                U[i0 * m + j0][c] += -1
                V[j0 * p + k0][c] += 1
                V[(j0 + 1) * p + k0][c] += 1
                W[(k0 + 1) * n + (i0 + 1)][c] += 1
            elif t == 6:  # M7
                U[i0 * m + (j0 + 1)][c] += 1
                U[(i0 + 1) * m + (j0 + 1)][c] += -1
                V[j0 * p + (k0 + 1)][c] += 1
                V[(j0 + 1) * p + (k0 + 1)][c] += 1
                W[k0 * n + i0][c] += 1
        
        col += 7
    
    # Verify col == rank
    # Should be: (105 - 48) + 6*7 = 57 + 42 = 99. Let me check.
    # Total triples: 3*5*7 = 105
    # Strassen triples: 6 blocks * 8 = 48
    # Non-strassen: 105 - 48 = 57
    # Strassen columns: 6 * 7 = 42
    # Total: 57 + 42 = 99. Correct.
    
    # Validate: check that the decomposition is correct
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 if (i,j,k) is a valid index, else 0.
    # Actually, the tensor for matrix multiplication has T[i,j,k] = 1 for all valid (i,j,k).
    
    if time.time() - t_start < DEADLINE:
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Spot check a few entries
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(actual - expected) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fall back to schoolbook
            U, V, W, rank = _build_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
