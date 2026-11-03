"""Matrix multiplication 4x5 by 5x6 — Hybrid Strassen/schoolbook with column reduction."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """
    Build the 2x2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
    U indices: i*m+j
    V indices: j*p+k
    W indices: k*n+i
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions into one global decomposition.
    Each decomposition is a tuple: (U_sub, V_sub, W_sub, i_start, j_start, k_start, i_size, j_size, k_size)
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, i_s, j_s, k_s, i_sz, j_sz, k_sz in decomps:
        rank_sub = len(U_sub[0])
        
        # Map A indices: i in [0, i_sz), j in [0, j_sz)
        for i in range(i_sz):
            for j in range(j_sz):
                u_row_sub = i * j_sz + j
                u_row_glob = (i_s + i) * m + (j_s + j)
                for c in range(rank_sub):
                    U[u_row_glob][current_rank + c] = U_sub[u_row_sub][c]
        
        # Map B indices: j in [0, j_sz), k in [0, k_sz)
        for j in range(j_sz):
            for k in range(k_sz):
                v_row_sub = j * k_sz + k
                v_row_glob = (j_s + j) * p + (k_s + k)
                for c in range(rank_sub):
                    V[v_row_glob][current_rank + c] = V_sub[v_row_sub][c]
        
        # Map C indices: k in [0, k_sz), i in [0, i_sz)
        # Convention: W[k*n + i]
        for k in range(k_sz):
            for i in range(i_sz):
                w_row_sub = k * i_sz + i
                w_row_glob = (k_s + k) * n + (i_s + i)
                for c in range(rank_sub):
                    W[w_row_glob][current_rank + c] = W_sub[w_row_sub][c]
        
        current_rank += rank_sub
    
    return U, V, W

def _build_kronecker_strassen_4x4x4():
    """
    Build 4x4x4 Strassen decomposition using Kronecker product of 2x2x2 Strassen.
    Rank is 49.
    """
    U2, V2, W2 = _build_strassen_2x2x2()
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for t1 in range(7):
        for t2 in range(7):
            col = t1 * 7 + t2
            # U[i1*2+i2, j1*2+j2] = U2[i1,j1,t1] * U2[i2,j2,t2]
            for i1 in range(2):
                for j1 in range(2):
                    c1 = U2[i1*2+j1][t1]
                    if c1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            c2 = U2[i2*2+j2][t2]
                            if c2 == 0: continue
                            u_row = (i1*2+i2)*4 + (j1*2+j2)
                            U[u_row][col] += c1 * c2
                            
            # V[j1*2+j2, k1*2+k2] = V2[j1,k1,t1] * V2[j2,k2,t2]
            for j1 in range(2):
                for k1 in range(2):
                    c1 = V2[j1*2+k1][t1]
                    if c1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            c2 = V2[j2*2+k2][t2]
                            if c2 == 0: continue
                            v_row = (j1*2+j2)*4 + (k1*2+k2)
                            V[v_row][col] += c1 * c2
                            
            # W[k1*2+k2, i1*2+i2] = W2[k1,i1,t1] * W2[k2,i2,t2]
            for k1 in range(2):
                for i1 in range(2):
                    c1 = W2[k1*2+i1][t1]
                    if c1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            c2 = W2[k2*2+i2][t2]
                            if c2 == 0: continue
                            w_row = (k1*2+k2)*4 + (i1*2+i2)
                            W[w_row][col] += c1 * c2
                            
    return U, V, W

def _remove_redundant_columns(U, V, W, n, m, p, tol=1e-10):
    """
    Remove linearly dependent columns from the combined decomposition.
    A column is redundant if its tensor (u_col, v_col, w_col) is a linear combination
    of previously kept columns.
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    num_rows_u = U_np.shape[0]
    num_rows_v = V_np.shape[0]
    num_rows_w = W_np.shape[0]
    rank = U_np.shape[1]
    
    # We'll track which columns to keep
    keep = []
    # Store the tensor slices for kept columns
    kept_tensors = []
    
    for c in range(rank):
        # Check if this column's tensor is in the span of kept tensors
        # The tensor is (U[:,c], V[:,c], W[:,c])
        # Two tensors are linearly dependent if one is a scalar multiple of the other
        # But here we need to check if the new column is a linear combination of previous ones
        # in the tensor space
        
        # Simple approach: check if U[:,c] is in the span of kept U columns
        # This is necessary but not sufficient. For exact decomposition, we need
        # the full tensor to be in the span.
        
        # Let's use a different approach: check if the column is linearly dependent
        # in the combined matrix [U; V; W] (stacked vertically)
        # This is sufficient for our purposes since the decomposition structure
        # means that linear dependence in the stacked matrix implies tensor dependence.
        
        if len(kept_tensors) == 0:
            keep.append(c)
            kept_tensors.append((U_np[:, c].copy(), V_np[:, c].copy(), W_np[:, c].copy()))
            continue
        
        # Check if this column is linearly dependent on previous kept columns
        # by checking if U[:,c] is in the span of kept U columns
        U_kept = np.column_stack([t[0] for t in kept_tensors])
        try:
            # Solve U_kept * x = U[:,c]
            x, residuals, rank_u, s = np.linalg.lstsq(U_kept, U_np[:, c], rcond=tol)
            if residuals.size > 0 and residuals[0] < tol:
                # U column is in span, now check if V and W columns match
                V_pred = U_kept @ x  # This is wrong, we need V_kept @ x
                V_kept = np.column_stack([t[1] for t in kept_tensors])
                W_kept = np.column_stack([t[2] for t in kept_tensors])
                
                # Solve for V
                x_v, _, _, _ = np.linalg.lstsq(V_kept, V_np[:, c], rcond=tol)
                # Solve for W
                x_w, _, _, _ = np.linalg.lstsq(W_kept, W_np[:, c], rcond=tol)
                
                # Check if the same x works for all three
                # Actually, we need the tensor to be in the span, which means
                # there exist scalars x_i such that:
                # U[:,c] = sum_i x_i * U_kept[:,i]
                # V[:,c] = sum_i x_i * V_kept[:,i]
                # W[:,c] = sum_i x_i * W_kept[:,i]
                # This is a joint linear system.
                
                # Build the joint system
                # Stack [U_kept, V_kept, W_kept] vertically
                joint = np.vstack([U_kept, V_kept, W_kept])
                rhs = np.concatenate([U_np[:, c], V_np[:, c], W_np[:, c]])
                
                x_joint, res_joint, rank_joint, _ = np.linalg.lstsq(joint, rhs, rcond=tol)
                
                if res_joint.size > 0 and res_joint[0] < tol:
                    # Column is in the span, skip it
                    continue
                else:
                    keep.append(c)
                    kept_tensors.append((U_np[:, c].copy(), V_np[:, c].copy(), W_np[:, c].copy()))
            else:
                keep.append(c)
                kept_tensors.append((U_np[:, c].copy(), V_np[:, c].copy(), W_np[:, c].copy()))
        except:
            keep.append(c)
            kept_tensors.append((U_np[:, c].copy(), V_np[:, c].copy(), W_np[:, c].copy()))
    
    # Build reduced matrices
    U_reduced = [[0]*len(keep) for _ in range(num_rows_u)]
    V_reduced = [[0]*len(keep) for _ in range(num_rows_v)]
    W_reduced = [[0]*len(keep) for _ in range(num_rows_w)]
    
    for new_c, old_c in enumerate(keep):
        for i in range(num_rows_u):
            U_reduced[i][new_c] = U[i][old_c]
        for i in range(num_rows_v):
            V_reduced[i][new_c] = V[i][old_c]
        for i in range(num_rows_w):
            W_reduced[i][new_c] = W[i][old_c]
    
    return U_reduced, V_reduced, W_reduced

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    
    decomps = []
    
    # 1. Block 4x4x4 (i=0..3, j=0..3, k=0..3)
    # Using Kronecker Strassen (Rank 49)
    U444, V444, W444 = _build_kronecker_strassen_4x4x4()
    decomps.append((U444, V444, W444, 0, 0, 0, 4, 4, 4))
    
    # 2. Block 4x4x2 (i=0..3, j=0..3, k=4..5)
    # Split into four 2x2x2 Strassen blocks (Rank 4 * 7 = 28)
    U222, V222, W222 = _build_strassen_2x2x2()
    
    # Sub-block 2.1
    decomps.append((U222, V222, W222, 0, 0, 4, 2, 2, 2))
    # Sub-block 2.2
    decomps.append((U222, V222, W222, 0, 2, 4, 2, 2, 2))
    # Sub-block 2.3
    decomps.append((U222, V222, W222, 2, 0, 4, 2, 2, 2))
    # Sub-block 2.4
    decomps.append((U222, V222, W222, 2, 2, 4, 2, 2, 2))
    
    # 3. Block 4x1x6 (i=0..3, j=4, k=0..5)
    # Using Schoolbook (Rank 24)
    U416, V416, W416 = _build_schoolbook(4, 1, 6)
    decomps.append((U416, V416, W416, 0, 4, 0, 4, 1, 6))
    
    U, V, W = _combine_decompositions(decomps, n, m, p)
    
    # Remove redundant columns
    U, V, W = _remove_redundant_columns(U, V, W, n, m, p)
    
    # Verify time
    elapsed = time.time() - start_time
    if elapsed > 140:
        # Should not happen, but safety check
        pass
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
