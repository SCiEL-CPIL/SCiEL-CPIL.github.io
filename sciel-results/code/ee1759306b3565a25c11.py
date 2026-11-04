"""Matrix multiplication 2x3x4 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE = 150.0
START_TIME = time.time()

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Try to find a rank-21 decomposition by optimizing a rank-22 candidate
    # and then attempting to zero out one factor.
    
    def build_rank22():
        rank = 22
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def add_u(i, j, c, val):
            U[i * m + j, c] += val
        def add_v(j, k, c, val):
            V[j * p + k, c] += val
        def add_w(k, i, c, val):
            W[k * n + i, c] += val
            
        def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, offset):
            a00 = (a_rows[0], a_cols[0])
            a01 = (a_rows[0], a_cols[1])
            a10 = (a_rows[1], a_cols[0])
            a11 = (a_rows[1], a_cols[1])
            
            b00 = (b_rows[0], b_cols[0])
            b01 = (b_rows[0], b_cols[1])
            b10 = (b_rows[1], b_cols[0])
            b11 = (b_rows[1], b_cols[1])
            
            c00 = (c_rows[0], c_cols[0])
            c01 = (c_rows[0], c_cols[1])
            c10 = (c_rows[1], c_cols[0])
            c11 = (c_rows[1], c_cols[1])
            
            # M1 = (a00+a11)(b00+b11)
            add_u(a00[0], a00[1], offset+0, 1)
            add_u(a11[0], a11[1], offset+0, 1)
            add_v(b00[0], b00[1], offset+0, 1)
            add_v(b11[0], b11[1], offset+0, 1)
            add_w(c00[0], c00[1], offset+0, 1)
            add_w(c11[0], c11[1], offset+0, 1)
            
            # M2 = (a10+a11)(b00)
            add_u(a10[0], a10[1], offset+1, 1)
            add_u(a11[0], a11[1], offset+1, 1)
            add_v(b00[0], b00[1], offset+1, 1)
            add_w(c10[0], c10[1], offset+1, 1)
            add_w(c11[0], c11[1], offset+1, -1)
            
            # M3 = a00(b01-b11)
            add_u(a00[0], a00[1], offset+2, 1)
            add_v(b01[0], b01[1], offset+2, 1)
            add_v(b11[0], b11[1], offset+2, -1)
            add_w(c01[0], c01[1], offset+2, 1)
            add_w(c11[0], c11[1], offset+2, 1)
            
            # M4 = a11(b10-b00)
            add_u(a11[0], a11[1], offset+3, 1)
            add_v(b10[0], b10[1], offset+3, 1)
            add_v(b00[0], b00[1], offset+3, -1)
            add_w(c00[0], c00[1], offset+3, -1)
            add_w(c10[0], c10[1], offset+3, 1)
            
            # M5 = (a00+a01)(b11)
            add_u(a00[0], a00[1], offset+4, 1)
            add_u(a01[0], a01[1], offset+4, 1)
            add_v(b11[0], b11[1], offset+4, 1)
            add_w(c00[0], c00[1], offset+4, -1)
            add_w(c01[0], c01[1], offset+4, 1)
            
            # M6 = (a10-a00)(b00+b10)
            add_u(a10[0], a10[1], offset+5, 1)
            add_u(a00[0], a00[1], offset+5, -1)
            add_v(b00[0], b00[1], offset+5, 1)
            add_v(b10[0], b10[1], offset+5, 1)
            add_w(c11[0], c11[1], offset+5, 1)
            
            # M7 = (a01-a11)(b10+b11)
            add_u(a01[0], a01[1], offset+6, 1)
            add_u(a11[0], a11[1], offset+6, -1)
            add_v(b10[0], b10[1], offset+6, 1)
            add_v(b11[0], b11[1], offset+6, 1)
            add_w(c00[0], c00[1], offset+6, 1)
            
        add_strassen_block([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0)
        add_strassen_block([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], 7)
        
        offset = 14
        for i in range(2):
            for k in range(4):
                add_u(i, 2, offset, 1)
                add_v(2, k, offset, 1)
                add_w(k, i, offset, 1)
                offset += 1
                
        return U, V, W
    
    def verify(U, V, W):
        R = U.shape[1]
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        tensor = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        expected = np.ones((n, m, p))
        if not np.allclose(tensor, expected, atol=1e-9):
            return False
        return True
    
    # Start with the rank-22 decomposition
    U, V, W = build_rank22()
    
    # Try to reduce to rank 21 by optimizing and sparsifying
    if time.time() - START_TIME < DEADLINE - 30:
        try:
            # Apply a random orthogonal transformation to the factor space
            # to see if it leads to a sparser representation
            best_rank = 22
            best_factors = (U, V, W)
            
            for attempt in range(5):
                if time.time() - START_TIME > DEADLINE - 30:
                    break
                
                # Random rotation in the factor space
                R_rot = np.eye(22)
                # Apply a few random Givens rotations
                for _ in range(10):
                    i, j = np.random.choice(22, 2, replace=False)
                    theta = np.random.uniform(0, 2*np.pi)
                    c, s = np.cos(theta), np.sin(theta)
                    G = np.eye(22)
                    G[i,i] = c
                    G[j,j] = c
                    G[i,j] = -s
                    G[j,i] = s
                    R_rot = R_rot @ G
                
                U_rot = U @ R_rot
                V_rot = V @ R_rot
                W_rot = W @ R_rot
                
                # Try to zero out small entries
                for thresh in [0.01, 0.005, 0.001]:
                    U_sp = np.where(np.abs(U_rot) < thresh, 0, U_rot)
                    V_sp = np.where(np.abs(V_rot) < thresh, 0, V_rot)
                    W_sp = np.where(np.abs(W_rot) < thresh, 0, W_rot)
                    
                    # Check if we can drop a column
                    for col in range(22):
                        if np.allclose(U_sp[:, col], 0) or np.allclose(V_sp[:, col], 0) or np.allclose(W_sp[:, col], 0):
                            # Drop this column
                            mask = [i for i in range(22) if i != col]
                            U_new = U_sp[:, mask]
                            V_new = V_sp[:, mask]
                            W_new = W_sp[:, mask]
                            if verify(U_new, V_new, W_new):
                                best_rank = 21
                                best_factors = (U_new, V_new, W_new)
                                break
                        if best_rank == 21:
                            break
                    if best_rank == 21:
                        break
                
                if best_rank == 21:
                    break
        
        except Exception:
            pass
    
    if best_rank == 21:
        U, V, W = best_factors
    else:
        # Use the rank-22 decomposition
        U, V, W = build_rank22()
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook rank 24
        U_school = np.zeros((n * m, 24))
        V_school = np.zeros((m * p, 24))
        W_school = np.zeros((p * n, 24))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, idx] = 1
                    V_school[j * p + k, idx] = 1
                    W_school[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
