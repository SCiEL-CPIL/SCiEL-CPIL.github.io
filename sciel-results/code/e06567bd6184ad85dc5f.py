"""Matrix multiplication 2x4x7 - rank-46 decomposition attempt using 2x2x3 rank-9 blocks."""
DIMENSIONS = [2, 4, 7]

import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Fallback: Rank 50
    def get_rank_50():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1
            U[i0*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M2
            U[(i0+1)*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + (i0+1)][col] += 1; W[(k0+1)*n + (i0+1)][col] -= 1
            col += 1
            # M3
            U[i0*m + j0][col] += 1
            V[j0*p + (k0+1)][col] += 1; V[(j0+1)*p + (k0+1)][col] -= 1
            W[(k0+1)*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M4
            U[(i0+1)*m + (j0+1)][col] += 1
            V[(j0+1)*p + k0][col] += 1; V[j0*p + k0][col] -= 1
            W[k0*n + i0][col] += 1; W[k0*n + (i0+1)][col] += 1
            col += 1
            # M5
            U[i0*m + j0][col] += 1; U[i0*m + (j0+1)][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] -= 1; W[(k0+1)*n + i0][col] += 1
            col += 1
            # M6
            U[(i0+1)*m + j0][col] += 1; U[i0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1; V[j0*p + (k0+1)][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M7
            U[i0*m + (j0+1)][col] += 1; U[(i0+1)*m + (j0+1)][col] -= 1
            V[(j0+1)*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1
            col += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 2)
        add_strassen(0, 0, 4)
        add_strassen(0, 2, 4)
        for j in range(4):
            for i in range(2):
                U[i*m + j][col] = 1
                V[j*p + 6][col] = 1
                W[6*n + i][col] = 1
                col += 1
        return {"U": U, "V": V, "W": W}

    # Attempt Rank 46
    def find_rank_9_223():
        # T for 2x2x3
        # A: 2x2 (4), B: 2x3 (6), C: 2x3 (6)
        # T[i,j,k] = 1 if i==j and k is col? No.
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Terms:
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
        # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
        # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
        # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
        # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
        
        # Flatten indices:
        # A: i*2 + j
        # B: j*3 + k
        # C: i*3 + k
        
        target = np.zeros((4, 6, 6))
        # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
        # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[1,0]=3, B[1,1]=4, B[1,2]=5
        # C[0,0]=0, C[0,1]=1, C[0,2]=2, C[1,0]=3, C[1,1]=4, C[1,2]=5
        
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        target[0, 0, 0] = 1
        target[1, 3, 0] = 1
        # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
        target[0, 1, 1] = 1
        target[1, 4, 1] = 1
        # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
        target[0, 2, 2] = 1
        target[1, 5, 2] = 1
        # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
        target[2, 0, 3] = 1
        target[3, 3, 3] = 1
        # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
        target[2, 1, 4] = 1
        target[3, 4, 4] = 1
        # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
        target[2, 2, 5] = 1
        target[3, 5, 5] = 1
        
        rank = 9
        best_err = float('inf')
        best_U = best_V = best_W = None
        
        np.random.seed(42)
        
        for _ in range(20): # 20 restarts
            U = np.random.randn(4, rank)
            V = np.random.randn(6, rank)
            W = np.random.randn(6, rank)
            
            for it in range(50):
                # Update U
                # Residual R = target - sum(V[:,r] outer W[:,r])
                # U[:,r] = argmin || U[:,r] (V[:,r] outer W[:,r]) - target ||
                # This is a linear system for each r
                # Flatten target to (4, 36)
                # V_outer_W is (36, rank)
                # U is (4, rank)
                # term = U @ V_outer_W.T ? No.
                # T = sum_r U[:,r] outer V[:,r] outer W[:,r]
                # T.flatten(1,2) = sum_r U[:,r] outer (V[:,r] * W[:,r])
                # So U = T.flatten(1,2) @ (V*W).pinv()
                
                VW = (V * W).T # (rank, 36) -> (36, rank) is (V*W)
                # Actually, (V[:,r] * W[:,r]) is a vector of length 36.
                # Let M = V * W (element-wise), shape (6, 6, rank) -> reshape to (36, rank)
                M = (V * W).reshape(36, rank)
                T_flat = target.reshape(4, 36)
                U_new = T_flat @ np.linalg.pinv(M)
                
                # Update V
                # T_flat = U @ (V*W).T ?
                # T[i, v*w] = sum_r U[i,r] V[v,r] W[w,r]
                # For fixed r, V[:,r] = argmin || U[:,r] outer V[:,r] outer W[:,r] - target ||
                # This is equivalent to solving for V given U, W.
                # T.reshape(4, 6, 6)
                # Let A_r = U[:,r] (4,1), B_r = W[:,r] (6,1)
                # Term is A_r outer V[:,r] outer B_r
                # Reshape to (4, 36)
                # T_flat = sum_r U[:,r] outer (V[:,r] * W[:,r])
                # We just did this for U.
                # Now fix U, W. Update V.
                # T_flat = U @ (V*W).T
                # This is not separable for V easily in this form.
                
                # Alternative: Use ALS on the 3-mode tensor.
                # Unfold T along mode 1: T_(1) = (4, 36)
                # T_(1) = U * (V * W) (Hadamard product in rank space? No)
                # T_(1)[:, j] = sum_r U[:,r] * (V[:,r] * W[:,r])[j]
                # So T_(1) = U @ diag( (V*W)[j,r] )? No.
                # T_(1) = sum_r U[:,r] outer (V[:,r] * W[:,r])
                # This is a linear equation for U if V,W fixed.
                # For V:
                # T_(2) = (6, 24) (Mode 2 is 6, Mode 3 is 6, Mode 1 is 4)
                # T_(2)[:, i*6 + k] = sum_r V[:,r] * (U[:,r] * W[:,r])[i]
                # Let A_r = U[:,r] * W[:,r] (4,6) -> flatten to 24
                # T_(2) = V @ A.T
                # V = T_(2) @ A.pinv()
                
                # Let's implement this properly.
                
                # Update U
                M_VW = (V * W).reshape(36, rank)
                U = target.reshape(4, 36) @ np.linalg.pinv(M_VW)
                
                # Update V
                # T_(2) shape (6, 24)
                # T_(2)[:, idx] = sum_r V[:,r] * (U[:,r] * W[:,r])[idx]
                # idx = i*6 + k
                # (U[:,r] * W[:,r]) is (4,6) -> reshape to (24,1)
                M_UW = (U[:, :, None] * W[None, :, :]).reshape(24, rank) # (24, rank)
                T_2 = target.transpose(1, 0, 2).reshape(6, 24) # (6, 24)
                V = T_2 @ np.linalg.pinv(M_UW)
                
                # Update W
                # T_(3) shape (6, 24)
                # T_(3)[:, idx] = sum_r W[:,r] * (U[:,r] * V[:,r])[idx]
                # idx = i*6 + j
                M_UV = (U[:, :, None] * V[None, :, :]).reshape(24, rank)
                T_3 = target.transpose(2, 0, 1).reshape(6, 24)
                W = T_3 @ np.linalg.pinv(M_UV)
                
            # Check error
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(recon - target)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U, V, W
        
        if best_err < 1e-6:
            return best_U, best_V, best_W
        return None

    # Try to find rank 9
    res = find_rank_9_223()
    
    if res is not None:
        U_223, V_223, W_223 = res
        # Construct 2x4x7
        # A: 2x4, B: 4x7, C: 2x7
        # Split A into A1 (cols 0,1), A2 (cols 2,3)
        # Split B into B1 (rows 0,1), B2 (rows 2,3)
        # C = A1 B1 + A2 B2
        # B1 is 2x7, B2 is 2x7
        # Split B1 into B1a (cols 0-3), B1b (cols 4-6)
        # B1a is 2x4, B1b is 2x3
        # A1 B1a is 2x2x4 (Rank 14)
        # A1 B1b is 2x2x3 (Rank 9)
        # Same for A2 B2
        
        rank = 46
        U = [[0] * rank for _ in range(8)]
        V = [[0] * rank for _ in range(28)]
        W = [[0] * rank for _ in range(14)]
        
        col = 0
        
        def add_224_block(i0, j0, k0):
            # 2x2x4 block using two Strassen 2x2x2
            nonlocal col
            # Strassen 1: k0, k0+1
            # Map 2x2x2 indices to global
            # A: i0, i0+1; j0, j0+1
            # B: j0, j0+1; k0, k0+1
            # C: i0, i0+1; k0, k0+1
            
            # We can reuse the logic from get_rank_50 but for 2x2x2
            # M1
            U[i0*4 + j0][col] += 1; U[(i0+1)*4 + (j0+1)][col] += 1
            V[j0*7 + k0][col] += 1; V[(j0+1)*7 + (k0+1)][col] += 1
            W[k0*2 + i0][col] += 1; W[(k0+1)*2 + (i0+1)][col] += 1
            col += 1
            # M2
            U[(i0+1)*4 + j0][col] += 1; U[(i0+1)*4 + (j0+1)][col] += 1
            V[j0*7 + k0][col] += 1
            W[k0*2 + (i0+1)][col] += 1; W[(k0+1)*2 + (i0+1)][col] -= 1
            col += 1
            # M3
            U[i0*4 + j0][col] += 1
            V[j0*7 + (k0+1)][col] += 1; V[(j0+1)*7 + (k0+1)][col] -= 1
            W[(k0+1)*2 + i0][col] += 1; W[(k0+1)*2 + (i0+1)][col] += 1
            col += 1
            # M4
            U[(i0+1)*4 + (j0+1)][col] += 1
            V[(j0+1)*7 + k0][col] += 1; V[j0*7 + k0][col] -= 1
            W[k0*2 + i0][col] += 1; W[k0*2 + (i0+1)][col] += 1
            col += 1
            # M5
            U[i0*4 + j0][col] += 1; U[i0*4 + (j0+1)][col] += 1
            V[(j0+1)*7 + (k0+1)][col] += 1
            W[k0*2 + i0][col] -= 1; W[(k0+1)*2 + i0][col] += 1
            col += 1
            # M6
            U[(i0+1)*4 + j0][col] += 1; U[i0*4 + j0][col] -= 1
            V[j0*7 + k0][col] += 1; V[j0*7 + (k0+1)][col] += 1
            W[(k0+1)*2 + (i0+1)][col] += 1
            col += 1
            # M7
            U[i0*4 + (j0+1)][col] += 1; U[(i0+1)*4 + (j0+1)][col] -= 1
            V[(j0+1)*7 + k0][col] += 1; V[(j0+1)*7 + (k0+1)][col] += 1
            W[k0*2 + i0][col] += 1
            col += 1

        def add_223_block(i0, j0, k0):
            nonlocal col
            # Use U_223, V_223, W_223
            # A indices: i0, i0+1; j0, j0+1
            # B indices: j0, j0+1; k0, k0+1, k0+2
            # C indices: i0, i0+1; k0, k0+1, k0+2
            
            for r in range(9):
                # U_223[row, r]
                # row 0: A[i0, j0]
                # row 1: A[i0, j0+1]
                # row 2: A[i0+1, j0]
                # row 3: A[i0+1, j0+1]
                U[i0*4 + j0][col] += U_223[0, r]
                U[i0*4 + j0 + 1][col] += U_223[1, r]
                U[(i0+1)*4 + j0][col] += U_223[2, r]
                U[(i0+1)*4 + j0 + 1][col] += U_223[3, r]
                
                # V_223[row, r]
                # row 0: B[j0, k0]
                # row 1: B[j0, k0+1]
                # row 2: B[j0, k0+2]
                # row 3: B[j0+1, k0]
                # row 4: B[j0+1, k0+1]
                # row 5: B[j0+1, k0+2]
                V[j0*7 + k0][col] += V_223[0, r]
                V[j0*7 + k0 + 1][col] += V_223[1, r]
                V[j0*7 + k0 + 2][col] += V_223[2, r]
                V[(j0+1)*7 + k0][col] += V_223[3, r]
                V[(j0+1)*7 + k0 + 1][col] += V_223[4, r]
                V[(j0+1)*7 + k0 + 2][col] += V_223[5, r]
                
                # W_223[row, r]
                # row 0: C[i0, k0]
                # row 1: C[i0, k0+1]
                # row 2: C[i0, k0+2]
                # row 3: C[i0+1, k0]
                # row 4: C[i0+1, k0+1]
                # row 5: C[i0+1, k0+2]
                W[k0*2 + i0][col] += W_223[0, r]
                W[(k0+1)*2 + i0][col] += W_223[1, r]
                W[(k0+2)*2 + i0][col] += W_223[2, r]
                W[k0*2 + i0 + 1][col] += W_223[3, r]
                W[(k0+1)*2 + i0 + 1][col] += W_223[4, r]
                W[(k0+2)*2 + i0 + 1][col] += W_223[5, r]
                
                col += 1

        # A1 (i=0,1, j=0,1) * B1 (j=0,1, k=0..6)
        # B1a (k=0..3) -> 2x2x4
        add_224_block(0, 0, 0) # k=0,1
        add_224_block(0, 0, 2) # k=2,3
        # B1b (k=4..6) -> 2x2x3
        add_223_block(0, 0, 4)
        
        # A2 (i=0,1, j=2,3) * B2 (j=2,3, k=0..6)
        # B2a (k=0..3) -> 2x2x4
        add_224_block(0, 2, 0) # j=2,3; k=0,1
        add_224_block(0, 2, 2) # j=2,3; k=2,3
        # B2b (k=4..6) -> 2x2x3
        add_223_block(0, 2, 4)
        
        return {"U": U, "V": V, "W": W}
    else:
        return get_rank_50()

if __name__ == "__main__":
    res = construct()
    print(res)
