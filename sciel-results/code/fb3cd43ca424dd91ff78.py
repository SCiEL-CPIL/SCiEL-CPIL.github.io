import numpy as np
import time
import scipy.optimize as opt

def construct():
    n, m, p = 2, 3, 8
    
    # Target tensor for 2x3x2 block
    # U: 6xR, V: 6xR, W: 4xR
    # T[u, v, w] = 1 if u=i*3+j, v=j*2+k, w=k*2+i
    target_232 = np.zeros((6, 6, 4))
    for i in range(2):
        for j in range(3):
            for k in range(2):
                target_232[i*3+j, j*2+k, k*2+i] = 1.0

    def get_rank_11_block():
        # Strassen for 2x2x2 part + Schoolbook for 2x1x2 part
        # Split A into A1 (2x2) and A2 (2x1)
        # Split B into B1 (2x2) and B2 (1x2)
        # C = A1 B1 + A2 B2
        
        R = 11
        U = np.zeros((6, R))
        V = np.zeros((6, R))
        W = np.zeros((4, R))
        
        col = 0
        # Strassen for A1 (rows 0,1 of A -> indices 0,1? No, A is 2x3. A1 is first 2 cols)
        # A indices: 0,1 (col 0), 2,3 (col 1)? No.
        # A is 2x3. Rows i=0,1. Cols j=0,1,2.
        # A1 uses j=0,1. Indices: i*3+j.
        # i=0, j=0 -> 0. i=0, j=1 -> 1.
        # i=1, j=0 -> 3. i=1, j=1 -> 4.
        # B1 uses j=0,1. B is 3x2. Rows j=0,1,2. Cols k=0,1.
        # B1 uses rows j=0,1. Indices: j*2+k.
        # j=0, k=0 -> 0. j=0, k=1 -> 1.
        # j=1, k=0 -> 2. j=1, k=1 -> 3.
        
        # Strassen 2x2x2 on A1, B1
        # a00=A[0,0]=0, a01=A[0,1]=1, a10=A[1,0]=3, a11=A[1,1]=4
        # b00=B[0,0]=0, b01=B[0,1]=1, b10=B[1,0]=2, b11=B[1,1]=3
        
        # 1. (a00+a11)(b00+b11)
        U[0, col] = 1; U[4, col] = 1
        V[0, col] = 1; V[3, col] = 1
        W[0, col] = 1; W[3, col] = 1
        col += 1
        
        # 2. (a10+a11)b00
        U[3, col] = 1; U[4, col] = 1
        V[0, col] = 1
        W[2, col] = 1; W[3, col] = -1
        col += 1
        
        # 3. a00(b01-b11)
        U[0, col] = 1
        V[1, col] = 1; V[3, col] = -1
        W[1, col] = 1; W[3, col] = 1
        col += 1
        
        # 4. a11(b10-b00)
        U[4, col] = 1
        V[2, col] = 1; V[0, col] = -1
        W[0, col] = 1; W[2, col] = 1
        col += 1
        
        # 5. (a00+a01)b11
        U[0, col] = 1; U[1, col] = 1
        V[3, col] = 1
        W[0, col] = -1; W[1, col] = 1
        col += 1
        
        # 6. (a10-a00)(b00+b01)
        U[3, col] = 1; U[0, col] = -1
        V[0, col] = 1; V[1, col] = 1
        W[3, col] = 1
        col += 1
        
        # 7. (a01-a11)(b10+b11)
        U[1, col] = 1; U[4, col] = -1
        V[2, col] = 1; V[3, col] = 1
        W[0, col] = 1
        col += 1
        
        # Schoolbook for A2 (2x1) * B2 (1x2)
        # A2 is col 2 of A. Indices: i*3+2.
        # i=0 -> 2. i=1 -> 5.
        # B2 is row 2 of B. Indices: 2*2+k.
        # k=0 -> 4. k=1 -> 5.
        
        # 8. A[0,2] * B[2,0]
        U[2, col] = 1
        V[4, col] = 1
        W[0, col] = 1
        col += 1
        
        # 9. A[0,2] * B[2,1]
        U[2, col] = 1
        V[5, col] = 1
        W[1, col] = 1
        col += 1
        
        # 10. A[1,2] * B[2,0]
        U[5, col] = 1
        V[4, col] = 1
        W[2, col] = 1
        col += 1
        
        # 11. A[1,2] * B[2,1]
        U[5, col] = 1
        V[5, col] = 1
        W[3, col] = 1
        col += 1
        
        return U, V, W

    # Verify rank 11
    U11, V11, W11 = get_rank_11_block()
    T11 = np.einsum('ur,vr,wr->uvw', U11, V11, W11)
    if not np.allclose(T11, target_232):
        # Fallback to full 44
        pass

    # Try to find rank 10
    target_rank = 10
    best_result = None
    best_norm = np.inf
    
    deadline = time.time() + 140
    
    # Initial guess: Rank 11 with one column removed and randomized?
    # Or better, use the structure.
    # We can try dropping each of the 11 terms and solving.
    
    for drop_idx in range(11):
        if time.time() > deadline - 10:
            break
            
        U_init = np.delete(U11, drop_idx, axis=1)
        V_init = np.delete(V11, drop_idx, axis=1)
        W_init = np.delete(W11, drop_idx, axis=1)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def residuals(params):
            U = params[:target_rank * 6].reshape(target_rank, 6).T
            V = params[target_rank * 6 : target_rank * 12].reshape(target_rank, 6).T
            W = params[target_rank * 12 :].reshape(target_rank, 4).T
            T = np.einsum('ur,vr,wr->uvw', U, V, W)
            return (T - target_232).flatten()
        
        try:
            result = opt.least_squares(residuals, x0, method='trf', max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if result.success and np.linalg.norm(result.fun) < 1e-8:
                if np.linalg.norm(result.fun) < best_norm:
                    best_norm = np.linalg.norm(result.fun)
                    best_result = result
        except:
            continue

    if best_result is not None and best_norm < 1e-8:
        U_block = best_result.x[:target_rank * 6].reshape(target_rank, 6).T
        V_block = best_result.x[target_rank * 6 : target_rank * 12].reshape(target_rank, 6).T
        W_block = best_result.x[target_rank * 12 :].reshape(target_rank, 4).T
        
        # Construct full solution
        # 4 blocks
        U_full = np.zeros((6, 4 * target_rank))
        V_full = np.zeros((24, 4 * target_rank))
        W_full = np.zeros((16, 4 * target_rank))
        
        for b in range(4):
            # Map block indices to full indices
            # A is same for all blocks? No, A is 2x3.
            # The block corresponds to columns 2b, 2b+1 of B.
            # So V indices need shifting.
            # B is 3x8. Block b uses rows 0,1,2 of B? No.
            # B is 3x8.
            # Block b corresponds to B[:, 2b:2b+2].
            # So V indices: j*8 + k.
            # In block, V index was j*2 + k.
            # Mapping: v_block = j*2 + k -> v_full = j*8 + 2b + k.
            
            U_full[:, b*target_rank:(b+1)*target_rank] = U_block
            
            for r in range(target_rank):
                for j in range(3):
                    for k in range(2):
                        v_idx_block = j*2 + k
                        v_idx_full = j*8 + 2*b + k
                        V_full[v_idx_full, b*target_rank + r] = V_block[v_idx_block, r]
            
            # W indices: k*2 + i.
            # In full, W index is k*2 + i? No.
            # W is p*n = 8*2 = 16.
            # Index is k*2 + i.
            # In block, W index is k*2 + i.
            # But k is local 0,1. Global k is 2b, 2b+1.
            # So w_block = k_local*2 + i.
            # w_full = k_global*2 + i = (2b + k_local)*2 + i = 4b + 2*k_local + i.
            
            for r in range(target_rank):
                for k_local in range(2):
                    for i in range(2):
                        w_idx_block = k_local*2 + i
                        w_idx_full = (2*b + k_local)*2 + i
                        W_full[w_idx_full, b*target_rank + r] = W_block[w_idx_block, r]
                        
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback to rank 44
        # Re-use the logic from previous program but optimized
        def get_rank_44():
            rank = 44
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            col = 0
            # Part 1: A[:,0] (2x1) * B[0,:] (1x8) -> 2x8
            for i in range(2):
                for k in range(8):
                    U[i * 3 + 0][col] = 1
                    V[0 * 8 + k][col] = 1
                    W[k * 2 + i][col] = 1
                    col += 1
            
            # Part 2: A[:,1:3] (2x2) * B[1:3,:] (2x8) -> 2x8
            for block in range(4):
                b00 = 1*8 + block*2
                b01 = 1*8 + block*2 + 1
                b10 = 2*8 + block*2
                b11 = 2*8 + block*2 + 1
                
                U[1][col] = 1; U[5][col] = 1
                V[b00][col] = 1; V[b11][col] = 1
                W[(block*2)*2 + 0][col] = 1
                W[(block*2+1)*2 + 1][col] = 1
                col += 1
                
                U[4][col] = 1; U[5][col] = 1
                V[b00][col] = 1
                W[(block*2)*2 + 1][col] = 1
                W[(block*2+1)*2 + 1][col] = -1
                col += 1
                
                U[1][col] = 1
                V[b01][col] = 1; V[b11][col] = -1
                W[(block*2+1)*2 + 0][col] = 1
                W[(block*2+1)*2 + 1][col] = 1
                col += 1
                
                U[5][col] = 1
                V[b10][col] = 1; V[b00][col] = -1
                W[(block*2)*2 + 0][col] = 1
                W[(block*2)*2 + 1][col] = 1
                col += 1
                
                U[1][col] = 1; U[2][col] = 1
                V[b11][col] = 1
                W[(block*2)*2 + 0][col] = -1
                W[(block*2+1)*2 + 0][col] = 1
                col += 1
                
                U[4][col] = 1; U[1][col] = -1
                V[b00][col] = 1; V[b01][col] = 1
                W[(block*2+1)*2 + 1][col] = 1
                col += 1
                
                U[2][col] = 1; U[5][col] = -1
                V[b10][col] = 1; V[b11][col] = 1
                W[(block*2)*2 + 0][col] = 1
                col += 1
                
            return {"U": U, "V": V, "W": W}
            
        return get_rank_44()
