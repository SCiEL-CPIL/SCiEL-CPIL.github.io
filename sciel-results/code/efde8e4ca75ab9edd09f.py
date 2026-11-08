import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_idx, v_idx, w_idx, U_block, V_block, W_block):
        rank = U_block.shape[1]
        for r in range(rank):
            U_parts.append(np.zeros(n * m))
            for i, idx in enumerate(u_idx):
                U_parts[-1][idx] = U_block[i, r]
            
            V_parts.append(np.zeros(m * p))
            for i, idx in enumerate(v_idx):
                V_parts[-1][idx] = V_block[i, r]
            
            W_parts.append(np.zeros(p * n))
            for i, idx in enumerate(w_idx):
                W_parts[-1][idx] = W_block[i, r]

    # Block 1: 3x3x3 Schoolbook
    # A: 3x3, B: 3x3, C: 3x3
    # U: 9x27, V: 9x27, W: 9x27
    def schoolbook_3x3x3():
        U = np.zeros((9, 27))
        V = np.zeros((9, 27))
        W = np.zeros((9, 27))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 3 + j * 3 + k # This is not right, rank is 27
                    # Actually, we need 27 terms.
                    # Term (i, j, k) contributes to C[i, k]
                    # A[i, j] * B[j, k]
                    # U index: i*3+j
                    # V index: j*3+k
                    # W index: k*3+i (W[k*n+i])
                    pass
        # Let's just fill it properly
        r = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j, r] = 1
                    V[j*3+k, r] = 1
                    W[k*3+i, r] = 1
                    r += 1
        return U, V, W

    # Block 2: 3x3x2 Rank 3
    # A: 3x3, B: 3x2, C: 3x2
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Terms:
    # j=0: A[:, 0] * B[0, :]
    # j=1: A[:, 1] * B[1, :]
    # j=2: A[:, 2] * B[2, :]
    # Each term is rank 1.
    def rank3_3x3x2():
        U = np.zeros((9, 3))
        V = np.zeros((6, 3))
        W = np.zeros((6, 3))
        for j in range(3):
            # U: A[:, j]
            for i in range(3):
                U[i*3+j, j] = 1
            # V: B[j, :]
            for k in range(2):
                V[j*3+k, j] = 1 # B is 3x2, flat index j*2+k? No, B is 3x2.
                # B indices: 0,1,2,3,4,5.
                # B[j, 0] is j*2+0. B[j, 1] is j*2+1.
                # Wait, V shape is (m*p, rank). m=3, p=2? No, this is a sub-block.
                # The global V index mapping is handled by add_block.
                # So local V index 0..5.
                # B[j, k] local index j*2+k.
                pass
        # Let's rewrite properly
        U = np.zeros((9, 3))
        V = np.zeros((6, 3))
        W = np.zeros((6, 3))
        for j in range(3):
            for i in range(3):
                U[i*3+j, j] = 1
            for k in range(2):
                V[j*2+k, j] = 1
                W[k*3+i, j] = 1 # W index k*n+i. n=3.
                # Wait, W is for C. C is 3x2.
                # C[i, k]. W index k*3+i.
                # But i varies.
                # The term is sum over i?
                # No, the term is A[:, j] * B[j, :].
                # This produces a 3x2 matrix.
                # C[i, k] gets A[i, j] * B[j, k].
                # So for a fixed j, the contribution to C is the outer product of A[:, j] and B[j, :].
                # So W should have 1s at all (i, k) positions?
                # No, W is a vector.
                # The rank-1 term is u * v * w.
                # u is A[:, j]. v is B[j, :].
                # w is the all-ones vector for C?
                # No, the term is $u \otimes v \otimes w$.
                # $C = \sum_r u_r v_r w_r$.
                # Here $u_r = A[:, j]$, $v_r = B[j, :]$.
                # $w_r$ must be such that $u_r v_r w_r$ gives the matrix.
                # $u_r v_r$ is a scalar? No, $u_r$ is a vector, $v_r$ is a vector.
                # The product of two vectors is a matrix (outer product).
                # So $w_r$ must be a scalar 1?
                # But W is a matrix of shape (p*n, rank).
                # So $w_r$ is a vector of length $p*n$.
                # If $w_r$ is all 1s, then $u_r \otimes v_r \otimes w_r$ is not a matrix.
                # The tensor decomposition is $T = \sum u_r \otimes v_r \otimes w_r$.
                # $T_{i, j, k} = \sum u_{i, r} v_{j, r} w_{k, r}$.
                # Here $i$ is A row, $j$ is B index, $k$ is C index.
                # For the term $j$, $u_{i, r} = 1$ if $A$ index is $(i, j)$.
                # $v_{b, r} = 1$ if $B$ index is $(j, k)$.
                # $w_{c, r} = 1$ if $C$ index is $(i, k)$.
                # So $w_r$ has 1s at all $(i, k)$?
                # No, $w_{c, r}$ depends on $i, k$.
                # But $r$ is fixed for the term $j$.
                # So $w_r$ must be a vector that is 1 for all $(i, k)$?
                # Then $\sum_i, k u_{i, r} v_{j, r} w_{c, r}$?
                # The sum is over $r$.
                # For a specific $r$ (corresponding to $j$),
                # $T_{i, b, c} = u_{i, r} v_{b, r} w_{c, r}$.
                # $u_{i, r} = 1$ if $i$ is row and $j$ is col?
                # $A$ index is $(i, j)$.
                # So $u_{(i, j), r} = 1$.
                # $B$ index is $(j, k)$.
                # So $v_{(j, k), r} = 1$.
                # $C$ index is $(i, k)$.
                # So $w_{(i, k), r} = 1$.
                # So yes, $w_r$ has 1s at all $(i, k)$ pairs.
                pass
        
        # Correct implementation
        U = np.zeros((9, 3))
        V = np.zeros((6, 3))
        W = np.zeros((6, 3))
        for j in range(3):
            for i in range(3):
                U[i*3+j, j] = 1
            for k in range(2):
                V[j*2+k, j] = 1
            for i in range(3):
                for k in range(2):
                    W[k*3+i, j] = 1
        return U, V, W

    # Block 3: 3x1x5 Rank 3
    # A: 3x1, B: 1x5, C: 3x5
    # C[i, k] = A[i, 0] * B[0, k]
    # Terms:
    # i=0: A[0, 0] * B[0, :]
    # i=1: A[1, 0] * B[0, :]
    # i=2: A[2, 0] * B[0, :]
    def rank3_3x1x5():
        U = np.zeros((3, 3))
        V = np.zeros((5, 3))
        W = np.zeros((15, 3))
        for i in range(3):
            U[i, i] = 1
            for k in range(5):
                V[k, i] = 1
                W[k*3+i, i] = 1
        return U, V, W

    # Assemble
    # Block 1: 3x3x3
    U_blk, V_blk, W_blk = schoolbook_3x3x3()
    u_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8] # A[0:3, 0:3]
    v_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8] # B[0:3, 0:3]
    w_idx = [0, 3, 6, 1, 4, 7, 2, 5, 8] # C[0:3, 0:3] -> k*3+i
    # C[0,0]=0, C[1,0]=3, C[2,0]=6
    # C[0,1]=1, C[1,1]=4, C[2,1]=7
    # C[0,2]=2, C[1,2]=5, C[2,2]=8
    # w_idx order must match W_block order.
    # W_block is filled with k*3+i.
    # So w_idx is just [0..8] mapped to global.
    # Global W index for C[i, k] is k*3+i.
    # Local W index is k*3+i.
    # So w_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8].
    add_block(u_idx, v_idx, [0, 1, 2, 3, 4, 5, 6, 7, 8], U_blk, V_blk, W_blk)

    # Block 2: 3x3x2
    U_blk, V_blk, W_blk = rank3_3x3x2()
    u_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8] # A[0:3, 0:3]
    v_idx = [3, 4, 6, 7, 9, 10] # B[0:3, 3:5]
    # B[0,3]=3, B[0,4]=4
    # B[1,3]=6, B[1,4]=7
    # B[2,3]=9, B[2,4]=10
    # Local V index j*2+k.
    # j=0, k=0 -> 0 -> Global 3
    # j=0, k=1 -> 1 -> Global 4
    # j=1, k=0 -> 2 -> Global 6
    # j=1, k=1 -> 3 -> Global 7
    # j=2, k=0 -> 4 -> Global 9
    # j=2, k=1 -> 5 -> Global 10
    v_idx = [3, 4, 6, 7, 9, 10]
    w_idx = [9, 12, 10, 13, 11, 14] # C[0:3, 3:5]
    # C[0,3]=9, C[1,3]=12, C[2,3]=15? No, C is 3x5.
    # C[i, k] index k*3+i.
    # C[0,3] = 3*3+0 = 9.
    # C[1,3] = 3*3+1 = 10.
    # C[2,3] = 3*3+2 = 11.
    # C[0,4] = 4*3+0 = 12.
    # C[1,4] = 4*3+1 = 13.
    # C[2,4] = 4*3+2 = 14.
    # Local W index k*3+i.
    # k=0 (local), i=0 -> 0 -> Global 9
    # k=0, i=1 -> 1 -> Global 10
    # k=0, i=2 -> 2 -> Global 11
    # k=1, i=0 -> 3 -> Global 12
    # k=1, i=1 -> 4 -> Global 13
    # k=1, i=2 -> 5 -> Global 14
    w_idx = [9, 10, 11, 12, 13, 14]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)

    # Block 3: 3x1x5
    U_blk, V_blk, W_blk = rank3_3x1x5()
    u_idx = [3, 4, 5] # A[0:3, 3]
    # A[i, 3] index i*4+3.
    # i=0 -> 3
    # i=1 -> 7
    # i=2 -> 11
    u_idx = [3, 7, 11]
    v_idx = [12, 13, 14, 15, 16] # B[3, 0:5]
    # B[3, k] index 3*5+k = 15+k.
    # 15, 16, 17, 18, 19.
    v_idx = [15, 16, 17, 18, 19]
    w_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] # C[0:3, 0:5]
    # Local W index k*3+i.
    # Global W index k*3+i.
    # So w_idx is [0..14].
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)

    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
