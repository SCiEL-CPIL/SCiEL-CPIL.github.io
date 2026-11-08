import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def get_strassen_2x2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def get_schoolbook_2x2x1():
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def get_schoolbook_1x2x2():
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
    U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
    U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def get_schoolbook_2x1x2():
    U = np.zeros((2, 4))
    V = np.zeros((2, 4))
    W = np.zeros((4, 4))
    
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
    U[1, 2] = 1; V[0, 2] = 1; W[2, 2] = 1
    U[1, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
    
    return U, V, W

def get_schoolbook_1x1x2():
    U = np.zeros((1, 2))
    V = np.zeros((2, 2))
    W = np.zeros((2, 2))
    
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
    
    return U, V, W

def get_schoolbook_2x1x1():
    U = np.zeros((2, 2))
    V = np.zeros((1, 2))
    W = np.zeros((2, 2))
    
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[0, 1] = 1; W[1, 1] = 1
    
    return U, V, W

def get_schoolbook_1x1x1():
    U = np.zeros((1, 1))
    V = np.zeros((1, 1))
    W = np.zeros((1, 1))
    
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    
    return U, V, W

def get_outer_product_3x1x3():
    # A (3x1) * B (1x3) -> C (3x3)
    # Rank 3
    U = np.zeros((3, 3))
    V = np.zeros((3, 3))
    W = np.zeros((9, 3))
    
    for i in range(3):
        U[i, i] = 1
        V[0, i] = 1 # B is 1x3, so V index 0,1,2? 
        # Wait, B is 1x3. V dimension is 3.
        # B_j corresponds to V index j.
        # So V is a 3x3 matrix? No, V is (m*p, rank). m=1, p=3. m*p=3.
        # So V is 3x3.
        # Term i: u_i = A_i (e_i in U). v_i = B (e_0+e_1+e_2? No).
        # v_i must be the vector B. But B is a variable.
        # In the tensor decomposition, the vectors are fixed.
        # $T = \sum u_k v_k w_k^T$.
        # We want $C_{ij} = A_i B_j$.
        # If we choose $u_k = e_k$ (for A_k), $v_k = \sum_j e_j$ (for B), $w_k = e_k \otimes I$?
        # No.
        # Term 1: $u_1 = A_1, v_1 = B_1, w_1 = E_{11}$.
        # Term 2: $u_2 = A_1, v_2 = B_2, w_2 = E_{12}$.
        # This is 9 terms.
        # Can we do 3?
        # Term 1: $u_1 = A_1, v_1 = B, w_1 = E_{1 \cdot}$.
        # $v_1$ is a vector in $V$ (dim 3). $B$ is a vector in $V$.
        # So $v_1$ is the standard basis vector corresponding to the whole B? No.
        # The decomposition is linear.
        # $C = \sum_{k=1}^3 A_k B W_k$.
        # $W_k$ is a matrix? No, $w_k$ is a vector in $W$ (dim 9).
        # $w_k$ corresponds to the $k$-th row of $C$.
        # So $w_k$ has 1s at positions $3(k-1)+j$ for $j=0,1,2$.
        # $v_k$ must be $B$. So $v_k = [1, 1, 1]^T$? No, $v_k$ is a fixed vector.
        # If $v_k = [1, 1, 1]^T$, then $v_k \cdot B_{vec}$? No, $v_k$ is multiplied by $B$?
        # The term is $u_k v_k w_k^T$.
        # $u_k$ is in $U$ (dim 3). $v_k$ is in $V$ (dim 3). $w_k$ is in $W$ (dim 9).
        # The product is $u_k \otimes v_k \otimes w_k^T$.
        # We want $\sum u_k v_k w_k^T = A B^T$.
        # If we set $u_k = e_k$, $v_k = [1,1,1]^T$, $w_k = e_k \otimes I_3$.
        # Then $\sum_{k=1}^3 e_k [1,1,1]^T (e_k \otimes I_3)^T$.
        # This equals $\sum_k A_k B (e_k \otimes I_3)^T$.
        # This is correct.
        # So $v_k$ is the same for all k: $[1, 1, 1]^T$.
        # $u_k = e_k$.
        # $w_k = e_k \otimes I_3$ (flattened).
        
        V[:, i] = 1
        for j in range(3):
            W[3*i + j, i] = 1
            
    return U, V, W

def get_outer_product_3x1x2():
    # A (3x1) * B (1x2) -> C (3x2)
    # Rank 3
    U = np.zeros((3, 3))
    V = np.zeros((2, 3))
    W = np.zeros((6, 3))
    
    for i in range(3):
        U[i, i] = 1
        V[:, i] = 1
        for j in range(2):
            W[2*i + j, i] = 1
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=5
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_indices, v_indices, w_indices, U_block, V_block, W_block):
        rank = U_block.shape[1]
        for r in range(rank):
            U_vec = np.zeros(n * m)
            V_vec = np.zeros(m * p)
            W_vec = np.zeros(p * n)
            
            for i, idx in enumerate(u_indices):
                U_vec[idx] = U_block[i, r]
            for i, idx in enumerate(v_indices):
                V_vec[idx] = V_block[i, r]
            for i, idx in enumerate(w_indices):
                W_vec[idx] = W_block[i, r]
                
            U_parts.append(U_vec)
            V_parts.append(V_vec)
            W_parts.append(W_vec)

    # Block 1: 3x3x3 (Rank 26)
    # A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
    # U indices: 0..8
    # V indices: 0..8
    # W indices: 0..8 (C is 3x3, n=3. W[k*n+i]. k=0..2, i=0..2)
    
    # Sub-blocks of 3x3x3
    # 1. 2x2x2 (A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2])
    u_idx = [0, 1, 3, 4]
    v_idx = [0, 1, 3, 4]
    w_idx = [0, 1, 3, 4]
    add_block(u_idx, v_idx, w_idx, *get_strassen_2x2x2())
    
    # 2. 2x1x2 (A[0:2,2:3] * B[2:3,0:2] -> C[0:2,0:2])
    u_idx = [2, 5]
    v_idx = [6, 7]
    w_idx = [0, 1, 3, 4]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_2x1x2())
    
    # 3. 2x2x1 (A[0:2,0:2] * B[0:2,2:3] -> C[0:2,2:3])
    u_idx = [0, 1, 3, 4]
    v_idx = [2, 5, 8, 11] # B is 3x3. B[0:2, 2:3] is cols 2. Rows 0,1.
    # V indices for B[0:2, 2:3]:
    # B is 3x3. V is 9x9.
    # B[r, c]. V index r*3+c.
    # r=0, c=2 -> 2. r=1, c=2 -> 5.
    # Wait, B is 3x3. m=3, p=3.
    # V index = r*3+c.
    # B[0,2]=2, B[1,2]=5.
    # So v_idx = [2, 5].
    # But get_schoolbook_2x2x1 expects V of size 2x4.
    # So v_idx length must be 2.
    v_idx = [2, 5]
    w_idx = [6, 7] # C[0:2, 2:3]. C[0,2]=2, C[1,2]=5?
    # C is 3x3. n=3. W index = k*n+i.
    # C[0,2] -> k=0, i=2 -> 2.
    # C[1,2] -> k=1, i=2 -> 5.
    # So w_idx = [2, 5].
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_2x2x1())
    
    # 4. 2x1x1 (A[0:2,2:3] * B[2:3,2:3] -> C[0:2,2:3])
    u_idx = [2, 5]
    v_idx = [8] # B[2,2]=8.
    w_idx = [2, 5]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_2x1x1())
    
    # 5. 1x2x2 (A[2:3,0:2] * B[0:2,0:2] -> C[2:3,0:2])
    u_idx = [6, 7]
    v_idx = [0, 1, 3, 4]
    w_idx = [6, 7] # C[2,0]=6, C[2,1]=7.
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_1x2x2())
    
    # 6. 1x1x2 (A[2:3,2:3] * B[2:3,0:2] -> C[2:3,0:2])
    u_idx = [8]
    v_idx = [6, 7]
    w_idx = [6, 7]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_1x1x2())
    
    # 7. 1x2x1 (A[2:3,0:2] * B[0:2,2:3] -> C[2:3,2:3])
    u_idx = [6, 7]
    v_idx = [2, 5]
    w_idx = [8] # C[2,2]=8.
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_1x2x1())
    
    # 8. 1x1x1 (A[2:3,2:3] * B[2:3,2:3] -> C[2:3,2:3])
    u_idx = [8]
    v_idx = [8]
    w_idx = [8]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_1x1x1())
    
    # Block 2: 3x3x2 (Rank 17)
    # A[0:3, 0:3] * B[0:3, 3:5] -> C[0:3, 3:5]
    # B[0:3, 3:5] is 3x2.
    # V indices: B[r, c]. c=3,4.
    # r=0: 3, 4. r=1: 6, 7. r=2: 9, 10.
    # W indices: C[0:3, 3:5].
    # C[k, i]. i=3,4.
    # k=0: 3, 4. k=1: 6, 7. k=2: 9, 10.
    
    # Sub-blocks of 3x3x2
    # 1. 2x2x2 (A[0:2,0:2] * B[0:2, 3:5] -> C[0:2, 3:5])
    u_idx = [0, 1, 3, 4]
    v_idx = [3, 4, 6, 7]
    w_idx = [3, 4, 6, 7]
    add_block(u_idx, v_idx, w_idx, *get_strassen_2x2x2())
    
    # 2. 2x1x2 (A[0:2,2:3] * B[2:3, 3:5] -> C[0:2, 3:5])
    u_idx = [2, 5]
    v_idx = [9, 10]
    w_idx = [3, 4, 6, 7]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_2x1x2())
    
    # 3. 1x2x2 (A[2:3,0:2] * B[0:2, 3:5] -> C[2:3, 3:5])
    u_idx = [6, 7]
    v_idx = [3, 4, 6, 7]
    w_idx = [9, 10]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_1x2x2())
    
    # 4. 1x1x2 (A[2:3,2:3] * B[2:3, 3:5] -> C[2:3, 3:5])
    u_idx = [8]
    v_idx = [9, 10]
    w_idx = [9, 10]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_1x1x2())
    
    # Block 3: 3x1x3 (Rank 3)
    # A[0:3, 3:4] * B[3:4, 0:3] -> C[0:3, 0:3]
    # A[0:3, 3:4] is 3x1. U indices: 3, 7, 11?
    # A is 3x4. U index = r*4+c.
    # r=0, c=3 -> 3. r=1, c=3 -> 7. r=2, c=3 -> 11.
    # B[3:4, 0:3] is 1x3. V indices: r=3, c=0,1,2 -> 12, 13, 14.
    # W indices: C[0:3, 0:3]. 0..8.
    
    U3, V3, W3 = get_outer_product_3x1x3()
    u_idx = [3, 7, 11]
    v_idx = [12, 13, 14]
    w_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    add_block(u_idx, v_idx, w_idx, U3, V3, W3)
    
    # Block 4: 3x1x2 (Rank 3)
    # A[0:3, 3:4] * B[3:4, 3:5] -> C[0:3, 3:5]
    # A indices: 3, 7, 11.
    # B[3:4, 3:5] is 1x2. V indices: r=3, c=3,4 -> 15, 16.
    # W indices: C[0:3, 3:5]. 3,4, 6,7, 9,10.
    
    U4, V4, W4 = get_outer_product_3x1x2()
    u_idx = [3, 7, 11]
    v_idx = [15, 16]
    w_idx = [3, 4, 6, 7, 9, 10]
    add_block(u_idx, v_idx, w_idx, U4, V4, W4)
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
