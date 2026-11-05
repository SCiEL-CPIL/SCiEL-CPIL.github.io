import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses Strassen's algorithm for 2x2x2 blocks to reduce the rank from the schoolbook 60
    to 56 by splitting the 3x4x5 multiplication into a 2x4x5 part (rank 36) and a 1x4x5 part (rank 20).
    The decomposition is given by U, V, W such that T = sum_r U[:, r] outer V[:, r] outer W[:, r].
    Indices:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank) with W[k*n+i, r]
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe within 120s/150s
    
    n, m, p = DIMENSIONS
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def strassen_2x2x2():
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
    
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W
    
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W
    
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    def w_idx_c(i, k):
        return k * n + i

    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            u = np.zeros(n * m)
            for i, idx in enumerate(u_idx):
                u[idx] = U_blk[i, r]
            U_parts.append(u)
            
            v = np.zeros(m * p)
            for i, idx in enumerate(v_idx):
                v[idx] = V_blk[i, r]
            V_parts.append(v)
            
            w = np.zeros(p * n)
            for i, idx in enumerate(w_idx):
                w[idx] = W_blk[i, r]
            W_parts.append(w)
            
            if time.time() > deadline:
                # In case of timeout, return what we have? 
                # But we need valid dimensions. 
                # The generation is fast, this is unlikely to trigger.
                pass

    # Part 1: Rows 0,1 of A
    add_block([0,1,4,5], [0,1,5,6], [w_idx_c(0,0),w_idx_c(0,1),w_idx_c(1,0),w_idx_c(1,1)], *strassen_2x2x2())
    add_block([0,1,4,5], [2,3,7,8], [w_idx_c(0,2),w_idx_c(0,3),w_idx_c(1,2),w_idx_c(1,3)], *strassen_2x2x2())
    add_block([0,1,4,5], [4,9], [w_idx_c(0,4),w_idx_c(1,4)], *schoolbook_2x2x1())
    
    add_block([2,3,6,7], [10,11,15,16], [w_idx_c(0,0),w_idx_c(0,1),w_idx_c(1,0),w_idx_c(1,1)], *strassen_2x2x2())
    add_block([2,3,6,7], [12,13,17,18], [w_idx_c(0,2),w_idx_c(0,3),w_idx_c(1,2),w_idx_c(1,3)], *strassen_2x2x2())
    add_block([2,3,6,7], [14,19], [w_idx_c(0,4),w_idx_c(1,4)], *schoolbook_2x2x1())

    # Part 2: Row 2 of A
    add_block([8,9], [0,1,5,6], [w_idx_c(2,0),w_idx_c(2,1)], *schoolbook_1x2x2())
    add_block([8,9], [2,3,7,8], [w_idx_c(2,2),w_idx_c(2,3)], *schoolbook_1x2x2())
    add_block([8,9], [4,9], [w_idx_c(2,4)], *schoolbook_1x2x1())
    
    add_block([10,11], [10,11,15,16], [w_idx_c(2,0),w_idx_c(2,1)], *schoolbook_1x2x2())
    add_block([10,11], [12,13,17,18], [w_idx_c(2,2),w_idx_c(2,3)], *schoolbook_1x2x2())
    add_block([10,11], [14,19], [w_idx_c(2,4)], *schoolbook_1x2x1())

    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
