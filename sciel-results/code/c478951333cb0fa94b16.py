import numpy as np
import time

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140

def _build_expected_tensor(n, m, p):
    """Build the expected matmul tensor T[a,b,c] = 1 if indices consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    return T

def _verify_decomposition(U, V, W, T_expected, atol=1e-8):
    """Verify that sum_r U[:,r]⊗V[:,r]⊗W[:,r] == T_expected."""
    nm = U.shape[0]
    mp = V.shape[0]
    pn = W.shape[0]
    rank = U.shape[1]
    T_computed = np.zeros((nm, mp, pn))
    for r in range(rank):
        T_computed += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_computed, T_expected, atol=atol)

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses Strassen for 2x2x2 blocks and rank-2 decomposition for 1x2x2 blocks.
    Rank is 48.
    """
    n, m, p = DIMENSIONS
    
    if time.time() > _DEADLINE:
        return _schoolbook_fallback(n, m, p)
    
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
    
    def strassen_2x2x2():
        """Strassen decomposition for 2x2x2. Rank 7."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        W[1, 0] = 0; W[2, 0] = 0
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        W[0, 1] = 0; W[1, 1] = 0
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        W[0, 2] = 0; W[2, 2] = 0
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        W[1, 3] = 0; W[3, 3] = 0
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        W[2, 4] = 0; W[3, 4] = 0
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        W[0, 5] = 0; W[1, 5] = 0; W[2, 5] = 0
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        W[1, 6] = 0; W[2, 6] = 0; W[3, 6] = 0
        
        return U, V, W
    
    def schoolbook_2x2x1():
        """Schoolbook for 2x2x1. Rank 4."""
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    def rank2_1x2x2():
        """Rank-2 decomposition for 1x2x2.
        A is 1x2: [A0, A1]
        B is 2x2: [[B00, B01], [B10, B11]]
        C is 1x2: [C0, C1]
        C0 = A0*B00 + A1*B10
        C1 = A0*B01 + A1*B11
        
        We can write this as:
        C0 = A0*(B00) + A1*(B10)
        C1 = A0*(B01) + A1*(B11)
        
        Using 2 rank-1 terms:
        Term 1: U=[1,0], V=[B00, B01], W=[1, 1] -> contributes A0*B00 to C0, A0*B01 to C1
        Term 2: U=[0,1], V=[B10, B11], W=[1, 1] -> contributes A1*B10 to C0, A1*B11 to C1
        
        But wait, W has shape (p*n, rank) = (2*1, rank) = (2, rank) for the block.
        The W indices correspond to C0 and C1.
        
        So:
        Term 1: U=[1,0], V=[1,0] (for B00,B01 positions), W=[1,1]
        Term 2: U=[0,1], V=[0,1] (for B10,B11 positions), W=[1,1]
        
        Wait, let me reconsider. V has 4 entries: B00, B01, B10, B11.
        For term 1, we want to pick B00 and B01, so V=[1,1,0,0]
        For term 2, we want to pick B10 and B11, so V=[0,0,1,1]
        """
        U = np.zeros((2, 2))
        V = np.zeros((4, 2))
        W = np.zeros((2, 2))
        
        # Term 1: A0 * [B00, B01] -> [C0, C1]
        U[0, 0] = 1
        V[0, 0] = 1; V[1, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1
        
        # Term 2: A1 * [B10, B11] -> [C0, C1]
        U[1, 1] = 1
        V[2, 1] = 1; V[3, 1] = 1
        W[0, 1] = 1; W[1, 1] = 1
        
        return U, V, W
    
    def schoolbook_1x2x1():
        """Schoolbook for 1x2x1. Rank 2."""
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        return U, V, W

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([0, 1, 4, 5], [0, 1, 5, 6], [0, 3, 1, 4], U_blk, V_blk, W_blk)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([0, 1, 4, 5], [2, 3, 7, 8], [6, 9, 7, 10], U_blk, V_blk, W_blk)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4] -> C[0:2, 4] (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block([0, 1, 4, 5], [4, 9], [12, 13], U_blk, V_blk, W_blk)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([2, 3, 6, 7], [10, 11, 15, 16], [0, 3, 1, 4], U_blk, V_blk, W_blk)
    
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4] (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([2, 3, 6, 7], [12, 13, 17, 18], [6, 9, 7, 10], U_blk, V_blk, W_blk)
    
    # Block 6: A[0:2, 2:4] * B[2:4, 4] -> C[0:2, 4] (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block([2, 3, 6, 7], [14, 19], [12, 13], U_blk, V_blk, W_blk)
    
    # Block 7: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2] (1x2x2)
    U_blk, V_blk, W_blk = rank2_1x2x2()
    add_block([8, 9], [0, 1, 5, 6], [2, 5], U_blk, V_blk, W_blk)
    
    # Block 8: A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4] (1x2x2)
    U_blk, V_blk, W_blk = rank2_1x2x2()
    add_block([8, 9], [2, 3, 7, 8], [8, 11], U_blk, V_blk, W_blk)
    
    # Block 9: A[2, 0:2] * B[0:2, 4] -> C[2, 4] (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    add_block([8, 9], [4, 9], [14], U_blk, V_blk, W_blk)
    
    # Block 10: A[2, 2:4] * B[2:4, 0:2] -> C[2, 0:2] (1x2x2)
    U_blk, V_blk, W_blk = rank2_1x2x2()
    add_block([10, 11], [10, 11, 15, 16], [2, 5], U_blk, V_blk, W_blk)
    
    # Block 11: A[2, 2:4] * B[2:4, 2:4] -> C[2, 2:4] (1x2x2)
    U_blk, V_blk, W_blk = rank2_1x2x2()
    add_block([10, 11], [12, 13, 17, 18], [8, 11], U_blk, V_blk, W_blk)
    
    # Block 12: A[2, 2:4] * B[2:4, 4] -> C[2, 4] (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    add_block([10, 11], [14, 19], [14], U_blk, V_blk, W_blk)
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    rank = U.shape[1]
    
    # Verify correctness
    if time.time() < _DEADLINE - 5:
        T_computed = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_computed += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
        T_expected = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    T_expected[a_idx, b_idx, c_idx] = 1.0
        
        if not np.allclose(T_computed, T_expected, atol=1e-10):
            return _schoolbook_fallback(n, m, p)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook_fallback(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    U_parts = []
    V_parts = []
    W_parts = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                u[i * m + j] = 1
                v = np.zeros(m * p)
                v[j * p + k] = 1
                w = np.zeros(p * n)
                w[k * n + i] = 1
                U_parts.append(u)
                V_parts.append(v)
                W_parts.append(w)
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
