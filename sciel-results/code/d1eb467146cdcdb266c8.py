import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses Strassen's algorithm for 2x2x2 blocks to reduce the rank from the schoolbook 60
    to 56 by splitting the 3x4x5 multiplication into structured blocks.
    Then attempts to reduce rank further by finding linear dependencies.
    The decomposition is given by U, V, W such that T = sum_r U[:, r] outer V[:, r] outer W[:, r].
    Indices:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank) with W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
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

    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(U_blk, V_blk, W_blk, u_idx, v_idx, w_idx):
        for r in range(U_blk.shape[1]):
            U_part = np.zeros(n * m)
            for i, idx in enumerate(u_idx):
                U_part[idx] = U_blk[i, r]
            U_parts.append(U_part)
            
            V_part = np.zeros(m * p)
            for i, idx in enumerate(v_idx):
                V_part[idx] = V_blk[i, r]
            V_parts.append(V_part)
            
            W_part = np.zeros(p * n)
            for i, idx in enumerate(w_idx):
                W_part[idx] = W_blk[i, r]
            W_parts.append(W_part)
    
    # Block 1: A1_top * B_top_left (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(U_blk, V_blk, W_blk, [0, 1, 4, 5], [0, 1, 5, 6], [0, 3, 1, 4])
    
    # Block 2: A1_top * B_top_mid (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(U_blk, V_blk, W_blk, [0, 1, 4, 5], [2, 3, 7, 8], [6, 9, 7, 10])
    
    # Block 3: A1_top * B_top_right (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block(U_blk, V_blk, W_blk, [0, 1, 4, 5], [4, 9], [12, 13])
    
    # Block 4: A1_bottom * B_bottom_left (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(U_blk, V_blk, W_blk, [2, 3, 6, 7], [10, 11, 15, 16], [0, 3, 1, 4])
    
    # Block 5: A1_bottom * B_bottom_mid (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(U_blk, V_blk, W_blk, [2, 3, 6, 7], [12, 13, 17, 18], [6, 9, 7, 10])
    
    # Block 6: A1_bottom * B_bottom_right (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block(U_blk, V_blk, W_blk, [2, 3, 6, 7], [14, 19], [12, 13])
    
    # Block 7: A2_top * B_top_left (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block(U_blk, V_blk, W_blk, [8, 9], [0, 1, 5, 6], [2, 5])
    
    # Block 8: A2_top * B_top_mid (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block(U_blk, V_blk, W_blk, [8, 9], [2, 3, 7, 8], [8, 11])
    
    # Block 9: A2_top * B_top_right (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    add_block(U_blk, V_blk, W_blk, [8, 9], [4, 9], [14])
    
    # Block 10: A2_bottom * B_bottom_left (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block(U_blk, V_blk, W_blk, [10, 11], [10, 11, 15, 16], [2, 5])
    
    # Block 11: A2_bottom * B_bottom_mid (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block(U_blk, V_blk, W_blk, [10, 11], [12, 13, 17, 18], [8, 11])
    
    # Block 12: A2_bottom * B_bottom_right (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    add_block(U_blk, V_blk, W_blk, [10, 11], [14, 19], [14])
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    # Verify the decomposition
    def verify(U, V, W, tol=1e-7):
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    acc = 0.0
                    for c in range(r):
                        acc += W[k * n + i, c] * U[i * m + j, c] * V[j * p + k, c]
                    T[i, j, k] = acc
        return np.allclose(T, np.ones((n, m, p)), atol=tol)
    
    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now try to reduce the rank by finding linear dependencies
    # The tensor equation is: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is a bilinear system. We can try to find if one column can be expressed as a
    # linear combination of others.
    
    # For a given column c0, we want to find alpha_c (c != c0) such that:
    # U[:, c0] = sum_{c != c0} alpha_c * U[:, c]  (in some sense)
    # But this is not straightforward because the decomposition is not linear in each term.
    
    # Alternative approach: try to find a rank-(R-1) decomposition by solving:
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # If we fix all but one term, we can solve for the last term's coefficients.
    
    # Let's try a greedy approach: try to eliminate each column and check if the
    # remaining system is consistent.
    
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    def can_reduce_rank(U, V, W, cols_to_remove, tol=1e-7):
        """Check if removing the specified columns still gives a valid decomposition."""
        if len(cols_to_remove) == 0:
            return True, U, V, W
        
        keep_cols = [c for c in range(U.shape[1]) if c not in cols_to_remove]
        U_new = U[:, keep_cols]
        V_new = V[:, keep_cols]
        W_new = W[:, keep_cols]
        
        # Verify
        r = U_new.shape[1]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    acc = 0.0
                    for c in range(r):
                        acc += W_new[k * n + i, c] * U_new[i * m + j, c] * V_new[j * p + k, c]
                    T[i, j, k] = acc
        return np.allclose(T, np.ones((n, m, p)), atol=tol), U_new, V_new, W_new
    
    # Try to remove one column at a time
    for attempt in range(current_rank):
        if time.time() > deadline - 10:
            break
        
        reduced = False
        for col in range(current_rank):
            if time.time() > deadline - 10:
                break
            
            valid, U_new, V_new, W_new = can_reduce_rank(current_U, current_V, current_W, [col])
            if valid:
                current_U = U_new
                current_V = V_new
                current_W = W_new
                current_rank -= 1
                reduced = True
                break
        
        if not reduced:
            break
    
    # Final verification
    if not verify(current_U, current_V, current_W):
        # Revert to the original rank-56 solution
        current_U = U
        current_V = V
        current_W = W
    
    return {
        "U": current_U.tolist(),
        "V": current_V.tolist(),
        "W": current_W.tolist()
    }
