"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    # Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 formulas for 2x2 matrix multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U rows: A00(row0), A01(row1), A10(row2), A11(row3)
    # V rows: B00(row0), B01(row1), B10(row2), B11(row3)
    # W rows: C00(row0), C01(row1), C10(row2), C11(row3)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """Return schoolbook decomposition for n x m by m x p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _combine_blocks(U1, V1, W1, U2, V2, W2):
    """Combine two block decompositions into a direct sum.
    Block 1: U1 (n1*m, r1), V1 (m*p1, r1), W1 (p1*n1, r1)
    Block 2: U2 (n2*m, r2), V2 (m*p2, r2), W2 (p2*n2, r2)
    Result: U (n*m, r1+r2), V (m*p, r1+r2), W (p*n, r1+r2)
    where n = n1+n2, p = p1+p2
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    total_rank = r1 + r2
    
    n1 = U1.shape[0] // U1.shape[1] if False else None
    # We need to figure out dimensions from the shapes
    # U1: (n1*m, r1), so n1*m = U1.shape[0]
    # This is tricky. Let's pass dimensions explicitly.
    # Actually, let me restructure.
    raise NotImplementedError("Use _combine_blocks_with_dims instead")

def _combine_blocks_with_dims(n1, m1, p1, U1, V1, W1, n2, m2, p2, U2, V2, W2):
    """Combine two block decompositions.
    Block 1: U1 (n1*m1, r1), V1 (m1*p1, r1), W1 (p1*n1, r1)
    Block 2: U2 (n2*m2, r2), V2 (m2*p2, r2), W2 (p2*n2, r2)
    Requires m1 == m2 (shared inner dimension)
    Result: U (n*m, r1+r2), V (m*p, r1+r2), W (p*n, r1+r2)
    where n = n1+n2, p = p1+p2
    """
    assert m1 == m2, "Inner dimensions must match"
    m = m1
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    total_rank = r1 + r2
    
    n = n1 + n2
    p = p1 + p2
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Place block 1
    U[:n1*m, :r1] = U1
    V[:m*p1, :r1] = V1
    W[:p1*n1, :r1] = W1
    
    # Place block 2
    U[n1*m:, r1:] = U2
    V[m*p1:, r1:] = V2
    W[p1*n1:, r1:] = W2
    
    return U, V, W

def construct():
    _start = time.time()
    _deadline = _start + 140.0  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Strategy: Recursive Strassen-like decomposition
    # Split inner dimension m=4 into m1=2, m2=2
    # Split output dimension p=5 into p1=4, p2=1 (or other splits)
    # 
    # Approach 1: Split p=5 into p1=4, p2=1
    #   Block 1: 2x4x4 -> can further decompose
    #   Block 2: 2x4x1 -> schoolbook rank 8
    #
    # Approach 2: Use Strassen on 2x2x2 blocks
    #   For 2x4x4: split m=4 into m1=2, m2=2
    #     Block 1a: 2x2x4
    #     Block 1b: 2x2x4
    #   For each 2x2x4: split p=4 into p1=2, p2=2
    #     2x2x2 (Strassen, rank 7) + 2x2x2 (Strassen, rank 7) = rank 14
    #   So 2x4x4 = rank 14 + 14 = 28
    #   2x4x1 = schoolbook rank 8
    #   Total: 28 + 8 = 36
    #
    # Let's try this approach.
    
    # First, build the schoolbook fallback
    U_school, V_school, W_school = _schoolbook_block(n, m, p)
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p  # 40
    
    # Build Strassen 2x2x2 decomposition
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Verify Strassen decomposition
    if not _verify_decomposition(U_s.tolist(), V_s.tolist(), W_s.tolist(), 2, 2, 2):
        # Fallback to schoolbook
        return best_result
    
    # Build 2x2x4 decomposition: two 2x2x2 Strassen blocks
    # Split p=4 into p1=2, p2=2
    U_224, V_224, W_224 = _combine_blocks_with_dims(
        2, 2, 2, U_s, V_s, W_s,
        2, 2, 2, U_s, V_s, W_s
    )
    # U_224: (4, 14), V_224: (8, 14), W_224: (8, 14)
    
    # Build 2x4x4 decomposition: two 2x2x4 blocks (split m=4 into m1=2, m2=2)
    U_244, V_244, W_244 = _combine_blocks_with_dims(
        2, 2, 4, U_224, V_224, W_224,
        2, 2, 4, U_224, V_224, W_224
    )
    # U_244: (8, 28), V_244: (16, 28), W_244: (16, 28)
    
    # Build 2x4x1 schoolbook decomposition
    U_241, V_241, W_241 = _schoolbook_block(2, 4, 1)
    # U_241: (8, 8), V_241: (4, 8), W_241: (2, 8)
    
    # Combine 2x4x4 and 2x4x1 to get 2x4x5
    # Split p=5 into p1=4, p2=1
    U_total, V_total, W_total = _combine_blocks_with_dims(
        2, 4, 4, U_244, V_244, W_244,
        2, 4, 1, U_241, V_241, W_241
    )
    # U_total: (8, 36), V_total: (20, 36), W_total: (10, 36)
    
    # Verify the total decomposition
    if _verify_decomposition(U_total.tolist(), V_total.tolist(), W_total.tolist(), n, m, p):
        best_result = {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
        best_rank = 36
    
    # Check deadline
    if time.time() > _deadline - 5:
        return best_result
    
    # Try to improve further: use Strassen for 2x2x3 as well?
    # 2x2x3 schoolbook rank = 12
    # Can we do better? Let's try a few ranks with random search.
    
    # Actually, let's try another decomposition strategy:
    # Split m=4 into m1=2, m2=2 and p=5 into p1=2, p2=3
    # Block 1: 2x2x2 (Strassen, rank 7)
    # Block 2: 2x2x3 (need to find best rank)
    # Block 3: 2x2x2 (Strassen, rank 7)
    # Block 4: 2x2x3 (need to find best rank)
    # Total: 14 + 2 * rank(2x2x3)
    
    # For 2x2x3, schoolbook is 12. Let's see if we can do better.
    # Try ranks 11, 10, 9 with random search
    
    def random_search_223(max_rank, num_trials=2000):
        """Try to find a rank-max_rank decomposition for 2x2x3."""
        np.random.seed(42)
        n2, m2, p2 = 2, 2, 3
        
        for trial in range(num_trials):
            if time.time() > _deadline - 10:
                break
            
            U = np.random.randn(n2*m2, max_rank) * 0.5
            V = np.random.randn(m2*p2, max_rank) * 0.5
            W = np.random.randn(p2*n2, max_rank) * 0.5
            
            best_err = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for step in range(30):
                if time.time() > _deadline - 10:
                    break
                
                U_3d = U.reshape(n2, m2, max_rank)
                V_3d = V.reshape(m2, p2, max_rank)
                W_3d = W.reshape(p2, n2, max_rank)
                
                err = 0.0
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            s = 0.0
                            for r in range(max_rank):
                                s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                            err += (s - 1.0) ** 2
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_err < 1e-9:
                    return best_U, best_V, best_W
                
                U += np.random.randn(n2*m2, max_rank) * 0.1
                V += np.random.randn(m2*p2, max_rank) * 0.1
                W += np.random.randn(p2*n2, max_rank) * 0.1
            
            if best_err < 1e-9:
                return best_U, best_V, best_W
        
        return None
    
    # Try to find 2x2x3 decomposition with rank < 12
    for rank_223 in [11, 10, 9]:
        if time.time() > _deadline - 15:
            break
        
        result = random_search_223(rank_223, num_trials=500)
        if result is not None:
            U_223, V_223, W_223 = result
            if _verify_decomposition(U_223.tolist(), V_223.tolist(), W_223.tolist(), 2, 2, 3):
                # Build 2x4x5 using this 2x2x3
                # Split m=4 into m1=2, m2=2 and p=5 into p1=2, p2=3
                # Block 1: 2x2x2 (Strassen)
                # Block 2: 2x2x3 (found)
                # Block 3: 2x2x2 (Strassen)
                # Block 4: 2x2x3 (found)
                
                U_b1, V_b1, W_b1 = U_s, V_s, W_s  # 2x2x2
                U_b2, V_b2, W_b2 = U_223, V_223, W_223  # 2x2x3
                U_b3, V_b3, W_b3 = U_s, V_s, W_s  # 2x2x2
                U_b4, V_b4, W_b4 = U_223, V_223, W_223  # 2x2x3
                
                # Combine block 1 and block 2: 2x2x5
                U_225a, V_225a, W_225a = _combine_blocks_with_dims(
                    2, 2, 2, U_b1, V_b1, W_b1,
                    2, 2, 3, U_b2, V_b2, W_b2
                )
                
                # Combine block 3 and block 4: 2x2x5
                U_225b, V_225b, W_225b = _combine_blocks_with_dims(
                    2, 2, 2, U_b3, V_b3, W_b3,
                    2, 2, 3, U_b4, V_b4, W_b4
                )
                
                # Combine 2x2x5 and 2x2x5: 2x4x5
                U_new, V_new, W_new = _combine_blocks_with_dims(
                    2, 2, 5, U_225a, V_225a, W_225a,
                    2, 2, 5, U_225b, V_225b, W_225b
                )
                
                if _verify_decomposition(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
                    new_rank = U_new.shape[1]
                    if new_rank < best_rank:
                        best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                        best_rank = new_rank
                
                break  # Found a 2x2x3, no need to try lower ranks
    
    return best_result
# EVOLVE-BLOCK-END
