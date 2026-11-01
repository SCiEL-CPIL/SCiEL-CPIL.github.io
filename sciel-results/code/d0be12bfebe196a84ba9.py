import time
import numpy as np

DEADLINE = 120.0
START_TIME = time.time()

def _check_deadline():
    return time.time() - START_TIME > DEADLINE - 10

def _construct_trivial(n, m, p):
    """
    Construct the trivial (schoolbook) tensor decomposition.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1.0
                V[b_idx, r] = 1.0
                W[c_idx, r] = 1.0
                r += 1
                
    return U, V, W

def _tensor_check(U, V, W, n, m, p):
    """
    Check if the decomposition is valid.
    Condition: For all i, k, the matrix M_{i,k} with entries 
    M[j, j'] = sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] 
    must be the Identity matrix I_m.
    """
    I_m = np.eye(m)
    for i in range(n):
        for k in range(p):
            U_sub = U[i*m : i*m+m, :] # (m, R)
            V_indices = np.arange(m) * p + k
            V_sub = V[V_indices, :] # (m, R)
            W_vec = W[k*n + i, :] # (R,)
            
            # M[j, j'] = sum_r U_sub[j, r] * V_sub[j', r] * W_vec[r]
            M = (U_sub * W_vec) @ V_sub.T
            
            if not np.allclose(M, I_m, atol=1e-9):
                return False
    return True

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 7
    
    try:
        # Strategy: Tensor Slicing and Rank Subadditivity
        # Partition m=4 into two sets of size 2: {0,1} and {2,3}.
        # The total tensor is the sum of two sub-tensors T_0 and T_1.
        # T_0 corresponds to m' = 2, using columns 0,1 of A and B.
        # T_1 corresponds to m' = 2, using columns 2,3 of A and B.
        
        # We need the optimal decomposition for 2x2x7.
        # Rank(2x2x7) = Rank(2x2x4) + Rank(2x2x3)
        # Rank(2x2x4) = 2 * Rank(2x2x2) = 2 * 7 = 14 (using two 2x2x2 blocks)
        # Rank(2x2x3) = 4 * 1 * 3 = 12 (schoolbook for 2x2x3? No, Rank(2x2x3) is known to be 12? 
        # Actually, Rank(2x2x3) = 12 is the trivial bound. Is it optimal?
        # For 2x2x3, trivial is 12. Strassen doesn't directly apply.
        # However, we can use 2x2x2 (rank 7) for first 2 cols and 2x2x1 (rank 4) for last col?
        # Rank(2x2x1) = 4.
        # So Rank(2x2x3) <= 7 + 4 = 11?
        # Let's check if 2x2x3 can be done in 11.
        # The trivial is 12.
        # If we use 2x2x2 for k=0,1 and 2x2x1 for k=2, rank is 7+4=11.
        # Is 11 the minimum? 
        # It is known that mu(2,2,3) = 12? No, let's verify.
        # Actually, for 2x2x3, the trivial is 12. 
        # Can we do better? 
        # A 2x2x3 multiplication is C = AB where A is 2x2, B is 2x3.
        # This is equivalent to multiplying a 2x2 matrix by a 2x3 matrix.
        # This can be seen as two 2x2x2 multiplications if we pad B with zeros? No.
        # Let's stick to the safe construction:
        # 2x2x4: Two 2x2x2 Strassen blocks. Rank 14.
        # 2x2x3: One 2x2x2 Strassen block (for k=0,1) + One 2x2x1 Schoolbook (for k=2). Rank 7+4=11.
        # Total Rank(2x2x7) = 14 + 11 = 25.
        
        # Wait, the current program uses:
        # 2x4x4: 4 * 7 = 28.
        # 2x4x3: 2 * (7+4) = 22.
        # Total 50.
        
        # My proposed slicing:
        # Slice 1 (m=0,1): 2x2x7.
        #   - 2x2x4 part: 2 * Rank(2x2x2) = 14.
        #   - 2x2x3 part: Rank(2x2x2) + Rank(2x2x1) = 7 + 4 = 11.
        #   Total for slice 1: 25.
        # Slice 2 (m=2,3): Same, 25.
        # Total Rank = 50.
        
        # This is the same rank as the current program! 
        # Why did I think it would be better?
        # The current program constructs 2x4x4 using 4 blocks of 2x2x2.
        # 2x4x4 = 4 * 7 = 28.
        # My slice approach for 2x4x4:
        # Slice 1 (m=0,1): 2x2x4 = 14.
        # Slice 2 (m=2,3): 2x2x4 = 14.
        # Total 28.
        
        # For 2x4x3:
        # Current: 2 * (2x2x3).
        # If 2x2x3 is 11, total is 22.
        # If 2x2x3 is 12 (trivial), total is 24.
        
        # So the bottleneck is the rank of 2x2x3.
        # If I can find a rank 11 decomposition for 2x2x3, I save 2 ranks total (1 per slice).
        # Total would be 28 + 22 = 50.
        # Wait, 28 + 22 = 50.
        # Current is 50.
        
        # Is there a better decomposition for 2x2x3?
        # The trivial rank is 12.
        # Strassen for 2x2x2 is 7.
        # Can we combine?
        # C = A B. A is 2x2, B is 2x3.
        # B = [B1 B2 B3] where Bj are 2x1.
        # C = [A B1, A B2, A B3].
        # This is 3 separate 2x2x1 multiplications.
        # Rank(2x2x1) = 4.
        # Total 12.
        
        # Is it possible to do 2x2x3 in rank 11?
        # According to literature, mu(2,2,3) = 12. 
        # So 50 is likely optimal for this slicing strategy.
        
        # However, the current program *might* be buggy or suboptimal in its construction.
        # Let's ensure our construction is clean and verify the rank.
        # If the current program is already 50, and 50 is optimal, we can't improve rank.
        # BUT, the prompt says "score is the rank". Lower is better.
        # If I can't beat 50, I should at least ensure I don't regress.
        
        # Let's look for a different slicing.
        # Slice n=2 into two n=1?
        # 1x4x7.
        # Rank(1x4x7) = 4*7 = 28 (trivial).
        # Two slices: 28 + 28 = 56. Worse.
        
        # Slice p=7 into 3+4?
        # 2x4x3 and 2x4x4.
        # We did this.
        
        # What if we use a different base?
        # 2x4x7.
        # Maybe 2x4x2 (rank 14) + 2x4x2 (rank 14) + 2x4x2 (rank 14) + 2x4x1 (rank 8)?
        # 2x4x2: 2 slices of 2x2x2 (rank 7) -> 14.
        # 2x4x1: 4 slices of 2x2x1 (rank 4) -> 16.
        # Total: 14*3 + 16 = 58. Worse.
        
        # It seems 50 is a hard floor for this slicing approach.
        
        # Is there a global optimization that beats slicing?
        # For 2x4x7, the trivial is 56.
        # Strassen-like methods usually help when dimensions are powers of 2.
        # 4 is power of 2. 2 is power of 2. 7 is not.
        
        # Let's try to construct the 50-rank solution explicitly and cleanly.
        # The current program does this.
        # I will implement the same logic but with a more robust check and potentially
        # different variable ordering to ensure no bugs.
        
        # If I cannot improve the rank, I should return the best known rank.
        
        def build_strassen_2x2x2():
            n, m, p = 2, 2, 2
            R = 7
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            
            # P1 = (A11+A22)(B11+B22)
            U[0, 0] = 1; U[3, 0] = 1
            V[0, 0] = 1; V[3, 0] = 1
            W[0, 0] = 1; W[3, 0] = 1
            
            # P2 = (A21+A22)A11
            U[2, 1] = 1; U[3, 1] = 1
            V[0, 1] = 1
            W[2, 1] = 1; W[3, 1] = -1
            
            # P3 = A11(B12-B22)
            U[0, 2] = 1
            V[1, 2] = 1; V[3, 2] = -1
            W[1, 2] = 1; W[3, 2] = 1
            
            # P4 = A22(B21-B11)
            U[3, 3] = 1
            V[2, 3] = 1; V[0, 3] = -1
            W[0, 3] = 1; W[2, 3] = 1
            
            # P5 = (A11+A12)B22
            U[0, 4] = 1; U[1, 4] = 1
            V[3, 4] = 1
            W[0, 4] = -1; W[3, 4] = 1
            
            # P6 = (A21-A11)(B11+B12)
            U[2, 5] = 1; U[0, 5] = -1
            V[0, 5] = 1; V[1, 5] = 1
            W[3, 5] = 1
            
            # P7 = (A12-A22)(B21+B22)
            U[1, 6] = 1; U[3, 6] = -1
            V[2, 6] = 1; V[3, 6] = 1
            W[0, 6] = 1; W[2, 6] = -1
            
            return U, V, W

        def build_schoolbook(n, m, p):
            R = n * m * p
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, r] = 1
                        V[j*p+k, r] = 1
                        W[k*n+i, r] = 1
                        r += 1
            return U, V, W

        def embed(U_src, V_src, W_src, n_src, m_src, p_src, n_tgt, m_tgt, p_tgt, n_map, m_map, p_map):
            U = np.zeros((n_tgt*m_tgt, U_src.shape[1]))
            V = np.zeros((m_tgt*p_tgt, V_src.shape[1]))
            W = np.zeros((p_tgt*n_tgt, W_src.shape[1]))
            
            for i in range(n_src):
                for j in range(m_src):
                    si = i*m_src + j
                    ti = n_map[i]*m_tgt + m_map[j]
                    U[ti, :] = U_src[si, :]
                    
            for j in range(m_src):
                for k in range(p_src):
                    sj = j*p_src + k
                    tj = m_map[j]*p_tgt + p_map[k]
                    V[tj, :] = V_src[sj, :]
                    
            for k in range(p_src):
                for i in range(n_src):
                    sk = k*n_src + i
                    tk = p_map[k]*n_tgt + n_map[i]
                    W[tk, :] = W_src[sk, :]
                    
            return U, V, W

        # Build 2x2x2 Strassen
        U222, V222, W222 = build_strassen_2x2x2()
        
        # Build 2x2x1 Schoolbook
        U221, V221, W221 = build_schoolbook(2, 2, 1)
        
        # Build 2x2x3: 2x2x2 (k=0,1) + 2x2x1 (k=2)
        U223_1, V223_1, W223_1 = embed(U222, V222, W222, 2, 2, 2, 2, 2, 3, [0,1], [0,1], [0,1])
        U223_2, V223_2, W223_2 = embed(U221, V221, W221, 2, 2, 1, 2, 2, 3, [0,1], [0,1], [2])
        
        U223 = np.hstack([U223_1, U223_2])
        V223 = np.hstack([V223_1, V223_2])
        W223 = np.hstack([W223_1, W223_2])
        
        # Build 2x4x3: Two 2x2x3 blocks (m=0,1 and m=2,3)
        U243_1, V243_1, W243_1 = embed(U223, V223, W223, 2, 2, 3, 2, 4, 3, [0,1], [0,1], [0,1,2])
        U243_2, V243_2, W243_2 = embed(U223, V223, W223, 2, 2, 3, 2, 4, 3, [0,1], [2,3], [0,1,2])
        
        U243 = np.hstack([U243_1, U243_2])
        V243 = np.hstack([V243_1, V243_2])
        W243 = np.hstack([W243_1, W243_2])
        
        # Build 2x4x4: Four 2x2x2 blocks
        U244_1, V244_1, W244_1 = embed(U222, V222, W222, 2, 2, 2, 2, 4, 4, [0,1], [0,1], [0,1])
        U244_2, V244_2, W244_2 = embed(U222, V222, W222, 2, 2, 2, 2, 4, 4, [0,1], [2,3], [0,1])
        U244_3, V244_3, W244_3 = embed(U222, V222, W222, 2, 2, 2, 2, 4, 4, [0,1], [0,1], [2,3])
        U244_4, V244_4, W244_4 = embed(U222, V222, W222, 2, 2, 2, 2, 4, 4, [0,1], [2,3], [2,3])
        
        U244 = np.hstack([U244_1, U244_2, U244_3, U244_4])
        V244 = np.hstack([V244_1, V244_2, V244_3, V244_4])
        W244 = np.hstack([W244_1, W244_2, W244_3, W244_4])
        
        # Build 2x4x7: 2x4x4 (p=0..3) + 2x4x3 (p=4..6)
        U247_1, V247_1, W247_1 = embed(U244, V244, W244, 2, 4, 4, 2, 4, 7, [0,1], [0,1,2,3], [0,1,2,3])
        U247_2, V247_2, W247_2 = embed(U243, V243, W243, 2, 4, 3, 2, 4, 7, [0,1], [0,1,2,3], [4,5,6])
        
        U_final = np.hstack([U247_1, U247_2])
        V_final = np.hstack([V247_1, V247_2])
        W_final = np.hstack([W247_1, W247_2])
        
        # Verify
        if _tensor_check(U_final, V_final, W_final, n, m, p):
            return {
                "U": U_final.tolist(),
                "V": V_final.tolist(),
                "W": W_final.tolist()
            }
        else:
            # Fallback to trivial if construction failed
            U_t, V_t, W_t = _construct_trivial(n, m, p)
            return {
                "U": U_t.tolist(),
                "V": V_t.tolist(),
                "W": W_t.tolist()
            }

    except Exception as e:
        # Fallback to trivial
        U_t, V_t, W_t = _construct_trivial(n, m, p)
        return {
            "U": U_t.tolist(),
            "V": V_t.tolist(),
            "W": W_t.tolist()
        }

if __name__ == "__main__":
    res = construct()
    print("Constructed.")
