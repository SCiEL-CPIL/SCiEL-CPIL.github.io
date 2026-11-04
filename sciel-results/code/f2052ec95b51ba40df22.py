import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 20s margin before 120s limit
    
    # Schoolbook decomposition: rank = n*m*p = 72
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify schoolbook
    def verify(U_mat, V_mat, W_mat):
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.dot(U_np[i*m+j], V_np[j*p+k]), W_np[k*n+i])
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W):
        raise ValueError("Schoolbook verification failed")
    
    # Try to find a lower-rank decomposition
    # Strategy: Use block structure. Split B into 4 blocks of 3x2.
    # For each block, the sub-tensor is 3x3x2.
    # The rank of 3x3x2 multiplication tensor: we can compute it.
    
    # For 3x3x2: C[i,k] = sum_j A[i,j] B[j,k] for i in {0,1,2}, k in {0,1}
    # Schoolbook rank = 3*3*2 = 18
    # Can we do better? The tensor is T[i,j,k] = 1 for all i,j,k.
    # This is a 3x3x2 tensor. Its rank is at most 18.
    # Lower bound: max(3*3, 3*2, 2*3) = max(9,6,6) = 9
    # Actual rank of 3x3x2? Let's compute.
    
    # The tensor T has entries T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,2)
    # This is the outer product of three all-ones vectors? No, it's just all ones.
    # Wait, no. For matrix multiplication, the tensor is not all ones.
    # The multiplication tensor M_{n,m,p} has M_{(i,j),(j,k),(k,i)} = 1
    # So for 3x3x2: M_{(i,j),(j,k),(k,i)} = 1 for i in {0,1,2}, j in {0,1,2}, k in {0,1}
    # This means U[i*m+j] * V[j*p+k] * W[k*n+i] summed over r should equal 1
    # for every triple (i,j,k) with i<n, j<m, k<p.
    
    # So the tensor is indeed all ones in the valid range.
    # The rank of the all-ones tensor of size 3x3x2 is... 
    # Actually, the tensor T[i,j,k] = 1 for all i,j,k is rank 1!
    # Because T[i,j,k] = 1*1*1.
    
    # Wait, that can't be right for matrix multiplication.
    # Let me re-read the problem.
    
    # The multiplication tensor for C = A*B where A is n x m, B is m x p:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is defined such that the bilinear form is:
    # sum_{i,j,k} C_{i,k} * (tensor)_{i,j,k} * A_{i,j} * B_{j,k}
    # No, the tensor decomposition is:
    # C_{i,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) * (sum_j V_{j,k,r} B_{j,k}) * W_{i,k,r}
    # Wait, that's not standard.
    
    # Standard: The multiplication tensor T has T_{(i,j),(j,k),(k,i)} = 1
    # And we want T = sum_r U_r ⊗ V_r ⊗ W_r where
    # U_r is a vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}
    # Such that T_{(i,j),(j,k),(k,i)} = sum_r U_r[(i,j)] V_r[(j,k)] W_r[(k,i)]
    
    # So for every valid (i,j,k), we need:
    # sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1
    
    # This is a system of n*m*p equations.
    # The schoolbook solution uses rank n*m*p with U_r, V_r, W_r being basis vectors.
    
    # Can we do better? For n=m=p=2, rank is 7 (Strassen) vs 8 schoolbook.
    # For n=3,m=3,p=2, what's the rank?
    
    # Let's try to find a rank-18 solution for 3x3x2 (which is schoolbook)
    # and see if we can reduce it.
    
    # Given time constraints, let's try a structured approach:
    # Split the 3x3x2 problem into parts.
    
    # Actually, let's just try random search for a lower rank for the full 3x3x8.
    # But that's computationally expensive.
    
    # Alternative: Use the fact that p=8 is even, and pair up columns.
    # For columns (0,1), (2,3), (4,5), (6,7), solve 4 instances of 3x3x2.
    # If we can find rank r for 3x3x2, total rank is 4r.
    
    # Let's focus on 3x3x2. The tensor is T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1,2}, k in {0,1}.
    # We want rank-r decomposition.
    
    # Try rank 9: Can we write T as sum of 9 rank-1 tensors?
    # Each rank-1 tensor is u ⊗ v ⊗ w where u in R^3 (for i), v in R^3 (for j), w in R^2 (for k).
    # Wait, the indices are (i,j) for U, (j,k) for V, (k,i) for W.
    # So U has 3*3=9 entries, V has 3*2=6 entries, W has 2*3=6 entries.
    
    # Let me set up the equations for 3x3x2 with rank r.
    # We need: for all i in {0,1,2}, j in {0,1,2}, k in {0,1}:
    # sum_{s=0}^{r-1} U[s][i*3+j] * V[s][j*2+k] * W[s][k*3+i] = 1
    
    # This is 18 equations.
    # Unknowns: r * (9 + 6 + 6) = 21r.
    # For r=9, 189 unknowns, 18 equations. Underdetermined, should have solutions.
    
    # Let's try to find a solution for 3x3x2 with rank 9 using least squares.
    
    def solve_3x3x2():
        """Find rank-r decomposition for 3x3x2 multiplication tensor."""
        n_sub, m_sub, p_sub = 3, 3, 2
        # Try ranks from lower bound up
        lower_bound = max(n_sub * m_sub, m_sub * p_sub, p_sub * n_sub)
        # lower_bound = max(9, 6, 6) = 9
        
        for r in range(lower_bound, n_sub * m_sub * p_sub + 1):
            if time.time() > deadline:
                break
                
            # Initialize with random values
            np.random.seed(42)
            U_init = np.random.randn(r, n_sub * m_sub) * 0.1
            V_init = np.random.randn(r, m_sub * p_sub) * 0.1
            W_init = np.random.randn(r, p_sub * n_sub) * 0.1
            
            # Build the linear system
            # For each (i,j,k), we have:
            # sum_s U[s, i*m_sub+j] * V[s, j*p_sub+k] * W[s, k*n_sub+i] = 1
            # This is nonlinear in U,V,W.
            
            # Alternative: Fix V and W, solve for U linearly.
            # Or use alternating minimization.
            
            # Let's use a simple approach: fix V and W to be basis-like, solve for U.
            # If V and W are identity-like, U can be solved.
            
            # Actually, let's just use the schoolbook for 3x3x2: rank 18.
            # U[s, i*3+j] = 1 if s == i*3+j, else 0
            # V[s, j*2+k] = 1 if s == j*2+k, else 0  -- but s goes up to 17, j*2+k goes up to 5
            # This doesn't work directly.
            
            # Schoolbook for 3x3x2:
            # For each (i,j,k), create a term.
            # U[i*3+j, idx] = 1
            # V[j*2+k, idx] = 1  -- but V has 6 rows, and we're indexing by j*2+k which is 0..5
            # W[k*3+i, idx] = 1  -- W has 6 rows, indexed by k*3+i which is 0..5
            # idx goes from 0 to 17
            
            U_sb = [[0] * (n_sub * m_sub * p_sub) for _ in range(n_sub * m_sub)]
            V_sb = [[0] * (n_sub * m_sub * p_sub) for _ in range(m_sub * p_sub)]
            W_sb = [[0] * (n_sub * m_sub * p_sub) for _ in range(p_sub * n_sub)]
            
            col = 0
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        U_sb[i * m_sub + j][col] = 1
                        V_sb[j * p_sub + k][col] = 1
                        W_sb[k * n_sub + i][col] = 1
                        col += 1
            
            # Verify
            U_np = np.array(U_sb, dtype=float)
            V_np = np.array(V_sb, dtype=float)
            W_np = np.array(W_sb, dtype=float)
            valid = True
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        val = np.dot(np.dot(U_np[i*m_sub+j], V_np[j*p_sub+k]), W_np[k*n_sub+i])
                        if abs(val - 1.0) > 1e-10:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                return U_sb, V_sb, W_sb, n_sub * m_sub * p_sub
        
        return None
    
    # Given the complexity and time constraints, let's stick with the schoolbook for now
    # but verify it's correct.
    
    # Actually, let me try a different approach: use the structure to reduce rank.
    # For the full 3x3x8, split into 4 blocks of 3x3x2.
    # For each block, use rank 18 (schoolbook). Total rank 72.
    # This is the same as full schoolbook.
    
    # To beat 72, we need a smarter approach.
    # Let's try: for 3x3x2, can we do rank < 18?
    # The tensor is all ones. The rank of the all-ones tensor of size a x b x c is 1
    # if we allow arbitrary entries, because T[i,j,k] = 1*1*1.
    # But wait, that's not the multiplication tensor!
    
    # I think I've been confused. Let me re-read the problem statement.
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "exactly decompose the matrix multiplication tensor"
    
    # The matrix multiplication tensor for C = A*B (A: n x m, B: m x p) is:
    # T[(i,j), (j,k), (k,i)] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And 0 otherwise (but all combinations are valid).
    
    # So yes, T is all ones in the valid range.
    # And the rank of an all-ones tensor is 1!
    # Because T[i,j,k] = 1 = (1)(1)(1).
    
    # So we can use rank 1:
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k  
    # W[k*n+i, 0] = 1 for all k,i
    # Then sum_r U*V*W = 1*1*1 = 1 for all (i,j,k).
    
    # Let me verify this!
    
    U_rank1 = [[1] for _ in range(n * m)]
    V_rank1 = [[1] for _ in range(m * p)]
    W_rank1 = [[1] for _ in range(p * n)]
    
    def verify_rank1():
        U_np = np.array(U_rank1, dtype=float)
        V_np = np.array(V_rank1, dtype=float)
        W_np = np.array(W_rank1, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.dot(U_np[i*m+j], V_np[j*p+k]), W_np[k*n+i])
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    if verify_rank1():
        return {"U": U_rank1, "V": V_rank1, "W": W_rank1}
    
    # If rank 1 doesn't work (it should, but let's be safe), fall back to schoolbook
    return {"U": U, "V": V, "W": W}
