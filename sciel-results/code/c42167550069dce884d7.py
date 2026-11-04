"""Matrix multiplication 3x4x6 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]
START_TIME = time.time()
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    rank_schoolbook = n * m * p  # 72
    
    # Build the multiplication tensor T[i,j,k] = 1 if we use term (i,j,k)
    # T has shape (n*m, m*p, p*n) with T[i*m+j, j*p+k, k*n+i] = 1
    # We want to decompose this as sum of rank-1 tensors
    
    # Strategy: Use structured Kronecker decomposition
    # 4 = 2*2, 6 = 2*3
    # The 3x4x6 mult can be seen as (3x2x3) x (2x2x2) structure
    # T_{3,4,6} = T_{3,2,3} ⊗ T_{1,2,2} in some sense
    
    # Let's use the approach: split j into (j1,j2) where j = j1*2 + j2, j1 in {0,1}, j2 in {0,1}
    # split k into (k1,k2) where k = k1*3 + k2, k1 in {0,1}, k2 in {0,1,2}
    
    # Then A[i, j1*2+j2] * B[j1*2+j2, k1*3+k2]
    # = sum over j1,j2 of A[i,j1,j2] * B[j1,j2,k1,k2]
    
    # This is a 2x2 block multiply of 3x3 matrices (with the 6 split as 2x3 blocks)
    
    # Actually, let me try a direct rank construction.
    # For 3x4x6, use the "Larsen" type construction or direct enumeration.
    
    # Let's try: rank = n * ceil(m*p/n) or similar heuristics
    # Known: for 3x4x6, best known is around 54
    
    # Use a structured decomposition based on tensor products:
    # T_{3,4,6} can be built from T_{3,2,3} (rank 18) and T_{1,2,2} (rank 7)
    # but the tensor product gives rank 18*7 = 126, which is worse.
    
    # Better: use the "sum" structure. 
    # Split m=4 into 2+2, p=6 into 3+3
    # A = [A1 | A2] where A1, A2 are 3x2
    # B = [B1; B2] where B1, B2 are 2x3
    # C = A1*B1 + A2*B2 (sum of two 3x2 x 2x3 products)
    # Each 3x2x2 product has rank 12, so total rank <= 24? No, that's not how it works.
    
    # The tensor for 3x2x2 has rank 12 (schoolbook) but can be lower.
    # Actually for 3x2x2, the rank is at most 3*2*2=12, and it's known to be 12.
    
    # Let me try a different approach: direct construction with rank ~54
    
    # Use the fact that we can write the multiplication as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # with j = 0,1,2,3 and k = 0,1,2,3,4,5
    
    # Group terms: for each i, the row of C is a linear combination of rows of B
    # weighted by entries of A
    
    # Let me try a structured rank-1 decomposition:
    # Use terms of the form: e_i ⊗ f_j ⊗ g_k where f_j and g_k have structure
    
    # For now, let's use a greedy rank-1 peeling approach with random initialization
    # and keep the best result found within the time budget.
    
    np.random.seed(42)
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    best_rank = rank_schoolbook
    best_decomp = None
    
    # Schoolbook as fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    for idx, (i, j, k) in enumerate([(i, j, k) for i in range(n) for j in range(m) for k in range(p)]):
        U_sb[i * m + j, idx] = 1.0
        V_sb[j * p + k, idx] = 1.0
        W_sb[k * n + i, idx] = 1.0
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Try structured decompositions
    # Approach 1: Use Kronecker structure 4=2*2, 6=2*3
    # The tensor T_{3,4,6} can be decomposed using T_{3,2,3} and T_{1,2,2}
    # T_{3,2,3} has rank at most 18, T_{1,2,2} has rank at most 4
    # But we need to be more clever.
    
    # Let's try: express j = 2*j1 + j2 (j1 in {0,1}, j2 in {0,1})
    # and k = 3*k1 + k2 (k1 in {0,1}, k2 in {0,1,2})
    
    # Then A[i, 2*j1+j2] * B[2*j1+j2, 3*k1+k2]
    # The sum over j1,j2 of these gives C[i, 3*k1+k2]
    
    # This is equivalent to a 2x2 block matrix multiply where:
    # A_blocks: A[i, 2*j1+j2] for fixed i, varying j1,j2 -> shape (2,2) per i
    # B_blocks: B[2*j1+j2, 3*k1+k2] -> shape (2, 2*3) per... 
    
    # This is getting complex. Let me just do iterative peeling.
    
    def peel_rank1(T_res, max_iter=200):
        """Peel rank-1 components from residual tensor."""
        U_list, V_list, W_list = [], [], []
        T_work = T_res.copy()
        for it in range(max_iter):
            if time.time() > DEADLINE - 5:
                break
            # Find best rank-1 approximation via power iteration on the tensor
            # Flatten and use SVD-like approach
            # For speed, use a simple approach: random init + few iterations
            shape = T_work.shape
            # Use the fact that rank-1 peeling can be done via:
            # u = T_work matricized along mode 1, etc.
            # Simplified: use numpy's tensordot
            U1 = np.zeros(shape[0])
            V1 = np.zeros(shape[1])
            W1 = np.zeros(shape[2])
            U1[0] = 1.0
            V1[0] = 1.0
            W1[0] = 1.0
            for inner in range(20):
                # Update U1: U1[i] = sum_{j,k} T_work[i,j,k] * V1[j] * W1[k]
                U1_new = np.tensordot(T_work, np.outer(V1, W1), axes=([1,2],[0,1]))
                norm = np.linalg.norm(U1_new)
                if norm < 1e-10:
                    break
                U1_new /= norm
                # Update V1
                V1_new = np.tensordot(T_work, np.outer(U1_new, W1), axes=([0,2],[0,1]))
                norm = np.linalg.norm(V1_new)
                if norm < 1e-10:
                    break
                V1_new /= norm
                # Update W1
                W1_new = np.tensordot(T_work, np.outer(U1_new, V1_new), axes=([0,1],[0,1]))
                norm = np.linalg.norm(W1_new)
                if norm < 1e-10:
                    break
                W1_new /= norm
                U1, V1, W1 = U1_new, V1_new, W1_new
            
            # Subtract this rank-1 component
            T_work -= np.einsum('i,j,k->ijk', U1, V1, W1)
            U_list.append(U1)
            V_list.append(V1)
            W_list.append(W1)
            
            # Check if residual is small
            if np.linalg.norm(T_work) < 1e-6:
                break
        
        return U_list, V_list, W_list
    
    # Try peeling with different initializations
    for seed in range(10):
        if time.time() > DEADLINE - 10:
            break
        np.random.seed(seed)
        # Add small noise to break symmetry
        T_noisy = T + 0.01 * np.random.randn(*T.shape)
        U_l, V_l, W_l = peel_rank1(T_noisy, max_iter=100)
        rank = len(U_l)
        if rank < best_rank:
            # Verify exactness: check that sum of rank-1 terms equals T
            U_arr = np.array(U_l).T  # shape (rank, n*m)
            V_arr = np.array(V_l).T  # shape (rank, m*p)
            W_arr = np.array(W_l).T  # shape (rank, p*n)
            T_recon = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
            if np.max(np.abs(T_recon - T)) < 1e-6:
                best_rank = rank
                best_decomp = (U_arr, V_arr, W_arr)
    
    # Also try a structured construction
    # For 3x4x6, use the "4 = 2+2" split with shared structure
    # Rank contribution: for each of the 2 blocks, we need rank for 3x2x3 mult
    # 3x2x3 schoolbook rank = 18, two blocks = 36? But we can share.
    
    # Actually, let's try a direct construction with rank 54:
    # Use the structure where we have 3x3 "core" with 6x6 "expansion"
    
    # For the final answer, use the best found
    if best_decomp is not None:
        U_final, V_final, W_final = best_decomp
        # Convert to lists as required
        U_out = U_final.tolist()
        V_out = V_final.tolist()
        W_out = W_final.tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Fallback to schoolbook
        U_out = U_sb.tolist()
        V_out = V_sb.tolist()
        W_out = W_sb.tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
