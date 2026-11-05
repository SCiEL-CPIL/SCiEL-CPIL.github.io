import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We target finishing by 130s.
    deadline = start_time + 130.0
    
    def time_check():
        return time.time() > deadline - 5.0

    # Initial Rank 33 Solution Construction
    # Structure:
    # - Columns 0-11: Schoolbook for A[:, 0] * B[0, :] (2x1 * 1x6)
    # - Columns 12-32: Strassen for A[:, 1:3] * B[1:, :] (2x2 * 2x6, split into 3 blocks of 2x2x2)
    
    rank_init = 33
    U = np.zeros((n * m, rank_init), dtype=np.float64)
    V = np.zeros((m * p, rank_init), dtype=np.float64)
    W = np.zeros((p * n, rank_init), dtype=np.float64)
    
    col_idx = 0
    
    # Part 1: Schoolbook for column 0 of A (indices 0, 1 in U)
    # A[0,0] and A[1,0] multiply with B[0, 0..5]
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col_idx] = 1.0
            V[0 * p + k, col_idx] = 1.0
            W[k * n + i, col_idx] = 1.0
            col_idx += 1
            
    # Part 2: Strassen for the 2x2 block (A[:, 1:3] and B[1:, :])
    # We split the 6 columns of B into 3 blocks of 2 columns each.
    # Block t corresponds to B columns 2t and 2t+1.
    # A columns 1 and 2 correspond to U indices 3, 4 (A[0,1], A[0,2]) and 6, 7 (A[1,1], A[1,2])
    # Note: U index = i * m + j. 
    # i=0, j=1 -> idx 1? No.
    # U shape is (n*m, rank). Index is i*m + j.
    # n=2, m=3.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=0, j=2 -> 2
    # i=1, j=0 -> 3
    # i=1, j=1 -> 4
    # i=1, j=2 -> 5
    
    # Wait, my previous manual trace in the thought process had:
    # U[0*m+1] is i=0, j=1.
    # U[1*m+2] is i=1, j=2.
    # Let's re-verify indices carefully.
    # A is 2x3.
    # A[0,0] -> U[0]
    # A[0,1] -> U[1]
    # A[0,2] -> U[2]
    # A[1,0] -> U[3]
    # A[1,1] -> U[4]
    # A[1,2] -> U[5]
    
    # The "2x2 block" for Strassen usually involves rows 0,1 and cols 1,2 of A?
    # Yes, A[:, 1:3] is a 2x2 matrix.
    # Let A_sub = [[A[0,1], A[0,2]], [A[1,1], A[1,2]]]
    # A_sub[0,0] = A[0,1] -> U[1]
    # A_sub[0,1] = A[0,2] -> U[2]
    # A_sub[1,0] = A[1,1] -> U[4]
    # A_sub[1,1] = A[1,2] -> U[5]
    
    # B is 3x6.
    # B[1:, :] is a 2x6 matrix.
    # We split into 3 blocks of 2 columns:
    # Block 0: B[1, 0], B[1, 1] -> V[0*p+0]=V[0], V[0*p+1]=V[1]?
    # V shape is (m*p, rank). Index is j*p + k.
    # j=1, k=0 -> V[6]
    # j=1, k=1 -> V[7]
    # j=2, k=0 -> V[12]
    # j=2, k=1 -> V[13]
    # j=1, k=2 -> V[8]
    # j=1, k=3 -> V[9]
    # j=2, k=2 -> V[14]
    # j=2, k=3 -> V[15]
    # j=1, k=4 -> V[10]
    # j=1, k=5 -> V[11]
    # j=2, k=4 -> V[16]
    # j=2, k=5 -> V[17]
    
    # W is 6x2. Index is k*n + i.
    # Output C is 2x6.
    
    for t in range(3):
        # k0, k1 are the column indices in the full B matrix for this block
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # V indices for B[1, k0], B[1, k1], B[2, k0], B[2, k1]
        v_1_k0 = 1 * p + k0
        v_1_k1 = 1 * p + k1
        v_2_k0 = 2 * p + k0
        v_2_k1 = 2 * p + k1
        
        # U indices for A_sub
        u_00 = 1  # A[0,1]
        u_01 = 2  # A[0,2]
        u_10 = 4  # A[1,1]
        u_11 = 5  # A[1,2]
        
        # W indices for C[?, k0], C[?, k1]
        # C[0, k0] -> W[k0*n + 0]
        # C[1, k0] -> W[k0*n + 1]
        # C[0, k1] -> W[k1*n + 0]
        # C[1, k1] -> W[k1*n + 1]
        
        w_0_k0 = k0 * n + 0
        w_1_k0 = k0 * n + 1
        w_0_k1 = k1 * n + 0
        w_1_k1 = k1 * n + 1
        
        # Strassen Formulas for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        U[u_00, col_idx] += 1.0
        U[u_11, col_idx] += 1.0
        V[v_1_k0, col_idx] += 1.0
        V[v_2_k1, col_idx] += 1.0
        W[w_0_k0, col_idx] += 1.0
        W[w_1_k1, col_idx] += 1.0
        col_idx += 1
        
        # M2 = (A10+A11)B00
        U[u_10, col_idx] += 1.0
        U[u_11, col_idx] += 1.0
        V[v_1_k0, col_idx] += 1.0
        W[w_1_k0, col_idx] += 1.0
        W[w_1_k1, col_idx] += -1.0
        col_idx += 1
        
        # M3 = A00(B10-B11)
        U[u_00, col_idx] += 1.0
        V[v_2_k0, col_idx] += 1.0
        V[v_2_k1, col_idx] += -1.0
        W[w_0_k1, col_idx] += 1.0
        W[w_1_k1, col_idx] += 1.0
        col_idx += 1
        
        # M4 = A11(B01-B00)
        U[u_11, col_idx] += 1.0
        V[v_1_k1, col_idx] += 1.0
        V[v_1_k0, col_idx] += -1.0
        W[w_0_k0, col_idx] += 1.0
        W[w_1_k0, col_idx] += 1.0
        col_idx += 1
        
        # M5 = (A00+A01)B11
        U[u_00, col_idx] += 1.0
        U[u_01, col_idx] += 1.0
        V[v_2_k1, col_idx] += 1.0
        W[w_0_k0, col_idx] += -1.0
        W[w_0_k1, col_idx] += 1.0
        col_idx += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[u_10, col_idx] += 1.0
        U[u_00, col_idx] += -1.0
        V[v_1_k0, col_idx] += 1.0
        V[v_1_k1, col_idx] += 1.0
        W[w_1_k1, col_idx] += 1.0
        col_idx += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[u_01, col_idx] += 1.0
        U[u_11, col_idx] += -1.0
        V[v_2_k0, col_idx] += 1.0
        V[v_2_k1, col_idx] += 1.0
        W[w_0_k0, col_idx] += 1.0
        col_idx += 1

    def verify_decomp(U_mat, V_mat, W_mat, tol=1e-9):
        """
        Verifies if U, V, W exactly decompose the matrix multiplication tensor.
        Tensor T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} should be 1 if i,j,k form a valid path, 0 otherwise.
        Actually, the standard tensor for matmul A(i,j)B(j,k) -> C(i,k) is T_{i,k,j} or similar depending on convention.
        Convention given: W[k*n+i, r].
        U is (n*m, r) -> U[i*m+j, r]
        V is (m*p, r) -> V[j*p+k, r]
        W is (p*n, r) -> W[k*n+i, r]
        
        The tensor decomposition identity is:
        delta_{i,i} delta_{j,j} delta_{k,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
        Wait, the product C_{i,k} = sum_j A_{i,j} B_{j,k}.
        The tensor T_{i,j,k} = 1 if it's a valid matmul term? No.
        The tensor for matmul is T_{i,k,j} = 1?
        Let's check the contraction:
        sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
        If we fix i,j,k, and sum over r, do we get 1?
        For schoolbook: U_{i,j,r}=1, V_{j,k,r}=1, W_{k,i,r}=1 for specific r.
        The sum is 1.
        So we need T_{i,j,k} = 1 for all i,j,k?
        No, the tensor T is defined such that T_{i,j,k} = 1.
        Because A_{i,j} B_{j,k} contributes to C_{i,k}.
        The "tensor" of multiplication is usually defined on indices (i, k, j) where T_{i,k,j} = 1.
        Or (i, j, k) where T_{i,j,k} = 1.
        Let's verify with schoolbook:
        U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
        Product is 1.
        So yes, the target tensor is all ones?
        Wait. If I pick i=0, j=1, k=0.
        C_{0,0} = sum_j A_{0,j} B_{j,0}.
        Term j=1: A_{0,1} B_{1,0}.
        This term exists.
        So for ANY i,j,k, there is a term in the sum?
        Yes, the tensor is indeed all 1s?
        No, that's not right.
        If the tensor is all 1s, then rank is just the dimension? No.
        Let's re-read the definition of matrix multiplication tensor.
        Usually, T_{i,k,j} = 1.
        Here, the indices are mapped to vector indices.
        U index: i*m + j.
        V index: j*p + k.
        W index: k*n + i.
        
        The equation is:
        C_{i,k} = sum_j A_{i,j} B_{j,k}
        In tensor form:
        C_{i,k} = sum_{j,r} A_{i,j} B_{j,k} ... no.
        
        The decomposition is:
        T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
        where T_{i,j,k} = 1 for all valid i,j,k.
        Is that true?
        If T_{i,j,k} = 1, then we are decomposing the all-ones tensor.
        This seems wrong for matrix multiplication.
        
        Let's look at the standard "matrix multiplication tensor" $\mu$.
        $\mu = \sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}$?
        No, $\mu = \sum_{i,j,k} e_{i} \otimes e_{k} \otimes e_{j}$?
        
        Let's check the contraction result.
        If we contract the tensor $T$ with vectors $a, b, c$:
        $\sum_{i,j,k} T_{i,j,k} a_{i,j} b_{j,k} c_{k,i}$?
        Usually, we contract to get the scalar product of the matmul.
        $\text{Tr}(C^T D) = \sum_{i,k} C_{i,k} D_{i,k} = \sum_{i,k} D_{i,k} (\sum_j A_{i,j} B_{j,k}) = \sum_{i,j,k} A_{i,j} B_{j,k} D_{i,k}$.
        So $T_{i,j,k} = 1$ for all $i,j,k$.
        
        Yes! The tensor for matrix multiplication $A(n \times m) \times B(m \times p) \to C(n \times p)$ is indeed the all-ones tensor on indices $i \in \{0..n-1\}, j \in \{0..m-1\}, k \in \{0..p-1\}$.
        Why? Because every term $A_{i,j} B_{j,k}$ contributes to some $C_{i,k}$.
        There are no "zeros" in the structural support of the tensor itself.
        The "zeros" come from the fact that we only sum over $j$ for a fixed $i,k$.
        But the tensor rank is about representing the multilinear form.
        The multilinear form is $L(A,B,C) = \sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}$?
        Wait, $C_{i,k}$ is the output.
        The bilinear map is $A, B \to C$.
        The tensor is defined on the dual spaces?
        
        Let's stick to the verification provided in the prompt's implied logic or standard Strassen verification.
        Standard verification:
        Construct $A, B$ with random values.
        Compute $C = A @ B$.
        Compute $\hat{C}$ via decomposition:
        $\hat{C}_{i,k} = \sum_r W_{k,n+i,r} ( \sum_j U_{i,m+j,r} A_{i,j} ) ( \sum_j V_{j,p+k,r} B_{j,k} )$?
        No, the decomposition is:
        $C_{i,k} = \sum_r \left( \sum_j U_{i,m+j,r} A_{i,j} \right) \left( \sum_k' V_{j,p+k',r} B_{j,k'} \right)$?
        
        Let's look at the indices again.
        $U_{i*m+j, r}$
        $V_{j*p+k, r}$
        $W_{k*n+i, r}$
        
        The decomposition identity is:
        $\delta_{i, i'} \delta_{k, k'} \delta_{j, j'}$? No.
        
        Let's just verify numerically.
        If the tensor is all ones, then:
        $\sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1$.
        
        Let's test this with schoolbook.
        $r$ runs over $i,j,k$ triples.
        $U_{i,j, (i,j,k)} = 1$ only if indices match.
        $V_{j,k, (i,j,k)} = 1$ only if indices match.
        $W_{k,i, (i,j,k)} = 1$ only if indices match.
        Sum is 1.
        
        So, **YES, the target tensor is all 1s.**
        My verification function must check if $\sum_r U_{i,m+j,r} V_{j,p+k,r} W_{k,n+i,r} = 1$ for all $i,j,k$.
        
        Using einsum:
        `np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)`
        Where:
        `U_r` has shape `(n, m, r)` -> `U[i, j, r]` corresponds to `U[i*m+j, r]`.
        `V_r` has shape `(m, p, r)` -> `V[j, k, r]` corresponds to `V[j*p+k, r]`.
        `W_r` has shape `(p, n, r)` -> `W[k, i, r]` corresponds to `W[k*n+i, r]`.
        
        Result `T[i, j, k]` should be 1.
        """
        if U_mat.size == 0:
            return False
        r = U_mat.shape[1]
        U_r = U_mat.reshape(n, m, -1)
        V_r = V_mat.reshape(m, p, -1)
        W_r = W_mat.reshape(p, n, -1)
        
        # Compute T_{i,j,k}
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        
        # Check if T is all ones
        return np.allclose(T, 1.0, atol=1e-9)

    # Verify initial rank 33
    if not verify_decomp(U, V, W):
        # Fallback to pure schoolbook (Rank 36)
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school))
        V_s = np.zeros((m * p, rank_school))
        W_s = np.zeros((p * n, rank_school))
        col_s = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_s[ii * m + jj, col_s] = 1.0
                    V_s[jj * p + kk, col_s] = 1.0
                    W_s[kk * n + ii, col_s] = 1.0
                    col_s += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_init

    # Optimization: Try to reduce rank by removing columns
    # We iterate from best_rank - 1 down to a lower bound (e.g., 25)
    # For each target rank, we try random subsets of columns of the current best solution.
    
    lower_bound = 25
    
    for target_rank in range(best_rank - 1, lower_bound - 1, -1):
        if time_check():
            break
            
        found_better = False
        
        # Heuristic: If target_rank is close to best_rank, try removing single columns first
        if target_rank == best_rank - 1:
            for col_to_remove in range(best_rank):
                if time_check():
                    break
                mask = np.ones(best_rank, dtype=bool)
                mask[col_to_remove] = False
                U_test = best_U[:, mask]
                V_test = best_V[:, mask]
                W_test = best_W[:, mask]
                
                if verify_decomp(U_test, V_test, W_test):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_test, V_test, W_test
                    found_better = True
                    break
        
        # Random Subset Search
        if not found_better:
            # Use a fixed seed for reproducibility within the run, but vary by target_rank
            np.random.seed(42 + target_rank)
            max_attempts = 1000 if (best_rank - target_rank) > 1 else 100
            for _ in range(max_attempts):
                if time_check():
                    break
                cols = np.random.choice(best_rank, target_rank, replace=False)
                U_sub = best_U[:, cols]
                V_sub = best_V[:, cols]
                W_sub = best_W[:, cols]
                
                if verify_decomp(U_sub, V_sub, W_sub):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_sub, V_sub, W_sub
                    found_better = True
                    break
                    
        if not found_better:
            # If we couldn't reduce to this rank, we might still try lower ranks 
            # from the current best, but it's unlikely to succeed if we failed here.
            # However, sometimes a rank-30 solution exists inside rank-32 even if no rank-31 exists.
            # So we continue the loop to lower bounds.
            continue

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
