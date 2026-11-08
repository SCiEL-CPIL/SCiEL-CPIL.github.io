"""
Tensor decomposition for matrix multiplication 3x3 * 3x8.
Attempts to find a rank-minimal exact tensor decomposition.
Starts with the known rank-68 construction (Strassen + Direct) and 
attempts to reduce rank via randomized ALS optimization.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # 1. Construct the base rank-68 decomposition
    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)
        
        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)
        
        # m1 = (a+d)(e+h)
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])
        
        # m2 = (c+d)*e
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])
        
        # m3 = a*(f-h)
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])
        
        # m4 = d*(g-e)
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])
        
        # m5 = (a+b)*h
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])
        
        # m6 = (c-a)*(e+f)
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])
        
        # m7 = (b-d)*(g+h)
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])
        
    def direct_term(i, j, k, U_t, V_t, W_t):
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])
        
    U_t = []
    V_t = []
    W_t = []
    
    # 4 Strassen blocks: A[0:2,0:2] * B[0:2, k:k+2] for k=0,2,4,6
    for k_off in [0, 2, 4, 6]:
        strassen_decomp(0, 0, 0, k_off, U_t, V_t, W_t)
        
    # A[0:2, 2] * B[2, k:k+2]
    for k_off in [0, 2, 4, 6]:
        for i in range(2):
            for k in range(k_off, k_off + 2):
                direct_term(i, 2, k, U_t, V_t, W_t)
                
    # A[2, 0:2] * B[0:2, k:k+2]
    for k_off in [0, 2, 4, 6]:
        for j in range(2):
            for k in range(k_off, k_off + 2):
                direct_term(2, j, k, U_t, V_t, W_t)
                
    # A[2, 2] * B[2, k:k+2]
    for k_off in [0, 2, 4, 6]:
        for k in range(k_off, k_off + 2):
            direct_term(2, 2, k, U_t, V_t, W_t)
            
    base_rank = len(W_t)
    
    # Build base factor matrices
    U_base = np.zeros((n * m, base_rank))
    V_base = np.zeros((m * p, base_rank))
    W_base = np.zeros((p * n, base_rank))
    
    for idx in range(base_rank):
        for (i, j), coeff in U_t[idx]:
            U_base[i * m + j, idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V_base[j * p + k, idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W_base[k * n + i, idx] = coeff
            
    # Verification function
    def verify_factors(U, V, W):
        # Check if U, V, W exactly decompose the tensor
        # Tensor T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # We want T[i, j, k] = 1 if i,j,k correspond to a valid product term? 
        # No, the tensor for matrix multiplication M_{n,m,p} has entries 1 at (i,j,k) 
        # where the output index k corresponds to the product of A[i,j] and B[j,k].
        # In the trace convention W[k*n+i, r], the index in W is k*n+i.
        # The tensor entry T[i, j, k] should be 1.
        
        # Let's reconstruct the tensor from factors
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # We can check a subset or all. n*m*p = 72 entries.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify_factors(U_base, V_base, W_base):
        # Fallback to schoolbook if base is broken (should not happen)
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = base_rank
    
    # Attempt to reduce rank to 67
    target_rank = base_rank - 1
    if target_rank >= 1:
        # Time limit guard
        start_time = time.time()
        deadline = 120.0 # Safety margin for 150s limit
        
        # Initialize with base factors, drop one column
        # We can try dropping each column one by one or random subsets
        # Simpler: Random restarts with rank 67
        # Initialize U, V, W with random noise + base projection?
        # Better: Take base factors, remove one column, and optimize.
        
        for attempt in range(10): # Limited attempts to stay within time
            if time.time() > deadline - 5:
                break
                
            # Pick a random column to drop from the base
            drop_idx = np.random.randint(0, base_rank)
            keep_mask = np.ones(base_rank, dtype=bool)
            keep_mask[drop_idx] = False
            
            U_init = U_base[:, keep_mask].copy()
            V_init = V_base[:, keep_mask].copy()
            W_init = W_base[:, keep_mask].copy()
            
            # Add small noise to escape local minima
            U_init += np.random.normal(0, 0.1, U_init.shape)
            V_init += np.random.normal(0, 0.1, V_init.shape)
            W_init += np.random.normal(0, 0.1, W_init.shape)
            
            # ALS Optimization
            # Minimize || T - sum_r U_r V_r W_r ||^2
            # T is the matrix multiplication tensor (ones at specific positions)
            
            # Precompute indices
            idx_i = np.arange(n)
            idx_j = np.arange(m)
            idx_k = np.arange(p)
            
            # We update U, V, W in turns.
            # For each r, U[:, r] is a vector.
            # Fix V, W. Update U.
            # U[i, r] = sum_j,k T[i,j,k] V[j, r] W[k, r] / sum_j,k V[j, r]^2 W[k, r]^2 ?
            # Actually, the tensor T is sparse (only 72 non-zeros, all 1).
            
            # Let's use a simpler gradient-free or fixed-point iteration.
            # For a fixed r, U[:, r] should be proportional to the "profile" of the tensor along i.
            
            # Standard ALS for CP decomposition of a sparse tensor:
            # U[:, r] = T_unfold_1 @ (V[:, r] * W[:, r]) / (V[:, r] * W[:, r]) @ (V[:, r] * W[:, r])
            
            U_curr = U_init
            V_curr = V_init
            W_curr = W_init
            
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                # For each i: U[i, r] = sum_{j,k} T[i,j,k] V[j, r] W[k, r] / norm
                # T[i,j,k] is 1 for all i,j,k in the multiplication tensor?
                # YES, the tensor for matrix multiplication M_{n,m,p} has 1s at all (i,j,k) 
                # because A_{i,j} B_{j,k} contributes to C_{i,k}.
                # Wait, is T[i,j,k] always 1?
                # The tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k.
                # Yes, every combination (i,j,k) appears exactly once with coefficient 1.
                
                # So U[:, r] is proportional to sum_{j,k} V[j, r] W[k, r] e_i ?
                # No, U[i, r] = sum_{j,k} 1 * V[j, r] W[k, r] / (sum_{j',k'} V[j', r]^2 W[k', r]^2)
                # This implies U[i, r] is constant for all i?
                # That can't be right. The tensor is not separable in that way.
                # The tensor T_{ijk} = 1.
                # Unfolding T along mode 1: T_1[i, (j,k)] = 1.
                # U[:, r] = T_1 @ (V[:, r] * W[:, r]) / ||V[:, r] * W[:, r]||^2
                # Since T_1 is all ones, U[:, r] is constant for all i.
                # This suggests that a rank-1 decomposition of the all-ones tensor is trivial,
                # but we are decomposing the *multiplication* tensor, which is NOT all ones in the standard basis?
                
                # Correction:
                # The tensor T for matrix multiplication is defined such that:
                # C_{i,k} = sum_j A_{i,j} B_{j,k}
                # This is a multilinear map. The tensor T has components T_{i, j, k} = 1.
                # Yes, T_{i,j,k} = 1 for all i, j, k.
                # If T_{i,j,k} = 1 for all i,j,k, then the tensor is the outer product of 
                # vectors of ones?
                # T = 1_i \otimes 1_j \otimes 1_k ?
                # If so, the rank is 1!
                
                # WAIT.
                # The tensor decomposition of matrix multiplication is NOT the all-ones tensor.
                # The tensor is $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$.
                # This IS the all-ones tensor in the standard basis.
                # But the *rank* of this tensor is not 1.
                # Why? Because the decomposition must satisfy the *structure* of matrix multiplication?
                # No, the tensor rank is a property of the tensor itself.
                # The all-ones tensor $J_{n,m,p}$ has rank 1?
                # $J_{ijk} = 1 = 1_i 1_j 1_k$.
                # So the rank is 1.
                
                # I am confused. Let's re-read the definition.
                # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 8 matrix."
                # The tensor is the *matrix multiplication tensor* $\mathcal{M}_{n,m,p}$.
                # $\mathcal{M}_{n,m,p} = \sum_{i=1}^n \sum_{j=1}^m \sum_{k=1}^p e_i \otimes e_j \otimes e_k$.
                # This tensor has entries 1 at every index.
                # Its tensor rank is 1?
                # No, the tensor rank of the all-ones tensor is 1.
                
                # So why is the rank of matrix multiplication 68?
                # Ah, the tensor is NOT the all-ones tensor.
                # The tensor is $\mathcal{M} = \sum_{i,j,k} E_{ij} \otimes E_{jk} \otimes E_{ik}$? No.
                
                # Let's check the standard definition.
                # The matrix multiplication tensor $M_{n,m,p}$ is an element of $\mathbb{R}^{n \times m} \otimes \mathbb{R}^{m \times p} \otimes \mathbb{R}^{n \times p}$.
                # Its components are $M_{(i,j), (j,k), (i,k)} = 1$?
                # No, the indices are:
                # 1. Input A index: $(i, j)$
                # 2. Input B index: $(j, k)$
                # 3. Output C index: $(i, k)$
                # The tensor entry $T_{(i,j), (j,k), (i,k)}$ is 1 if the indices match, 0 otherwise?
                # No, the tensor is defined by the bilinear map.
                # $T((i,j), (j',k), (i',k')) = 1$ if $i=i', j=j', k=k'$?
                # No.
                
                # Correct definition:
                # $T$ is the tensor such that $\langle T, A \otimes B \otimes C \rangle = \text{trace}(A B C^T)$?
                # Or $C_{ik} = \sum_j A_{ij} B_{jk}$.
                # The tensor $T$ has non-zero entries at $T_{(i,j), (j,k), (i,k)} = 1$.
                # Note that the middle index $j$ is shared.
                # So $T_{\alpha, \beta, \gamma} = 1$ if $\alpha=(i,j), \beta=(j,k), \gamma=(i,k)$.
                # This is NOT the all-ones tensor. It is sparse in the sense that the indices are linked.
                # In the flattened vector space:
                # Index 1: $i \cdot m + j$
                # Index 2: $j \cdot p + k$
                # Index 3: $k \cdot n + i$
                # $T_{I, J, K} = 1$ if $J$'s row index equals $I$'s col index, and $K$'s col index equals $J$'s col index, and $K$'s row index equals $I$'s row index.
                
                # So $T$ is a 0-1 tensor with specific structure.
                
                # My ALS update rule:
                # $U[:, r] = T_1 @ (V[:, r] * W[:, r]) / ||V[:, r] * W[:, r]||^2$
                # $T_1$ is the unfolding of T along mode 1.
                # $T_1[I, (J, K)] = T_{I, J, K}$.
                # This is not all ones.
                
                # We can compute $T_1 @ v$ efficiently?
                # $v = V[:, r] * W[:, r]$.
                # $res[I] = \sum_{J, K} T_{I, J, K} v[J] v[K]$.
                # $I = i \cdot m + j$.
                # $T_{I, J, K} = 1$ if $J = j \cdot p + k$ and $K = k \cdot n + i$.
                # So for a fixed $I=(i,j)$, the non-zero $J$ must have row $j$.
                # $J \in \{ j \cdot p + k \mid k \in 0 \dots p-1 \}$.
                # And for each such $J=(j,k)$, $K$ must be $k \cdot n + i$.
                # So $res[i \cdot m + j] = \sum_{k=0}^{p-1} v[j \cdot p + k] \cdot v[k \cdot n + i]$.
                
                # This is computable in $O(n \cdot m \cdot p)$.
                
                def compute_T1_dot(v):
                    # v is a vector of size m*p * p*n? No, v is V[:,r] * W[:,r] ?
                    # Wait, in ALS, we fix V and W.
                    # We want to update U.
                    # U[:, r] is a vector of size n*m.
                    # U[i*m+j, r] = \sum_{j', k'} T_{i*m+j, j'*p+k', k'*n+i} V[j'*p+k', r] W[k'*n+i, r]
                    # The condition for T=1 is j'=j and k'=k (where k is from K index).
                    # Let's re-verify indices.
                    # T_{I, J, K} = 1 iff:
                    # I = i*m + j
                    # J = j*p + k
                    # K = k*n + i
                    # So for fixed I=(i,j), J must be (j,k) for some k, and K must be (k,i).
                    # Sum over k: V[j*p+k, r] * W[k*n+i, r].
                    
                    res = np.zeros(n*m)
                    for i in range(n):
                        for j in range(m):
                            s = 0.0
                            for k in range(p):
                                s += V_curr[j*p+k, r] * W_curr[k*n+i, r]
                            res[i*m+j] = s
                    return res

                # This is too slow for 50 iterations * 10 attempts * 67 rank.
                # n=3, m=3, p=8. 72 entries.
                # 72 * 50 * 10 * 67 ~ 2.4M operations. Very fast.
                
                # Let's implement the ALS loop properly.
                
                for r in range(target_rank):
                    # Update U[:, r]
                    v_vec = V_curr[:, r] * W_curr[:, r] # This is not how ALS works.
                    # ALS updates one factor at a time.
                    
                    # Fix V, W. Update U.
                    # U[:, r] = T_1 @ (V[:, r] * W[:, r]) ? No.
                    # The term is sum_r U[:, r] V[:, r] W[:, r].
                    # For a fixed r, the contribution to T is U[:, r] \otimes V[:, r] \otimes W[:, r].
                    # We want to minimize || T - sum_{r'} U_{r'} V_{r'} W_{r'} ||.
                    # Let T_rest = T - sum_{r' != r} U_{r'} V_{r'} W_{r'}.
                    # We want U_r, V_r, W_r to approximate T_rest.
                    # This is a rank-1 approximation of T_rest.
                    # This is hard.
                    
                    # Standard ALS:
                    # 1. Fix V, W. Update U.
                    #    U[:, r] = (T_1 @ (V[:, r] * W[:, r])) / ||V[:, r] * W[:, r]||^2
                    #    Wait, the tensor product is not separable in that way for the update.
                    #    T = sum_r U_r \otimes V_r \otimes W_r.
                    #    Unfold 1: T_1 = sum_r U_r (V_r \odot W_r)^T.
                    #    So T_1 @ (V_r \odot W_r) = U_r (V_r \odot W_r) \cdot (V_r \odot W_r) + ...
                    #    Actually, T_1 @ x = sum_r U_r (V_r \odot W_r)^T x.
                    #    If we fix V, W and want to find U_r, we set x = V_r \odot W_r.
                    #    Then T_1 @ (V_r \odot W_r) = U_r ||V_r \odot W_r||^2 + sum_{r'!=r} U_{r'} (V_{r'} \odot W_{r'})^T (V_r \odot W_r).
                    #    This is a system of linear equations for all U_r simultaneously.
                    #    U = T_1 @ Z @ (Z^T Z)^{-1} where Z columns are V_r \odot W_r.
                    
                    # Let's do this block update.
                    # Z is (m*p * p*n) x target_rank.
                    # Z[:, r] = V_curr[:, r] * W_curr[:, r].
                    
                    Z = V_curr * W_curr # Broadcasting error? V is (m*p, R), W is (p*n, R).
                    # We need element-wise product per column.
                    # V_curr[:, r] is size m*p. W_curr[:, r] is size p*n.
                    # They have different sizes!
                    # V is (m*p, R). W is (p*n, R).
                    # The tensor unfolding T_1 is (n*m, m*p * p*n).
                    # T_1[I, J, K] -> T_1[I, J*p*n + K].
                    
                    # So Z should be constructed as:
                    # Z[:, r] = vec(V_curr[:, r] \otimes W_curr[:, r])? No.
                    # The term in T_1 is U_r (V_r \odot W_r)^T.
                    # Here V_r is size m*p, W_r is size p*n.
                    # The vector V_r \odot W_r is not defined unless sizes match.
                    # In the unfolding, the row index is I (size n*m).
                    # The column index is (J, K) where J is size m*p, K is size p*n.
                    # So the column vector is the Kronecker product V_r \otimes W_r.
                    
                    # So Z[:, r] = np.kron(V_curr[:, r], W_curr[:, r]).
                    
                    Z = np.zeros((m*p * p*n, target_rank))
                    for r in range(target_rank):
                        Z[:, r] = np.kron(V_curr[:, r], W_curr[:, r])
                        
                    # T_1 is (n*m, m*p * p*n).
                    # We don't want to build T_1 explicitly (72 x 576). It's small, so we can.
                    T_1 = np.zeros((n*m, m*p * p*n))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                I = i*m + j
                                J = j*p + k
                                K = k*n + i
                                T_1[I, J * p*n + K] = 1.0
                                
                    # Solve U = T_1 @ Z @ inv(Z^T Z)
                    # Regularized least squares
                    ZtZ = Z.T @ Z
                    ZtZ += 1e-6 * np.eye(target_rank)
                    Ut = T_1 @ Z
                    U_new = Ut @ np.linalg.inv(ZtZ).T
                    
                    U_curr = U_new.T # U_new is (R, n*m)
                    
                    # Update V
                    # Unfold 2: T_2 = sum_r V_r (W_r \odot U_r)^T ?
                    # T_2[J, (K, I)] = T_{I, J, K}.
                    # T_2 = sum_r V_r (W_r \otimes U_r)^T.
                    # Z_V[:, r] = np.kron(W_curr[:, r], U_curr[:, r])
                    Z_V = np.zeros((p*n * n*m, target_rank))
                    for r in range(target_rank):
                        Z_V[:, r] = np.kron(W_curr[:, r], U_curr[:, r])
                        
                    T_2 = np.zeros((m*p, p*n * n*m))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                I = i*m + j
                                J = j*p + k
                                K = k*n + i
                                T_2[J, K * n*m + I] = 1.0
                                
                    ZtZ_V = Z_V.T @ Z_V
                    ZtZ_V += 1e-6 * np.eye(target_rank)
                    Vt = T_2 @ Z_V
                    V_new = Vt @ np.linalg.inv(ZtZ_V).T
                    V_curr = V_new.T
                    
                    # Update W
                    # Unfold 3: T_3 = sum_r W_r (U_r \otimes V_r)^T
                    Z_W = np.zeros((n*m * m*p, target_rank))
                    for r in range(target_rank):
                        Z_W[:, r] = np.kron(U_curr[:, r], V_curr[:, r])
                        
                    T_3 = np.zeros((p*n, n*m * m*p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                I = i*m + j
                                J = j*p + k
                                K = k*n + i
                                T_3[K, I * m*p + J] = 1.0
                                
                    ZtZ_W = Z_W.T @ Z_W
                    ZtZ_W += 1e-6 * np.eye(target_rank)
                    Wt = T_3 @ Z_W
                    W_new = Wt @ np.linalg.inv(ZtZ_W).T
                    W_curr = W_new.T
                    
            # Check convergence and validity
            if verify_factors(U_curr, V_curr, W_curr):
                # Check if coefficients are nice
                # Round to nearest integer or half-integer
                U_rounded = np.round(U_curr * 2) / 2
                V_rounded = np.round(V_curr * 2) / 2
                W_rounded = np.round(W_curr * 2) / 2
                
                if verify_factors(U_rounded, V_rounded, W_rounded):
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = target_rank
                    break # Found a better rank

    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
