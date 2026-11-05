"""Matrix multiplication 3x5x7 - attempting rank-minimal decomposition."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np

    # Step 1: Schoolbook decomposition for 3x3x3 (Rank 27)
    n1, m1, p1 = 3, 3, 3
    rank1 = n1 * m1 * p1
    U1 = np.zeros((n1 * m1, rank1))
    V1 = np.zeros((m1 * p1, rank1))
    W1 = np.zeros((p1 * n1, rank1))
    
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j, col] = 1
                V1[j * p1 + k, col] = 1
                W1[k * n1 + i, col] = 1
                col += 1

    # Step 2: Schoolbook decomposition for 1x2x3 (Rank 2)
    # M_{1,2,3}: C_k = sum_j A_j B_{j,k}
    n2, m2, p2 = 1, 2, 3
    rank2 = m2 # Rank is 2
    U2 = np.zeros((n2 * m2, rank2))
    V2 = np.zeros((m2 * p2, rank2))
    W2 = np.zeros((p2 * n2, rank2))
    
    # U2: indices (i=0, j). Rank index b=j.
    # U2[0, 0] = 1 (for b=0, j=0)
    # U2[1, 1] = 1 (for b=1, j=1)
    U2[0, 0] = 1
    U2[1, 1] = 1
    
    # V2: indices (j, k). Rank index b=j.
    # V2[j*3+k, j] = 1
    for j in range(m2):
        for k in range(p2):
            V2[j * p2 + k, j] = 1
            
    # W2: indices (k, i=0). Rank index b=j? No, W corresponds to output.
    # Term b is u_b (from U) v_b (from V) w_b (from W).
    # u_b is e_b in R^2? No, u_b is vector in R^{n*m} = R^2.
    # v_b is vector in R^{m*p} = R^6.
    # w_b is vector in R^{p*n} = R^3.
    # For b=0: u_0 = [1, 0]^T. v_0 = [1, 1, 1, 0, 0, 0]^T (j=0 part).
    # w_0 must be such that sum u v w = tensor.
    # Tensor T_{i,j,k} = delta_{i,0} delta_{j,k}? No.
    # T_{0,j,k} = delta_{j,k}? No, T_{0,j,k} is 1 if j=k? No.
    # Matrix mult: C_{0,k} = sum_j A_{0,j} B_{j,k}.
    # Tensor T_{i,j,k} = delta_{i,0} delta_{j,k}?
    # If i=0, T_{0,j,k} = 1 if j=k? No, the tensor entry is 1 if the indices match the multiplication structure.
    # T_{i,j,k} = 1 if i=0 and j=k? No.
    # The tensor for matrix multiplication $M_{n,m,p}$ has entries $T_{i,j,k} = \delta_{i,0}$? No.
    # $T_{i,j,k} = 1$ if the term $A_{i,j} B_{j,k}$ contributes to $C_{i,k}$.
    # So $T_{i,j,k} = 1$ for all valid i,j,k? No.
    # $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # The tensor is $\sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{i,k}^*$.
    # So $T_{i,j,k} = 1$ for all i,j,k.
    # Wait, for $1 \times 2 \times 3$:
    # $C_0 = A_0 B_{0,0} + A_1 B_{1,0}$? No, $C_k = \sum_j A_j B_{j,k}$.
    # So $C_0 = A_0 B_{0,0} + A_1 B_{1,0}$.
    # $C_1 = A_0 B_{0,1} + A_1 B_{1,1}$.
    # $C_2 = A_0 B_{0,2} + A_1 B_{1,2}$.
    # Tensor terms:
    # $A_0 B_{0,0} C_0 \to T_{0,0,0} = 1$.
    # $A_1 B_{1,0} C_0 \to T_{0,1,0} = 1$.
    # $A_0 B_{0,1} C_1 \to T_{0,0,1} = 1$.
    # $A_1 B_{1,1} C_1 \to T_{0,1,1} = 1$.
    # $A_0 B_{0,2} C_2 \to T_{0,0,2} = 1$.
    # $A_1 B_{1,2} C_2 \to T_{0,1,2} = 1$.
    
    # Decomposition:
    # Term b=0: $u_0 = [1, 0]^T$ (A_0). $v_0$ has 1s at $B_{0,k}$. $w_0$ has 1s at $C_k$.
    # $v_0 = [1, 1, 1, 0, 0, 0]^T$.
    # $w_0 = [1, 1, 1]^T$.
    # Term b=1: $u_1 = [0, 1]^T$ (A_1). $v_1$ has 1s at $B_{1,k}$. $w_1$ has 1s at $C_k$.
    # $v_1 = [0, 0, 0, 1, 1, 1]^T$.
    # $w_1 = [1, 1, 1]^T$.
    
    U2[0, 0] = 1
    U2[1, 1] = 1
    
    V2[0, 0] = 1
    V2[1, 0] = 1
    V2[2, 0] = 1
    V2[3, 1] = 1
    V2[4, 1] = 1
    V2[5, 1] = 1
    
    W2[0, 0] = 1
    W2[1, 0] = 1
    W2[2, 0] = 1
    W2[0, 1] = 1
    W2[1, 1] = 1
    W2[2, 1] = 1

    # Step 3: Kronecker Product to get 3x6x9 decomposition
    # Rank 27 * 2 = 54
    rank_total = rank1 * rank2
    
    U_large = np.zeros((n1 * m2 * 2, rank_total)) # n1=3, m_total=6. 3*6=18
    V_large = np.zeros((m1 * m2 * p1 * p2, rank_total)) # m_total=6, p_total=9. 6*9=54
    W_large = np.zeros((p1 * p2 * n1, rank_total)) # p_total=9, n1=3. 9*3=27
    
    for a in range(rank1):
        for b in range(rank2):
            c = a * rank2 + b
            
            # U_large: indices (i, j_total). j_total = j1 * 2 + j2.
            # U_large[i, j1*2+j2, c] = U1[i, j1, a] * U2[0, j2, b]
            for i in range(n1):
                for j1 in range(m1):
                    for j2 in range(m2):
                        idx_u = i * (m1 * m2) + j1 * m2 + j2
                        U_large[idx_u, c] = U1[i * m1 + j1, a] * U2[j2, b] # U2 row is j2 (since n2=1, idx=j2)
                        
            # V_large: indices (j_total, k_total). j_total = j1*2+j2, k_total = k1*3+k2.
            # V_large[j1*2+j2, k1*3+k2, c] = V1[j1, k1, a] * V2[j2, k2, b]
            for j1 in range(m1):
                for j2 in range(m2):
                    for k1 in range(p1):
                        for k2 in range(p2):
                            idx_v = (j1 * m2 + j2) * (p1 * p2) + k1 * p2 + k2
                            V_large[idx_v, c] = V1[j1 * p1 + k1, a] * V2[j2 * p2 + k2, b]
                            
            # W_large: indices (k_total, i). k_total = k1*3+k2.
            # W_large[k1*3+k2, i, c] = W1[k1, i, a] * W2[k2, 0, b]
            for k1 in range(p1):
                for k2 in range(p2):
                    for i in range(n1):
                        idx_w = (k1 * p2 + k2) * n1 + i
                        W_large[idx_w, c] = W1[k1 * n1 + i, a] * W2[k2, b] # W2 col is i? No, W2 is p*n. n2=1. idx=k2.

    # Step 4: Restrict to 3x5x7
    # n=3, m=5, p=7
    n, m, p = 3, 5, 7
    
    U_rows = []
    for i in range(n):
        for j in range(m):
            # j = j1 * 2 + j2
            j1 = j // 2
            j2 = j % 2
            idx_u = i * (m1 * m2) + j1 * m2 + j2
            U_rows.append(idx_u)
            
    V_rows = []
    for j in range(m):
        j1 = j // 2
        j2 = j % 2
        for k in range(p):
            k1 = k // 3
            k2 = k % 3
            idx_v = (j1 * m2 + j2) * (p1 * p2) + k1 * p2 + k2
            V_rows.append(idx_v)
            
    W_rows = []
    for k in range(p):
        k1 = k // 3
        k2 = k % 3
        for i in range(n):
            idx_w = (k1 * p2 + k2) * n1 + i
            W_rows.append(idx_w)

    U_sub = U_large[U_rows, :]
    V_sub = V_large[V_rows, :]
    W_sub = W_large[W_rows, :]

    # Step 5: Remove zero columns
    mask = (np.any(U_sub != 0, axis=0) & np.any(V_sub != 0, axis=0) & np.any(W_sub != 0, axis=0))
    U_final = U_sub[:, mask]
    V_final = V_sub[:, mask]
    W_final = W_sub[:, mask]

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
