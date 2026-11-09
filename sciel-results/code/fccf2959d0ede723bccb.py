import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Returns a rank-56 decomposition for 2x4x8 matrix multiplication.
    This is achieved by decomposing the problem into 8 independent 2x2x2 
    matrix multiplications, each solved using Strassen's algorithm (rank 7).
    Total rank = 8 * 7 = 56.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen forms for 2x2 * 2x2
    # X_forms[s][i*2+j'] for s=0..6, i=0,1, j'=0,1
    # Corresponds to factors in A
    X_forms = [
        [1, 0, 0, 1],   # m1: (A00 + A11)
        [0, 0, 1, 1],   # m2: (A10 + A11)
        [1, 0, 0, 0],   # m3: A00
        [0, 0, 0, 1],   # m4: A11
        [1, 1, 0, 0],   # m5: (A00 + A01)
        [-1, 0, 1, 0],  # m6: (A10 - A00)
        [0, 1, 0, -1],  # m7: (A01 - A11)
    ]
    
    # Y_forms[s][j'*2+k'] for s=0..6, j'=0,1, k'=0,1
    # Corresponds to factors in B
    Y_forms = [
        [1, 0, 0, 1],   # m1: (B00 + B11)
        [1, 0, 0, 0],   # m2: B00
        [0, 1, 0, -1],  # m3: (B01 - B11)
        [-1, 0, 1, 0],  # m4: (B10 - B00)
        [0, 0, 0, 1],   # m5: B11
        [1, 1, 0, 0],   # m6: (B00 + B01)
        [0, 0, 1, 1],   # m7: (B10 + B11)
    ]
    
    # Z_combos[z_idx][s] for z_idx=0..3 (C00,C01,C10,C11), s=0..6
    # Corresponds to reconstruction of C
    Z_combos = [
        [1, 0, 0, 1, -1, 0, 1],   # C00 = m1 + m4 - m5 + m7
        [0, 0, 1, 0, 1, 0, 0],    # C01 = m3 + m5
        [0, 1, 0, 1, 0, 0, 0],    # C10 = m2 + m4
        [1, -1, 1, 0, 0, 1, 0],   # C11 = m1 - m2 + m3 + m6
    ]
    
    num_groups = 4      # 4 groups of 2 columns in B (k offsets: 0, 2, 4, 6)
    num_ablocks = 2     # 2 blocks of 2 columns in A (j offsets: 0, 2)
    num_strassen = 7    # 7 Strassen products per 2x2x2 block
    
    rank = num_groups * num_ablocks * num_strassen  # 56
    
    # Initialize matrices
    # U: (n*m) x rank = 8 x 56
    # V: (m*p) x rank = 32 x 56
    # W: (p*n) x rank = 16 x 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Iterate over the 8 disjoint 2x2x2 blocks
    # g: column group in B (0..3), corresponds to k offset 2*g
    # b: column block in A (0..1), corresponds to j offset 2*b
    # Note: The row offset in B is determined by the A block 'b' because 
    #       A[i, j] multiplies B[j, k].
    #       If b=0, we use A[:, 0:2] which multiplies B[0:2, :].
    #       If b=1, we use A[:, 2:4] which multiplies B[2:4, :].
    #       So B row offset is also 2*b.
    
    for g in range(num_groups):
        for b in range(num_ablocks):
            # Offsets
            a_col_off = 2 * b
            b_row_off = 2 * b
            b_col_off = 2 * g
            
            for s in range(num_strassen):
                # Fill U (A factors)
                # U index: i * m + (a_col_off + jp)
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + a_col_off + jp
                        U[a_idx][r] = X_forms[s][i * 2 + jp]
                
                # Fill V (B factors)
                # V index: (b_row_off + jp) * p + (b_col_off + kp)
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b_row_off + jp) * p + b_col_off + kp
                        V[b_idx][r] = Y_forms[s][jp * 2 + kp]
                
                # Fill W (C factors)
                # W index: (b_col_off + kp) * n + i
                # Note: The Strassen output C00..C11 corresponds to 
                #       C[i, b_col_off + kp]
                for i in range(n):
                    for kp in range(2):
                        k_global = b_col_off + kp
                        c_idx = k_global * n + i
                        # Z_combos index: i * 2 + kp (C00, C01, C10, C11)
                        W[c_idx][r] = Z_combos[i * 2 + kp][s]
                
                r += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple self-test to ensure validity if run directly
    import numpy as np
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    n, m, p = 2, 4, 8
    # Random test matrices
    np.random.seed(42)
    A = np.random.randint(-5, 5, (n, m))
    B = np.random.randint(-5, 5, (m, p))
    
    # Standard multiplication
    C_expected = A @ B
    
    # Tensor decomposition multiplication
    # C[i, k] = sum_r U[i, r] * V[r, k] * W[k, i] ? 
    # No, the tensor decomposition of multiplication is:
    # C[i, k] = sum_r (sum_j U[i, j, r] A[i, j]) * (sum_k V[j, k, r] B[j, k]) ... wait.
    
    # The standard tensor decomposition form is:
    # C_{i,k} = \sum_r U_{i,r} V_{r,k} W_{k,i} is NOT correct for matrix mult.
    # Matrix mult tensor T has indices (i, j, k).
    # T_{i,j,k} = \sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} ?
    
    # Let's check the interface definition:
    # 'U', 'V', 'W' factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # U has size (n*m) x R. Index (i*m+j, r)
    # V has size (m*p) x R. Index (j*p+k, r)
    # W has size (p*n) x R. Index (k*n+i, r)
    
    # The decomposition is:
    # C_{i,k} = \sum_r ( \sum_j U_{i*m+j, r} A_{i,j} ) * ( \sum_{j',k'} V_{j'*p+k', r} B_{j',k'} ) ... NO.
    
    # Actually, the bilinear form for matrix multiplication is:
    # C_{i,k} = \sum_j A_{i,j} B_{j,k}
    
    # The tensor decomposition expresses the *trilinear* form?
    # No, it's a bilinear map.
    # The rank of the bilinear map is the minimum R such that:
    # C_{i,k} = \sum_{r=1}^R L_{i,r}(A) M_{r,k}(B) ? No.
    
    # Standard definition of Tensor Rank for Matrix Multiplication:
    # The multiplication tensor $T_{n,m,p}$ is defined by $T_{i,j,k} = 1$ if $i,j,k$ are compatible.
    # A rank-1 term is $u_i v_j w_k$.
    # The decomposition is:
    # $T_{i,j,k} = \sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}$?
    
    # Let's look at the shapes provided in the prompt:
    # U: (n*m, rank) -> Flattened A indices.
    # V: (m*p, rank) -> Flattened B indices.
    # W: (p*n, rank) -> Flattened C indices.
    
    # This implies the decomposition is:
    # $A_{i,j} B_{j,k} \delta_{i,k} \text{ (identity)} $... no.
    
    # The condition for exact decomposition is:
    # $\sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i', r}$?
    
    # Actually, for the *multiplication* tensor, the rank decomposition satisfies:
    # $\sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} = \delta_{i,i'} \delta_{k,k'}$?
    
    # Let's assume the standard "Strassen" verification:
    # The product $C = A B$ is computed as:
    # $C_{i,k} = \sum_r (\sum_j U_{i*m+j, r} A_{i,j}) \cdot (\sum_{j'} V_{j'*p+k, r} B_{j',k})$?
    # No, V is indexed by (j, k).
    
    # Let's test the hypothesis:
    # $C_{i,k} = \sum_r (\sum_j U_{i*m+j, r} A_{i,j}) \times (\sum_{j',k'} V_{j'*p+k', r} B_{j',k'})$?
    # This doesn't make sense dimensionally for V.
    
    # Correct Interpretation of the Interface:
    # The tensor $T$ is in $A^* \otimes B^* \otimes C^*$.
    # A rank-1 tensor is $a \otimes b \otimes c$.
    # The decomposition is $T = \sum_r a_r \otimes b_r \otimes c_r$.
    # Where $a_r$ is a vector in $A$ (size n*m), $b_r$ in $B$ (size m*p), $c_r$ in $C$ (size p*n).
    # Wait, the prompt says U, V, W are matrices of shape (dim, rank).
    # So column r of U is $a_r$, column r of V is $b_r$, column r of W is $c_r$.
    
    # The condition for matrix multiplication is:
    # $T_{i,j,k} = \delta_{i,j} \delta_{j,k}$? No.
    # $T_{i,j,k}$ corresponds to the coefficient of $A_{i,j} B_{j,k}$ in $C_{i,k}$.
    # So $T_{i,j,k} = 1$ if the indices form a valid path $i \leftarrow j \rightarrow k$.
    # Specifically, $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # So the tensor is $\sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}$.
    
    # The decomposition requires:
    # $\sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = \delta_{i,i} \delta_{k,k}$?
    # No, it must equal the tensor itself.
    # The tensor itself is:
    # $T_{i,j,k} = 1$ if we identify the indices correctly.
    # Actually, the tensor is:
    # $\sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}$.
    
    # So we need:
    # $\sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = \begin{cases} 1 & \text{if } (i,j,k) \text{ is a valid triple} \\ 0 & \text{otherwise} \end{cases}$
    
    # In our construction:
    # For a specific r (which corresponds to a specific Strassen block),
    # U is non-zero only for specific i, j.
    # V is non-zero only for specific j, k.
    # W is non-zero only for specific i, k.
    
    # Let's verify with a single r (e.g., r=0, g=0, b=0, s=0).
    # g=0, b=0, s=0 (Strassen m1: (A00+A11)*(B00+B11))
    # a_col_off=0, b_row_off=0, b_col_off=0.
    # U[0,0] = 1, U[3,0] = 1 (A00, A11). Others 0.
    # V[0,0] = 1, V[3,0] = 1 (B00, B11). Others 0.
    # W[0,0] = 1, W[1,0] = 0, W[2,0] = 0, W[3,0] = 1 (C00, C11).
    # Wait, Z_combos[0] for C00 is 1. Z_combos[3] for C11 is 1.
    # So W[0,0] = 1 (C00), W[3,0] = 1 (C11).
    
    # Let's check the sum $\sum_r U_{i,j} V_{j,k} W_{i,k}$.
    # This is not how it works.
    # The decomposition is $T = \sum_r U_r \otimes V_r \otimes W_r$.
    # The entry $T_{i,j,k}$ is $\sum_r U_r(i,j) V_r(j,k) W_r(i,k)$.
    # We need this sum to be 1 for valid triples and 0 otherwise.
    
    # Let's test a valid triple: i=0, j=0, k=0.
    # Which r contribute?
    # Only r where U[0,0]!=0, V[0,0]!=0, W[0,0]!=0.
    # r=0 (g=0,b=0,s=0): U[0,0]=1, V[0,0]=1, W[0,0]=1. Product = 1.
    # r=1 (g=0,b=0,s=1): U[0,0]=0 (m2 is A10+A11).
    # ...
    # We need to ensure that for every valid (i,j,k), the sum is 1, and for invalid, 0.
    
    # This is guaranteed by the Strassen algebra.
    
    # Verification code
    # T_expected[i, j, k] = 1 if valid else 0
    # T_calc[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Note: The indices in U, V, W are flattened.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Let's run a quick check in the code.
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        T_calc = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    # Sum over r
                    T_calc[u_idx, v_idx, w_idx] = np.sum(U[:, u_idx] * V[:, v_idx] * W[:, w_idx])
        
        # Expected T
        T_exp = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    # Valid if indices match?
                    # A[i,j] * B[j,k] -> C[i,k]
                    # So yes, every (i,j,k) is a valid term in the sum for C[i,k].
                    # The tensor T is exactly the identity map in this bilinear sense?
                    # No, the tensor T is defined such that <T, A \otimes B \otimes C> = tr(C^T A B).
                    # So T_{i,j,k} = 1 for all i,j,k?
                    # Let's check:
                    # <T, A \otimes B \otimes C> = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
                    # We want this to equal sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}.
                    # So T_{i,j,k} must be 1 for all i,j,k.
                    
                    T_exp[u_idx, v_idx, w_idx] = 1
        
        # Check if T_calc == T_exp
        if not np.allclose(T_calc, T_exp):
            print("ERROR: Tensor decomposition is invalid!")
            diff = np.abs(T_calc - T_exp)
            print("Max diff:", np.max(diff))
        else:
            print("SUCCESS: Tensor decomposition is valid.")
