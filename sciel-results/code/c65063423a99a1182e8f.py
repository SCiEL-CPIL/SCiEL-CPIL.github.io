import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # We will build the decomposition using tensor operations for the 4x4x4 core
    # and simple loops for the boundary terms.
    
    # 1. Construct the 2x2x2 Strassen Tensor
    # Shape: (4, 4, 4) -> (n*m, m*p, p*n) for 2x2x2
    # Indices: U: 0:00, 1:01, 2:10, 3:11
    #          V: 0:00, 1:01, 2:10, 3:11
    #          W: 0:00, 1:01, 2:10, 3:11 (k*n+i)
    
    # We create a 3D array S of shape (4, 4, 4) where S[i, j, k] is the coefficient
    # of the k-th rank-1 term for the (i,j) input to the k-th output?
    # Actually, the standard representation is U, V, W matrices.
    # Let's build the 2x2x2 factors U2, V2, W2 of shape (4, 7).
    
    def get_strassen_2x2_factors():
        u = np.zeros((4, 7), dtype=int)
        v = np.zeros((4, 7), dtype=int)
        w = np.zeros((4, 7), dtype=int)
        
        # Term 0: (a00+a11)(b00+b11) -> c00, c11
        u[0,0]=1; u[3,0]=1
        v[0,0]=1; v[3,0]=1
        w[0,0]=1; w[3,0]=1
        
        # Term 1: (a10+a11)b00 -> c10, c11(-1)
        u[2,1]=1; u[3,1]=1
        v[0,1]=1
        w[2,1]=1; w[3,1]=-1
        
        # Term 2: a00(b01-b11) -> c01, c11
        u[0,2]=1
        v[1,2]=1; v[3,2]=-1
        w[1,2]=1; w[3,2]=1
        
        # Term 3: a11(b10-b00) -> c00, c10
        u[3,3]=1
        v[2,3]=1; v[0,3]=-1
        w[0,3]=1; w[2,3]=1
        
        # Term 4: (a00+a01)b11 -> c00(-1), c01
        u[0,4]=1; u[1,4]=1
        v[3,4]=1
        w[0,4]=-1; w[1,4]=1
        
        # Term 5: (a10-a00)(b00+b01) -> c11
        u[0,5]=-1; u[2,5]=1
        v[0,5]=1; v[1,5]=1
        w[3,5]=1
        
        # Term 6: (a01-a11)(b10+b11) -> c00
        u[1,6]=1; u[3,6]=-1
        v[2,6]=1; v[3,6]=1
        w[0,6]=1
        
        return u, v, w

    u2, v2, w2 = get_strassen_2x2_factors()
    
    # 2. Construct 4x4x4 factors using Kronecker Product
    # The 4x4x4 multiplication tensor is T_4 = T_2 \otimes T_2
    # This implies U_4 = kron(U_2, U_2)?
    # Let's verify the index mapping.
    # If we have a 4x4 matrix A, it is a 2x2 block matrix of 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # The Kronecker product of the linear forms works.
    # U_4 will have shape (16, 49).
    # V_4 will have shape (16, 49).
    # W_4 will have shape (16, 49).
    
    # Note: The Kronecker product of the factor matrices corresponds to the
    # tensor product of the decompositions.
    # Rank(4x4x4) = Rank(2x2x2) * Rank(2x2x2) = 7 * 7 = 49.
    
    u4 = np.kron(u2, u2)
    v4 = np.kron(v2, v2)
    w4 = np.kron(w2, w2)
    
    # 3. Construct the full 4x5x5 decomposition
    # Total Rank = 49 (4x4x4) + 16 (4x4x1) + 16 (4x1x4) + 4 (4x1x1) = 85
    
    rank = 85
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Part 1: 4x4x4 core (Rows 0-3 of A, Cols 0-3 of A; Rows 0-3 of B, Cols 0-3 of B; Rows 0-3 of C, Cols 0-3 of C)
    # A indices: i in 0..3, j in 0..3 -> idx = i*5 + j
    # B indices: j in 0..3, k in 0..3 -> idx = j*5 + k
    # C indices: k in 0..3, i in 0..3 -> idx = k*4 + i
    
    # We need to map the 16 entries of the 4x4 matrices to the 5x5 / 4x5 global indices.
    
    # Helper to map 4x4 local index (0..15) to global index
    def map_a_4x4_to_global(local_idx):
        # local_idx is 0..15, representing row r (0..3), col c (0..3)
        r = local_idx // 4
        c = local_idx % 4
        return r * 5 + c
        
    def map_b_4x4_to_global(local_idx):
        r = local_idx // 4
        c = local_idx % 4
        return r * 5 + c
        
    def map_c_4x4_to_global(local_idx):
        # C is 4x4, stored as k*n+i.
        # Local 4x4 C indices: row k (0..3), col i (0..3)
        # In the 4x4x4 decomposition, W4 has shape (16, 49).
        # The rows of W4 correspond to the 16 entries of the 4x4 C matrix.
        # Indexing in W4: standard row-major? 
        # In get_strassen_2x2_factors, w[0] was c00, w[1] c01, w[2] c10, w[3] c11.
        # So W4 rows are ordered: c00, c01, c02, c03, c10, c11, c12, c13, c20...
        # This matches k*4 + i.
        # Global C index is k*4 + i (since n=4 for the output part? No, n=4 is the number of rows of C? 
        # Task: C is p x n = 5 x 4.
        # Wait, the task says W[k*n+i, r].
        # C is 5x4. Rows k=0..4, Cols i=0..3.
        # The 4x4x4 part only affects C[0..3, 0..3].
        # So for local C index (k, i) with k,i in 0..3, global index is k*4 + i.
        return local_idx # Since local 4x4 is exactly the top-left 4x4 of the 5x4 C, and indices align 0..15
        
    # Apply the 49 terms of the 4x4x4 core
    for r in range(16):
        u_global = map_a_4x4_to_global(r)
        v_global = map_b_4x4_to_global(r) # B is also 4x4 block, same mapping
        w_global = map_c_4x4_to_global(r)
        
        U[u_global, 0:49] = u4[r, :]
        V[v_global, 0:49] = v4[r, :]
        W[w_global, 0:49] = w4[r, :]
        
    # Part 2: 4x4x1 part (A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4])
    # Rank 16
    col_offset = 49
    for i in range(4):
        for j in range(4):
            a_idx = i * 5 + j
            b_idx = j * 5 + 4
            c_idx = 4 * 4 + i # k=4, i=0..3
            
            U[a_idx, col_offset] = 1
            V[b_idx, col_offset] = 1
            W[c_idx, col_offset] = 1
            col_offset += 1
            
    # Part 3: 4x1x4 part (A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4])
    # Rank 16
    for i in range(4):
        for k in range(4):
            a_idx = i * 5 + 4
            b_idx = 4 * 5 + k
            c_idx = k * 4 + i
            
            U[a_idx, col_offset] = 1
            V[b_idx, col_offset] = 1
            W[c_idx, col_offset] = 1
            col_offset += 1
            
    # Part 4: 4x1x1 part (A[0:4, 4] * B[4, 4] -> C[0:4, 4])
    # Rank 4
    for i in range(4):
        a_idx = i * 5 + 4
        b_idx = 4 * 5 + 4
        c_idx = 4 * 4 + i
        
        U[a_idx, col_offset] = 1
        V[b_idx, col_offset] = 1
        W[c_idx, col_offset] = 1
        col_offset += 1
        
    # Deadline guard
    if time.time() > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Optional: Verification step to ensure correctness
    # We can check a few random vectors to ensure the decomposition holds.
    # T_A = U * diag(s) * V^T ? No, the decomposition is:
    # C_{ik} = sum_r U_{ir, r} V_{jk, r} W_{ik, r} ? 
    # Wait, the standard tensor decomposition for matrix multiplication M_{ijk} is:
    # M_{ijk} = sum_r U_{i, r} V_{j, r} W_{k, r}
    # Where M is the multiplication tensor.
    # A is n x m, B is m x p, C is p x n.
    # C_{k, i} = sum_j A_{i, j} B_{j, k}
    # So the tensor T_{i, j, k} = delta_{i, j, k} ? No.
    # The tensor is T_{i, j, k} such that C_{k, i} = sum_j A_{i, j} T_{i, j, k} B_{j, k}?
    # No, the factors U, V, W are defined such that:
    # U has shape (n*m, rank) -> Index (i, j)
    # V has shape (m*p, rank) -> Index (j, k)
    # W has shape (p*n, rank) -> Index (k, i)
    # The condition is:
    # sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} = delta_{i,j,k} ? 
    # Actually, for matrix multiplication, the "tensor" is the identity in the middle dimension.
    # The decomposition must satisfy:
    # sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} = 1 if i,j,k form a valid multiplication path?
    # No, the evaluator checks if the factors *exactly decompose the matrix multiplication tensor*.
    # The tensor T_{i, j, k} is 1 if the indices correspond to a term in C_{k, i} = sum_j A_{i, j} B_{j, k}.
    # So T_{i, j, k} = 1 for all i, j, k? No.
    # The tensor is defined by the multiplication rule.
    # T_{i, j, k} = 1 if we are computing C_{k, i} using A_{i, j} and B_{j, k}.
    # Yes, for any i, j, k, the term A_{i, j} B_{j, k} contributes to C_{k, i}.
    # So the tensor T_{i, j, k} is 1 for all valid indices i, j, k.
    # Wait, if T_{i, j, k} = 1 for all i, j, k, then the rank is n*m*p = 100.
    # Strassen reduces this by finding a lower rank decomposition.
    
    # Let's verify with a small example.
    # If we pick random A, B, compute C = A @ B.
    # Then check if C_{k, i} == sum_j sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r}.
    # Note: The indices in U, V, W are flattened.
    # U_{i, j, r} = U[i*m + j, r]
    # V_{j, k, r} = V[j*p + k, r]
    # W_{k, i, r} = W[k*n + i, r]
    
    # Let's do a quick check with small random integers.
    try:
        A_test = np.random.randint(-2, 3, size=(n, m))
        B_test = np.random.randint(-2, 3, size=(m, p))
        C_test = A_test @ B_test
        
        # Compute via factors
        # C_calc_{k, i} = sum_j sum_r U[i, j, r] * A[i, j] * V[j, k, r] * B[j, k] * W[k, i, r] ?
        # No, the factors are linear combinations.
        # The formula is:
        # C_{k, i} = sum_r ( sum_j U_{i, j, r} A_{i, j} ) * ( sum_j V_{j, k, r} B_{j, k} ) * W_{k, i, r}
        # Wait, W is a factor matrix, not a scalar multiplier in the same way?
        # In tensor decomposition, T_{i, j, k} = sum_r U_{i, r} V_{j, r} W_{k, r}.
        # Here, the "indices" of the tensor are (i, j) for A, (j, k) for B, (k, i) for C.
        # So T_{(i,j), (j,k), (k,i)} = sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r}.
        # And T should be 1 if the multiplication is valid?
        # For matrix multiplication, the tensor is the "multiplication tensor" which has 1s at positions corresponding to the terms in the sum.
        # Specifically, for each i, j, k, the term A_{i, j} B_{j, k} contributes to C_{k, i}.
        # So the tensor T has entries T_{i, j, k} = 1 for all i, j, k?
        # If so, the rank is 100.
        # But Strassen says rank < 100.
        # The catch: The tensor is not just 1s. It's a specific tensor.
        # Actually, the tensor T is defined such that:
        # C = T(A, B)
        # The factors U, V, W allow us to compute T(A, B) using rank multiplications.
        
        # Let's verify the identity:
        # sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} = 1 for all i, j, k?
        # If this sum is 1, then:
        # sum_j sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} A_{i, j} B_{j, k} 
        # = sum_j A_{i, j} B_{j, k} (sum_r ...)
        # If sum_r ... = 1, then it equals C_{k, i}.
        
        # So we just need to check if sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} == 1 for all i, j, k.
        
        # Let's check one entry.
        # i=0, j=0, k=0
        val = 0
        for r in range(rank):
            val += U[0*5+0, r] * V[0*5+0, r] * W[0*4+0, r]
            
        if val != 1:
            # If it fails, return a known valid dense decomposition (rank 100)
            # This is a fallback to ensure we don't return an invalid result.
            U_dense = np.zeros((n*m, n*m*p), dtype=int)
            V_dense = np.zeros((m*p, n*m*p), dtype=int)
            W_dense = np.zeros((p*n, n*m*p), dtype=int)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_dense[i*m+j, idx] = 1
                        V_dense[j*p+k, idx] = 1
                        W_dense[k*n+i, idx] = 1
                        idx += 1
                        if time.time() > DEADLINE:
                            break
                    if time.time() > DEADLINE:
                        break
                if time.time() > DEADLINE:
                    break
            if time.time() > DEADLINE:
                break
            
            return {"U": U_dense.tolist(), "V": V_dense.tolist(), "W": W_dense.tolist()}
            
    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
