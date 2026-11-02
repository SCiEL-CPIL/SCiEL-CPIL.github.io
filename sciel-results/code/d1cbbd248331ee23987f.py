import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    start_time = time.time()
    deadline = start_time + 140  # Leave some margin before 150s limit
    
    # Schoolbook rank is n*m*p = 60
    # We know that for 2x3x2, the rank is at most 8 (or maybe 7?).
    # Let's try to construct a valid decomposition for 2x3x2 first.
    # If we can get rank R for 2x3x2, then for 2x3x10 (5 blocks of 2x3x2),
    # we can use rank 5*R.
    # Schoolbook for 2x3x2 is 12.
    # If rank is 11, total 55.
    # If rank is 10, total 50.
    # If rank is 9, total 45.
    # If rank is 8, total 40.
    
    # Let's try to find a low-rank decomposition for 2x3x2.
    # Variables:
    # A: a00, a01, a02, a10, a11, a12
    # B: b00, b01, b10, b11, b20, b21
    # C: c00, c01, c10, c11
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    
    # We can treat this as a bilinear form.
    # Let's try to use a search algorithm for the 2x3x2 tensor.
    # We can represent the tensor as a matrix T of size (n*m) x (m*p) ? No, it's a 3-tensor.
    # T[i, j, k] = 1 if the monomial A[i,j] B[j,k] appears in C[i,k].
    # Here i in {0,1}, j in {0,1,2}, k in {0,1}.
    # So T is a 2x3x2 tensor.
    
    # We want to find U, V, W such that sum_r U[i,j,r] V[j,k,r] W[i,k,r] = T[i,j,k] (with appropriate indexing).
    # Actually, the standard decomposition is:
    # C[i,k] = sum_j sum_r U[i,j,r] V[j,k,r] ... wait.
    # The factorization is usually:
    # C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k]) ... No.
    # The standard tensor decomposition for matrix multiplication M_{n,m,p}:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # We want: C_{i,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) * (sum_j V_{j,k,r} B_{j,k}) ? No, that's not right either.
    # The correct form is:
    # C_{i,k} = sum_r W_{i,k,r} * (sum_j U_{i,j,r} A_{i,j}) * (sum_j V_{j,k,r} B_{j,k}) ? No, that's cubic.
    
    # The correct tensor decomposition form for matrix multiplication is:
    # C_{i,k} = sum_r ( (sum_j U_{i,j,r} A_{i,j}) * (sum_j V_{j,k,r} B_{j,k}) ) ? No.
    # It is:
    # C_{i,k} = sum_r W_{i,k,r} * M_r
    # Where M_r are intermediate products.
    # Actually, the interface says:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # This suggests:
    # C_{i,k} = sum_r U_{i*m+j, r} ... no.
    
    # Let's look at the seed program's indexing:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # This implies:
    # C_{i,k} = sum_j sum_r U_{i*m+j, r} * A_{i,j} * V_{j*p+k, r} * B_{j,k} * W_{k*n+i, r} ?
    # No, that would be a sum of products of 3 terms.
    
    # Let's re-read the task description carefully.
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "exactly decompose the matrix multiplication tensor"
    # The standard tensor rank decomposition for M_{n,m,p} is:
    # M_{n,m,p} = sum_r U_r \otimes V_r \otimes W_r
    # Where U_r is a matrix of size n x m, V_r is m x p, W_r is p x n?
    # Or U_r is a vector?
    
    # In the seed code:
    # U is (n*m) x rank. Index i*m+j.
    # V is (m*p) x rank. Index j*p+k.
    # W is (p*n) x rank. Index k*n+i.
    
    # The tensor T has indices i, j, k.
    # T[i, j, k] = 1 if the term A[i,j] B[j,k] contributes to C[i,k].
    # Actually, the multiplication tensor M has M[i, j, k] = 1 for all i, j, k.
    # The decomposition is:
    # M[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ?
    # Let's check the schoolbook case.
    # For a specific (i,j,k), U=1, V=1, W=1.
    # Sum over r: only one r is non-zero for this (i,j,k).
    # So M[i,j,k] = 1.
    # This matches.
    
    # So the condition is:
    # For all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of equations.
    # For 2x3x2:
    # i in {0,1}, j in {0,1,2}, k in {0,1}.
    # Total 2*3*2 = 12 equations.
    # Variables: U (2*3 * R), V (3*2 * R), W (2*2 * R).
    # 6R + 6R + 4R = 16R variables.
    
    # We want to find integer/half-integer solutions.
    # Let's try to solve for R=8, 9, 10, 11.
    
    # We can use a randomized search or a linear algebra approach over a finite field?
    # Or just try to find a solution in Z or Q.
    
    # Let's try to construct a solution for 2x3x2 with R=8.
    # We can use the fact that 2x3x2 is close to 2x2x2.
    # 2x2x2 has rank 7 (Strassen).
    # 2x3x2 might have rank 8 or 9.
    
    # Let's try a simple heuristic:
    # Initialize U, V, W with random integers in {-1, 0, 1}.
    # Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    
    # This is a 3SAT-like problem. Random search might work for small R.
    
    def verify_block(U, V, W, R, n_block, m_block, p_block):
        # U: (n_block*m_block, R)
        # V: (m_block*p_block, R)
        # W: (p_block*n_block, R)
        for i in range(n_block):
            for j in range(m_block):
                for k in range(p_block):
                    s = 0
                    for r in range(R):
                        s += U[i*m_block + j, r] * V[j*p_block + k, r] * W[k*n_block + i, r]
                    if s != 1:
                        return False
        return True

    def find_block_decomp(n_block, m_block, p_block, R, max_iters=10000):
        # Try to find a decomposition of rank R for n_block x m_block x p_block
        for it in range(max_iters):
            if time.time() > deadline - 20:
                return None
            
            U = np.random.randint(-2, 3, size=(n_block*m_block, R))
            V = np.random.randint(-2, 3, size=(m_block*p_block, R))
            W = np.random.randint(-2, 3, size=(p_block*n_block, R))
            
            if verify_block(U, V, W, R, n_block, m_block, p_block):
                return U, V, W
        return None

    # Try to find a decomposition for 2x3x2
    block_decomp = None
    for R in range(8, 13):
        block_decomp = find_block_decomp(2, 3, 2, R, max_iters=5000)
        if block_decomp is not None:
            print(f"Found 2x3x2 decomposition with rank {R}")
            break
    
    if block_decomp is not None:
        U_block, V_block, W_block = block_decomp
        R_block = U_block.shape[1]
        total_rank = 5 * R_block
        
        # Construct the full 2x3x10 decomposition
        # We have 5 blocks of 2x3x2.
        # B is 3x10. We split it into 5 blocks of 3x2: B_0, B_1, ..., B_4.
        # C is 2x10. C = [C_0, C_1, ..., C_4] where C_t = A * B_t.
        
        U_full = [[0] * total_rank for _ in range(n*m)]
        V_full = [[0] * total_rank for _ in range(m*p)]
        W_full = [[0] * total_rank for _ in range(p*n)]
        
        for t in range(5):
            # Block t corresponds to columns 2t and 2t+1 of B and C.
            # U is the same for all blocks (since A is the same).
            # V is different for each block (different columns of B).
            # W is different for each block (different columns of C).
            
            # U_block has shape (2*3, R_block) = (6, R_block)
            # U_full has shape (2*3, total_rank) = (6, 5*R_block)
            # We can just repeat U_block 5 times?
            # Wait, the indices in U are i*m+j. This is independent of k.
            # So yes, U_full[i*m+j, t*R_block + r] = U_block[i*m+j, r]
            
            for i in range(n):
                for j in range(m):
                    for r in range(R_block):
                        U_full[i*m + j][t*R_block + r] = U_block[i*m + j, r]
            
            # V_block has shape (3*2, R_block) = (6, R_block)
            # V_full has shape (3*10, total_rank) = (30, 5*R_block)
            # V_full[j*p + k, t*R_block + r] = V_block[j*2 + (k-2t), r]
            for j in range(m):
                for k_local in range(2):
                    k = 2*t + k_local
                    for r in range(R_block):
                        V_full[j*p + k][t*R_block + r] = V_block[j*2 + k_local, r]
            
            # W_block has shape (2*2, R_block) = (4, R_block)
            # W_full has shape (10*2, total_rank) = (20, 5*R_block)
            # W_full[k*n + i, t*R_block + r] = W_block[k_local*2 + i, r]
            for k_local in range(2):
                k = 2*t + k_local
                for i in range(n):
                    for r in range(R_block):
                        W_full[k*n + i][t*R_block + r] = W_block[k_local*2 + i, r]
        
        # Verify the full decomposition
        # C[i,k] = sum_j sum_r U_full[i*m+j, r] * A[i,j] * V_full[j*p+k, r] * B[j,k] * W_full[k*n+i, r] ?
        # No, the verification condition is:
        # sum_r U_full[i*m+j, r] * V_full[j*p+k, r] * W_full[k*n+i, r] = 1 for all i,j,k.
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(total_rank):
                        s += U_full[i*m + j][r] * V_full[j*p + k][r] * W_full[k*n + i][r]
                    if s != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            print(f"Full decomposition valid with rank {total_rank}")
            return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
