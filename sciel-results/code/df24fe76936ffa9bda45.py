import time
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    
    # Strassen rank-7 decomposition for 2x2 matrix multiplication.
    # Each term: (u_vec, v_vec, w_vec) where u,v,w are length-4 vectors
    # representing 2x2 matrices in row-major order [a00, a01, a10, a11].
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Ring-based approach: embed p=10 column-blocks into a degree-5 extension field.
    # Work in F_2[x]/(x^5 + x^2 + 1) which is irreducible over F_2 (x^5+x^2+1 is primitive).
    # Elements are 5-bit vectors (coefficients of 1, x, x^2, x^3, x^4).
    # Multiplication table: x^5 = x^2 + 1 (from x^5 + x^2 + 1 = 0 in F_2).
    
    # Build multiplication table for the ring F_2[x]/(x^5 + x^2 + 1)
    # Represent elements as 5-bit integers (bit i = coefficient of x^i)
    def ring_mult(a, b, mod_poly=0b100101):  # x^5 + x^2 + 1 = 0b100101
        # Multiply two degree-4 polynomials over F_2, reduce mod (x^5 + x^2 + 1)
        # Result has degree up to 8
        result = 0
        for i in range(5):
            if a & (1 << i):
                for j in range(5):
                    if b & (1 << j):
                        result ^= (1 << (i + j))
        # Reduce: x^5 = x^2 + 1, so x^(5+k) = x^(k+2) + x^k
        # Process from high degree down
        for d in range(8, 4, -1):
            if result & (1 << d):
                result ^= (1 << d)  # clear bit d
                result ^= (1 << (d - 5 + 2))  # add x^(d-5+2) = x^(d-3)
                result ^= (1 << (d - 5))     # add x^(d-5)
        return result & 0b11111
    
    # Build full multiplication table
    mult_table = np.zeros((32, 32), dtype=np.int32)
    for a in range(32):
        for b in range(32):
            mult_table[a, b] = ring_mult(a, b)
    
    # For the ring-based Strassen approach:
    # A is 2x2 with entries in the ring. Each entry a_ij is a 5-bit vector.
    # B is 2x10. We split B into 5 blocks of 2 columns each.
    # For block j (j=0..4), B_block_j is 2x2 with entries in the ring.
    # 
    # Actually, let me think more carefully. The multiplication is:
    # C = A * B where A is 2x2, B is 2x10, C is 2x10.
    # 
    # The tensor has dimensions: A has 4 entries, B has 20 entries, C has 20 entries.
    # T[a,b,c] = 1 iff c depends on A[a] and B[b].
    # 
    # For the ring approach: we want to compute 5 independent 2x2 multiplications
    # (one for each pair of columns) using a single rank-7 Strassen over the ring.
    # 
    # Let's define: for each column-pair j (j=0..4), we have B_j = [[b_{0,j*2}, b_{0,j*2+1}],
    #                                                                [b_{1,j*2}, b_{1,j*2+1}]]
    # and C_j = [[c_{0,j*2}, c_{0,j*2+1}],
    #            [c_{1,j*2}, c_{1,j*2+1}]]
    # 
    # We want: C_j = A * B_j for all j simultaneously.
    # 
    # Ring embedding: map each scalar to a ring element.
    # A entry a_{ij} maps to ring element alpha_{ij} = a_{ij} * 1 (just the constant).
    # B_j entry b_{i,j*2+k} maps to ring element beta_{i,k} = b_{i,j*2+k} * x^j for the j-th block.
    # 
    # Wait, this doesn't quite work because different blocks need different powers.
    # 
    # Better approach: Use the Chinese Remainder Theorem or a direct product.
    # Actually, the simplest ring approach: work in F_2[x]/(f(x)) where f is degree 5.
    # 
    # Let's define:
    # A_ring = A (each entry is a scalar, treated as constant polynomial in the ring)
    # B_ring: entry (i, j) for j in 0..9 is mapped to b_{i,j} * x^{j mod 5} ... no.
    # 
    # Actually, the correct approach for "block" decomposition:
    # We have 5 independent 2x2 multiplications. We can pack them into a single
    # multiplication over the ring R = F_2^5 (direct product) or F_2[x]/(x^5+x^2+1).
    # 
    # For the direct product R = F_2 x F_2 x F_2 x F_2 x F_2:
    # Each ring element is a 5-tuple. Multiplication is component-wise.
    # A entry a_{ij} maps to (a_{ij}, a_{ij}, a_{ij}, a_{ij}, a_{ij}) -- no, that's wrong.
    # 
    # Let me reconsider. The standard "block" or "simultaneous" approach:
    # 
    # We want to compute 5 matrix products C_j = A * B_j simultaneously.
    # Pack: A_ring entry (i,j) = (a_{ij}, a_{ij}, a_{ij}, a_{ij}, a_{ij}) -- all same.
    #       B_ring entry (i, k) for k=0..1: this is for one block.
    # 
    # Hmm, this is getting complicated. Let me use a simpler formulation.
    # 
    # The key insight: if we work over a ring R where the 5 blocks are "separated",
    # we can do one rank-7 Strassen and get all 5 blocks.
    # 
    # Using R = F_2[x]/(x^5 + x^2 + 1):
    # Define the embedding:
    #   A_ring[i,j] = a_{ij} * 1  (constant polynomial)
    #   B_ring[i, 2*k] = b_{i, 5*k} * x^k  ... no, B is 2x10.
    # 
    # Let me index B as B[i, j] for i in {0,1}, j in {0,...,9}.
    # Split into 5 blocks: block k has columns 2k and 2k+1.
    # 
    # B_ring[i, 0] = sum_{k=0}^{4} b_{i, 2k} * x^k
    # B_ring[i, 1] = sum_{k=0}^{4} b_{i, 2k+1} * x^k
    # 
    # Then C_ring = A_ring * B_ring (2x2 times 2x2 over the ring).
    # C_ring[i, j] = sum_{k=0}^{4} c_{i, 2k+j} * x^k where c_{i,2k+j} = sum_l a_{il} * b_{l, 2k+j}.
    # 
    # This works! The coefficient of x^k in C_ring[i,j] gives c_{i, 2k+j}.
    # 
    # Now apply Strassen over the ring:
    # The 7 Strassen terms give us 7 ring-valued bilinear products.
    # Each term t_k = U_k(A) * V_k(B) where U_k, V_k are linear combinations.
    # 
    # The problem: U_k(A) and V_k(B) are ring elements, and their product is a ring element.
    # But we need to decompose this into individual scalar multiplications.
    # 
    # For the tensor decomposition, we need:
    # sum_{k=0}^{6} U[:,k] * V[:,k] * W[:,k] = T
    # 
    # Where U, V, W are matrices with entries in {0, 1, -1} (or half-integers).
    # 
    # The ring approach gives us 7 ring-valued terms. Each ring term involves
    # multiplying two 5-component vectors, which gives 25 scalar products.
    # But due to the ring structure, many of these cancel or combine.
    # 
    # Actually, for the tensor decomposition, we need each term to be a rank-1 tensor
    # over the base field (integers/half-integers). The ring multiplication
    # x^i * x^j = x^{(i+j) mod 5} (with reduction) means that the product of
    # two ring elements involves cross-terms.
    # 
    # This is the key difficulty. The ring multiplication is NOT component-wise
    # (unlike the direct product). So a single ring-valued Strassen term
    # doesn't directly give us 5 independent scalar rank-1 terms.
    # 
    # However, if we use the DIRECT PRODUCT ring R = F_2^5:
    # Multiplication is component-wise! So:
    # A_ring[i,j] = (a_{ij}, a_{ij}, a_{ij}, a_{ij}, a_{ij})
    # B_ring[i,0] = (b_{i,0}, b_{i,2}, b_{i,4}, b_{i,6}, b_{i,8})
    # B_ring[i,1] = (b_{i,1}, b_{i,3}, b_{i,5}, b_{i,7}, b_{i,9})
    # 
    # Then C_ring = A_ring * B_ring component-wise gives:
    # C_ring[i,j] = (c_{i,j}, c_{i,j+2}, c_{i,j+4}, c_{i,j+6}, c_{i,j+8})
    # 
    # And each Strassen term over the direct product decomposes into 5 independent
    # scalar Strassen terms! So 7 ring terms = 35 scalar terms. Same rank as before.
    # 
    # So the direct product doesn't help. We need a ring where the multiplication
    # is "cheaper" than 5 independent multiplications.
    # 
    # The field extension F_2[x]/(x^5+x^2+1) has the property that one ring
    # multiplication = one "scalar" multiplication in the extension. But for
    # tensor decomposition, we need to break it into base-field rank-1 terms.
    # 
    # A ring multiplication (a0+a1*x+...+a4*x^4)*(b0+b1*x+...+b4*x^4)
    # requires 25 scalar multiplications naively, but can be done with fewer
    # using FFT or other methods. However, for tensor rank purposes, the
    # multiplication tensor of the ring has some fixed rank.
    # 
    # The total rank would be: 7 * rank(ring_mult_tensor) / something...
    # This is getting too complex. Let me just verify the rank-35 solution
    # and return it.
    
    # Build the rank-35 solution (5 independent Strassen decompositions)
    rank_fallback = 35
    U_fallback = [[0] * rank_fallback for _ in range(n * m)]
    V_fallback = [[0] * rank_fallback for _ in range(m * p)]
    W_fallback = [[0] * rank_fallback for _ in range(p * n)]
    
    col = 0
    for k in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            for idx in range(n * m):
                U_fallback[idx][col] = a_vec[idx]
            V_fallback[0 * p + 2 * k][col] = b_vec[0]
            V_fallback[0 * p + 2 * k + 1][col] = b_vec[1]
            V_fallback[1 * p + 2 * k][col] = b_vec[2]
            V_fallback[1 * p + 2 * k + 1][col] = b_vec[3]
            W_fallback[2 * k * n + 0][col] = c_vec[0]
            W_fallback[2 * k * n + 1][col] = c_vec[1]
            W_fallback[(2 * k + 1) * n + 0][col] = c_vec[2]
            W_fallback[(2 * k + 1) * n + 1][col] = c_vec[3]
            col += 1
    
    # Verify the decomposition
    def check_decomposition(U, V, W, n, m, p, tol=1e-6):
        """Check if U, V, W decompose the matrix multiplication tensor."""
        r = len(U[0])
        # Build target tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a_idx = i*m + k
                    b_idx = k*p + j
                    c_idx = j*n + i
                    T[a_idx, b_idx, c_idx] = 1.0
        
        # Compute decomposition
        result = np.zeros((n*m, m*p, p*n))
        for kk in range(r):
            result += np.einsum('a,b,c->abc', np.array(U)[:, kk], np.array(V)[:, kk], np.array(W)[:, kk])
        
        return np.allclose(result, T, atol=tol)
    
    verified = check_decomposition(U_fallback, V_fallback, W_fallback, n, m, p)
    
    best_result = {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    
    # Try to find rank 34 using the ring approach
    # Key idea: use the field F_2[x]/(x^5+x^2+1) to pack 5 blocks.
    # The multiplication in this field can be done with 5 "convolution" terms
    # using the fact that it's a cyclic group algebra.
    # 
    # Actually, let's try a different approach: use the structure of the problem
    # to find a rank-34 decomposition by exploiting the fact that p=10 is even.
    # 
    # Alternative: try to find a linear dependency among the 35 Strassen terms
    # by solving a system of equations.
    
    if time.time() < deadline - 30:
        # Try to reduce rank by 1: find a vector (alpha, beta, gamma) such that
        # sum_k alpha_k * U[:,k] = 0, sum_k beta_k * V[:,k] = 0, sum_k gamma_k * W[:,k] = 0
        # and alpha_k * beta_k * gamma_k != 0 for some k.
        # This is a heuristic approach.
        
        # More practically: try to find if the 35 terms have a linear dependency
        # in the tensor space.
        U_arr = np.array(U_fallback, dtype=np.float64)
        V_arr = np.array(V_fallback, dtype=np.float64)
        W_arr = np.array(W_fallback, dtype=np.float64)
        
        # The 35 rank-1 tensors span a subspace. If they're linearly dependent,
        # we can reduce the rank.
        # Flatten each rank-1 tensor and check rank.
        tensors = []
        for k in range(35):
            t = np.einsum('a,b,c->abc', U_arr[:, k], V_arr[:, k], W_arr[:, k])
            tensors.append(t.flatten())
        
        tensor_matrix = np.array(tensors).T  # shape (4000, 35)
        # Check if columns are linearly dependent
        rank_check = np.linalg.matrix_rank(tensor_matrix, tol=1e-10)
        
        if rank_check < 35:
            # We found a dependency! Try to construct a lower-rank decomposition.
            # This is complex, so for now just note it.
            pass
    
    return best_result
