"""Matrix multiplication 4x5x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _winograd_4x4():
    """
    Winograd Fourier Transform Algorithm for 4x4 matrix multiplication.
    Returns U, V, W matrices for the 4x4x4 tensor with rank 16.
    
    The Winograd scheme uses:
    - M = 4x4 matrix of multipliers (from FFT of size 4)
    - U: (4*4, 16) - input A transform
    - V: (4*4, 16) - input B transform  
    - W: (4*4, 16) - output C transform
    
    The 4-point DFT matrix (circulant) gives us the structure.
    For Winograd FMA with n=4:
    - We use the cyclic group Z_4
    - The transform matrices are derived from the DFT
    """
    n = 4
    rank = 16
    
    # The Winograd 4x4 scheme:
    # M = [[1, 1, 1, 1],
    #      [1, -1, 1, -1],
    #      [1, 1, -1, -1],
    #      [1, -1, -1, 1]]
    # This is the Hadamard-like transform for Z_4
    
    # Actually, the standard Winograd for 4x4 uses:
    # U = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],  -- row 0 of A
    #      [1,-1,1,-1,1,1,1,1,1,-1,1,-1,1,1,1,1],  -- row 1
    #      ...]
    
    # Let me use the known Winograd 4x4 decomposition.
    # The rank-16 Winograd scheme for 4x4 matrix multiplication:
    
    # The transform matrices for Winograd FMA(4,4,4):
    # Using the group Z_4 = {0,1,2,3}
    
    # The DFT matrix for size 4 (with omega = i):
    # DFT = [[1, 1, 1, 1],
    #        [1, i, -1, -i],
    #        [1, -1, 1, -1],
    #        [1, -i, -1, i]]
    
    # But for Winograd, we use a different factorization.
    # The standard Winograd 4x4 uses these transforms:
    
    # Let me construct it properly.
    # For the 4x4 Winograd scheme, the decomposition has the form:
    # C = W * (M * (U * A)) * V where M is diagonal with 16 entries
    
    # Actually, let me use the explicit known decomposition.
    # The Winograd 4x4 matrix multiplication can be written as:
    # 
    # Let's use the following approach:
    # Partition into blocks and use the FFT-based approach.
    
    # For a 4x4 matrix multiply C = A * B:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    
    # Winograd FMA approach:
    # We use the cyclic convolution structure.
    # For 4x4, we can view it as computing 4 cyclic convolutions of length 4.
    
    # The key insight: a 4x4 matrix multiply can be decomposed using
    # the fact that it involves 16 independent multiplications in the
    # frequency domain.
    
    # Let me use the explicit Winograd 4x4 tables.
    # From Winograd's paper, the 4x4 scheme has:
    # - 16 multiplications
    # - Specific transform matrices
    
    # Here's the standard Winograd 4x4 decomposition:
    
    # U matrix (maps A to intermediate): shape (16, 16)
    # The rows correspond to the 16 elements of A (4x4), columns to 16 multipliers
    
    # Let me use a different, verified approach.
    # The Winograd 4x4 can be constructed as follows:
    
    # We use the tensor decomposition based on the group Z_4.
    # The structure constants give us the decomposition.
    
    # For Z_4 = {e, g, g^2, g^3}:
    # The group algebra has 4 irreducible representations (1-dim each for abelian group)
    # This gives rank 4 * 4 = 16 for the 4x4 case.
    
    # The characters of Z_4:
    # chi_0: e->1, g->1, g^2->1, g^3->1
    # chi_1: e->1, g->i, g^2->-1, g^3->-i
    # chi_2: e->1, g->-1, g^2->1, g^3->-1
    # chi_3: e->1, g->-i, g^2->-1, g^3->i
    
    # For the matrix multiplication tensor, we use:
    # T[i,j,k] = 1 if i+j+k ≡ 0 (mod 4) in the appropriate sense
    
    # Actually, for matrix multiplication C = AB where C[i,j] = sum_k A[i,k]B[k,j]:
    # The tensor is T[i,k,j] = 1 (for all i,k,j)
    # This is the standard matrix multiplication tensor.
    
    # The Winograd approach exploits that this tensor can be decomposed
    # using the structure of the group Z_4.
    
    # Let me just use the known explicit matrices for Winograd 4x4.
    
    # From the literature, the Winograd 4x4 decomposition:
    
    # The 16 rank-1 terms are indexed by (a, b) where a, b in {0,1,2,3}
    # Each term contributes to specific entries of A, B, C.
    
    # Let me construct it using the FFT approach:
    # 
    # For 4x4 matrix multiplication:
    # C = A * B
    # 
    # We can write: C = F^{-1} * diag(f) * F * (A * B) * F^{-1} * diag(g) * F
    # where f and g are the Fourier transforms of the appropriate vectors.
    
    # Actually, the cleanest way is:
    # 
    # Let ω = exp(2πi/4) = i
    # 
    # The DFT matrix F (4x4):
    # F[i,j] = ω^(ij)
    
    # For the Winograd scheme, we use:
    # U[i, (a,b)] = ω^((i-a)*b) for appropriate indices
    # V[k, (a,b)] = ω^((k-b)*a) 
    # W[j, (a,b)] = ω^((j+b)*a) / 4  (with normalization)
    
    # Let me be more careful. The matrix multiplication tensor is:
    # T = sum_{i,k,j} e_i ⊗ e_k ⊗ e_j (where e_i is basis vector)
    # 
    # The Winograd decomposition for 4x4:
    # 
    # Using the group Z_4, the decomposition is:
    # T = (1/4) sum_{a=0}^{3} sum_{b=0}^{3} 
    #     (sum_i ω^(ia) e_i) ⊗ (sum_k ω^(-kb) e_k) ⊗ (sum_j ω^((j-b)a) e_j)
    
    # Wait, this doesn't quite work for standard matrix multiplication.
    # Let me think again.
    
    # For standard matrix multiplication: C[i,j] = sum_k A[i,k] * B[k,j]
    # The tensor is: T = sum_{i,j,k} e_i ⊗ e_k ⊗ e_j
    
    # This is actually the tensor for the bilinear map (A,B) -> A*B.
    
    # The Winograd FMA for n x n matrix multiplication uses:
    # T_n = sum_{i,j,k} e_i ⊗ e_k ⊗ e_j
    
    # For the cyclic convolution approach, we need a different tensor.
    # The standard matrix multiplication is NOT a convolution.
    
    # However, Winograd showed that matrix multiplication CAN be done
    # with O(n^2 log n) multiplications using the FMA approach.
    
    # The key is to use the fact that:
    # A*B can be computed using cyclic convolutions with appropriate padding.
    
    # For 4x4, the Winograd scheme uses 16 multiplications.
    # Let me use the explicit known decomposition.
    
    # From Winograd 1980, the 4x4 matrix multiplication:
    
    # Let me use a verified construction.
    # The 4x4 Winograd decomposition:
    
    # Define the transform matrices:
    # For the input A (4x4), we compute 16 intermediate values
    # For the input B (4x4), we compute 16 intermediate values
    # Then 16 multiplications
    # Then combine to get C (4x4)
    
    # The explicit Winograd 4x4:
    
    # Multiplier indices: (r, s) for r, s in {0, 1, 2, 3}
    
    # U[i, (r,s)] = value for A[i, ?]
    # Actually, U has shape (16, 16) where rows are A elements and columns are multipliers
    
    # Let me use the following known construction:
    # 
    # The Winograd 4x4 uses the following structure:
    # 
    # For each pair (r, s) in {0,1,2,3} x {0,1,2,3}:
    #   u_{r,s} = sum_{i=0}^{3} A[i, (i+r) mod 4] * ω^(i*r)
    #   v_{r,s} = sum_{k=0}^{3} B[(k+s) mod 4, k] * ω^(k*s)  
    #   c_{r,s} = u_{r,s} * v_{r,s}
    # 
    # Then: C[i,j] = (1/4) sum_{r,s} c_{r,s} * ω^(-((i+r) mod 4) * r + ((j-s) mod 4) * s)
    # 
    # Hmm, this is getting complicated. Let me just use the schoolbook for now
    # and focus on getting a valid decomposition.
    
    # Actually, let me use a simpler approach: just use the schoolbook decomposition
    # which has rank 4*4*4 = 64 for the 4x4 part.
    
    # But wait - the task is 4x5x6, not 4x4x4.
    # Let me reconsider the overall approach.
    
    # For 4x5x6:
    # Schoolbook rank = 4*5*6 = 120
    
    # The strategy from the directive suggests:
    # - Partition 4x5x6 into 4x4x4 block + 4x1x6 strip
    # - Use Winograd for 4x4x4 (rank 16)
    # - Use schoolbook for 4x1x6 (rank 24)
    # - Total: 40
    
    # But I need to verify this is valid.
    
    # Let me first just return the schoolbook decomposition to ensure correctness,
    # then try to improve.
    
    return None


def _schoolbook_decomposition(n, m, p):
    """Returns the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return U, V, W


def _validate_decomposition(n, m, p, U, V, W):
    """Validate that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    
    # Check all tensor entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check that invalid entries are 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                
                if not (j == j2 and i == i2 and k == k2):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    
    return True


def _winograd_4x4x4():
    """
    Construct the Winograd 4x4x4 decomposition with rank 16.
    Uses the FFT-based approach for cyclic convolution.
    
    For 4x4 matrix multiplication C = A*B:
    C[i,j] = sum_{k=0}^{3} A[i,k] * B[k,j]
    
    We use the fact that this can be computed using 4-point DFTs.
    """
    n = 4
    rank = 16
    
    # The 4-point DFT matrix
    omega = np.exp(2j * np.pi / 4)  # = i
    F = np.array([[omega ** (i * j) for j in range(n)] for i in range(n)], dtype=complex)
    
    # For the Winograd scheme, we use the following:
    # The matrix multiplication tensor for n x n is:
    # T[i,k,j] = 1 for all i,k,j
    # 
    # This can be decomposed using the group Z_n structure.
    # For n=4, we get rank 16.
    
    # The decomposition:
    # T = (1/n) * sum_{a=0}^{n-1} sum_{b=0}^{n-1} 
    #     (sum_i omega^(i*a) e_i) ⊗ (sum_k omega^(-k*b) e_k) ⊗ (sum_j omega^((j-k+a)*b) ... )
    
    # Actually, let me use a different, verified approach.
    # 
    # The Winograd 4x4 matrix multiplication:
    # 
    # Let's use the explicit construction from the literature.
    # 
    # For 4x4, the Winograd scheme uses:
    # - 16 multiplications
    # - Transform matrices derived from the 4-point FFT
    
    # The key matrices:
    # U (for A): 16 x 16
    # V (for B): 16 x 16  
    # W (for C): 16 x 16
    
    # Let me construct these using the FFT structure.
    
    # For the 4x4 case, we can use:
    # 
    # Define the transform:
    # For A: U[i, (r,s)] = omega^((i+r) mod 4 * s) * delta... 
    # 
    # This is getting complex. Let me use a simpler verified decomposition.
    
    # Alternative: Use the fact that 4x4 matrix multiplication has
    # a known decomposition with rank 16 (Winograd).
    
    # Let me construct it explicitly:
    
    # The 16 rank-1 terms are indexed by (a, b) where a, b in {0, 1, 2, 3}
    
    # For term (a, b):
    # U[i, (a,b)] = omega^(i*a) for the A part
    # V[k, (a,b)] = omega^(-k*b) for the B part
    # W[j, (a,b)] = omega^(j*b) for the C part
    
    # But this gives: sum_{a,b} U[i,(a,b)] * V[k,(a,b)] * W[j,(a,b)]
    # = sum_{a,b} omega^(i*a) * omega^(-k*b) * omega^(j*b)
    # = sum_{a,b} omega^(i*a + (j-k)*b)
    # = (sum_a omega^(i*a)) * (sum_b omega^((j-k)*b))
    # = 4*delta(i,0) * 4*delta(j-k,0)  [by orthogonality of characters]
    # = 16 * delta(i,0) * delta(j,k)
    
    # This is NOT the matrix multiplication tensor. It's something else.
    
    # The matrix multiplication tensor is T[i,k,j] = 1 for ALL i,k,j.
    # So I need a different decomposition.
    
    # Let me reconsider. The tensor for matrix multiplication is:
    # T = sum_{i,k,j} e_i ⊗ e_k ⊗ e_j
    # 
    # This is just the all-ones tensor of size 4x4x4.
    # The rank of this tensor is 4 (it's just the sum of 4 rank-1 tensors
    # if we group by i, or by k, or by j).
    
    # Wait, that's not right either. The tensor T[i,k,j] = 1 means:
    # For each (i,k,j), the coefficient is 1.
    # A rank-1 tensor is u ⊗ v ⊗ w where (u⊗v⊗w)[i,k,j] = u[i]*v[k]*w[j].
    # 
    # So T[i,k,j] = 1 for all i,k,j means T = (sum_i e_i) ⊗ (sum_k e_k) ⊗ (sum_j e_j)
    # which is rank 1!
    
    # But that can't be right for matrix multiplication. Let me re-examine.
    
    # The matrix multiplication bilinear map is:
    # (A, B) -> C where C[i,j] = sum_k A[i,k] * B[k,j]
    # 
    # The tensor for this is:
    # T[i,j,k] = 1 (coefficient of A[i,k] * B[k,j] in C[i,j])
    # 
    # Wait, the tensor indices are:
    # T[a, b, c] where a indexes A, b indexes B, c indexes C
    # a = (i, k), b = (k, j), c = (i, j)
    # T[(i,k), (k,j), (i,j)] = 1
    # 
    # So the tensor is:
    # T = sum_{i,j,k} e_{(i,k)} ⊗ e_{(k,j)} ⊗ e_{(i,j)}
    # 
    # This is NOT the all-ones tensor. It's a specific tensor where
    # the indices are correlated.
    
    # OK so for the 4x4x4 case:
    # T = sum_{i=0}^{3} sum_{j=0}^{3} sum_{k=0}^{3} e_{(i,k)} ⊗ e_{(k,j)} ⊗ e_{(i,j)}
    
    # This tensor has 64 non-zero entries (one for each (i,j,k) triple).
    # The schoolbook decomposition has rank 64.
    # Winograd showed it can be done with rank 16.
    
    # Let me use the Winograd FMA approach properly.
    # 
    # The Winograd FMA for n x n matrix multiplication:
    # 
    # Key idea: Use the cyclic group Z_n to structure the computation.
    # 
    # For n=4, the decomposition uses the 4 irreducible representations
    # of Z_4 (which are all 1-dimensional since Z_4 is abelian).
    
    # The Winograd 4x4 decomposition:
    # 
    # Let ω = i (primitive 4th root of unity)
    # 
    # Define:
    # u_{r,s} = sum_{i=0}^{3} A[i, (i+r) mod 4] * ω^(i*r)  for r,s in {0,1,2,3}
    # v_{r,s} = sum_{k=0}^{3} B[(k+s) mod 4, k] * ω^(k*s)  for r,s in {0,1,2,3}
    # 
    # Wait, I'm overcomplicating this. Let me just use the schoolbook
    # for now and ensure correctness. The rank will be 120 for 4x5x6.
    
    # Let me return None to indicate this function doesn't have a working
    # Winograd implementation yet.
    
    return None


def construct():
    """
    Returns the best matrix multiplication tensor decomposition found.
    Starts with schoolbook and tries to find lower-rank alternatives.
    """
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Validate the schoolbook decomposition
    assert _validate_decomposition(n, m, p, U, V, W), "Schoolbook decomposition failed validation"
    
    # Try to find better decompositions using Winograd approach
    # For 4x5x6, we can try to use the Winograd 4x4x4 for the 4x4x4 part
    # and schoolbook for the remaining parts
    
    # The 4x5x6 multiplication:
    # C[i,j] = sum_{k=0}^{4} A[i,k] * B[k,j]  for i in 0..3, j in 0..5
    # 
    # We can split k into k=0..3 and k=4:
    # C[i,j] = sum_{k=0}^{3} A[i,k] * B[k,j] + A[i,4] * B[4,j]
    # 
    # The first term is a 4x4x6 multiplication (A is 4x4, B is 4x6, C is 4x6)
    # The second term is a 4x1x6 multiplication (A is 4x1, B is 1x6, C is 4x6)
    
    # For the 4x4x6 part:
    # We can further split j into j=0..3 and j=4..5:
    # C[i,j] for j=0..3: sum_{k=0}^{3} A[i,k] * B[k,j]  (4x4x4)
    # C[i,j] for j=4..5: sum_{k=0}^{3} A[i,k] * B[k,j]  (4x4x2)
    
    # So we have:
    # 4x4x4 part: can use Winograd (rank 16) if we can implement it
    # 4x4x2 part: schoolbook (rank 32)
    # 4x1x6 part: schoolbook (rank 24)
    # Total: 16 + 32 + 24 = 72
    
    # This is better than 120!
    
    # Let me implement this decomposition.
    
    # First, let me try to implement the Winograd 4x4x4.
    
    # For now, let's use schoolbook for all parts to ensure correctness:
    # 4x4x4: rank 64
    # 4x4x2: rank 32
    # 4x1x6: rank 24
    # Total: 120 (same as full schoolbook)
    
    # So we need the Winograd 4x4x4 to get an improvement.
    
    # Let me try to implement the Winograd 4x4x4 using the FFT approach.
    
    # The Winograd 4x4x4 decomposition:
    # 
    # Using the group Z_4 = {0, 1, 2, 3}
    # 
    # The tensor is:
    # T = sum_{i,k,j in Z_4} e_{(i,k)} ⊗ e_{(k,j)} ⊗ e_{(i,j)}
    # 
    # where the indices are interpreted modulo 4.
    
    # Wait, that's not the standard matrix multiplication tensor.
    # The standard tensor has:
    # T[(i,k), (k,j), (i,j)] = 1
    # 
    # where i, k, j are ordinary indices (not modulo).
    
    # For the 4x4 case, all indices are in {0,1,2,3}, so modulo 4 is the same.
    
    # The Winograd approach uses the following:
    # 
    # Define the transform:
    # For each (r, s) in Z_4 x Z_4:
    #   u_{r,s} = sum_{i in Z_4} A[i, i+r] * ω^(i*r)
    #   v_{r,s} = sum_{k in Z_4} B[k+r, k+s] * ω^(k*s)
    #   c_{r,s} = u_{r,s} * v_{r,s}
    # 
    # Then: C[i, j] = (1/16) * sum_{r,s} c_{r,s} * ω^(-i*r + (j-r-s)*s)
    # 
    # Hmm, I'm not sure this is correct. Let me just use the schoolbook.
    
    # Actually, let me try a different approach: use the known fact that
    # the 4x4x4 matrix multiplication tensor has rank 16 (Winograd).
    
    # I'll construct it using the following method:
    # 
    # The tensor T for 4x4x4 matrix multiplication:
    # T[(i,k), (k,j), (i,j)] = 1
    # 
    # This can be written as:
    # T = sum_{i,k,j} e_{(i,k)} ⊗ e_{(k,j)} ⊗ e_{(i,j)}
    
    # Using the FFT, we can diagonalize this tensor.
    # 
    # Let F be the 4x4 DFT matrix.
    # 
    # Define:
    # U[i, (a,b)] = F[i,a] * F[i,b]  (for the A part)
    # V[k, (a,b)] = F[k,a] * F[-k,b]  (for the B part)
    # W[j, (a,b)] = F[j,a] * F[j,b]  (for the C part)
    
    # Then: sum_{a,b} U[i,(a,b)] * V[k,(a,b)] * W[j,(a,b)]
    # = sum_{a,b} F[i,a] * F[i,b] * F[k,a] * F[-k,b] * F[j,a] * F[j,b]
    # = (sum_a F[i,a] * F[k,a] * F[j,a]) * (sum_b F[i,b] * F[-k,b] * F[j,b])
    
    # Using the orthogonality of the DFT:
    # sum_a F[i,a] * F[k,a] * F[j,a] = ?
    
    # This is the product of three DFT entries, which doesn't simplify nicely.
    
    # Let me give up on the Winograd approach for now and just use schoolbook.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
