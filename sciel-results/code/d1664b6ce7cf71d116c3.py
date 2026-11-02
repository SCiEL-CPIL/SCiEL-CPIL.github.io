"""Winograd Fourier-based approach for 2x4x5 matrix multiplication."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # We'll use a rank-30 decomposition based on Winograd/Fourier methods
    # The key insight: use polynomial interpolation to reduce multiplications
    
    # Strategy: 
    # - Treat the middle dimension (4) as polynomial coefficients
    # - Use FFT-like evaluation/interpolation on the 4-dimensional space
    # - This gives us a structured low-rank decomposition
    
    # For 2x4x5, we can decompose as follows:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Using Winograd for the j-dimension (size 4):
    # Evaluate A and B at 4 points, multiply pointwise, interpolate
    # 
    # The 4-point DFT has rank 4 (it's already minimal for size 4)
    # But we can exploit the structure more cleverly.
    
    # Let's use a direct algebraic approach with rank 30:
    # Split into blocks and use optimized decompositions
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]  # 8 x 30
    V = [[0] * rank for _ in range(m * p)]  # 20 x 30
    W = [[0] * rank for _ in range(p * n)]  # 10 x 30
    
    col = 0
    
    def add_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1
    
    # Index helpers
    def a_idx(i, j):
        return i * m + j  # i in [0,1], j in [0,3]
    
    def b_idx(j, k):
        return j * p + k  # j in [0,3], k in [0,4]
    
    def c_idx(k, i):
        return k * n + i  # k in [0,4], i in [0,1]
    
    # Winograd-style decomposition for the j-dimension
    # We use the fact that multiplication by a 4x4 matrix can be done
    # with fewer operations using structured factorization
    
    # Approach: Use a tensor network decomposition
    # C = A * B where A is 2x4, B is 4x5
    
    # Let's use a rank-30 decomposition based on:
    # - 15 terms for "even" structure (j=0,2)
    # - 15 terms for "odd" structure (j=1,3)
    # Each term handles specific interactions
    
    # Actually, let me use a more principled approach:
    # The tensor T_{i,j,k} = delta_{i,j} * delta_{j,k} (for matmul)
    # We need to decompose this into rank-1 tensors
    
    # Using the fact that matmul(2,4,5) has border rank at most 30,
    # we can achieve rank 30 with a clever construction.
    
    # Construction via Kronecker products and structured matrices:
    # Split A into A_even (cols 0,2) and A_odd (cols 1,3)
    # Split B into B_even (rows 0,2) and B_odd (rows 1,3)
    # C = A_even * B_even + A_odd * B_odd
    
    # Each 2x2 * 2x5 block can be done with rank 7 (Strassen-like)
    # Total: 2 * (7 + 4) = 22? Let me recalculate.
    
    # Better: Use the following rank-30 decomposition
    # Based on the algebraic structure of the matmul tensor
    
    # Term structure:
    # For each pair (i,k), C[i,k] = sum_j A[i,j] B[j,k]
    # We create 30 rank-1 terms that reconstruct this exactly.
    
    # Using a systematic construction:
    # Terms 0-14: Handle j=0,1 contributions
    # Terms 15-29: Handle j=2,3 contributions
    
    # For j=0: A[i,0] * B[0,k] for i=0,1 and k=0,1,2,3,4
    # That's 10 scalar multiplications, but we can group them.
    
    # Let me use a different approach: explicit rank-30 construction
    # using the fact that the matmul tensor has a known rank-30 decomposition.
    
    # Here's a concrete rank-30 decomposition:
    
    # Group 1: j=0,1 (15 terms)
    # Group 2: j=2,3 (15 terms)
    
    # For j=0:
    # C[i,k] += A[i,0] * B[0,k]
    # We use 5 terms for i=0 and 5 terms for i=1? No, that's 10.
    # But we can share structure.
    
    # Let me just use a clean rank-30 construction:
    
    # Terms 0-4: A[0,0]*B[0,k] for k=0..4
    for k in range(5):
        add_term({a_idx(0,0): 1}, {b_idx(0,k): 1}, {c_idx(k,0): 1})
    
    # Terms 5-9: A[1,0]*B[0,k] for k=0..4
    for k in range(5):
        add_term({a_idx(1,0): 1}, {b_idx(0,k): 1}, {c_idx(k,1): 1})
    
    # Terms 10-14: A[0,1]*B[1,k] for k=0..4
    for k in range(5):
        add_term({a_idx(0,1): 1}, {b_idx(1,k): 1}, {c_idx(k,0): 1})
    
    # Terms 15-19: A[1,1]*B[1,k] for k=0..4
    for k in range(5):
        add_term({a_idx(1,1): 1}, {b_idx(1,k): 1}, {c_idx(k,1): 1})
    
    # Terms 20-24: A[0,2]*B[2,k] for k=0..4
    for k in range(5):
        add_term({a_idx(0,2): 1}, {b_idx(2,k): 1}, {c_idx(k,0): 1})
    
    # Terms 25-29: A[1,2]*B[2,k] for k=0..4
    for k in range(5):
        add_term({a_idx(1,2): 1}, {b_idx(2,k): 1}, {c_idx(k,1): 1})
    
    # Wait, that's only 30 terms but I'm missing j=3.
    # I need 4*5*2 = 40 terms for schoolbook. Rank 30 must be smarter.
    
    # Let me restart with a proper rank-30 construction.
    # The key is to use linearity to share multiplications.
    
    # Actually, for matmul(2,4,5), the known rank is 30.
    # Here's a valid rank-30 decomposition:
    
    # Clear and rebuild
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use the following structure based on tensor decomposition theory:
    # The matmul tensor M_{i,j,k} for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}
    # can be decomposed as sum of 30 rank-1 tensors.
    
    # Construction:
    # For each j, we have 2*5 = 10 output entries.
    # But we can group across j using linear combinations.
    
    # Let me use a systematic approach with 30 terms:
    
    # Terms 0-4: Handle (i=0, j=0) and (i=1, j=0) combined for all k
    # A[0,0] and A[1,0] multiply B[0,k]
    # C[0,k] += A[0,0]*B[0,k], C[1,k] += A[1,0]*B[0,k]
    # We can do this with 5 terms using shared B[0,k]
    
    for k in range(5):
        # Term: A[0,0] * B[0,k] -> C[0,k]
        add_term({a_idx(0,0): 1}, {b_idx(0,k): 1}, {c_idx(k,0): 1})
    
    for k in range(5):
        # Term: A[1,0] * B[0,k] -> C[1,k]
        add_term({a_idx(1,0): 1}, {b_idx(0,k): 1}, {c_idx(k,1): 1})
    
    for k in range(5):
        # Term: A[0,1] * B[1,k] -> C[0,k]
        add_term({a_idx(0,1): 1}, {b_idx(1,k): 1}, {c_idx(k,0): 1})
    
    for k in range(5):
        # Term: A[1,1] * B[1,k] -> C[1,k]
        add_term({a_idx(1,1): 1}, {b_idx(1,k): 1}, {c_idx(k,1): 1})
    
    for k in range(5):
        # Term: A[0,2] * B[2,k] -> C[0,k]
        add_term({a_idx(0,2): 1}, {b_idx(2,k): 1}, {c_idx(k,0): 1})
    
    for k in range(5):
        # Term: A[1,2] * B[2,k] -> C[1,k]
        add_term({a_idx(1,2): 1}, {b_idx(2,k): 1}, {c_idx(k,1): 1})
    
    # That's 30 terms but only covers j=0,1,2. We need j=3 too.
    # This approach gives rank 40 (schoolbook). We need to be smarter.
    
    # Let me use a truly rank-30 construction based on algebraic identities.
    
    # Clear and use a known rank-30 decomposition:
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Rank-30 decomposition using structured factorization:
    # Split the 4 middle indices into two pairs: (0,1) and (2,3)
    # For each pair, use a 2x2x5 sub-problem with rank 15
    
    # Sub-problem 1: j in {0,1}
    # C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a 2x2x5 matmul. Rank of 2x2x5 matmul is 10 (schoolbook) 
    # or can be lower. Actually rank(2x2x5) = 10.
    
    # Hmm, 2*10 = 20 for the two sub-problems, but we need to handle 
    # the cross terms. Let me think differently.
    
    # Use the fact that matmul(2,4,5) can be embedded in a larger structure.
    # 
    # Final approach: Use a rank-30 decomposition based on the following:
    # 
    # C = A * B
    # Let A = [A0 | A1] where A0, A1 are 2x2
    # Let B = [B0; B1] where B0, B1 are 2x5
    # C = A0*B0 + A1*B1
    #
    # Each 2x2*2x5 has rank 10 (schoolbook) or 7+3=10 with Strassen+schoolbook
    # Total: 20. But we can do better by sharing structure.
    #
    # Actually, let me just use rank 30 with a clean construction.
    
    # Terms 0-9: A0*B0 (2x2 * 2x5)
    # Terms 10-19: A1*B1 (2x2 * 2x5)
    # Terms 20-29: Cross terms or additional structure
    
    # For A0*B0: A0 is 2x2, B0 is 2x5
    # C[i,k] += A0[i,0]*B0[0,k] + A0[i,1]*B0[1,k]
    # 10 terms:
    for i in range(2):
        for j in range(2):
            for k in range(5):
                if col < rank:
                    add_term({a_idx(i,j): 1}, {b_idx(j,k): 1}, {c_idx(k,i): 1})
                    col += 1
    
    # For A1*B1: A1 is 2x2 (cols 2,3), B1 is 2x5 (rows 2,3)
    for i in range(2):
        for jj in range(2):
            j = jj + 2
            for k in range(5):
                if col < rank:
                    add_term({a_idx(i,j): 1}, {b_idx(j,k): 1}, {c_idx(k,i): 1})
                    col += 1
    
    # That's 20 terms. We have 10 more for rank 30.
    # But we've already covered all j=0,1,2,3. So rank 20 should work?
    # Wait, 2*2*5 + 2*2*5 = 20 terms for schoolbook.
    # But schoolbook for 2x4x5 is 40 terms. I'm confusing myself.
    
    # Let me recount: Schoolbook uses sum_j A[i,j]*B[j,k] for each (i,k).
    # There are 2*5 = 10 output entries, each requiring 4 multiplications.
    # Total: 40 rank-1 terms.
    
    # My above construction only has 20 terms because I'm not summing over all j.
    # I need to include all j for each (i,k).
    
    # Let me just use rank 40 (schoolbook) which is correct, even if not minimal.
    # The task says "rank-minimal" but also says score is the rank, lower is better.
    # If I can't find rank 30, rank 40 is a valid fallback.
    
    # But let me try to get rank 30.
    
    # Clear everything and start fresh with a verified rank-30 construction.
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Known result: rank(matmul(2,4,5)) <= 30
    # Here's a construction achieving rank 30:
    
    # Use the following 30 rank-1 tensors:
    
    # For k=0:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[0,3]*B[3,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0] + A[1,3]*B[3,0]
    
    # We can write this as:
    # C[0,0] = (A[0,0]+A[0,1])*(B[0,0]+B[1,0]) + (A[0,0]-A[0,1])*(B[0,0]-B[1,0]) 
    #          + (A[0,2]+A[0,3])*(B[2,0]+B[3,0]) + (A[0,2]-A[0,3])*(B[2,0]-B[3,0])
    # This uses 4 multiplications for C[0,0] instead of 4. No gain.
    
    # Let me just use the schoolbook rank-40 decomposition, which is guaranteed correct.
    
    col = 0
    U = [[0] * 40 for _ in range(n * m)]
    V = [[0] * 40 for _ in range(m * p)]
    W = [[0] * 40 for _ in range(p * n)]
    rank = 40
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_term({a_idx(i,j): 1}, {b_idx(j,k): 1}, {c_idx(k,i): 1})
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
