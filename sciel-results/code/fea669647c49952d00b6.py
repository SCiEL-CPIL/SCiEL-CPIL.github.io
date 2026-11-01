"""Matrix multiplication 2x2x12 rank-minimal decomposition."""
import time

DIMENSIONS = [2, 2, 12]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2x12 matrix multiplication.
    
    Strategy: Partition the 12 columns into two groups of 6.
    For 2x2x6, we can use a decomposition of rank 11 (better than schoolbook 24).
    Total rank = 2 * 11 = 22.
    
    The tensor is M_{i,j,k} for i in [0,1], j in [0,1], k in [0,11].
    We split k into k1 in [0,5] and k2 in [6,11].
    
    For each 2x2x6 block, we need rank 11.
    
    A known rank-11 decomposition for 2x2x6:
    We can think of it as multiplying [2x2] * [2x6] = [2x6].
    
    Let's use the following approach for 2x2x6:
    The standard schoolbook is rank 24. We can do better.
    
    Actually, let's think about this more carefully. The tensor for 2x2x6 is:
    T_{i,j,k} where i,j in {0,1}, k in {0,...,5}
    
    We want to express this as sum of rank-1 tensors.
    
    One approach: Use the fact that 2x2x6 can be decomposed using 
    a combination of Strassen-like ideas.
    
    For 2x2x2, Strassen gives rank 7.
    For 2x2x6, we can split the 6 into 2+2+2 and use Strassen on each pair?
    That would give 3*7 = 21, which is worse than schoolbook 24? No, 21 < 24.
    
    Wait, let me reconsider. If we split 6 into three groups of 2:
    - Group 1: k in {0,1}
    - Group 2: k in {2,3}  
    - Group 3: k in {4,5}
    
    Each 2x2x2 block can be done with rank 7 (Strassen).
    Total: 3 * 7 = 21.
    
    But can we do better? The known lower bound for 2x2x6 is... 
    Actually, I recall that the rank of 2x2x6 is 11 or 12.
    
    Let me try to construct a rank-11 or rank-12 decomposition.
    
    Alternative: Use a direct algebraic construction.
    
    For 2x2xn matrix multiplication, the rank is known to be:
    - n=1: 4 (schoolbook, optimal)
    - n=2: 7 (Strassen)
    - n=3: 10? or 11?
    - n=4: 14? 
    - n=5: 17?
    - n=6: 20? or 21?
    
    Actually, I'm not sure about the exact optimal ranks. Let me just use a 
    reasonable construction.
    
    For 2x2x6, let's use the following:
    Split into three 2x2x2 blocks, each with Strassen rank 7. Total = 21.
    
    For 2x2x12, split into six 2x2x2 blocks, each with Strassen rank 7. Total = 42.
    
    But wait, the current program already achieves 48 (schoolbook). So 42 is an improvement.
    
    Actually, let me reconsider. Can we share factors between blocks?
    
    For 2x2x12, if we split the 12 columns into 6 groups of 2, and use Strassen on each,
    we get rank 6*7 = 42.
    
    But maybe we can do better by not fully splitting.
    
    Let me try a different approach: use a rank-11 construction for 2x2x6.
    
    Here's a known result: The rank of matrix multiplication 2x2xn is at most 
    floor(3n/2) + 2 for n >= 2? Let me check: for n=2, floor(3) + 2 = 5, but Strassen is 7.
    That doesn't work.
    
    Let me just implement the Strassen-based approach for 2x2x12 with rank 42.
    
    Strassen's algorithm for 2x2x2:
    Given A (2x2), B (2x2), C = A*B (2x2)
    
    Let A = [[a11, a12], [a21, a22]]
    Let B = [[b11, b12], [b21, b22]]
    
    M1 = (a11 + a22) * (b11 + b22)
    M2 = (a21 + a22) * b11
    M3 = a11 * (b12 - b22)
    M4 = a22 * (b21 - b11)
    M5 = (a11 + a12) * b22
    M6 = (a21 - a11) * (b11 + b12)
    M7 = (a12 - a22) * (b21 + b22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    So for each 2x2x2 block, we have 7 rank-1 terms.
    
    For 2x2x12, we have 6 such blocks. Total rank = 42.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # We'll construct U, V, W for the full 2x2x12 multiplication.
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (24, rank)
    # W has shape (p*n, rank) = (24, rank)
    
    # Split the 12 columns (k dimension) into 6 groups of 2.
    # For each group, use Strassen's 7-term decomposition.
    
    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x 42
    
    # Strassen coefficients for 2x2x2:
    # M1 = (a11 + a22) * (b11 + b22)
    # M2 = (a21 + a22) * b11
    # M3 = a11 * (b12 - b22)
    # M4 = a22 * (b21 - b11)
    # M5 = (a11 + a12) * b22
    # M6 = (a21 - a11) * (b11 + b12)
    # M7 = (a12 - a22) * (b21 + b22)
    
    # For each Strassen term, we need:
    # - U coefficients (for A matrix elements)
    # - V coefficients (for B matrix elements)
    # - W coefficients (for C matrix elements)
    
    # A is 2x2, indexed as A[i,j] with i in {0,1}, j in {0,1}
    # B is 2x2, indexed as B[j,k] with j in {0,1}, k in {0,1} (local)
    # C is 2x2, indexed as C[i,k] with i in {0,1}, k in {0,1} (local)
    
    # U index: i*m + j = i*2 + j
    # V index: j*p + k (global) = j*12 + (group_start + k_local)
    # W index: k*n + i (global) = (group_start + k_local)*2 + i
    
    # Strassen terms (U_coeff, V_coeff, W_coeff):
    # Each term is a sum of products of linear forms in A and B.
    
    # Let's define the 7 Strassen terms:
    # Term 1: (a00 + a11) * (b00 + b11)
    # Term 2: (a10 + a11) * b00
    # Term 3: a00 * (b01 - b11)
    # Term 4: a11 * (b10 - b00)
    # Term 5: (a00 + a01) * b11
    # Term 6: (a10 - a00) * (b00 + b01)
    # Term 7: (a01 - a11) * (b10 + b11)
    
    # For each term, we need to express it as:
    # sum_i U_i * V_i * W_i
    
    # But wait, Strassen's algorithm uses 7 multiplications, where each multiplication
    # is of a linear form in A times a linear form in B. So each term IS a rank-1 tensor.
    
    # Term t: L_A^t * L_B^t * (contribution to C)
    
    # Let me be more precise. For term t:
    # L_A^t = sum_{i,j} alpha^t_{i,j} * A_{i,j}
    # L_B^t = sum_{j,k} beta^t_{j,k} * B_{j,k}
    # C contribution: gamma^t_{i,k} * (L_A^t * L_B^t)
    
    # So the rank-1 tensor for term t has:
    # U[i*2+j] = alpha^t_{i,j}
    # V[j*12 + k_global] = beta^t_{j,k_local}
    # W[k_global*2 + i] = gamma^t_{i,k_local}
    
    # Let's define the 7 terms:
    
    # Term 1: (a00 + a11) * (b00 + b11)
    # alpha: a00=1, a11=1, others=0
    # beta: b00=1, b11=1, others=0
    # gamma: C00 += 1, C01 += 0, C10 += 0, C11 += -1
    # Wait, from Strassen: C11 = M1 + M4 - M5 + M7, C12 = M3 + M5, C21 = M2 + M4, C22 = M1 - M2 + M3 + M6
    # Using 0-indexed: C[0,0] = M1 + M4 - M5 + M7, C[0,1] = M3 + M5, C[1,0] = M2 + M4, C[1,1] = M1 - M2 + M3 + M6
    
    # Let me redo with 0-indexed:
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1] - B[1,1])
    # M4 = A[1,1] * (B[1,0] - B[0,0])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # So for each M_t, the gamma coefficients (for C[0,0], C[0,1], C[1,0], C[1,1]) are:
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, -1]
    # M3: [0, 1, 0, 1]
    # M4: [1, 0, 1, 0]
    # M5: [-1, 1, 0, 0]
    # M6: [0, 0, 0, 1]
    # M7: [1, 0, 0, 0]
    
    # Wait, let me double-check:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # So:
    # M1: C[0,0] += 1, C[1,1] += 1 => gamma = [1, 0, 0, 1]
    # M2: C[1,0] += 1, C[1,1] += -1 => gamma = [0, 0, 1, -1]
    # M3: C[0,1] += 1, C[1,1] += 1 => gamma = [0, 1, 0, 1]
    # M4: C[0,0] += 1, C[1,0] += 1 => gamma = [1, 0, 1, 0]
    # M5: C[0,0] += -1, C[0,1] += 1 => gamma = [-1, 1, 0, 0]
    # M6: C[1,1] += 1 => gamma = [0, 0, 0, 1]
    # M7: C[0,0] += 1 => gamma = [1, 0, 0, 0]
    
    # Now for each M_t, the alpha (for A) and beta (for B):
    # M1: alpha = {A[0,0]: 1, A[1,1]: 1}, beta = {B[0,0]: 1, B[1,1]: 1}
    # M2: alpha = {A[1,0]: 1, A[1,1]: 1}, beta = {B[0,0]: 1}
    # M3: alpha = {A[0,0]: 1}, beta = {B[0,1]: 1, B[1,1]: -1}
    # M4: alpha = {A[1,1]: 1}, beta = {B[1,0]: 1, B[0,0]: -1}
    # M5: alpha = {A[0,0]: 1, A[0,1]: 1}, beta = {B[1,1]: 1}
    # M6: alpha = {A[1,0]: 1, A[0,0]: -1}, beta = {B[0,0]: 1, B[0,1]: 1}
    # M7: alpha = {A[0,1]: 1, A[1,1]: -1}, beta = {B[1,0]: 1, B[1,1]: 1}
    
    # Let's encode these:
    # alpha is a dict from (i,j) to coefficient
    # beta is a dict from (j,k) to coefficient
    # gamma is a dict from (i,k) to coefficient
    
    strassen_terms = [
        # (alpha, beta, gamma)
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Now construct U, V, W for the full 2x2x12 problem.
    # We have 6 groups of 2 columns each.
    # Group g (g=0,...,5) handles k in {2g, 2g+1}
    
    for g in range(6):
        k_start = 2 * g
        for t in range(7):
            col = g * 7 + t  # column index in U, V, W
            alpha, beta, gamma = strassen_terms[t]
            
            # Fill U: U[i*m + j, col] = alpha[(i,j)]
            for (i, j), coeff in alpha.items():
                U[i * m + j][col] = coeff
            
            # Fill V: V[j*p + k, col] = beta[(j, k_local)]
            for (j, k_local), coeff in beta.items():
                k_global = k_start + k_local
                V[j * p + k_global][col] = coeff
            
            # Fill W: W[k*n + i, col] = gamma[(i, k_local)]
            for (i, k_local), coeff in gamma.items():
                k_global = k_start + k_local
                W[k_global * n + i][col] = coeff
    
    return {"U": U, "V": V, "W": W}
