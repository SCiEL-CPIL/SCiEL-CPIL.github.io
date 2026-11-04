"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    terms = []  # List of (u_index, v_index, w_index, coeff)
    
    def add_term(u, v, w, c=1):
        if c != 0:
            terms.append((u, v, w, c))
    
    # --- Part 1: A1 (3x2) * B1 (2x7) ---
    # Schoolbook for 3x2x7: 3*2*7 = 42 terms
    # U rows: i*5 + j (i in 0..2, j in 0..1)
    # V rows: j*7 + k (j in 0..1, k in 0..6)
    # W rows: k*3 + i (k in 0..6, i in 0..2)
    
    for i in range(3):
        for j in range(2):
            for k in range(7):
                u = i * 5 + j
                v = j * 7 + k
                w = k * 3 + i
                add_term(u, v, w, 1)
    
    # --- Part 2: A2 (3x3) * B2 (3x7) ---
    # We'll use Laderman's rank-23 decomposition for 3x3x3, applied to the
    # first 3 columns of B2, then schoolbook for the remaining 4 columns.
    # 
    # Actually, let's use a block approach:
    # Split B2 (3x7) into B2a (3x3) and B2b (3x4).
    # A2 * B2a: use Laderman (rank 23)
    # A2 * B2b: schoolbook (3*3*4 = 36)
    # Total for Part 2: 23 + 36 = 59
    
    # Laderman's 23-term decomposition for 3x3x3
    # Using the standard Laderman terms. The tensor is T[i,j,k] = A[i,j]*B[j,k] = C[i,k]
    # U[i*3+j, t], V[j*3+k, t], W[k*3+i, t]
    
    # Laderman terms (i, j, k, coeff) where i,j,k are local indices 0..2
    laderman_terms = [
        (0, 0, 0, 1),
        (0, 0, 1, 1),
        (0, 0, 2, 1),
        (0, 1, 0, 1),
        (0, 1, 1, 1),
        (0, 1, 2, 1),
        (0, 2, 0, 1),
        (0, 2, 1, 1),
        (0, 2, 2, 1),
        (1, 0, 0, 1),
        (1, 0, 1, 1),
        (1, 0, 2, 1),
        (1, 1, 0, 1),
        (1, 1, 1, 1),
        (1, 1, 2, 1),
        (1, 2, 0, 1),
        (1, 2, 1, 1),
        (1, 2, 2, 1),
        (2, 0, 0, 1),
        (2, 0, 1, 1),
        (2, 0, 2, 1),
        (2, 1, 0, 1),
        (2, 1, 1, 1),
    ]
    # Wait, that's 23 terms but they're just schoolbook for 3x3x3 minus some? 
    # No, schoolbook for 3x3x3 has 27 terms. Let me recount.
    # Actually, the above list has 23 entries but they're all (i,j,k) with coeff 1.
    # That's NOT Laderman - that's just a subset of schoolbook.
    
    # Let me use the actual Laderman decomposition.
    # The Laderman algorithm for 3x3x3 has rank 23.
    # I'll use a verified set of terms.
    
    # Clear and redo Part 2 with proper Laderman
    terms = terms[:42]  # Keep only Part 1 terms
    
    # Laderman 3x3x3 decomposition - 23 terms
    # These are bilinear forms. Each term is a product of a linear form in A,
    # a linear form in B, contributing to a linear form in C.
    # 
    # Standard Laderman terms (from the literature):
    # We'll encode them as (u_indices, v_indices, w_indices, coeff)
    # where each *_indices is a dict mapping local index to coefficient.
    
    # For simplicity and correctness, let's just use schoolbook for 3x3x3 (rank 27)
    # and schoolbook for 3x3x4 (rank 36). Total: 27+36 = 63.
    # Combined with Part 1: 42+63 = 105. That's the same as full schoolbook.
    
    # Let me try a different partition to actually get improvement.
    # 
    # Alternative: Use the fact that we can share computations.
    # Let's just use schoolbook for the whole thing first to ensure correctness,
    # then optimize.
    
    # Actually, let's use a clean approach:
    # Split m=5 into 2+3.
    # Part 1: 3x2x7, schoolbook: 42 terms
    # Part 2: 3x3x7, schoolbook: 63 terms
    # Total: 105 terms. This is the same as full schoolbook.
    
    # To get improvement, we need to use non-schoolbook decompositions.
    # Let's use the known rank-7 Strassen for 2x2x2 where applicable.
    
    # For 3x2x7: 
    # We can view this as 3 rows of (1x2x7). Each 1x2x7 can be done in rank 7?
    # No, 1x2x7 is just 7 multiplications per row, total 3*7=21? No.
    # 1x2x7: C[k] = A[0]*B[k] for k=0..6. That's 7 products. 3 rows: 21.
    # Wait, that's not right either. A is 3x2, B is 2x7.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # For each (i,k), we need 2 products. Total 3*7*2 = 42 products. Schoolbook.
    # Can we do better? 
    # Group by i: for each i, we compute a 1x7 vector = A[i,:] (1x2) * B (2x7).
    # This is equivalent to 7 separate 1x2x1 multiplications? No.
    # It's a bilinear map R^2 x R^14 -> R^7.
    # The rank of this tensor is 2*7 = 14? No.
    # Actually, for fixed i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:].
    # This requires 2 "dot products" of length 7, which is 2*7 = 14 scalar mults?
    # No, it's 2 scalar mults per k, so 14 per row, 42 total.
    # 
    # I think for these small sizes, schoolbook might be close to optimal.
    # Let's just go with schoolbook for correctness, and the score will be 105/105 = 1.0
    # unless we find a better algorithm.
    
    # Let me just implement full schoolbook to be safe.
    terms = []  # Reset
    
    for i in range(3):
        for j in range(5):
            for k in range(7):
                u = i * 5 + j
                v = j * 7 + k
                w = k * 3 + i
                add_term(u, v, w, 1)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(3 * 5)]
    V = [[0] * rank for _ in range(5 * 7)]
    W = [[0] * rank for _ in range(7 * 3)]
    
    for t in range(rank):
        u, v, w, coeff = terms[t]
        U[u][t] = coeff
        V[v][t] = coeff
        W[w][t] = coeff
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
