"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use a rank-24 decomposition for M_{2,2,8}
    # Strategy: 2 groups of 4 columns, each with rank-12 decomposition
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For M_{2,2,4} with rank 12, we use a combination of:
    # - 4 trivial terms (rank 4) for direct products
    # - 8 Strassen-like shared terms
    # Actually, let's use a verified rank-12 scheme for 2x2 x 2x4
    
    # A indices: 00->0, 01->1, 10->2, 11->3
    # For a 2x4 block starting at column offset k0:
    # B indices: j=0,k0 -> j*p+k0, j=0,k0+1 -> j*p+k0+1, etc.
    # C indices: i=0,k0 -> k0*n+0, i=1,k0 -> k0*n+1, etc.
    
    # Rank-12 decomposition for M_{2,2,4}:
    # We use the following 12 terms:
    # Terms 0-3: Trivial (each handles one output column pair)
    # Terms 4-11: Shared Strassen-like terms
    
    # Actually, let me use a known rank-12 scheme:
    # Based on the idea that M_{2,2,4} can be decomposed with rank 12
    
    # Let me define the 12 rank-1 components for a 2x2 x 2x4 multiplication
    # where B columns are at offsets 0,1,2,3
    
    # Component definitions for local 2x4 block:
    # U local indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V local indices: 0=B0,0 1=B0,1 2=B0,2 3=B0,3 4=B1,0 5=B1,1 6=B1,2 7=B1,3
    # W local indices: 0=C0,0 1=C1,0 2=C0,1 3=C1,1 4=C0,2 5=C1,2 6=C0,3 7=C1,3
    
    # I'll use a systematic construction. For each of the 12 terms:
    
    # Term 0: A00 * B0,0 -> C0,0
    # Term 1: A01 * B1,0 -> C0,0
    # Term 2: A10 * B0,0 -> C1,0
    # Term 3: A11 * B1,0 -> C1,0
    # These 4 terms handle column 0 trivially.
    
    # For columns 1,2,3, we use shared Strassen factors.
    
    # Let me use a different approach: 3 Strassen blocks won't work since 4 isn't divisible by 2 in the right way.
    
    # Instead, let me just use 4 Strassen blocks but merge some factors.
    # Actually, the simplest improvement: use rank 24 by using a better scheme.
    
    # Let me try: use 2x2 matrix-vector mult for each of 8 columns = 8*4 = 32 (worse)
    # Or use 4x2 matrix-matrix with Strassen = 4*7 = 28 (current)
    
    # Let me try a rank-24 scheme using the following insight:
    # C = A*B where A is 2x2, B is 2x8
    # Write A = [[a, b], [c, d]], B = [B0, B1, B2, B3] where Bi are 2x2
    # C = [[a*B0[0]+b*B1[0], ...], [c*B0[0]+d*B1[0], ...]]
    
    # For each column k: C[:,k] = [a*B0[k]+b*B1[k], c*B0[k]+d*B1[k]]
    # = [a, b; c, d] * [B0[k]; B1[k]]
    
    # This is 8 independent 2x2 matrix-vector multiplies, each rank 4, total 32.
    # But we can do better by sharing.
    
    # Let me just implement rank 24 with a verified construction.
    # I'll use the fact that we can group columns.
    
    # For now, let me use a safe rank-24 construction:
    # 2 groups of 4 columns, each group uses a rank-12 decomposition
    
    # Rank-12 for M_{2,2,4}:
    # Based on literature, M_{2,2,4} has rank at most 12.
    
    # Let me construct it explicitly.
    # For a 2x2 x 2x4 multiplication:
    
    # I'll use the following 12 terms (u, v, w) where:
    # u is 4-vector (A coeffs), v is 8-vector (B coeffs), w is 8-vector (C coeffs)
    
    # The tensor T has T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2,3}
    # with the constraint that the j in A_{i,j} matches the j in B_{j,k}
    
    # Let me define the 12 terms:
    
    terms = []
    
    # Term 0: covers (i=0,j=0,k=0) and (i=1,j=0,k=0)
    # u = [1,0,0,0] (A00)
    # v = [1,0,0,0, 0,0,0,0] (B0,0)
    # w = [1,0, 0,0, 0,0, 0,0] (C0,0)
    # But this only covers C0,0 from A00*B0,0. We also need C1,0 from A10*B0,0.
    # So this term is: u=[1,0,0,0], v=[1,0,0,0,0,0,0,0], w=[1,0,0,0,0,0,0,0]
    # This contributes to T[0,0,0] = u[0]*v[0]*w[0] = 1. Good.
    
    # Actually, each rank-1 term u⊗v⊗w contributes to T[a,b,c] = u[a]*v[b]*w[c]
    # We need the sum over all terms to equal 1 for valid (a,b,c) and 0 otherwise.
    
    # Let me use a known rank-12 decomposition for M_{2,2,4}.
    # I'll look it up or derive it.
    
    # Derivation: We need to express the 8 equations:
    # C0,0 = A00*B0,0 + A01*B1,0
    # C0,1 = A00*B0,1 + A01*B1,1
    # C0,2 = A00*B0,2 + A01*B1,2
    # C0,3 = A00*B0,3 + A01*B1,3
    # C1,0 = A10*B0,0 + A11*B1,0
    # C1,1 = A10*B0,1 + A11*B1,1
    # C1,2 = A10*B0,2 + A11*B1,2
    # C1,3 = A10*B0,3 + A11*B1,3
    
    # as a sum of 12 rank-1 bilinear forms.
    
    # A rank-1 term: (u_A * A) * (u_B * B) where u_A is a linear functional on A,
    # u_B is a linear functional on B, and the product is placed in specific C entries.
    
    # More precisely, term r contributes:
    # ΔC[i,k] = (Σ_j U[i,j,r] * A[i,j]) * (Σ_j V[j,k,r] * B[j,k]) * ... 
    # No, that's not right.
    
    # The correct form: term r is defined by vectors u_r (4-dim), v_r (8-dim), w_r (8-dim)
    # and contributes to the tensor as u_r ⊗ v_r ⊗ w_r
    # So T[a,b,c] += u_r[a] * v_r[b] * w_r[c]
    
    # And the computation is: C[i,k] = Σ_{j} A[i,j] * B[j,k] = Σ_{a,b,c} T[a,b,c] * A[a] * B[b] * W[c]
    # Wait, that's not right either.
    
    # Let me just use the verification from the task: the decomposition must satisfy
    # T[a,b,c] = Σ_r U[a,r] * V[b,r] * W[c,r]
    # where T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k, else 0.
    
    # So for M_{2,2,4}:
    # T has 1s at positions:
    # (0,0,0), (0,4,1)  -- for i=0,j=0,k=0: a=0*2+0=0, b=0*4+0=0, c=0*2+0=0
    # Wait, I need to get the indexing right.
    
    # a = i*m + j = i*2 + j
    # b = j*p + k = j*4 + k  (for the local 2x4 block, p=4)
    # c = k*n + i = k*2 + i
    
    # For i=0,j=0,k=0: a=0, b=0, c=0
    # For i=0,j=0,k=1: a=0, b=1, c=2
    # For i=0,j=0,k=2: a=0, b=2, c=4
    # For i=0,j=0,k=3: a=0, b=3, c=6
    # For i=0,j=1,k=0: a=1, b=4, c=1
    # For i=0,j=1,k=1: a=1, b=5, c=3
    # For i=0,j=1,k=2: a=1, b=6, c=5
    # For i=0,j=1,k=3: a=1, b=7, c=7
    # For i=1,j=0,k=0: a=2, b=0, c=1
    # For i=1,j=0,k=1: a=2, b=1, c=3
    # For i=1,j=0,k=2: a=2, b=2, c=5
    # For i=1,j=0,k=3: a=2, b=3, c=7
    # For i=1,j=1,k=0: a=3, b=4, c=0
    # For i=1,j=1,k=1: a=3, b=5, c=2
    # For i=1,j=1,k=2: a=3, b=6, c=4
    # For i=1,j=1,k=3: a=3, b=7, c=6
    
    # So T has 16 ones at these 16 positions.
    
    # I need to find 12 vectors u_r (4-dim), v_r (8-dim), w_r (8-dim) such that
    # Σ_r u_r[a] * v_r[b] * w_r[c] = 1 for the 16 positions above, 0 elsewhere.
    
    # This is a system of equations. Let me try to construct it.
    
    # Given the complexity, let me just use the current rank-28 approach but try to verify it works,
    # and if I can find a rank-24 scheme, use that.
    
    # For now, let me stick with rank 28 but ensure correctness.
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A indices)
    U_coeffs = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {2: 1, 0: -1},      # M6
        {1: 1, 3: -1}       # M7
    ]
    
    # Strassen coefficients for V (B indices, local to block)
    V_coeffs = [
        {0: 1, 3: 1},       # M1
        {0: 1},             # M2
        {1: 1, 3: -1},      # M3
        {2: 1, 0: -1},      # M4
        {3: 1},             # M5
        {0: 1, 1: 1},       # M6
        {2: 1, 3: 1}        # M7
    ]
    
    # Strassen coefficients for W (C indices, local to block)
    W_coeffs = [
        {0: 1, 3: 1},       # M1
        {1: 1, 3: -1},      # M2
        {2: 1, 3: 1},       # M3
        {0: 1, 1: 1},       # M4
        {0: -1, 2: 1},      # M5
        {3: 1},             # M6
        {0: 1}              # M7
    ]

    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            
            for u_idx, val in U_coeffs[s].items():
                U[u_idx][r] = val
                
            for v_local_idx, val in V_coeffs[s].items():
                j = v_local_idx // 2
                k_local = v_local_idx % 2
                k = 2 * q + k_local
                v_idx = j * p + k
                V[v_idx][r] = val
                
            for w_local_idx, val in W_coeffs[s].items():
                i = w_local_idx % 2
                k_local = w_local_idx // 2
                k = 2 * q + k_local
                w_idx = k * n + i
                W[w_idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
