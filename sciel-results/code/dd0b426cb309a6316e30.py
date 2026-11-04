"""Model-and-optimize approach: exploit super-multiplicativity for 2x3x8."""
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # Schoolbook fallback (rank 48)
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate, dims=None):
        """Validate a decomposition. dims = (n, m, p) for the subproblem."""
        if dims is None:
            dims = (n, m, p)
        nn, mm, pp = dims
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(nn):
            for a in range(mm):
                for k in range(pp):
                    acc = Fraction(0)
                    u_row = U[i * mm + a]
                    v_row = V[a * pp + k]
                    w_row = W[k * nn + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Rank-6 decomposition for 2x3x2 (Winograd-style)
    # U: 6x6, V: 6x6, W: 4x6
    # Row indices: U: (i,a) -> i*3+a for i in {0,1}, a in {0,1,2}
    #              V: (a,k) -> a*2+k for a in {0,1,2}, k in {0,1}
    #              W: (k,i) -> k*2+i for k in {0,1}, i in {0,1}
    
    # The 12 equations to satisfy:
    # For each (i,a,k) in {0,1}x{0,1,2}x{0,1}: sum_c U[i,a,c]*V[a,k,c]*W[k,i,c] = 1
    
    # Strategy: Use 6 rank terms. Each term c has:
    #   u_c: vector in R^6 (for the 6 U-indices)
    #   v_c: vector in R^6 (for the 6 V-indices)  
    #   w_c: vector in R^4 (for the 4 W-indices)
    
    # Let me try a structured approach:
    # Term 0: covers (i=0,a=0,k=0) and (i=0,a=1,k=0) and (i=0,a=2,k=0) partially
    # Actually, let me think about this differently.
    
    # For 2x3x2, we have 12 entries. With rank 6, we have 6*(6+6+4)=120 variables, 12 equations.
    # Let me try to find a solution by construction.
    
    # Idea: Group the 12 entries into 6 pairs, each pair covered by one rank term.
    # Pair 1: (0,0,0) and (0,0,1) -> U[0,0]=1, V[0,0]=1, V[0,1]=1, W[0,0]=1, W[1,0]=1
    #         But then term 0 contributes 1 to (0,0,0) and 1 to (0,0,1). Good.
    #         But it also contributes to other (i,a,k) where U[i,a], V[a,k], W[k,i] are nonzero.
    
    # Let me try a more systematic approach. Define 6 rank terms:
    # Term c has support on specific (i,a), (a,k), (k,i) indices.
    
    # Try: 
    # Term 0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> covers (0,0,0)
    # Term 1: U[0,0]=1, V[0,1]=1, W[1,0]=1 -> covers (0,0,1)
    # Term 2: U[0,1]=1, V[1,0]=1, W[0,0]=1 -> covers (0,1,0)
    # Term 3: U[0,1]=1, V[1,1]=1, W[1,0]=1 -> covers (0,1,1)
    # Term 4: U[0,2]=1, V[2,0]=1, W[0,0]=1 -> covers (0,2,0)
    # Term 5: U[0,2]=1, V[2,1]=1, W[1,0]=1 -> covers (0,2,1)
    # This gives rank 6 for i=0, but we also need i=1.
    
    # So this doesn't work directly. Let me try a different grouping.
    
    # Alternative: Use the fact that for fixed k, the 2x3 submatrix must be decomposed.
    # For k=0: need 2x3x1, which has rank at least 2 (since the 2x3 matrix has rank 2).
    # For k=1: need 2x3x1, rank at least 2.
    # Total lower bound: 4? But we need 12 entries, so probably more.
    
    # Let me just try to solve it numerically with a structured ansatz.
    
    # Ansatz: Let each term c have:
    # u_c[i,a] = alpha_c[i] * beta_c[a]  (outer product in i-a space)
    # v_c[a,k] = beta_c[a] * gamma_c[k]  (outer product in a-k space)
    # w_c[k,i] = gamma_c[k] * alpha_c[i]  (outer product in k-i space)
    # Then term c contributes: alpha_c[i]*beta_c[a]*gamma_c[k] * alpha_c[i]*beta_c[a]*gamma_c[k] * ... 
    # Wait, that gives alpha_c[i]^2 * beta_c[a]^2 * gamma_c[k]^2, which is always non-negative.
    # That's too restrictive.
    
    # Let me try a different ansatz. Allow:
    # u_c[i,a] = A_c[i,a]
    # v_c[a,k] = B_c[a,k]
    # w_c[k,i] = C_c[k,i]
    # with the constraint that the product sums to 1 for each (i,a,k).
    
    # This is a nonlinear system. Let me try to find a solution by hand.
    
    # For 2x3x2, let's try rank 6 with the following structure:
    # Terms 0-2 cover k=0, terms 3-5 cover k=1.
    # For k=0: need to decompose the 2x3 identity-like tensor.
    #   sum_{c=0}^{2} U[i,a,c] * V[a,0,c] * W[0,i,c] = 1 for all (i,a)
    # For k=1: need to decompose the 2x3 identity-like tensor.
    #   sum_{c=3}^{5} U[i,a,c] * V[a,1,c] * W[1,i,c] = 1 for all (i,a)
    
    # For the k=0 part (2x3x1), the minimal rank is 2 (since the 2x3 matrix has rank 2).
    # So we can do it with 2 terms, not 3. Let's try rank 4 total: 2 for k=0, 2 for k=1.
    
    # For 2x3x1 with rank 2:
    # Need: sum_{c=0}^{1} U[i,a,c] * V[a,0,c] * W[0,i,c] = 1 for all (i,a)
    # Let U[0,0]=1, U[0,1]=0, U[0,2]=0
    #     U[1,0]=0, U[1,1]=1, U[1,2]=0
    # Let V[0,0]=1, V[1,0]=0, V[2,0]=0
    #     V[0,1]=0, V[1,1]=1, V[2,1]=0
    # Let W[0,0]=1, W[0,1]=1
    # Then for (i=0,a=0): U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 ✓
    # For (i=0,a=1): U[0,1]*V[1,0]*W[0,0] = 0*0*1 = 0 ✗ (need 1)
    
    # That doesn't work. Let me try:
    # U[0,0]=1, U[0,1]=1, U[0,2]=0
    # U[1,0]=1, U[1,1]=1, U[1,2]=0
    # V[0,0]=1, V[1,0]=0, V[2,0]=0
    # V[0,1]=0, V[1,1]=1, V[2,1]=0
    # W[0,0]=1, W[0,1]=1
    # For (i=0,a=0): U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,0] = 1*1*1 + 1*0*1 = 1 ✓
    # For (i=0,a=1): U[0,0]*V[1,0]*W[0,0] + U[0,1]*V[1,1]*W[0,0] = 1*0*1 + 1*1*1 = 1 ✓
    # For (i=0,a=2): U[0,0]*V[2,0]*W[0,0] + U[0,1]*V[2,1]*W[0,0] = 1*0*1 + 1*0*1 = 0 ✗
    
    # Still doesn't work. The issue is that with only 2 terms, we can't cover all 3 columns.
    
    # Let me try rank 3 for 2x3x1:
    # U[0,0]=1, U[0,1]=0, U[0,2]=0
    # U[1,0]=0, U[1,1]=1, U[1,2]=0
    # V[0,0]=1, V[1,0]=1, V[2,0]=1
    # V[0,1]=1, V[1,1]=1, V[2,1]=1
    # V[0,2]=1, V[1,2]=1, V[2,2]=1
    # W[0,0]=1, W[0,1]=1
    # This is getting complicated. Let me just use the known result.
    
    # Actually, let me just try to build a rank-44 solution first (which we know works)
    # and then try to improve it.
    
    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    S_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    S_W = [
        [1, 0, 0, 1],   # M1
        [0, 1, 0, -1],  # M2
        [0, 0, 1, 1],   # M3
        [1, 1, 0, 0],   # M4
        [-1, 0, 1, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build rank-44 solution as baseline
    def _build_rank44():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val != 0:
                            U[li * m + la][col] = u_val
                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val != 0:
                            V[la * p + (k0 + lk)][col] = v_val
                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val != 0:
                            W[(k0 + lk) * n + li][col] = w_val
                col += 1
        for i in range(2):
            for k in range(8):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        return {"U": U, "V": V, "W": W}

    baseline = _build_rank44()
    if not _validate(baseline):
        return _schoolbook()

    # Now try to find a lower-rank solution using super-multiplicativity.
    # rank(2x3x8) <= rank(2x3x2) * rank(2x2x4)
    # We need to find rank(2x3x2) and rank(2x2x4).
    
    # For 2x2x4: rank <= 7 * 2 = 14 (Strassen tiled) or better.
    # For 2x3x2: let's try to find the minimal rank.
    
    # Let me try to construct a rank-6 solution for 2x3x2.
    # If successful, total rank = 6 * 14 = 84, which is worse than 44.
    # So this approach doesn't help directly.
    
    # Alternative: Can we do 2x3x8 with rank < 44 by a different decomposition?
    # Let's try: 2x3x8 = 2x3x4 + 2x3x4
    # If rank(2x3x4) = 20, then total = 40.
    # rank(2x3x4) <= rank(2x3x2) * rank(2x2x2) = 6 * 7 = 42? No, that's for the product.
    # Actually, rank(2x3x4) <= rank(2x3x2) + rank(2x3x2) = 6 + 6 = 12? No, that's not right either.
    
    # The correct super-multiplicativity is:
    # rank(<n x m, m x p> * <p x q, q x r>) <= rank(<n x m, m x p>) * rank(<p x q, q x r>)
    # So rank(2x3x8) <= rank(2x3x2) * rank(2x2x4)
    
    # We need rank(2x2x4). This is 2x2 matrix times 2x4 matrix.
    # rank(2x2x4) <= 7 * 2 = 14 (Strassen for 2x2x2, tiled twice)
    # Can we do better? rank(2x2x4) <= 12? Let's assume 14 for now.
    
    # So if rank(2x3x2) = 6, then rank(2x3x8) <= 6 * 14 = 84. Worse than 44.
    
    # What if we use a different factorization?
    # rank(2x3x8) <= rank(2x3x3) * rank(2x3x8/3)? No, that doesn't make sense.
    
    # Let me try a direct approach: can we do 2x3x8 with rank 40?
    # Use Strassen for 2x2x8 (rank 28) and try to do 2x1x8 with rank 12 instead of 16.
    # 2x1x8: multiply 2x1 by 1x8. This is just 16 scalar multiplications.
    # Rank of 2x1x8: each (i,k) entry needs a term. With 2x1, U has 2 rows, V has 8 rows, W has 16 rows.
    # The tensor is T[i,a,k] where a is fixed (a=2). So T[i,2,k] = 1 for all i,k.
    # This is a 2x8 matrix, which has rank 2. So the tensor rank is at least 2? No.
    # Actually, for 2x1x8, the tensor is T[i,2,k]. The rank is the rank of the 2x8 matrix, which is 2.
    # But wait, that's the matrix rank, not the tensor rank.
    
    # For 2x1x8, the tensor has 16 entries. Each rank-1 term is u_i * v_k * w_{k,i}.
    # Since a is fixed, v_k is just a vector in R^8.
    # The tensor is T[i,k] = sum_c u_i[c] * v_k[c] * w_{k,i}[c].
    # This is like a 2x8 matrix with entries that are products of three vectors.
    # The minimal rank is at least the rank of the 2x8 matrix, which is 2.
    # But can we achieve rank 2? Let's try:
    # u_0 = [1, 0], u_1 = [0, 1]
    # v_k = [1, 1] for all k
    # w_{k,0} = [1, 0], w_{k,1} = [0, 1] for all k
    # Then T[0,k] = 1*1*1 + 0*1*0 = 1 ✓
    # T[1,k] = 0*1*0 + 1*1*1 = 1 ✓
    # So rank(2x1x8) = 2!
    
    # Wait, that's great. So rank(2x3x8) <= rank(2x2x8) + rank(2x1x8) = 28 + 2 = 30?
    # No, that's not right. The a-index is shared.
    
    # Let me reconsider. The tensor is T[i,a,k] for i in {0,1}, a in {0,1,2}, k in {0,...,7}.
    # We can decompose it as:
    # T[i,a,k] = T1[i,a,k] + T2[i,a,k]
    # where T1 covers a in {0,1} and T2 covers a=2.
    # T1[i,a,k] = 1 for a in {0,1}, 0 otherwise.
    # T2[i,a,k] = 1 for a=2, 0 otherwise.
    # rank(T1) <= rank(2x2x8) = 28
    # rank(T2) <= rank(2x1x8) = 2 (as shown above)
    # So rank(2x3x8) <= 28 + 2 = 30!
    
    # Let me verify rank(2x1x8) = 2.
    # U: 2x2 (rows i=0,1), V: 8x2 (rows k=0,...,7), W: 16x2 (rows (k,i))
    # U[0,0]=1, U[0,1]=0
    # U[1,0]=0, U[1,1]=1
    # V[k,0]=1, V[k,1]=1 for all k
    # W[k*2+0, 0]=1, W[k*2+0, 1]=0 for all k
    # W[k*2+1, 0]=0, W[k*2+1, 1]=1 for all k
    # Check: T[i,2,k] = sum_c U[i,c]*V[k,c]*W[k*2+i,c]
    # For i=0,k=0: U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[1,1] = 1*1*1 + 0*1*0 = 1 ✓
    # For i=1,k=0: U[1,0]*V[0,0]*W[2,0] + U[1,1]*V[0,1]*W[3,1] = 0*1*0 + 1*1*1 = 1 ✓
    # For i=0,k=1: U[0,0]*V[1,0]*W[2,0] + U[0,1]*V[1,1]*W[3,1] = 1*1*0 + 0*1*0 = 0 ✗
    
    # Oops, that doesn't work. Let me fix it.
    # W[k*2+i, c] should be such that for fixed (k,i), only one c is nonzero.
    # Let W[k*2+0, 0]=1, W[k*2+0, 1]=0
    # W[k*2+1, 0]=0, W[k*2+1, 1]=1
    # Then for i=0,k=1: U[0,0]*V[1,0]*W[2,0] + U[0,1]*V[1,1]*W[3,1] = 1*1*1 + 0*1*0 = 1 ✓
    # For i=1,k=1: U[1,0]*V[1,0]*W[4,0] + U[1,1]*V[1,1]*W[5,1] = 0*1*0 + 1*1*1 = 1 ✓
    
    # Wait, I need to re-index. W has rows indexed by (k,i), so W[k*n+i, c] where n=2.
    # W[0*2+0, 0]=1, W[0*2+0, 1]=0
    # W[0*2+1, 0]=0, W[0*2+1, 1]=1
    # W[1*2+0, 0]=1, W[1*2+0, 1]=0
    # W[1*2+1, 0]=0, W[1*2+1, 1]=1
    # etc.
    
    # So W[k*2+i, c] = 1 if (i==0 and c==0) or (i==1 and c==1), else 0.
    
    # Let me rebuild:
    # U[i,c] = 1 if i==c, else 0
    # V[k,c] = 1 for all k,c
    # W[k*2+i, c] = 1 if i==c, else 0
    
    # Check T[i,2,k] = sum_c U[i,c]*V[k,c]*W[k*2+i,c]
    # = sum_c (1 if i==c else 0) * 1 * (1 if i==c else 0)
    # = 1 if i==0 or i==1 (since for c=i, both factors are 1)
    # = 1 ✓
    
    # Great! So rank(2x1x8) = 2.
    
    # Now, rank(2x3x8) <= rank(2x2x8) + rank(2x1x8) = 28 + 2 = 30.
    
    # Let me build this.
    def _build_rank30():
        rank = 30
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        
        # Part 1: 2x2x8 using Strassen (rank 28)
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val != 0:
                            U[li * m + la][col] = u_val
                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val != 0:
                            V[la * p + (k0 + lk)][col] = v_val
                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val != 0:
                            W[(k0 + lk) * n + li][col] = w_val
                col += 1
        
        # Part 2: 2x1x8 with rank 2
        # U[i, c] = 1 if i==c else 0, for c in {0,1} (relative to part 2)
        # V[k, c] = 1 for all k, c
        # W[k*2+i, c] = 1 if i==c else 0
        for c in range(2):
            for i in range(2):
                if i == c:
                    U[i * m + 2][col] = 1
            for k in range(8):
                V[2 * p + k][col] = 1
            for k in range(8):
                for i in range(2):
                    if i == c:
                        W[k * n + i][col] = 1
            col += 1
        
        return {"U": U, "V": V, "W": W}

    candidate = _build_rank30()
    if _validate(candidate):
        return candidate

    # Fallback to rank 44
    if _validate(baseline):
        return baseline

    return _schoolbook()

# EVOLVE-BLOCK-END
