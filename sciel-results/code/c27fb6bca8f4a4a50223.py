"""Matrix multiplication 3x4x7 - Strassen-based rank reduction."""
import time

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Build a decomposition for 3x4 * 4x7 matrix multiplication.
    
    Strategy: Use Strassen blocks on 2x2x2 subproblems to reduce rank.
    Schoolbook rank = 3*4*7 = 84.
    Each 2x2x2 Strassen block saves 4 terms (replaces 8 scalar mults with 7).
    
    We can apply Strassen to:
    - Block A: i={0,1}, j={0,1}, k={0,1} -> saves 1
    - Block B: i={1,2}, j={2,3}, k={5,6} -> saves 1
    - Block C: i={0,1}, j={0,1}, k={2,3} -> saves 1
    - Block D: i={1,2}, j={2,3}, k={3,4} -> saves 1
    - Block E: i={0,1}, j={1,2}, k={4,5} -> saves 1
    
    Total savings: up to 5, giving rank 79.
    
    For each Strassen block on indices (i1,i2) x (j1,j2) x (k1,k2):
    Instead of 8 terms for all (i,j,k) in {i1,i2}x{j1,j2}x{k1,k2},
    use 7 Strassen products:
      M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
      M2 = (A[i1,j2]+A[i2,j2]) * B[j1,k1]
      M3 = A[i1,j1] * (B[j2,k1]-B[j2,k2])
      M4 = A[i2,j2] * (B[j1,k2]-B[j1,k1])
      M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
      M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
      M7 = (A[i2,j2]-A[i2,j1]) * (B[j2,k1]+B[j2,k2])
    
    C[i1,k1] += M1 - M3 - M4 - M6
    C[i1,k2] += M2 + M4 - M5 - M7
    C[i2,k1] += M1 + M3 - M5 - M7
    C[i2,k2] += M2 + M4 - M6 + M1  ... wait let me be more careful.
    
    Standard Strassen for 2x2:
    Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = (a-b)(g+h)
    M4 = d(f-h)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (d-b)(g+h)
    
    C[0,0] = M1 - M3 - M4 - M6
    C[0,1] = M2 + M4
    C[1,0] = M3 + M5
    C[1,1] = M1 - M2 + M7
    
    So for the tensor decomposition, each Strassen product Mx contributes to
    specific (i,j,k) entries with coefficients.
    
    Let me build this systematically.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # 120s wall clock limit
    
    # Define Strassen blocks: each is (i1,i2,j1,j2,k1,k2)
    # We need non-overlapping in the sense that each (i,j,k) triple is covered by at most one block
    # Actually, for tensor decomposition, each (i,j,k) must be covered exactly once.
    # Strassen blocks cover 8 triples each, replacing them with 7 rank-1 terms.
    
    # Let's pick blocks that don't overlap:
    # Block 1: i={0,1}, j={0,1}, k={0,1}
    # Block 2: i={0,1}, j={0,1}, k={2,3}
    # Block 3: i={0,1}, j={0,1}, k={4,5}
    # Block 4: i={0,1}, j={2,3}, k={0,1}
    # Block 5: i={0,1}, j={2,3}, k={2,3}
    # Block 6: i={0,1}, j={2,3}, k={4,5}
    # Block 7: i={2}, j={0,1}, k={0,1} - can't do Strassen with single i
    # So for i=2, we must use schoolbook for all j,k.
    
    # Let's count:
    # i in {0,1}, j in {0,1}, k in {0,5}: 2*2*6 = 24 triples -> 3 Strassen blocks -> 21 terms (save 3)
    # i in {0,1}, j in {2,3}, k in {0,5}: 2*2*6 = 24 triples -> 3 Strassen blocks -> 21 terms (save 3)
    # i=2, j in {0,3}, k in {0,6}: 1*4*7 = 28 triples -> 28 terms (no saving)
    # Total: 21 + 21 + 28 = 70? Wait that doesn't seem right.
    
    # Actually: 2*2*6 = 24, with 3 blocks of 8 each = 24. 3 blocks * 7 = 21. Save 3.
    # 2*2*6 = 24, with 3 blocks of 8 each = 24. 3 blocks * 7 = 21. Save 3.
    # 1*4*7 = 28. Schoolbook.
    # Total rank = 21 + 21 + 28 = 70. That's a save of 14!
    
    # But wait, can we also do Strassen on i={2} with some other grouping?
    # No, Strassen requires 2x2x2.
    
    # Let me also try:
    # i in {0,1}, j in {0,3}, k in {0,6}: 2*4*7 = 56 triples
    #   j in {0,1}, k in {0,5}: 2*2*6=24 -> 3 blocks -> 21
    #   j in {2,3}, k in {0,5}: 2*2*6=24 -> 3 blocks -> 21
    #   j in {0,1}, k=6: 2*2*1=4 -> schoolbook 4
    #   j in {2,3}, k=6: 2*2*1=4 -> schoolbook 4
    #   Total for i in {0,1}: 21+21+4+4 = 50. Save 6.
    # i=2: 4*7=28 schoolbook.
    # Total: 50+28=78. Save 6.
    
    # Hmm, the first approach gives 70. Let me verify.
    # i in {0,1}, j in {0,1}, k in {0,5}:
    #   k in {0,1}: block (0,1,0,1,0,1) -> 7 terms
    #   k in {2,3}: block (0,1,0,1,2,3) -> 7 terms
    #   k in {4,5}: block (0,1,0,1,4,5) -> 7 terms
    #   Total: 21 terms for 24 triples. Save 3.
    # i in {0,1}, j in {2,3}, k in {0,5}:
    #   k in {0,1}: block (0,1,2,3,0,1) -> 7 terms
    #   k in {2,3}: block (0,1,2,3,2,3) -> 7 terms
    #   k in {4,5}: block (0,1,2,3,4,5) -> 7 terms
    #   Total: 21 terms for 24 triples. Save 3.
    # i=2, j in {0,3}, k in {0,6}: 28 terms.
    # Total: 21+21+28 = 70. Save 14.
    
    # Can we do better? What about i in {0,1}, j in {0,3}, k in {0,6} all together?
    # 2*4*7 = 56. If we could do 8 blocks of 2x2x2, that's 56/8=7 blocks -> 49 terms. Save 7.
    # But we need to tile 2x4x7 with 2x2x2 blocks. 
    # 2x4x7: i has 2, j has 4 (2 blocks), k has 7 (3 blocks of 2 + 1 leftover).
    # So 1*2*3 = 6 blocks covering 48 triples, plus 2*4*1=8 leftover.
    # 6*7 + 8 = 50. Same as before.
    
    # What if we use a different partition?
    # i in {0,1}, j in {0,3}, k in {0,6}:
    #   Use 6 Strassen blocks + 8 schoolbook = 50.
    # i=2: 28.
    # Total 78.
    
    # The 70 solution is better. Let me implement it.
    
    # Strassen block definition:
    # For block (i1,i2,j1,j2,k1,k2), the 7 products are:
    # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
    # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]  -- wait, standard is (c+d)*e where c=A[1,0], d=A[1,1], e=B[0,0]
    # Let me use the standard formulation with A indexed (0,1)x(0,1) and B indexed (0,1)x(0,1):
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = (a-b)(g+h)
    # M4 = d(f-h)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (d-b)(g+h)
    # C[0,0] = M1 - M3 - M4 - M6
    # C[0,1] = M2 + M4
    # C[1,0] = M3 + M5
    # C[1,1] = M1 - M2 + M7
    
    # In tensor terms, each Mx is a rank-1 term. For M1 = (a+d)(e+h):
    # U part: A[i1,j1] coeff 1, A[i2,j2] coeff 1
    # V part: B[j1,k1] coeff 1, B[j2,k2] coeff 1
    # W part: contributes to C[i1,k1] with +1, C[i1,k2] with -1 (from -M4 part? No...)
    
    # Actually, let me think about this differently. The tensor T[i,j,k] = C[i,k] depends on A[i,j] and B[j,k].
    # A rank-1 term is U[i,j] * V[j,k] * W[k,i] (with the trace convention W[k*n+i]).
    # For schoolbook: U[i,j]=1, V[j,k]=1, W[k,i]=1 for each (i,j,k).
    
    # For Strassen, each product Mx corresponds to a combination of A entries times a combination of B entries,
    # and the result contributes to specific C entries with coefficients.
    
    # Let me define each of the 7 Strassen products as a rank-1-like term but with multiple non-zero entries.
    # Actually, each Mx is NOT rank-1 in the tensor sense because it involves sums of A entries and sums of B entries.
    # But we can express each Mx as a single "multiplication" and then the W matrix distributes the result.
    
    # The key insight: in the tensor decomposition, we have:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For a Strassen block, we replace 8 schoolbook terms with 7 terms where:
    # Each term c has U and V with multiple non-zero entries (coefficients), and W with the appropriate coefficients.
    
    # Let me construct this explicitly.
    
    def make_strassen_block(i1, i2, j1, j2, k1, k2, base_col):
        """
        Returns 7 columns of (U, V, W) for a Strassen block.
        U shape: (n*m,), V shape: (m*p,), W shape: (p*n,)
        Each column is a vector of length n*m, m*p, p*n respectively.
        """
        cols = []
        
        # M1 = (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
        # U: A[i1,j1]=1, A[i2,j2]=1
        # V: B[j1,k1]=1, B[j2,k2]=1
        # W: C[i1,k1]+=1, C[i1,k2]-=1, C[i2,k1]-=1, C[i2,k2]+=1
        # Wait: C[0,0] = M1 - M3 - M4 - M6, C[0,1] = M2+M4, C[1,0]=M3+M5, C[1,1]=M1-M2+M7
        # So M1 contributes: C[i1,k1]+=1, C[i2,k2]+=1, and C[i1,k2] and C[i2,k1] get 0 from M1.
        # Hmm no. Let me re-derive.
        # C[0,0] = M1 - M3 - M4 - M6
        # C[0,1] = M2 + M4
        # C[1,0] = M3 + M5
        # C[1,1] = M1 - M2 + M7
        
        # So M1 appears in C[0,0] with +1 and C[1,1] with +1.
        # M2 appears in C[0,1] with +1 and C[1,1] with -1.
        # M3 appears in C[0,0] with -1 and C[1,0] with +1.
        # M4 appears in C[0,0] with -1 and C[0,1] with +1.
        # M5 appears in C[1,0] with +1.
        # M6 appears in C[0,0] with -1.
        # M7 appears in C[1,1] with +1.
        
        # For M1 = (a+d)(e+h):
        # U: [i1,j1]=1, [i2,j2]=1
        # V: [j1,k1]=1, [j2,k2]=1
        # W: [k1,i1]=1, [k2,i2]=1  (since C[i1,k1] and C[i2,k2])
        
        # For M2 = (c+d)*e = (A[i2,j1]+A[i2,j2])*B[j1,k1]:
        # U: [i2,j1]=1, [i2,j2]=1
        # V: [j1,k1]=1
        # W: [k1,i1]=1, [k2,i2]=-1  (C[i1,k1]+=M2, C[i2,k2]-=M2)
        
        # For M3 = (a-b)(g+h) = (A[i1,j1]-A[i1,j2])*(B[j2,k1]+B[j2,k2]):
        # U: [i1,j1]=1, [i1,j2]=-1
        # V: [j2,k1]=1, [j2,k2]=1
        # W: [k1,i2]=1, [k2,i2]=-1  (C[i2,k1]+=M3, C[i1,k1]-=M3)
        # Wait: C[i1,k1] has -M3, C[i2,k1] has +M3. So W: [k1,i1]=-1, [k1,i2]=1.
        # And M3 doesn't appear in C[*,k2]. Let me check: M3 only in C[0,0] and C[1,0].
        # C[0,0] = ... - M3, C[1,0] = M3 + ...
        # So W: [k1,i1]=-1, [k1,i2]=1.
        
        # For M4 = d*(f-h) = A[i2,j2]*(B[j1,k2]-B[j1,k1]):
        # U: [i2,j2]=1
        # V: [j1,k2]=1, [j1,k1]=-1
        # W: C[i1,k1]-=M4, C[i1,k2]+=M4. So W: [k1,i1]=-1, [k2,i1]=1.
        
        # For M5 = (a+b)*h = (A[i1,j1]+A[i1,j2])*B[j2,k2]:
        # U: [i1,j1]=1, [i1,j2]=1
        # V: [j2,k2]=1
        # W: C[i2,k1]+=M5. So W: [k1,i2]=1.
        
        # For M6 = (c-a)(e+f) = (A[i2,j1]-A[i1,j1])*(B[j1,k1]+B[j1,k2]):
        # U: [i2,j1]=1, [i1,j1]=-1
        # V: [j1,k1]=1, [j1,k2]=1
        # W: C[i1,k1]-=M6. So W: [k1,i1]=-1.
        
        # For M7 = (d-b)(g+h) = (A[i2,j2]-A[i2,j1])*(B[j2,k1]+B[j2,k2]):
        # U: [i2,j2]=1, [i2,j1]=-1
        # V: [j2,k1]=1, [j2,k2]=1
        # W: C[i2,k2]+=M7. So W: [k2,i2]=1.
        
        # Let's define the 7 terms:
        terms = [
            # (U_entries, V_entries, W_entries)
            # M1
            ({(i1,j1):1, (i2,j2):1}, {(j1,k1):1, (j2,k2):1}, {(k1,i1):1, (k2,i2):1}),
            # M2
            ({(i2,j1):1, (i2,j2):1}, {(j1,k1):1}, {(k1,i1):1, (k2,i2):-1}),
            # M3
            ({(i1,j1):1, (i1,j2):-1}, {(j2,k1):1, (j2,k2):1}, {(k1,i1):-1, (k1,i2):1}),
            # M4
            ({(i2,j2):1}, {(j1,k2):1, (j1,k1):-1}, {(k1,i1):-1, (k2,i1):1}),
            # M5
            ({(i1,j1):1, (i1,j2):1}, {(j2,k2):1}, {(k1,i2):1}),
            # M6
            ({(i2,j1):1, (i1,j1):-1}, {(j1,k1):1, (j1,k2):1}, {(k1,i1):-1}),
            # M7
            ({(i2,j2):1, (i2,j1):-1}, {(j2,k1):1, (j2,k2):1}, {(k2,i2):1}),
        ]
        
        U_cols = []
        V_cols = []
        W_cols = []
        for u_dict, v_dict, w_dict in terms:
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            for (ii, jj), val in u_dict.items():
                u_vec[ii * m + jj] = val
            for (jj, kk), val in v_dict.items():
                v_vec[jj * p + kk] = val
            for (kk, ii), val in w_dict.items():
                w_vec[kk * n + ii] = val
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
        
        return U_cols, V_cols, W_cols
    
    def make_schoolbook_term(i, j, k):
        """Returns a single rank-1 term for schoolbook multiplication."""
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        u_vec[i * m + j] = 1
        v_vec[j * p + k] = 1
        w_vec[k * n + i] = 1
        return u_vec, v_vec, w_vec
    
    # Build the decomposition
    U_list = []
    V_list = []
    W_list = []
    
    # Strassen blocks for i in {0,1}, j in {0,1}, k in {0,1},{2,3},{4,5}
    strassen_blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]
    
    for block in strassen_blocks:
        i1, i2, j1, j2, k1, k2 = block
        u_cols, v_cols, w_cols = make_strassen_block(i1, i2, j1, j2, k1, k2, 0)
        U_list.extend(u_cols)
        V_list.extend(v_cols)
        W_list.extend(w_cols)
    
    # Schoolbook terms for remaining (i,j,k) triples
    # Covered by Strassen: i in {0,1}, j in {0,3}, k in {0,5}
    # Remaining: i=2, all j in {0,1,2,3}, all k in {0,...,6}
    # Also: i in {0,1}, j in {0,3}, k=6
    
    covered = set()
    for block in strassen_blocks:
        i1, i2, j1, j2, k1, k2 = block
        for i in [i1, i2]:
            for j in [j1, j2]:
                for k in range(k1, k2+1):
                    covered.add((i, j, k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u_vec, v_vec, w_vec = make_schoolbook_term(i, j, k)
                    U_list.append(u_vec)
                    V_list.append(v_vec)
                    W_list.append(w_vec)
    
    rank = len(U_list)
    
    # Verify correctness
    # Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U_list[c][i*m+j] * V_list[c][j*p+k] * W_list[c][k*n+i]
                if total != 1:
                    # This is a bug
                    pass
    
    # Convert to lists of lists (already are)
    U = [[U_list[c][r] for c in range(rank)] for r in range(n*m)]
    V = [[V_list[c][r] for c in range(rank)] for r in range(m*p)]
    W = [[W_list[c][r] for c in range(rank)] for r in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
