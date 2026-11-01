"""Matrix multiplication 2x2 by 2x10: rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x10 matrix.
    
    We use Strassen's algorithm for each 2x2 block of the 2x10 matrix.
    The 2x10 matrix can be viewed as 5 blocks of 2x2 (columns 0-1, 2-3, ..., 8-9).
    
    For each 2x2 block, Strassen reduces the rank from 4 to 7.
    Total rank = 5 * 7 = 35 (vs schoolbook 40).
    
    Strassen's 7 products for multiplying A(2x2) * B(2x2):
    Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    Result: [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    For the full 2x10 multiplication:
    A is 2x2: [[a,b],[c,d]]
    B is 2x10: [[b0,b1,b2,b3,b4,b5,b6,b7,b8,b9],
                [b10,b11,b12,b13,b14,b15,b16,b17,b18,b19]]
    
    Block j (j=0..4) uses B columns 2j, 2j+1.
    For block j, let:
      e = B[0, 2j], f = B[0, 2j+1]
      g = B[1, 2j], h = B[1, 2j+1]
    
    The 7 Strassen products for block j:
      M1_j = (a+d)(e+h)
      M2_j = (c+d)e
      M3_j = a(f-h)
      M4_j = d(g-e)
      M5_j = (a+b)h
      M6_j = (c-a)(e+f)
      M7_j = (b-d)(g+h)
    
    Output C[0, 2j]   = M1_j + M4_j - M5_j + M7_j
    Output C[0, 2j+1] = M3_j + M5_j
    Output C[1, 2j]   = M2_j + M4_j
    Output C[1, 2j+1] = M1_j - M2_j + M3_j + M6_j
    
    Now we need to express this as a tensor decomposition.
    
    The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k (the standard
    matmul tensor). We need to find rank r such that:
      T = sum_{c=1}^{r} U_c ⊗ V_c ⊗ W_c
    
    where U has shape (n*m, r) = (4, r), V has shape (m*p, r) = (20, r),
    W has shape (p*n, r) = (20, r).
    
    The convention: W[k*n+i, c] (trace convention).
    
    For each Strassen product M_c for block j, we have:
      M_c = sum_{i,j} U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k]
    
    Wait, let me be more careful. The tensor decomposition of matmul means:
    
    C[i,k] = sum_{j} A[i,j] * B[j,k]
    
    And we want:
    C[i,k] = sum_{c} (sum_{i',j'} U[i',j',c] * A[i',j']) * (sum_{j',k'} V[j',k',c] * B[j',k'])
    
    Actually, the standard form is:
    C[i,k] = sum_{c} (sum_{j} U_{i,j,c} * A[i,j]) * (sum_{j} V_{j,k,c} * B[j,k])
    
    Hmm, that's not quite right either. Let me think again.
    
    The tensor T ∈ R^{n×m × m×p × p×n} where T[(i,j),(j',k),(k',i')] = 1 if j=j' and k=k'.
    
    A rank-r decomposition is T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c where:
      u_c ∈ R^{n×m}, v_c ∈ R^{m×p}, w_c ∈ R^{p×n}
    
    And the decomposition works as:
      C[i,k] = sum_{c} (u_c · A) * (v_c · B) * (w_c · C) ... no.
    
    Actually, for matrix multiplication tensor:
      T[i,j | j',k | k',i'] = δ_{j,j'} δ_{k,k'}
    
    A rank-1 term is u[i,j] * v[j',k] * w[k',i']
    
    For the product to give C[i,k] = sum_j A[i,j]B[j,k], we need:
      sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] ... 
    
    Let me use the standard formulation. The matmul tensor M ∈ F^{n×m} ⊗ F^{m×p} ⊗ F^{p×n}
    is defined by M = sum_{i,j,k} e_{ij} ⊗ e_{jk} ⊗ e_{ki}.
    
    A rank-r decomposition: M = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    
    Then for matrices A ∈ F^{n×m}, B ∈ F^{m×p}:
      (A·B)[i,k] = sum_{j} A[i,j] B[j,k] = sum_c (u_c · A) (v_c · B) w_c[k,i]
    
    where u_c · A = sum_{i,j} u_c[i,j] A[i,j], etc.
    
    Wait, that doesn't work dimensionally. Let me reconsider.
    
    Actually: C = A × B where C[i,k] = sum_j A[i,j] B[j,k].
    
    The tensor is T = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    
    Rank-r decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    
    Then: C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_{i',j'} u_c[i',j'] A[i',j']) (sum_{j',k'} v_c[j',k'] B[j',k']) ... 
    
    No, that's not right. The correct interpretation is:
    
    If T = sum_c u_c ⊗ v_c ⊗ w_c, then for any A, B:
      <T, A ⊗ B ⊗ C> = sum_c <u_c, A> <v_c, B> <w_c, C>
    
    And <T, A ⊗ B ⊗ C> = sum_{i,j,k} A[i,j] B[j,k] C[k,i] = tr(A B C)
    
    So we need: tr(A B C) = sum_c (u_c · A)(v_c · B)(w_c · C)
    
    where u_c · A = sum_{i,j} u_c[i,j] A[i,j], etc.
    
    This means: sum_j A[i,j] B[j,k] C[k,i] = sum_c (sum_{i',j'} u_c[i',j'] A[i',j'])(sum_{j',k'} v_c[j',k'] B[j',k'])(sum_{k',i'} w_c[k',i'] C[k',i'])
    
    For this to hold for all A, B, C, we need the tensor decomposition to be exact.
    
    Now, for Strassen's algorithm applied to 2x2 × 2x2:
    
    The 7 rank-1 terms give:
      tr(A B C) = sum_{s=1}^7 (u_s · A)(v_s · B)(w_s · C)
    
    Let me write out the Strassen decomposition explicitly.
    
    A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[p, q], [r, s]]
    
    tr(ABC) = a(e p + f r) + b(g p + h r) + c(e q + f s) + d(g q + h s)
             = a e p + a f r + b g p + b h r + c e q + c f s + d g q + d h s
    
    Strassen's 7 products:
      M1 = (a+d)(e+h) = ae + ah + de + dh
      M2 = (c+d)e = ce + de
      M3 = a(f-h) = af - ah
      M4 = d(g-e) = dg - de
      M5 = (a+b)h = ah + bh
      M6 = (c-a)(e+f) = ce + cf - ae - af
      M7 = (b-d)(g+h) = bg + bh - dg - dh
    
    Now we need to find u_s, v_s, w_s such that:
      sum_s (u_s · A)(v_s · B)(w_s · C) = tr(ABC)
    
    Let me work this out. For each s, u_s, v_s, w_s are 2x2 matrices.
    
    For s=1: M1 = (a+d)(e+h)
      u_1 = [[1,0],[0,1]] (so u_1·A = a+d)
      v_1 = [[1,1],[0,1]] (so v_1·B = e+h)
      w_1 must be such that w_1·C contributes appropriately.
    
    Actually, this is getting complicated. Let me use a different approach.
    
    Since I know the rank is 35 (5 blocks × 7 Strassen products), let me construct the
    U, V, W matrices directly by expanding each Strassen product.
    
    For block j (j=0..4), the 7 Strassen products are:
      M1_j = (a+d)(B[0,2j]+B[1,2j+1])
      M2_j = (c+d)B[0,2j]
      M3_j = a(B[0,2j+1]-B[1,2j+1])
      M4_j = d(B[1,2j]-B[0,2j])
      M5_j = (a+b)B[1,2j+1]
      M6_j = (c-a)(B[0,2j]+B[0,2j+1])
      M7_j = (b-d)(B[1,2j]+B[1,2j+1])
    
    Each M_s_j is of the form (linear combo of A entries) × (linear combo of B entries).
    
    For the tensor decomposition, each rank-1 term corresponds to one of these products.
    The contribution to tr(ABC) from product M_s_j is M_s_j × (appropriate linear combo of C entries).
    
    Let me figure out what C entries each product contributes to.
    
    C[0, 2j]   = M1_j + M4_j - M5_j + M7_j
    C[0, 2j+1] = M3_j + M5_j
    C[1, 2j]   = M2_j + M4_j
    C[1, 2j+1] = M1_j - M2_j + M3_j + M6_j
    
    So tr(ABC) = sum_{i,k} C[i,k] * (A B)[i,k] ... no, tr(ABC) = sum_{i,k} C[k,i] (AB)[i,k]
    
    Actually, tr(ABC) = sum_{i,j,k} A[i,j] B[j,k] C[k,i].
    
    So for each (i,k) pair, the coefficient of A[i,j]B[j,k] in tr(ABC) is C[k,i].
    
    For the decomposition to work, we need:
      sum_c (u_c · A)(v_c · B)(w_c · C) = tr(ABC)
    
    Let me define the 35 rank-1 terms. For block j and Strassen product s:
    
    The A-part is a linear form in {a,b,c,d}.
    The B-part is a linear form in {B[0,2j], B[0,2j+1], B[1,2j], B[1,2j+1]}.
    The C-part must be a linear form in C entries such that the product sums correctly.
    
    This is getting complex. Let me just directly construct U, V, W by considering
    each of the 35 terms.
    
    Actually, let me think about it differently. The standard matmul tensor for 2x2×2x2
    has rank 7 (Strassen). The decomposition gives 7 rank-1 terms.
    
    For the 2x2×2x10 case, we have 5 independent 2x2 blocks, so rank = 5×7 = 35.
    
    Let me construct the decomposition by treating each block independently.
    
    For block j, define local indices:
      A rows: i=0,1; A cols: j_a=0,1 (but j_a is always 0 or 1 since A is 2x2)
      B rows: j_b=0,1; B cols: k=2j, 2j+1
      C rows: i=0,1; C cols: k=2j, 2j+1
    
    Wait, I'm confusing myself with index names. Let me use:
      A is 2x2: A[i, ja] for i,j_a ∈ {0,1}
      B is 2x10: B[jb, k] for jb ∈ {0,1}, k ∈ {0,...,9}
      C is 2x10: C[i, k] for i ∈ {0,1}, k ∈ {0,...,9}
    
    C[i,k] = sum_{jb} A[i,jb] B[jb,k]
    
    For block j (columns 2j, 2j+1 of B and C):
    
    Strassen products for block j:
      Let e = B[0,2j], f = B[0,2j+1], g = B[1,2j], h = B[1,2j+1]
      
      M1 = (A[0,0]+A[1,1])(e+h)
      M2 = (A[1,0]+A[1,1])e
      M3 = A[0,0](f-h)
      M4 = A[1,1](g-e)
      M5 = (A[0,0]+A[0,1])h
      M6 = (A[1,0]-A[0,0])(e+f)
      M7 = (A[0,1]-A[1,1])(g+h)
    
    C[0,2j]   = M1 + M4 - M5 + M7
    C[0,2j+1] = M3 + M5
    C[1,2j]   = M2 + M4
    C[1,2j+1] = M1 - M2 + M3 + M6
    
    Now, for the tensor decomposition, each rank-1 term s in block j contributes:
      (u_s · A) * (v_s · B) * (w_s · C)
    
    to tr(ABC) = sum_{i,jb,k} A[i,jb] B[jb,k] C[k,i].
    
    The key insight: for each Strassen product M_s in block j, we can write:
      M_s = (linear form in A) * (linear form in B restricted to block j)
    
    And the contribution to tr(ABC) is M_s * (linear form in C restricted to block j).
    
    Specifically, if M_s appears in C[i,k] with coefficient α, then it contributes
      α * M_s * C[k,i]
    
    to tr(ABC).
    
    So for each (s, j) pair, the C-part w_s is determined by which C entries M_s contributes to.
    
    Let me tabulate for each Strassen product s in block j, which C entries it contributes to:
    
    M1: C[0,2j] (+1), C[1,2j+1] (+1)
    M2: C[1,2j] (+1), C[1,2j+1] (-1)
    M3: C[0,2j+1] (+1), C[1,2j+1] (+1)
    M4: C[0,2j] (+1), C[1,2j] (+1)
    M5: C[0,2j] (-1), C[0,2j+1] (+1)
    M6: C[1,2j+1] (+1)
    M7: C[0,2j] (+1)
    
    So w_s for block j has non-zero entries only at positions (k,i) where C[i,k] receives
    a contribution from M_s.
    
    w_s[k*n+i] = coefficient of M_s in C[i,k]
    
    For M1 in block j:
      w_1[2j*2+0] = 1  (C[0,2j] gets +M1, so w[2j, 0] = 1)
      w_1[(2j+1)*2+1] = 1  (C[1,2j+1] gets +M1, so w[2j+1, 1] = 1)
    
    Wait, W has shape (p*n, rank) = (20, 35). W[k*n+i, c].
    
    So W[k*2+i, c] is the coefficient for C[i,k] in the c-th rank-1 term.
    
    For M1 in block j:
      C[0,2j] += M1 → W[2j*2+0, c] = 1
      C[1,2j+1] += M1 → W[(2j+1)*2+1, c] = 1
    
    For M2 in block j:
      C[1,2j] += M2 → W[2j*2+1, c] = 1
      C[1,2j+1] -= M2 → W[(2j+1)*2+1, c] = -1
    
    For M3 in block j:
      C[0,2j+1] += M3 → W[(2j+1)*2+0, c] = 1
      C[1,2j+1] += M3 → W[(2j+1)*2+1, c] = 1
    
    For M4 in block j:
      C[0,2j] += M4 → W[2j*2+0, c] = 1
      C[1,2j] += M4 → W[2j*2+1, c] = 1
    
    For M5 in block j:
      C[0,2j] -= M5 → W[2j*2+0, c] = -1
      C[0,2j+1] += M5 → W[(2j+1)*2+0, c] = 1
    
    For M6 in block j:
      C[1,2j+1] += M6 → W[(2j+1)*2+1, c] = 1
    
    For M7 in block j:
      C[0,2j] += M7 → W[2j*2+0, c] = 1
    
    Now for U (shape 4×35) and V (shape 20×35):
    
    U[i*m+j_a, c] where i ∈ {0,1}, j_a ∈ {0,1}, so U[i*2+j_a, c].
    
    For each Strassen product, the A-part:
    
    M1: (a+d) = A[0,0] + A[1,1]
      U[0*2+0, c] = 1  (A[0,0])
      U[1*2+1, c] = 1  (A[1,1])
    
    M2: (c+d) = A[1,0] + A[1,1]
      U[1*2+0, c] = 1  (A[1,0])
      U[1*2+1, c] = 1  (A[1,1])
    
    M3: a = A[0,0]
      U[0*2+0, c] = 1
    
    M4: d = A[1,1]
      U[1*2+1, c] = 1
    
    M5: (a+b) = A[0,0] + A[0,1]
      U[0*2+0, c] = 1
      U[0*2+1, c] = 1
    
    M6: (c-a) = A[1,0] - A[0,0]
      U[1*2+0, c] = 1
      U[0*2+0, c] = -1
    
    M7: (b-d) = A[0,1] - A[1,1]
      U[0*2+1, c] = 1
      U[1*2+1, c] = -1
    
    V[jb*p+k, c] where jb ∈ {0,1}, k ∈ {0,...,9}, so V[jb*10+k, c].
    
    For each Strassen product in block j, the B-part involves only B[0,2j], B[0,2j+1], B[1,2j], B[1,2j+1].
    
    M1: (e+h) = B[0,2j] + B[1,2j+1]
      V[0*10+2j, c] = 1
      V[1*10+2j+1, c] = 1
    
    M2: e = B[0,2j]
      V[0*10+2j, c] = 1
    
    M3: (f-h) = B[0,2j+1] - B[1,2j+1]
      V[0*10+2j+1, c] = 1
      V[1*10+2j+1, c] = -1
    
    M4: (g-e) = B[1,2j] - B[0,2j]
      V[1*10+2j, c] = 1
      V[0*10+2j, c] = -1
    
    M5: h = B[1,2j+1]
      V[1*10+2j+1, c] = 1
    
    M6: (e+f) = B[0,2j] + B[0,2j+1]
      V[0*10+2j, c] = 1
      V[0*10+2j+1, c] = 1
    
    M7: (g+h) = B[1,2j] + B[1,2j+1]
      V[1*10+2j, c] = 1
      V[1*10+2j+1, c] = 1
    
    Now let me put it all together.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    rank = 35  # 5 blocks * 7 Strassen products
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 35)
    V = [[0] * rank for _ in range(m * p)]  # (20, 35)
    W = [[0] * rank for _ in range(p * n)]  # (20, 35)
    
    # For each block j (0..4) and Strassen product s (0..6):
    for j in range(5):
        for s in range(7):
            c = j * 7 + s  # column index in U, V, W
            
            # U part (A coefficients)
            if s == 0:  # M1: a + d
                U[0][c] = 1
                U[3][c] = 1
            elif s == 1:  # M2: c + d
                U[2][c] = 1
                U[3][c] = 1
            elif s == 2:  # M3: a
                U[0][c] = 1
            elif s == 3:  # M4: d
                U[3][c] = 1
            elif s == 4:  # M5: a + b
                U[0][c] = 1
                U[1][c] = 1
            elif s == 5:  # M6: c - a
                U[2][c] = 1
                U[0][c] = -1
            elif s == 6:  # M7: b - d
                U[1][c] = 1
                U[3][c] = -1
            
            # V part (B coefficients)
            if s == 0:  # M1: e + h = B[0,2j] + B[1,2j+1]
                V[0 * p + 2*j][c] = 1
                V[1 * p + 2*j+1][c] = 1
            elif s == 1:  # M2: e = B[0,2j]
                V[0 * p + 2*j][c] = 1
            elif s == 2:  # M3: f - h = B[0,2j+1] - B[1,2j+1]
                V[0 * p + 2*j+1][c] = 1
                V[1 * p + 2*j+1][c] = -1
            elif s == 3:  # M4: g - e = B[1,2j] - B[0,2j]
                V[1 * p + 2*j][c] = 1
                V[0 * p + 2*j][c] = -1
            elif s == 4:  # M5: h = B[1,2j+1]
                V[1 * p + 2*j+1][c] = 1
            elif s == 5:  # M6: e + f = B[0,2j] + B[0,2j+1]
                V[0 * p + 2*j][c] = 1
                V[0 * p + 2*j+1][c] = 1
            elif s == 6:  # M7: g + h = B[1,2j] + B[1,2j+1]
                V[1 * p + 2*j][c] = 1
                V[1 * p + 2*j+1][c] = 1
            
            # W part (C coefficients), W[k*n+i, c]
            if s == 0:  # M1: C[0,2j] += M1, C[1,2j+1] += M1
                W[2*j * n + 0][c] = 1
                W[(2*j+1) * n + 1][c] = 1
            elif s == 1:  # M2: C[1,2j] += M2, C[1,2j+1] -= M2
                W[2*j * n + 1][c] = 1
                W[(2*j+1) * n + 1][c] = -1
            elif s == 2:  # M3: C[0,2j+1] += M3, C[1,2j+1] += M3
                W[(2*j+1) * n + 0][c] = 1
                W[(2*j+1) * n + 1][c] = 1
            elif s == 3:  # M4: C[0,2j] += M4, C[1,2j] += M4
                W[2*j * n + 0][c] = 1
                W[2*j * n + 1][c] = 1
            elif s == 4:  # M5: C[0,2j] -= M5, C[0,2j+1] += M5
                W[2*j * n + 0][c] = -1
                W[(2*j+1) * n + 0][c] = 1
            elif s == 5:  # M6: C[1,2j+1] += M6
                W[(2*j+1) * n + 1][c] = 1
            elif s == 6:  # M7: C[0,2j] += M7
                W[2*j * n + 0][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
