"""Randomized rank-reduction search for 3x4x8 matrix multiplication tensor."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    deadline = time.time() + 140
    
    # The tensor T_{ijk} = delta_{jk} (for A_{ij} B_{jk} -> C_{ik})
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r} = delta_{jk} if i,k match? No.
    # The multiplication tensor is T_{i,j,k} such that C_{ik} = sum_j A_{ij} B_{jk}.
    # In tensor notation: T_{i,j,k} is the coefficient of A_{ij} B_{jk} in C_{ik}.
    # Actually, the standard decomposition is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # This can be written as sum_r (sum_i,j U_{ij,r} A_{ij}) (sum_j,k V_{jk,r} B_{jk}) (sum_k,i W_{ki,r} C_{ki})? No.
    # The rank-r decomposition is:
    # T = sum_{r=1}^R u_r \otimes v_r \otimes w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
    # The condition is that for all A, B:
    # sum_{i,j,k} A_{ij} B_{jk} C_{ik} = sum_r (u_r^T A) (v_r^T B) (w_r^T C)
    # This implies that the tensor T_{i,j,k} (where C index is k*n+i? No, C is i,k)
    # Let's stick to the interface:
    # U shape (n*m, rank), index i*m+j
    # V shape (m*p, rank), index j*p+k
    # W shape (p*n, rank), index k*n+i  (Note: Task says W[k*n+i, r])
    
    # The tensor entry T_{(i,j), (j,k), (k,i)}? No.
    # The bilinear form is:
    # <A, B, C> = sum_{i,j,k} A_{ij} B_{jk} C_{ik}
    # We need: sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1 if (i,j,k) corresponds to a valid term?
    # Actually, the tensor is defined by the identity:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # So the coefficient of A_{ij} B_{jk} in C_{ik} is 1 if the indices match, 0 otherwise.
    # Wait, the "tensor" T has indices (i, j, k) where i is row of A, j is inner, k is col of B.
    # The output is C_{i, k}.
    # The standard tensor for matrix multiplication M_{n,p,m} (A n x m, B m x p -> C n x p)
    # has entries T_{i, j, k} = delta_{j, j'}? No.
    # T_{i, j, k} is non-zero only if... 
    # Let's look at the decomposition form:
    # sum_r U_{i, j, r} V_{j, k, r} W_{i, k, r} = delta_{j, j} (always 1?) No.
    # The condition is:
    # For all A, B, C: sum_{i,j,k} A_{ij} B_{jk} C_{ik} = sum_r (sum_{i,j} U_{ij,r} A_{ij}) (sum_{j,k} V_{jk,r} B_{jk}) (sum_{i,k} W_{ik,r} C_{ik})
    # Expanding the RHS:
    # sum_r (sum_{i,j} U_{ij,r} A_{ij}) (sum_{j',k} V_{j'k,r} B_{j'k}) (sum_{i',k'} W_{i'k',r} C_{i'k'})
    # = sum_{i,j,j',k,i',k'} (sum_r U_{ij,r} V_{j'k,r} W_{i'k',r}) A_{ij} B_{j'k} C_{i'k'}
    # We want this to equal sum_{i,j,k} A_{ij} B_{jk} C_{ik}.
    # Therefore, the coefficient of A_{ij} B_{j'k} C_{i'k'} must be:
    # 1 if j=j' and i=i' and k=k'
    # 0 otherwise.
    # So we need:
    # sum_r U_{i*m+j, r} V_{j'*p+k, r} W_{k'*n+i', r} = delta_{j,j'} delta_{i,i'} delta_{k,k'}
    
    # This is a system of equations for U, V, W.
    # Number of unknowns: 3 * (n*m + m*p + p*n) * R = 3 * (12 + 32 + 24) * R = 3 * 68 * R = 204 R
    # Number of equations: n*m*m*p*p*n? No.
    # The indices are (i, j, j', k, i', k').
    # i: 0..n-1, j: 0..m-1, j': 0..m-1, k: 0..p-1, i': 0..n-1, k': 0..p-1.
    # Total equations: n * m * m * p * n * p = n^2 * m^2 * p^2 = 9 * 16 * 64 = 9216.
    
    # We can set up this as a linear system? No, it's cubic in U,V,W.
    # However, we can use the "border rank" or "rank" search via random projections.
    # A simpler approach: Start with schoolbook (R=96) and try to reduce rank by finding linear dependencies.
    
    # Schoolbook construction:
    # For each i, j, k:
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # This gives 96 terms.
    
    # Strategy:
    # 1. Generate a random initial decomposition (or use schoolbook).
    # 2. Form the "flattened" matrix M of size (n*m) x (m*p * p*n)? No.
    # The condition sum_r U_{ij,r} V_{j'k,r} W_{i'k',r} = delta...
    # This is equivalent to saying that the tensor T has rank R.
    # We can try to solve for R by minimizing the number of terms.
    
    # Alternative: Use the fact that for small matrices, we can brute-force search for small R?
    # R=96 is the seed. Can we do better?
    # Known bounds:
    # 3x4x8:
    # Schoolbook: 96.
    # Strassen-like?
    # If we split 3x4x8 into 2x4x8 + 1x4x8.
    # 1x4x8 is rank 32 (schoolbook) or maybe lower?
    # 1xm x m x p: rank is mp? No, it's just dot products.
    # C_{1,k} = sum_j A_{1j} B_{jk}.
    # This is a set of p independent dot products of length m.
    # Rank of 1xm x m x p tensor?
    # It's the sum of p rank-1 tensors? No.
    # T_{1, j, k} = delta_{j,j} ? No.
    # The tensor is T_{1,j,k} such that C_k = sum_j A_j B_{jk}.
    # In decomposition form: sum_r U_{1j,r} V_{jk,r} W_{k,r} = delta_{j,j}? No.
    # The condition is: sum_r U_{1j,r} V_{jk,r} W_{k',r} = delta_{j,j} delta_{k,k'}?
    # Wait, the indices for 1xm x m x p are:
    # A has indices (1, j). B has indices (j, k). C has index (1, k).
    # U shape (1*m, R) = (m, R).
    # V shape (m*p, R).
    # W shape (p*1, R) = (p, R).
    # Condition: sum_r U_{j,r} V_{j'k,r} W_{k',r} = delta_{j,j'} delta_{k,k'}
    # This is the identity tensor of size m x (m*p) x p?
    # Actually, this is just the standard matrix multiplication tensor M_{1,p,m}.
    # The rank of M_{1,p,m} is mp? No, it's m*p?
    # For 1x2x2, rank is 4? No, 1*2*2=4.
    # Actually, M_{1,p,m} is isomorphic to M_{m,p,1} (transpose).
    # The rank of M_{1,p,m} is mp.
    # So 1x4x8 has rank 32.
    
    # 2x4x8:
    # Can we do better than 2*32=64?
    # Strassen on 2x2x2 blocks?
    # 2x4x8 can be viewed as 2 blocks of 2x4x4?
    # 2x4x4 rank?
    # If we use Strassen for 2x2x2, rank 7.
    # 2x4x4 = 4 blocks of 2x2x2? No.
    # 2x4x4: A is 2x4, B is 4x4, C is 2x4.
    # Split A into A1, A2 (2x2 each). B into B1, B2, B3, B4 (2x2 each).
    # C1 = A1 B1 + A2 B3.
    # C2 = A1 B2 + A2 B4.
    # This is 2 matrix multiplications of 2x2x2.
    # Rank = 2 * 7 = 14?
    # But we have 2 such blocks (cols 0-3 and 4-7).
    # So 2x4x8 rank could be 2 * 14 = 28?
    # Total rank = 32 (for 1x4x8) + 28 (for 2x4x8) = 60?
    # Let's verify the 2x4x4 decomposition.
    # A = [A11 A12; A21 A22] (2x4)
    # B = [B11 B12; B21 B22] (4x4) where B11 is 2x2, etc.
    # C = [C11 C12; C21 C22] (2x4)
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    # This is NOT two independent 2x2x2 multiplications.
    # It's 4 terms.
    # However, we can use Strassen on the 2x2x2 blocks if we view the 2x4x4 as a block matrix multiplication.
    # A (2x4) * B (4x4) -> C (2x4).
    # This is equivalent to multiplying a 2x2 block matrix A by a 2x2 block matrix B, where each block is 2x2?
    # No, A is 2x4 (1x2 blocks of 2x2). B is 4x4 (2x2 blocks of 2x2).
    # C is 2x4 (1x2 blocks of 2x2).
    # This is not square block multiplication.
    
    # Let's try a different split.
    # 3x4x8.
    # Split rows of A: 1 and 2.
    # Row 0: 1x4x8. Rank 32.
    # Rows 1-2: 2x4x8.
    # For 2x4x8, can we use Strassen?
    # Strassen works for 2x2x2.
    # If we group the 4 inner dimensions into 2 pairs?
    # j=0,1 and j=2,3.
    # C_{i,k} = sum_{j=0}^1 A_{ij} B_{jk} + sum_{j=2}^3 A_{ij} B_{jk}.
    # Let A1 = A[:, 0:2], B1 = B[0:2, :].
    # Let A2 = A[:, 2:4], B2 = B[2:4, :].
    # C = A1 B1 + A2 B2.
    # This is the sum of two 2x2x8 multiplications.
    # Rank of 2x2x8?
    # 2x2x8: A 2x2, B 2x8, C 2x8.
    # This is 2 independent 1x2x8 multiplications? No.
    # C1 = A11 B11 + A12 B21? No.
    # C_{1,k} = A_{1,1} B_{1,k} + A_{1,2} B_{2,k}.
    # C_{2,k} = A_{2,1} B_{1,k} + A_{2,2} B_{2,k}.
    # This is 2x2x8.
    # Can we use Strassen?
    # Strassen is for 2x2x2.
    # 2x2x8 can be seen as 4 independent 2x2x2 multiplications? No.
    # B is 2x8. Split B into 4 blocks of 2x2.
    # B = [B1, B2, B3, B4].
    # C = [C1, C2, C3, C4] where Ci is 2x2.
    # Ci = A Bi.
    # So 4 independent 2x2x2 multiplications.
    # Rank = 4 * 7 = 28.
    # So 2x4x8 = 2x2x8 + 2x2x8 = 28 + 28 = 56?
    # Wait, A is 2x4. A1 is 2x2 (cols 0,1), A2 is 2x2 (cols 2,3).
    # B1 is 2x8 (rows 0,1), B2 is 2x8 (rows 2,3).
    # C = A1 B1 + A2 B2.
    # A1 B1 is 2x2x8. Rank 28.
    # A2 B2 is 2x2x8. Rank 28.
    # Total rank for 2x4x8 is 56.
    # Total rank for 3x4x8 is 32 + 56 = 88.
    
    # This matches the "inspiration" finding!
    # So we can construct a rank-88 decomposition.
    
    # Construction:
    # 1. 1x4x8 schoolbook (rank 32).
    # 2. 2x4x8 using Strassen on 2x2x2 blocks (rank 56).
    
    # Let's build the 2x2x2 Strassen decomposition.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]].
    # i = (a+d)(e+h) - (d-c)(h-g) ? No.
    # Standard Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+h)
    # m7 = (b-d)(e+g)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # We need to map this to U, V, W.
    # For each of the 7 terms, we have:
    # U_{ij, r}, V_{jk, r}, W_{ki, r} (for 2x2x2, n=2, m=2, p=2).
    # Indices:
    # A: 00, 01, 10, 11.
    # B: 00, 01, 10, 11.
    # C: 00, 01, 10, 11.
    
    # Term 1: m1 = (a+d)(e+h)
    # U: a=1, d=1. Others 0.
    # V: e=1, h=1. Others 0.
    # W: contributes to i, j, k, l.
    # i = m1... so W_{00} += 1.
    # j = ... no m1 in j.
    # k = ... no m1 in k.
    # l = m1... so W_{11} += 1.
    # Wait, l = m1 - m2 + m3 + m6.
    # So W_{11} gets +1 from m1.
    
    # Let's list the 7 Strassen terms for 2x2x2:
    # r=0: m1
    # U: [1, 0, 0, 1] (a, b, c, d)
    # V: [1, 0, 0, 1] (e, f, g, h)
    # W: [1, 0, 0, 1] (i, j, k, l) -> i and l get 1.
    
    # r=1: m2
    # U: [0, 0, 1, 1] (c, d)
    # V: [1, 0, 0, 0] (e)
    # W: k gets 1. l gets -1.
    # W: [0, 0, 1, -1]
    
    # r=2: m3
    # U: [1, 0, 0, 0] (a)
    # V: [0, 1, 0, -1] (f, -h)
    # W: j gets 1. l gets 1.
    # W: [0, 1, 0, 1]
    
    # r=3: m4
    # U: [0, 0, 0, 1] (d)
    # V: [0, 0, 1, -1] (g, -e)
    # W: i gets 1. k gets 1.
    # W: [1, 0, 1, 0]
    
    # r=4: m5
    # U: [1, 1, 0, 0] (a, b)
    # V: [0, 0, 0, 1] (h)
    # W: i gets -1. j gets 1.
    # W: [-1, 1, 0, 0]
    
    # r=5: m6
    # U: [0, 0, 1, -1] (c, -d)
    # V: [0, 1, 0, 1] (f, h)
    # W: l gets 1.
    # W: [0, 0, 0, 1]
    
    # r=6: m7
    # U: [0, 1, 0, -1] (b, -d)
    # V: [1, 0, 1, 0] (e, g)
    # W: i gets 1.
    # W: [1, 0, 0, 0]
    
    # Let's verify one entry: C_{00} (i).
    # i = m1 + m4 - m5 + m7
    # W_{00} for r=0 is 1.
    # W_{00} for r=3 is 1.
    # W_{00} for r=4 is -1.
    # W_{00} for r=6 is 1.
    # Sum: 1 + 1 - 1 + 1 = 2? No, i should be 1.
    # Wait, Strassen formula:
    # C00 = m1 + m4 - m5 + m7
    # Let's re-verify the Strassen formulas.
    # m1 = (a+d)(e+h) = ae + ah + de + dh
    # m4 = d(g-e) = dg - de
    # m5 = (a+b)h = ah + bh
    # m7 = (b-d)(e+g) = be + bg - de - dg
    # Sum: ae + ah + de + dh + dg - de - ah - bh + be + bg - de - dg
    # = ae + dh + be + bg - de - bh
    # This is not ae + bg.
    # My Strassen formulas are slightly off or I'm summing wrong.
    # Standard Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-d)(f+h)
    # p7 = (b-d)(e+g)
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    
    # Let's re-check C00:
    # p1 = ae + ah + de + dh
    # p4 = dg - de
    # p5 = ah + bh
    # p7 = be + bg - de - dg
    # Sum: ae + ah + de + dh + dg - de - ah - bh + be + bg - de - dg
    # = ae + dh + be + bg - de - bh
    # Still not ae + bg.
    # Ah, p7 is (b-d)(e+g) = be + bg - de - dg.
    # Is there a sign error?
    # Let's check p1 - p5 + p4 + p7.
    # Maybe C00 = p1 + p4 - p5 + p7 is correct?
    # Let's test with a=1, others 0.
    # p1 = 1*e = e.
    # p4 = 0.
    # p5 = 1*h = h.
    # p7 = 0.
    # C00 = e - h. Should be e.
    # So the formula is wrong.
    
    # Correct Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-d)(f+h)
    # p7 = (b-d)(e+g)
    
    # C = [[p1 + p4 - p5 + p7, p3 + p5], [p2 + p4, p1 - p2 + p3 + p6]]
    
    # Let's re-evaluate C00 with a=1.
    # p1 = e.
    # p4 = 0.
    # p5 = h.
    # p7 = 0.
    # C00 = e - h.
    # But A B = [[e, f], [0, 0]]. So C00 should be e.
    # Why is there a -h?
    # Because p5 = (a+b)h = h.
    # And p1 = (a+d)(e+h) = e+h.
    # So p1 - p5 = e.
    # So C00 = p1 - p5 + p4 + p7.
    # If a=1, p1=e+h, p5=h, p4=0, p7=0. Sum = e. Correct.
    
    # So the W coefficients for 2x2x2 are:
    # r=0 (p1): W = [1, 0, 0, 1] (C00, C01, C10, C11)
    # r=1 (p2): W = [0, 0, 1, -1]
    # r=2 (p3): W = [0, 1, 0, 1]
    # r=3 (p4): W = [1, 0, 1, 0]
    # r=4 (p5): W = [-1, 1, 0, 0]
    # r=5 (p6): W = [0, 0, 0, 1]
    # r=6 (p7): W = [1, 0, 0, 0]
    
    # Now we need to assemble the 2x4x8 decomposition.
    # 2x4x8: A (2x4), B (2x8), C (2x8).
    # Split A into A1 (2x2, cols 0,1), A2 (2x2, cols 2,3).
    # Split B into B1 (2x8, rows 0,1), B2 (2x8, rows 2,3).
    # C = A1 B1 + A2 B2.
    
    # For A1 B1 (2x2x8):
    # B1 is 2x8. Split into 4 blocks of 2x2: B1_1, B1_2, B1_3, B1_4.
    # C1 is 2x8. Split into 4 blocks of 2x2: C1_1, C1_2, C1_3, C1_4.
    # Each Ci_j = A1 * B1_j.
    # This is 4 independent 2x2x2 multiplications.
    # Each uses 7 terms. Total 28 terms.
    
    # For A2 B2 (2x2x8):
    # Similarly, 4 independent 2x2x2 multiplications.
    # Total 28 terms.
    
    # Total 56 terms for 2x4x8.
    
    # Now we need to map these to the global U, V, W matrices.
    # Global dimensions: n=3, m=4, p=8.
    # U: 3x4 = 12 rows.
    # V: 4x8 = 32 rows.
    # W: 8x3 = 24 rows.
    
    # We will construct U, V, W with 88 columns (32 + 56).
    
    U = np.zeros((12, 88))
    V = np.zeros((32, 88))
    W = np.zeros((24, 88))
    
    # Part 1: 1x4x8 schoolbook (rows 0 of A)
    # A_{0,j} * B_{j,k} -> C_{0,k}
    # j=0..3, k=0..7.
    # 32 terms.
    col = 0
    for j in range(4):
        for k in range(8):
            # U index: i=0, j=j. idx = 0*4 + j = j.
            U[j, col] = 1
            # V index: j=j, k=k. idx = j*8 + k.
            V[j*8 + k, col] = 1
            # W index: k=k, i=0. idx = k*3 + 0 = k*3.
            W[k*3, col] = 1
            col += 1
            
    # Part 2: 2x4x8 Strassen (rows 1,2 of A)
    # A_{1,j}, A_{2,j} * B_{j,k} -> C_{1,k}, C_{2,k}
    
    # Helper to add a 2x2x2 Strassen block
    # a_rows: [i1, i2] (global row indices in A, 1 and 2)
    # b_rows: [j1, j2] (global row indices in B, 0..3)
    # c_cols: [k1, k2] (global col indices in C, 0..7)
    # c_rows: [i1, i2] (global row indices in C, 1 and 2)
    
    def add_strassen_block(a_rows, b_rows, c_cols, c_rows, start_col):
        # 7 terms
        # U indices: a_rows[0]*4 + b_rows[0], etc.
        # Wait, A is 3x4. a_rows are 1,2. b_rows are inner indices.
        # U index for A_{i, j} is i*4 + j.
        # V index for B_{j, k} is j*8 + k.
        # W index for C_{i, k} is k*3 + i.
        
        i1, i2 = a_rows
        j1, j2 = b_rows
        k1, k2 = c_cols
        ci1, ci2 = c_rows
        
        # U vectors for the 7 Strassen terms (local 2x2)
        # Local A indices: 00, 01, 10, 11
        # Global A indices:
        # 00 -> i1, j1
        # 01 -> i1, j2
        # 10 -> i2, j1
        # 11 -> i2, j2
        
        u_00 = i1*4 + j1
        u_01 = i1*4 + j2
        u_10 = i2*4 + j1
        u_11 = i2*4 + j2
        
        # V vectors (local 2x2)
        # 00 -> j1, k1
        # 01 -> j1, k2
        # 10 -> j2, k1
        # 11 -> j2, k2
        
        v_00 = j1*8 + k1
        v_01 = j1*8 + k2
        v_10 = j2*8 + k1
        v_11 = j2*8 + k2
        
        # W vectors (local 2x2)
        # 00 -> k1, ci1
        # 01 -> k2, ci1
        # 10 -> k1, ci2
        # 11 -> k2, ci2
        
        w_00 = k1*3 + ci1
        w_01 = k2*3 + ci1
        w_10 = k1*3 + ci2
        w_11 = k2*3 + ci2
        
        # Strassen coefficients
        # r=0: p1
        U[u_00, start_col] += 1
        U[u_11, start_col] += 1
        V[v_00, start_col] += 1
        V[v_11, start_col] += 1
        W[w_00, start_col] += 1
        W[w_11, start_col] += 1
        start_col += 1
        
        # r=1: p2
        U[u_10, start_col] += 1
        U[u_11, start_col] += 1
        V[v_00, start_col] += 1
        W[w_10, start_col] += 1
        W[w_11, start_col] += -1
        start_col += 1
        
        # r=2: p3
        U[u_00, start_col] += 1
        V[v_01, start_col] += 1
        V[v_11, start_col] += -1
        W[w_01, start_col] += 1
        W[w_11, start_col] += 1
        start_col += 1
        
        # r=3: p4
        U[u_11, start_col] += 1
        V[v_10, start_col] += 1
        V[v_00, start_col] += -1
        W[w_00, start_col] += 1
        W[w_10, start_col] += 1
        start_col += 1
        
        # r=4: p5
        U[u_00, start_col] += 1
        U[u_01, start_col] += 1
        V[v_11, start_col] += 1
        W[w_00, start_col] += -1
        W[w_01, start_col] += 1
        start_col += 1
        
        # r=5: p6
        U[u_10, start_col] += 1
        U[u_11, start_col] += -1
        V[v_01, start_col] += 1
        V[v_11, start_col] += 1
        W[w_11, start_col] += 1
        start_col += 1
        
        # r=6: p7
        U[u_01, start_col] += 1
        U[u_11, start_col] += -1
        V[v_00, start_col] += 1
        V[v_10, start_col] += 1
        W[w_00, start_col] += 1
        start_col += 1
        
        return start_col
    
    # A1 B1: a_rows=[1,2], b_rows=[0,1], c_cols=[0,1,2,3]? No, 4 blocks of 2x2.
    # Block 1: b_rows=[0,1], c_cols=[0,1]
    col = add_strassen_block([1,2], [0,1], [0,1], [1,2], col)
    # Block 2: b_rows=[0,1], c_cols=[2,3]
    col = add_strassen_block([1,2], [0,1], [2,3], [1,2], col)
    # Block 3: b_rows=[0,1], c_cols=[4,5]
    col = add_strassen_block([1,2], [0,1], [4,5], [1,2], col)
    # Block 4: b_rows=[0,1], c_cols=[6,7]
    col = add_strassen_block([1,2], [0,1], [6,7], [1,2], col)
    
    # A2 B2: a_rows=[1,2], b_rows=[2,3], c_cols=[0,1,2,3,4,5,6,7]
    # Block 5: b_rows=[2,3], c_cols=[0,1]
    col = add_strassen_block([1,2], [2,3], [0,1], [1,2], col)
    # Block 6: b_rows=[2,3], c_cols=[2,3]
    col = add_strassen_block([1,2], [2,3], [2,3], [1,2], col)
    # Block 7: b_rows=[2,3], c_cols=[4,5]
    col = add_strassen_block([1,2], [2,3], [4,5], [1,2], col)
    # Block 8: b_rows=[2,3], c_cols=[6,7]
    col = add_strassen_block([1,2], [2,3], [6,7], [1,2], col)
    
    # Verify col is 88
    assert col == 88, f"Expected 88, got {col}"
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
