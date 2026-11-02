import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    """
    Search for a rank-minimal exact tensor decomposition for 2x4 x 4x6 matrix multiplication.
    
    Strategy:
    1. Start with a valid schoolbook decomposition (rank 48).
    2. Use Gaussian elimination on the tensor flattening to reduce rank.
       The tensor T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}.
       We can view this as finding a low-rank completion of the tensor.
       A common heuristic is to start with a high rank, then iteratively try to 
       eliminate terms by finding linear dependencies or using "term elimination" 
       via Gaussian elimination on the coefficient matrix of the constraints.
       
       Specifically, we can formulate the problem as:
       Find U (8xR), V (24xR), W (12xR) such that the tensor is decomposed.
       
       A more direct approach for small dimensions:
       The tensor has 2*4*6 = 48 entries.
       A rank-R decomposition has 8R + 24R + 12R = 44R parameters (up to scaling).
       
       We will use a "greedy rank reduction" approach:
       1. Initialize with Rank 48 (Schoolbook).
       2. Try to reduce to Rank 47, 46, ... down to a limit.
       For each target rank R:
       - Randomly initialize U, V, W with rank R.
       - Use a local search or least squares to fit the tensor.
       - However, exact decomposition is hard. 
       
       Alternative: Use the known structure.
       The schoolbook rank is 48.
       Strassen on 2x2 blocks gave 42.
       
       Let's try to find if rank < 42 is possible.
       The tensor is T_{i,j,k} = delta_{i,i} delta_{j,j} delta_{k,k} ? No.
       T_{i,j,k} = 1 if the index corresponds to a valid multiplication term?
       Actually, the multiplication tensor T has T_{(i,j),(j,k),(i,k)} = 1.
       In the flattened form:
       Index U: i*m + j (0..7)
       Index V: j*p + k (0..23)
       Index W: k*n + i (0..11)
       
       The tensor T is a 8x24x12 tensor.
       T[u, v, w] = 1 if:
       i = u // 4, j = u % 4
       j' = v // 6, k = v % 6
       k' = w // 2, i' = w % 2
       
       Then u,v,w are linked if i==i', j==j', k==k'.
       So T[u, v, w] = 1 if (u//4 == w%2) AND (u%4 == v//6) AND (v%6 == w//2).
       
       We want to find minimal R such that T = sum_{r=1}^R u_r v_r w_r.
       
       We can use a randomized search for R.
       For a given R, we can set up a system of linear equations if we fix two of the matrices.
       But that's expensive.
       
       Let's try a simpler structural insight.
       2x4 x 4x6.
       Split A into 2x2 blocks A0, A1.
       Split B into 2x3 blocks B0, B1 (top 2 rows, bottom 2 rows).
       C = A0 B0 + A1 B1.
       
       We know rank(2x2 x 2x2) = 7 (Strassen).
       Here B0 is 2x3, B1 is 2x3.
       2x2 x 2x3 is equivalent to two 2x2 x 2x1 multiplications? No.
       2x2 x 2x3 can be seen as 2x2 x [2x2 | 2x1].
       Rank(2x2 x 2x2) = 7.
       Rank(2x2 x 2x1) = 4 (schoolbook, no Strassen benefit for vector).
       So 2x2 x 2x3 has rank 7 + 4 = 11?
       If we use this:
       C = A0 B0 + A1 B1.
       A0 B0: 2x2 * 2x3 -> Rank 11.
       A1 B1: 2x2 * 2x3 -> Rank 11.
       Total Rank = 22?
       
       Wait, the U factors for A0 and A1 are separate.
       The V factors for B0 and B1 are separate.
       The W factors are separate.
       
       Let's verify rank(2x2 x 2x3).
       X (2x2) * Y (2x3) = Z (2x3).
       Z = [Z0, Z1] where Z0, Z1 are 2x1.
       Z0 = X * Y0 (Y0 is 2x1).
       Z1 = X * Y1 (Y1 is 2x1).
       
       If we compute X * Y0 and X * Y1 separately:
       X * Y0 is 2x2 * 2x1. Rank 4.
       X * Y1 is 2x2 * 2x1. Rank 4.
       Total 8.
       
       Can we do better?
       Strassen for 2x2 x 2x2 uses 7 mults.
       For 2x2 x 2x3, we can pad Y to 2x4? No.
       
       Actually, 2x2 x 2x3 is equivalent to 2x2 x 2x2 plus 2x2 x 2x1.
       Rank(2x2 x 2x2) = 7.
       Rank(2x2 x 2x1) = 4.
       Total 11.
       
       Is there a better decomposition for 2x2 x 2x3?
       Let's check known results.
       Rank(2x2 x 2xn) for n=1 is 4.
       For n=2 is 7.
       For n=3?
       If we use the 7 Strassen terms for the first 2 columns, and 4 terms for the 3rd column, we get 11.
       But maybe we can share?
       
       Let's try to construct a rank 22 decomposition for the whole thing.
       A0 (2x2) * B0 (2x3): Rank 11.
       A1 (2x2) * B1 (2x3): Rank 11.
       Total 22.
       
       This is significantly better than 42.
       
       Let's verify the rank of 2x2 x 2x3.
       The tensor is T_{i,j,k} with i in {0,1}, j in {0,1}, k in {0,1,2}.
       T_{i,j,k} = 1 if indices match.
       Actually, the multiplication tensor for X(2x2) * Y(2x3) = Z(2x3).
       Z_{i,k} = sum_j X_{i,j} Y_{j,k}.
       The tensor T has indices (i,j) for X, (j,k) for Y, (i,k) for Z.
       Dimensions: 4 x 6 x 6.
       
       Let's see if we can decompose 2x2 x 2x3 with rank < 11.
       Schoolbook is 4 (for col 0) + 4 (for col 1) + 4 (for col 2) = 12?
       Wait, 2x2 x 2x1 has 4 terms:
       (0,0)*(0,0), (0,1)*(0,0), (1,0)*(0,0), (1,1)*(0,0) -> for Z00?
       No, Z is 2x1.
       Z0 = X00 Y0 + X01 Y1
       Z1 = X10 Y0 + X11 Y1
       Terms:
       1. X00 Y0 -> Z0
       2. X01 Y1 -> Z0
       3. X10 Y0 -> Z1
       4. X11 Y1 -> Z1
       Rank 4.
       
       So 2x2 x 2x3 (3 columns) would be 4 + 4 + 4 = 12 if done separately.
       But we can use Strassen for the first 2 columns (rank 7) and schoolbook for the 3rd (rank 4).
       Total 11.
       
       Can we do better for 2x2 x 2x3?
       Maybe 10?
       
       Let's try to find a rank 10 decomposition for 2x2 x 2x3.
       If rank(2x2 x 2x3) = 10, then total rank = 20.
       If rank(2x2 x 2x3) = 11, then total rank = 22.
       
       Let's try to search for rank 20 or 21.
       
       Actually, there is a known result: Rank(2x2 x 2xn) = 3n+1 for n>=2?
       For n=2, 3(2)+1 = 7. Correct.
       For n=3, 3(3)+1 = 10.
       So Rank(2x2 x 2x3) should be 10.
       
       If so, Total Rank = 10 + 10 = 20.
       
       Let's try to construct a rank 20 decomposition.
       
       We need a rank-10 decomposition for 2x2 x 2x3.
       
       Let's use the "3n+1" construction.
       For 2x2 x 2x2 (n=2), Strassen is rank 7.
       For 2x2 x 2x3 (n=3), rank 10.
       
       How to construct it?
       We can use the fact that 2x2 x 2x3 is equivalent to 2x2 x 2x2 plus 2x2 x 2x1, but sharing some terms?
       
       Let's try a randomized search for the 2x2 x 2x3 subproblem first to get the factors, then combine.
       
       However, implementing a robust random search for exact integer/half-integer factors is tricky.
       
       Alternative:
       Use the known Strassen rank 7 for 2x2x2.
       And add 3 more terms for the 3rd column?
       If we just append the 3rd column schoolbook (4 terms), we get 7+4=11.
       To get 10, we need to share one term.
       
       Let's try to find a rank 10 decomposition for 2x2 x 2x3 manually or via search.
       
       Actually, let's just try to search for the full 2x4 x 4x6 problem with rank 20, 21, 22.
       
       We can use a "tensor completion" approach.
       Fix R.
       Randomly initialize U, V, W.
       Minimize || T - sum U_r V_r W_r ||^2.
       If we can get close to 0, we can try to round to integers/half-integers.
       
       Given the time limit, a full non-linear optimization might be slow.
       
       Let's try a simpler structural decomposition.
       
       Decomposition for 2x4 x 4x6:
       A = [A0, A1] (2x2 blocks)
       B = [B0; B1] (2x3 blocks)
       C = A0 B0 + A1 B1.
       
       We need rank(A0 B0) + rank(A1 B1).
       If we use rank 11 for each, total 22.
       If we use rank 10 for each, total 20.
       
       Let's try to implement the rank 11 decomposition (Strassen 7 + Schoolbook 4) first, as it's easier to construct correctly.
       
       Rank 11 for 2x2 x 2x3:
       Columns 0,1 of Y use Strassen (7 terms).
       Column 2 of Y uses Schoolbook (4 terms).
       
       U (4x11), V (6x11), W (6x11).
       
       Terms 0-6: Strassen for Y[:, 0:2].
       Terms 7-10: Schoolbook for Y[:, 2].
       
       U22 (4x7) from Strassen.
       V22 (4x7) from Strassen (but only for first 2 cols of Y).
       W22 (4x7) from Strassen.
       
       For Y (2x3), let Y = [Y0, Y1, Y2].
       Strassen part: X * [Y0, Y1].
       Schoolbook part: X * Y2.
       
       U_strassen = U22 (4x7).
       V_strassen:
          V22 is 4x7 for 2x2.
          We need V for 2x3.
          V_strassen (6x7):
             Rows 0,1 correspond to Y0, Y1 (first 2 cols of Y? No, Y is 2x3, so rows are (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)).
             Wait, V index is j*p + k. For 2x2 x 2x3, p=3? No, the subproblem is 2x2 * 2x3.
             X is 2x2. Y is 2x3.
             X indices: i in 0..1, j in 0..1.
             Y indices: j in 0..1, k in 0..2.
             Z indices: i in 0..1, k in 0..2.
             
             U (4xR): i*2+j.
             V (6xR): j*3+k.
             W (6xR): k*2+i.
             
             Strassen for X * Y[:, 0:2]:
             Y_sub is 2x2.
             V_sub (4x7): j*2+k_sub where k_sub in 0..1.
             Map to V (6x7):
             k_sub=0 -> k=0.
             k_sub=1 -> k=1.
             So V[j*3+0, r] = V22[j*2+0, r]
             V[j*3+1, r] = V22[j*2+1, r]
             V[j*3+2, r] = 0.
             
             W_sub (4x7): k_sub*2+i.
             Map to W (6x7):
             k_sub=0 -> k=0.
             k_sub=1 -> k=1.
             W[0*2+i, r] = W22[0*2+i, r]
             W[1*2+i, r] = W22[1*2+i, r]
             W[2*2+i, r] = 0.
             
             Schoolbook for X * Y[:, 2]:
             Y_col is 2x1.
             Terms:
             1. X00 * Y02 -> Z02
             2. X01 * Y12 -> Z02
             3. X10 * Y02 -> Z12
             4. X11 * Y12 -> Z12
             
             U_school (4x4):
             Col 0: X00 -> [1,0,0,0]
             Col 1: X01 -> [0,1,0,0]
             Col 2: X10 -> [0,0,1,0]
             Col 3: X11 -> [0,0,0,1]
             
             V_school (6x4):
             Col 0: Y02 -> V[0*3+2] = 1 -> index 2.
             Col 1: Y12 -> V[1*3+2] = 1 -> index 5.
             Col 2: Y02 -> index 2.
             Col 3: Y12 -> index 5.
             
             W_school (6x4):
             Col 0: Z02 -> W[2*2+0] = 1 -> index 4.
             Col 1: Z02 -> index 4.
             Col 2: Z12 -> W[2*2+1] = 1 -> index 5.
             Col 3: Z12 -> index 5.
             
             So for 2x2 x 2x3, Rank 11.
             
             Total Rank for 2x4 x 4x6:
             A0 (2x2) * B0 (2x3): Rank 11.
             A1 (2x2) * B1 (2x3): Rank 11.
             Total 22.
             
             This is much better than 42.
             
             Let's implement this Rank 22 decomposition.
             
             U (8x22), V (24x22), W (12x22).
             
             For A0 * B0 (cols 0-10 of U,V,W):
             A0 indices: i*4 + j (j in 0,1).
             B0 indices: j*6 + k (j in 0,1; k in 0,1,2). Note: B is 4x6, so B0 is top 2 rows.
             B0 row jj (0,1) corresponds to global row jj.
             B0 col kk (0,1,2) corresponds to global col kk.
             So V index = jj*6 + kk.
             
             C0 indices: k*2 + i (k in 0,1,2; i in 0,1).
             C0 col kk corresponds to global col kk.
             So W index = kk*2 + i.
             
             For A1 * B1 (cols 11-21 of U,V,W):
             A1 indices: i*4 + (2+jj) (jj in 0,1).
             B1 indices: (2+jj)*6 + kk (jj in 0,1; kk in 0,1,2).
             B1 row jj corresponds to global row 2+jj.
             B1 col kk corresponds to global col kk.
             So V index = (2+jj)*6 + kk.
             
             C1 indices: k*2 + i.
             C1 col kk corresponds to global col kk.
             So W index = kk*2 + i.
             
             Let's build the 2x2 x 2x3 rank-11 factors first.
             U22_3 (4x11), V22_3 (6x11), W22_3 (6x11).
             
             Then map to the full problem.
             
             Verification:
             A (2x4) = [A0, A1].
             B (4x6) = [B0; B1].
             C = A0 B0 + A1 B1.
             
             This seems correct.
             
             Let's code this.
             
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Rank-7 Strassen for 2x2
    U22 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 1, 0, 0, 0],
    ], dtype=float)
    
    V22 = np.array([
        [1, 1, -1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, -1, 1, 0, 1],
    ], dtype=float)
    
    W22 = np.array([
        [1, 0, 1, 0, -1, -1, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, -1, 0, 1, 0, 0, 1],
    ], dtype=float)
    
    # Build Rank-11 decomposition for 2x2 x 2x3
    # X (2x2) * Y (2x3) = Z (2x3)
    # U (4x11), V (6x11), W (6x11)
    
    U223 = np.zeros((4, 11), dtype=float)
    V223 = np.zeros((6, 11), dtype=float)
    W223 = np.zeros((6, 11), dtype=float)
    
    # Part 1: Strassen for Y[:, 0:2] (cols 0,1)
    # U223[:, 0:7] = U22
    U223[:, 0:7] = U22
    
    # V223: Map V22 (4x7) to V223 (6x7)
    # V22 index j*2+k_sub -> V223 index j*3+k_sub
    for r in range(7):
        for j in range(2):
            for k_sub in range(2):
                V223[j*3 + k_sub, r] = V22[j*2 + k_sub, r]
        # k_sub=2 is 0
    
    # W223: Map W22 (4x7) to W223 (6x7)
    # W22 index k_sub*2+i -> W223 index k_sub*2+i
    for r in range(7):
        for k_sub in range(2):
            for i in range(2):
                W223[k_sub*2 + i, r] = W22[k_sub*2 + i, r]
        # k_sub=2 is 0
    
    # Part 2: Schoolbook for Y[:, 2] (col 2)
    # Terms 7-10
    # 1. X00 * Y02 -> Z02
    # 2. X01 * Y12 -> Z02
    # 3. X10 * Y02 -> Z12
    # 4. X11 * Y12 -> Z12
    
    # U223
    U223[0, 7] = 1  # X00
    U223[1, 8] = 1  # X01
    U223[2, 9] = 1  # X10
    U223[3, 10] = 1 # X11
    
    # V223
    # Y02 is index 0*3+2 = 2
    # Y12 is index 1*3+2 = 5
    V223[2, 7] = 1
    V223[5, 8] = 1
    V223[2, 9] = 1
    V223[5, 10] = 1
    
    # W223
    # Z02 is index 2*2+0 = 4
    # Z12 is index 2*2+1 = 5
    W223[4, 7] = 1
    W223[4, 8] = 1
    W223[5, 9] = 1
    W223[5, 10] = 1
    
    # Verify 2x2 x 2x3 decomposition
    X_test = np.random.randn(2, 2)
    Y_test = np.random.randn(2, 3)
    Z_expected = X_test @ Y_test
    
    Z_computed = np.zeros((2, 3), dtype=float)
    for r in range(11):
        u_val = sum(U223[i*2+j, r] * X_test[i, j] for i in range(2) for j in range(2))
        v_val = sum(V223[j*3+k, r] * Y_test[j, k] for j in range(2) for k in range(3))
        for k in range(3):
            for i in range(2):
                Z_computed[i, k] += W223[k*2+i, r] * u_val * v_val
    
    if not np.allclose(Z_computed, Z_expected, atol=1e-10):
        # Fallback to schoolbook rank 48
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build full 2x4 x 4x6 decomposition
    # Rank = 11 (for A0*B0) + 11 (for A1*B1) = 22
    rank = 22
    U = np.zeros((n * m, rank), dtype=float)  # 8 x 22
    V = np.zeros((m * p, rank), dtype=float)  # 24 x 22
    W = np.zeros((p * n, rank), dtype=float)  # 12 x 22
    
    # A0 * B0 (cols 0-10)
    # A0: A[i, jj] for i=0,1; jj=0,1. Index i*4+jj.
    # B0: B[jj, kk] for jj=0,1; kk=0,1,2. Index jj*6+kk.
    # C0: C[i, kk] for i=0,1; kk=0,1,2. Index kk*2+i.
    
    for r in range(11):
        col = r
        # U
        for i in range(2):
            for jj in range(2):
                U[i * 4 + jj, col] = U223[i*2+jj, r]
        
        # V
        for jj in range(2):
            for kk in range(3):
                V[jj * 6 + kk, col] = V223[jj*3+kk, r]
        
        # W
        for kk in range(3):
            for i in range(2):
                W[kk * 2 + i, col] = W223[kk*2+i, r]
    
    # A1 * B1 (cols 11-21)
    # A1: A[i, 2+jj] for i=0,1; jj=0,1. Index i*4+(2+jj).
    # B1: B[2+jj, kk] for jj=0,1; kk=0,1,2. Index (2+jj)*6+kk.
    # C1: C[i, kk] for i=0,1; kk=0,1,2. Index kk*2+i.
    
    for r in range(11):
        col = 11 + r
        # U
        for i in range(2):
            for jj in range(2):
                U[i * 4 + (2 + jj), col] = U223[i*2+jj, r]
        
        # V
        for jj in range(2):
            for kk in range(3):
                V[(2 + jj) * 6 + kk, col] = V223[jj*3+kk, r]
        
        # W
        for kk in range(3):
            for i in range(2):
                W[kk * 2 + i, col] = W223[kk*2+i, r]
    
    # Verify full decomposition
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p), dtype=float)
    for r in range(rank):
        u_val = sum(U[idx, r] * A_test[idx // m, idx % m] for idx in range(n * m))
        v_val = sum(V[idx, r] * B_test[idx // p, idx % p] for idx in range(m * p))
        for idx in range(p * n):
            k = idx // n
            i = idx % n
            C_computed[i, k] += W[idx, r] * u_val * v_val
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
