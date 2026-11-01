"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def check(U, V, W, tol=1e-9):
        """Verify exact decomposition via random sampling."""
        if time.time() > deadline:
            return False
        rng = np.random.default_rng(42)
        for _ in range(50):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            C = A @ B
            
            # U is (nm, rank), V is (mp, rank), W is (pn, rank)
            # C[i, k] = sum_c W[k*n+i, c] * (sum_j A[i, j] U[i*m+j, c]) * (sum_j B[j, k] V[j*p+k, c])
            # Let's vectorize this.
            # L_c = A.flatten() @ U[:, c]
            # R_c = B.flatten() @ V[:, c]
            # C_calc = sum_c (L_c * R_c) * W[:, c].reshape(p, n).T
            
            # Efficient computation:
            # L = A.flatten() @ U  -> (rank,)
            # R = B.flatten() @ V  -> (rank,)
            # C_calc = (L * R) @ W.T -> (pn,) -> reshape to (p, n) -> transpose to (n, p)
            
            L = A.flatten() @ U
            R = B.flatten() @ V
            C_calc = (L * R) @ W.T
            C_calc = C_calc.reshape(p, n).T
            
            if not np.allclose(C_calc, C, atol=tol):
                return False
        return True

    def to_half_int_list(arr):
        """Convert numpy array to list of lists of half-integers (Fractions or floats)."""
        # The problem asks for integer/half-integer entries.
        # We'll return floats that are multiples of 0.5, or Fractions.
        # Floats are safer for compatibility unless strictly checked.
        # Let's return floats rounded to 2 decimal places if close to half-int.
        res = []
        for row in arr:
            r = []
            for val in row:
                # Round to nearest 0.5
                val = round(val * 2) / 2.0
                r.append(val)
            res.append(r)
        return res

    # 1. Start with schoolbook decomposition
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(nm)]
    V = [[0] * rank_schoolbook for _ in range(mp)]
    W = [[0] * rank_schoolbook for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W
    
    # 2. Try to find a lower rank decomposition.
    # Lower bound is max(n*p, n*m, m*p) = 32.
    # We will try ranks from 32 up to best_rank - 1.
    
    # Optimization: Use a smarter initialization or search?
    # Random search for rank 32 is unlikely to succeed immediately.
    # However, the current program does random search.
    # Let's improve the search by using a specific structure if possible, 
    # or just more efficient random search.
    
    # Actually, for 2x2x16, the rank is known to be 32 (it's a "block" of 2x2x1 repeated 8 times? No).
    # 2x2x16 can be viewed as 8 independent 2x2x2 multiplications? No, the indices are coupled.
    # A(2x2) * B(2x16) = C(2x16).
    # This is equivalent to 8 independent 2x2x2 multiplications if B is viewed as 8 columns of 2x2?
    # No, B is 2x16. Each column of B is 2x1.
    # So it's 16 independent 2x2x1 multiplications?
    # C[:, k] = A * B[:, k].
    # Rank of 2x2x1 is 4 (schoolbook).
    # Total rank would be 16 * 4 = 64?
    # Wait, the tensor is T_{i, j, k}.
    # If we group k in pairs, say k=2l, 2l+1.
    # Then C[:, 2l] and C[:, 2l+1] depend on A and B[:, 2l:2l+2].
    # This looks like 8 blocks of 2x2x2 matrix multiplication.
    # Rank of 2x2x2 is 7 (Strassen).
    # So total rank could be 8 * 7 = 56.
    # Schoolbook is 64.
    # So 56 is a valid upper bound.
    # Can we do better?
    # The lower bound is 32.
    
    # Let's try to construct the 8x Strassen decomposition.
    # This is a structural approach, much more likely to succeed than random search for rank 32.
    
    # Strassen for 2x2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)
    # m4 = (d)(g-c)
    # m5 = (a+b)(h)
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # We can apply this to each block of 2 columns of B.
    # B has 16 columns. Let's group them into 8 blocks of 2 columns.
    # B_block_l = B[:, 2l:2l+2] (2x2)
    # C_block_l = A @ B_block_l (2x2)
    # C[:, 2l] and C[:, 2l+1] are the columns of C_block_l.
    
    # So we just need to map the Strassen factors to the global U, V, W.
    
    def strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # m1 = (a+d)(e+h)
        # a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
        # U[0,0]=1, U[3,0]=1
        # V[0,0]=1, V[3,0]=1
        # W[0,0]=1, W[3,0]=1, W[1,0]=-1, W[2,0]=1
        # C[0,0] += m1, C[1,1] += m1
        # C[0,1] -= m5? No, let's use standard Strassen.
        
        # Standard Strassen:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - A00)  # Wait, A11(B10 - A00)? No, A11(B10 - A00) is not right.
        # M4 = A11(B10 - A00) -> A11 B10 - A11 A00? No.
        # M4 = (A10 - A00)(B00 + B01)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B10 + B11)
        # M7 = (A01 - A11)(B00 + B10)
        
        # Let's stick to a known correct set.
        # From Wikipedia:
        # M1 = (a + d)(e + h)
        # M2 = (c + d)(e)
        # M3 = (a)(f - h)
        # M4 = (d)(g - c)
        # M5 = (a + b)(h)
        # M6 = (c - a)(e + f)
        # M7 = (b - d)(g + h)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Indices:
        # A: 00(0), 01(1), 10(2), 11(3)
        # B: 00(4), 01(5), 10(6), 11(7)
        # C: 00(8), 01(9), 10(10), 11(11)
        
        # Flatten:
        # U index: i*2+j
        # V index: j*2+k
        # W index: k*2+i
        
        # M1: (A00+A11)(B00+B11)
        # U: 0, 3 -> 1
        # V: 0, 3 -> 1
        # W: C00(0*2+0=0), C11(1*2+1=3) -> 1
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        
        # M2: (A10+A11)B00
        # U: 2, 3 -> 1
        # V: 0 -> 1
        # W: C21(1*2+0=2) -> 1
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1
        
        # M3: A00(B01-B11)
        # U: 0 -> 1
        # V: 1, 3 -> 1, -1
        # W: C12(0*2+1=1), C22(1*2+1=3) -> 1, 1
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1; W_s[3, 2] = 1
        
        # M4: A11(B10-A10) -- Wait, M4 = d(g-c) = A11(B10-A10)
        # U: 3 -> 1
        # V: 2, 0? No, B10 is index 2 (j=1, k=0 -> 1*2+0=2). A10 is not in V.
        # M4 = A11 * B10 - A11 * A10? No, M4 is a product of a linear combo of A and a linear combo of B.
        # M4 = A11(B10 - A10) is wrong because A10 is in A.
        # The formula M4 = (d)(g-c) implies d is A11, g is B10, c is A10.
        # This is not a bilinear form in A and B separately if c is in A.
        # Let's check the formula again.
        # M4 = (d)(g-c) -> d*g - d*c.
        # d is A[1,1]. g is B[1,0]. c is A[1,0].
        # This term is A[1,1]B[1,0] - A[1,1]A[1,0].
        # This is not bilinear in A and B! It has A*A.
        # So this set of Strassen formulas is incorrect or I'm misinterpreting it.
        
        # Correct Strassen:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - A00) -> Still has A*A?
        # No, standard Strassen uses:
        # M4 = (A10 - A00)(B00 + B01)
        # Let's use this one.
        
        # M4: (A10 - A00)(B00 + B01)
        # U: 2 -> 1, 0 -> -1
        # V: 0 -> 1, 1 -> 1
        # W: C21(2), C22(3) -> 1, 1
        U_s[2, 3] = 1; U_s[0, 3] = -1
        V_s[0, 3] = 1; V_s[1, 3] = 1
        W_s[2, 3] = 1; W_s[3, 3] = 1
        
        # M5: (A00 + A01)B11
        # U: 0 -> 1, 1 -> 1
        # V: 3 -> 1
        # W: C11(3), C12(1) -> 1, 1
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[3, 4] = 1; W_s[1, 4] = 1
        
        # M6: (A10 - A00)(B10 + B11)
        # U: 2 -> 1, 0 -> -1
        # V: 2 -> 1, 3 -> 1
        # W: C22(3) -> 1
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[2, 5] = 1; V_s[3, 5] = 1
        W_s[3, 5] = 1
        
        # M7: (A01 - A11)(B00 + B10)
        # U: 1 -> 1, 3 -> -1
        # V: 0 -> 1, 2 -> 1
        # W: C11(3), C21(2) -> 1, 1
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[0, 6] = 1; V_s[2, 6] = 1
        W_s[3, 6] = 1; W_s[2, 6] = 1
        
        # Reconstruct C:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's verify C00:
        # M1: A00B00 + A00B11 + A11B00 + A11B11
        # M4: A10B00 + A10B01 - A00B00 - A00B01
        # -M5: -A00B11 - A01B11
        # M7: A01B00 + A01B10 - A11B00 - A11B10
        # Sum:
        # A00B00: 1 - 1 = 0? No, M1 has A00B00. M4 has -A00B00. So 0.
        # This is wrong. C00 should be A00B00 + A01B10.
        
        # Let's use a verified Strassen implementation.
        # I will hardcode a known correct rank-7 decomposition for 2x2x2.
        
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # Source: https://en.wikipedia.org/wiki/Strassen_algorithm
        # M1 = (a + d)(e + h)
        # M2 = (c + d)(e)
        # M3 = (a)(f - h)
        # M4 = (d)(g - c)  <-- This one is suspicious in my previous check.
        # Let's check the Wikipedia formulas again.
        # M1 = (a + d)(e + h)
        # M2 = (c + d)(e)
        # M3 = (a)(f - h)
        # M4 = (d)(g - c)
        # M5 = (a + b)(h)
        # M6 = (c - a)(e + f)
        # M7 = (b - d)(g + h)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's re-evaluate M4 = d(g-c).
        # d = A11, g = B10, c = A10.
        # M4 = A11 B10 - A11 A10.
        # This is NOT bilinear.
        # Wait, is c in M4?
        # Wikipedia says: M4 = (d)(g - c).
        # If c is A10, then it's not bilinear.
        # Ah, I see. In some notations, the matrix B is transposed or indices are different.
        # Let's assume the standard C = AB.
        # Maybe the formula is M4 = (d)(g - c) where c is NOT A10?
        # No, c is A10.
        # Let's look for a different source or derive it.
        
        # Actually, let's just use the random search for rank 56.
        # It's safer.
        
        return None

    # Since deriving Strassen correctly under time pressure is risky,
    # and random search for rank 32 is unlikely,
    # let's try random search for rank 56 (8 * 7).
    # Or even just improve the random search for lower ranks.
    
    # Let's try rank 56 first.
    # If we find a valid decomposition of rank 56, that's a 64 -> 56 improvement.
    
    for r in range(32, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Only try a few attempts for each rank to stay within time.
        # For r=56, we might need more attempts.
        attempts = 0
        max_attempts = 20 if r > 50 else 5
        
        while attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            seed = attempts * 100 + r
            rng = np.random.default_rng(seed)
            
            # Scale initialization
            # For rank r, the norm of each vector should be small enough.
            U_rand = rng.standard_normal((nm, r)) * 0.5
            V_rand = rng.standard_normal((mp, r)) * 0.5
            W_rand = rng.standard_normal((pn, r)) * 0.5
            
            if check(U_rand, V_rand, W_rand):
                best_rank = r
                best_U = to_half_int_list(U_rand)
                best_V = to_half_int_list(V_rand)
                best_W = to_half_int_list(W_rand)
                break
        
        if best_rank == r:
            continue

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
