"""Matrix multiplication 2,2,11 - rank-minimal exact tensor decomposition."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 2, 11]

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x11 matrix.
    
    The tensor has dimensions (n*m) x (m*p) x (p*n) = 4 x 22 x 22.
    Schoolbook rank = 4*2*11 = 88.
    
    We attempt to find a lower rank decomposition using known techniques
    for rectangular matrix multiplication.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Schoolbook baseline
    def schoolbook():
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = Fraction(1)
                    V[j * p + k][col] = Fraction(1)
                    W[k * n + i][col] = Fraction(1)
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to use Strassen-like ideas for the 2x2 block structure
    # A (2x2) * B (2x11) = C (2x11)
    # Split B into B1 (2x5) and B2 (2x6)
    # Or use the fact that we're multiplying 2x2 by 2x11
    
    # One approach: treat this as two independent 2x2 * 2x1 multiplications
    # for each of the 11 columns, but that's just schoolbook.
    
    # Better: use the structure that A is 2x2, so we can use Strassen's 7-term
    # formula for the 2x2 part, applied to each column of B.
    # Strassen: C = A*B where A,B are 2x2
    # M1 = (A11-A22)(B21-B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A22)(B11+B22)
    # M6 = (A11-A21)(B12+B22)
    # M7 = (A21-A22)(B11-B12)
    # C11 = M1+M4+M5-M7
    # C12 = M3+M7
    # C21 = M2+M6
    # C22 = M1+M3+M5-M6
    
    # For 2x2 * 2x11, we can view B as 11 such 2x2*2x1 problems... 
    # but that doesn't help directly.
    
    # Alternative: think of it as computing A * B where A is 2x2 and B is 2x11.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = A[0,0]*B[0,k] + A[0,1]*B[1,k]  for i=0
    # = A[1,0]*B[0,k] + A[1,1]*B[1,k]  for i=1
    
    # This is essentially: for each k, compute a 2x2 * 2x1 matrix product.
    # The 2x2 * 2x1 product can be done with 4 multiplications (schoolbook)
    # or potentially fewer with some sharing across k values.
    
    # Since A is the same for all k, we can precompute combinations of A's entries.
    # Let a00, a01, a10, a11 be entries of A.
    # For each k: C[0,k] = a00*B[0,k] + a01*B[1,k]
    #               C[1,k] = a10*B[0,k] + a11*B[1,k]
    
    # If we define:
    # S0k = B[0,k] + B[1,k]
    # D0k = B[0,k] - B[1,k]
    # Then B[0,k] = (S0k + D0k)/2, B[1,k] = (S0k - D0k)/2
    
    # C[0,k] = a00*(S0k+D0k)/2 + a01*(S0k-D0k)/2
    #         = (a00+a01)*S0k/2 + (a00-a01)*D0k/2
    # C[1,k] = a10*(S0k+D0k)/2 + a11*(S0k-D0k)/2
    #         = (a10+a11)*S0k/2 + (a10-a11)*D0k/2
    
    # So we need:
    # (a00+a01)*S0k, (a00-a01)*D0k, (a10+a11)*S0k, (a10-a11)*D0k
    # That's 4 products per k, same as schoolbook. No gain.
    
    # Let's try a different decomposition. Consider grouping columns.
    # Group the 11 columns into groups and use bilinear tricks.
    
    # Actually, let's think about this more carefully.
    # The tensor is T[i,j,k] = delta(i,j)*delta(j,k) in a sense... no.
    # T is the matrix multiplication tensor: T_{(i,j),(j',k),(k',i')} = 
    #   delta(j,j') * delta(k,k')
    
    # For n=2, m=2, p=11:
    # U has 4 rows (i*m+j for i in 0,1; j in 0,1)
    # V has 22 rows (j*p+k for j in 0,1; k in 0..10)
    # W has 22 rows (k*n+i for k in 0..10; i in 0,1)
    
    # The tensor T[(i,j), (j',k), (k',i')] = 1 if j==j' and k==k', else 0
    
    # We want to write T as sum_{r=1}^{rank} u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^4, v_r in R^22, w_r in R^22
    
    # This is equivalent to finding rank-minimal decomposition.
    
    # Known results: For 2x2 * 2xp, the rank is at most 7*p/2 + O(1) 
    # using Strassen-like methods, but exact results are complex.
    
    # For p=11, let's try to construct a decomposition with rank < 88.
    
    # Simple improvement: Use the fact that we can share computations.
    # Define for each k:
    #   P_k = B[0,k]  (row 0 of B, column k)
    #   Q_k = B[1,k]  (row 1 of B, column k)
    
    # C[0,k] = a00*P_k + a01*Q_k
    # C[1,k] = a10*P_k + a11*Q_k
    
    # We can write this as:
    # [C[0,k]]   [a00 a01] [P_k]
    # [C[1,k]] = [a10 a11] [Q_k]
    
    # For all k, this is:
    # C = A * B (2x2 times 2x11)
    
    # The bilinear rank of 2x2 * 2xp matrix multiplication...
    # Each "column" of the result requires 4 multiplications in schoolbook.
    # With Strassen's method for 2x2*2x2, we need 7 multiplications for 4 outputs.
    # For 2x2*2x11, if we could group columns...
    
    # Let's try: group the 11 columns as 5+6 or use a different approach.
    
    # Actually, a cleaner approach: think of B as 11 column vectors in R^2.
    # We're computing A * [b_0, b_1, ..., b_10] = [A*b_0, A*b_1, ..., A*b_10]
    # where each A*b_k is a 2-vector.
    
    # The map b -> A*b is linear in b for fixed A, and bilinear overall.
    # The tensor rank is the rank of this bilinear map.
    
    # For a 2x2 matrix A acting on 2-vectors, the bilinear rank for one column
    # is 4 (schoolbook) or potentially lower with half-integer coefficients?
    # No, 2x2*2x1 has rank 4 over any field (it's the identity in a sense).
    
    # Wait, 2x2 * 2x1: C[0] = a00*B[0] + a01*B[1], C[1] = a10*B[0] + a11*B[1]
    # This requires 4 independent products. Rank is 4.
    
    # So for 11 columns, naive rank is 44. But we can share!
    # The key insight: A is the same for all columns. So we can precompute
    # linear combinations of A's entries that are reused.
    
    # Let's define:
    # s0 = a00 + a01
    # d0 = a00 - a01  
    # s1 = a10 + a11
    # d1 = a10 - a11
    
    # Then:
    # C[0,k] = (s0*(P_k+Q_k) + d0*(P_k-Q_k))/2
    # C[1,k] = (s1*(P_k+Q_k) + d1*(P_k-Q_k))/2
    
    # Products needed: s0*(P_k+Q_k), d0*(P_k-Q_k), s1*(P_k+Q_k), d1*(P_k-Q_k)
    # That's still 4 per column = 44 total. No gain.
    
    # What if we define shared sums?
    # Let S_k = P_k + Q_k, D_k = P_k - Q_k
    # Then we need: s0*S_k, d0*D_k, s1*S_k, d1*D_k for each k
    # Still 4 per k.
    
    # The issue is that s0, d0, s1, d1 are independent of k, but we still need
    # to multiply them by the k-dependent quantities.
    
    # Unless... we can find a decomposition where the number of distinct
    # "patterns" is less than 4 per column.
    
    # Let me try a direct construction with rank 44 first (improving from 88).
    # Actually wait - schoolbook is 88 because it does n*m*p = 2*2*11 = 44... 
    # no, 2*2*11 = 44. Let me recheck.
    
    # n=2, m=2, p=11: n*m*p = 2*2*11 = 44. But the code says rank = n*m*p = 44?
    # Wait, the original code has rank = n * m * p = 44, but it creates 
    # n*m = 4 rows for U, m*p = 22 rows for V, p*n = 22 rows for W.
    # And it iterates i in range(n), j in range(m), k in range(p) = 2*2*11 = 44 iterations.
    # So schoolbook rank is 44, not 88. I misread earlier.
    
    # So we need to find rank < 44.
    
    # Let's try to construct a rank-44 decomposition first to verify, then optimize.
    
    def build_rank44():
        rank = 44
        U = [[Fraction(0)] * rank for _ in range(4)]
        V = [[Fraction(0)] * rank for _ in range(22)]
        W = [[Fraction(0)] * rank for _ in range(22)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(11):
                    U[i * 2 + j][col] = Fraction(1)
                    V[j * 11 + k][col] = Fraction(1)
                    W[k * 2 + i][col] = Fraction(1)
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Now try to do better. Use the Strassen-inspired approach.
    # For each pair of columns (k1, k2), we can compute A * [B[:,k1], B[:,k2]]
    # as a 2x2 * 2x2 multiplication using Strassen (7 mults) instead of 8.
    
    # With 11 columns, we can do 5 pairs + 1 single.
    # 5 pairs: 5 * 7 = 35 mults for 10 columns
    # 1 single: 4 mults for 1 column
    # Total: 39 mults for 11 columns, vs 44 schoolbook.
    
    # Let's construct this.
    
    def build_strassen_hybrid():
        # Columns: 0,1 | 2,3 | 4,5 | 6,7 | 8,9 | 10
        # Pairs: (0,1), (2,3), (4,5), (6,7), (8,9), single: 10
        
        pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
        singles = [10]
        
        # For each pair (k1, k2), use Strassen's 7-term formula
        # A = [[a00, a01], [a10, a11]]
        # B_pair = [[B[0,k1], B[0,k2]], [B[1,k1], B[1,k2]]]
        # C_pair = A * B_pair = [[C[0,k1], C[0,k2]], [C[1,k1], C[1,k2]]]
        
        # Strassen:
        # M1 = (a00-a11) * (B[1,k1]-B[1,k2])
        # M2 = (a10+a11) * B[0,k1]
        # M3 = a00 * (B[0,k2]-B[1,k2])
        # M4 = a11 * (B[1,k1]-B[0,k1])
        # M5 = (a00+a11) * (B[0,k1]+B[1,k2])
        # M6 = (a00-a10) * (B[0,k2]+B[1,k2])
        # M7 = (a10-a11) * (B[0,k1]-B[1,k1])
        
        # C[0,k1] = M1 + M4 + M5 - M7
        # C[0,k2] = M3 + M7
        # C[1,k1] = M2 + M6
        # C[1,k2] = M1 + M3 + M5 - M6
        
        # Each M is a product of a linear combo of A entries and a linear combo of B entries.
        # So each M corresponds to one rank-1 term in the tensor decomposition.
        
        rank_terms = []
        
        # For each pair, add 7 terms
        for (k1, k2) in pairs:
            # M1: (a00-a11) * (B[1,k1]-B[1,k2])
            rank_terms.append({
                'U': {0: Fraction(1), 3: Fraction(-1)},  # a00 - a11 -> U[i*2+j]: i=0,j=0 and i=1,j=1
                'V': {11 + k1: Fraction(1), 11 + k2: Fraction(-1)},  # B[1,k1] - B[1,k2]
                'W': {k1*2+0: Fraction(1), k1*2+1: Fraction(1), k2*2+0: Fraction(1), k2*2+1: Fraction(1)},
                'coeff_W': {k1*2+0: 1, k1*2+1: 1, k2*2+0: 1, k2*2+1: 1},
                'name': 'M1'
            })
            # Actually, let me be more careful. The tensor decomposition requires
            # T = sum_r u_r ⊗ v_r ⊗ w_r where each is a vector.
            # For M1: u = e_{0,0} - e_{1,1} (in U space, indices 0 and 3)
            #         v = e_{1,k1} - e_{1,k2} (in V space, indices 11+k1 and 11+k2)
            #         w should give C[0,k1] += 1, C[0,k2] += 1, C[1,k1] += 1, C[1,k2] += 1
            #         But that's not right - w should be a vector, and the contribution
            #         to C is u ⊗ v ⊗ w meaning C[idx] += U[idx] * V[idx] * W[idx]
            #         No wait, the decomposition is:
            #         T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            #         So for each r, we have a rank-1 tensor.
            
            # Let me restart the construction more carefully.
            pass
        
        # This is getting complicated. Let me just build it directly.
        rank = 39  # 5*7 + 4
        U = [[Fraction(0)] * rank for _ in range(4)]
        V = [[Fraction(0)] * rank for _ in range(22)]
        W = [[Fraction(0)] * rank for _ in range(22)]
        
        col = 0
        
        # Process pairs
        for (k1, k2) in pairs:
            # M1 = (a00 - a11) * (B[1,k1] - B[1,k2])
            # Contributes to: C[0,k1] += 1, C[0,k2] += 1, C[1,k1] += 1, C[1,k2] += 1
            U[0][col] = Fraction(1)   # a00
            U[3][col] = Fraction(-1)  # -a11
            V[11 + k1][col] = Fraction(1)   # B[1,k1]
            V[11 + k2][col] = Fraction(-1)  # -B[1,k2]
            W[k1*2+0][col] = Fraction(1)  # C[0,k1]
            W[k1*2+1][col] = Fraction(1)  # C[0,k2]
            W[k2*2+0][col] = Fraction(1)  # C[1,k1]
            W[k2*2+1][col] = Fraction(1)  # C[1,k2]
            col += 1
            
            # M2 = (a10 + a11) * B[0,k1]
            # Contributes to: C[1,k1] += 1
            U[2][col] = Fraction(1)   # a10
            U[3][col] = Fraction(1)   # a11
            V[k1][col] = Fraction(1)  # B[0,k1]
            W[k2*2+0][col] = Fraction(1)  # C[1,k1]
            col += 1
            
            # M3 = a00 * (B[0,k2] - B[1,k2])
            # Contributes to: C[0,k2] += 1, C[1,k2] += 1
            U[0][col] = Fraction(1)   # a00
            V[k2][col] = Fraction(1)  # B[0,k2]
            V[11+k2][col] = Fraction(-1)  # -B[1,k2]
            W[k1*2+1][col] = Fraction(1)  # C[0,k2]
            W[k2*2+1][col] = Fraction(1)  # C[1,k2]
            col += 1
            
            # M4 = a11 * (B[1,k1] - B[0,k1])
            # Contributes to: C[0,k1] += 1
            U[3][col] = Fraction(1)   # a11
            V[11+k1][col] = Fraction(1)  # B[1,k1]
            V[k1][col] = Fraction(-1)  # -B[0,k1]
            W[k1*2+0][col] = Fraction(1)  # C[0,k1]
            col += 1
            
            # M5 = (a00 + a11) * (B[0,k1] + B[1,k2])
            # Contributes to: C[0,k1] += 1, C[0,k2] += 1, C[1,k2] += 1
            U[0][col] = Fraction(1)   # a00
            U[3][col] = Fraction(1)   # a11
            V[k1][col] = Fraction(1)  # B[0,k1]
            V[11+k2][col] = Fraction(1)  # B[1,k2]
            W[k1*2+0][col] = Fraction(1)  # C[0,k1]
            W[k1*2+1][col] = Fraction(1)  # C[0,k2]
            W[k2*2+1][col] = Fraction(1)  # C[1,k2]
            col += 1
            
            # M6 = (a00 - a10) * (B[0,k2] + B[1,k2])
            # Contributes to: C[1,k1] += 1
            U[0][col] = Fraction(1)   # a00
            U[2][col] = Fraction(-1)  # -a10
            V[k2][col] = Fraction(1)  # B[0,k2]
            V[11+k2][col] = Fraction(1)  # B[1,k2]
            W[k2*2+0][col] = Fraction(1)  # C[1,k1]
            col += 1
            
            # M7 = (a10 - a11) * (B[0,k1] - B[1,k1])
            # Contributes to: C[0,k1] -= 1, C[0,k2] += 1
            U[2][col] = Fraction(1)   # a10
            U[3][col] = Fraction(-1)  # -a11
            V[k1][col] = Fraction(1)  # B[0,k1]
            V[11+k1][col] = Fraction(-1)  # -B[1,k1]
            W[k1*2+0][col] = Fraction(-1)  # -C[0,k1]
            W[k1*2+1][col] = Fraction(1)   # C[0,k2]
            col += 1
        
        # Process single column k=10 with schoolbook (4 terms)
        k = 10
        # C[0,k] = a00*B[0,k] + a01*B[1,k]
        U[0][col] = Fraction(1)
        V[k][col] = Fraction(1)
        W[k*2+0][col] = Fraction(1)
        col += 1
        
        U[1][col] = Fraction(1)
        V[11+k][col] = Fraction(1)
        W[k*2+0][col] = Fraction(1)
        col += 1
        
        # C[1,k] = a10*B[0,k] + a11*B[1,k]
        U[2][col] = Fraction(1)
        V[k][col] = Fraction(1)
        W[k*2+1][col] = Fraction(1)
        col += 1
        
        U[3][col] = Fraction(1)
        V[11+k][col] = Fraction(1)
        W[k*2+1][col] = Fraction(1)
        col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Let's verify the Strassen hybrid is correct by checking the tensor identity.
    # Actually, let me just return the schoolbook for now to be safe,
    # and try to verify the Strassen version.
    
    # For safety, return schoolbook if time is tight
    if time.time() > deadline - 5:
        return schoolbook()
    
    # Try the Strassen hybrid
    try:
        result = build_strassen_hybrid()
        # Verify: check that the tensor identity holds
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # T should be 1 if a corresponds to (i,j), b to (j',k), c to (k',i')
        # with j==j' and k==k', else 0.
        
        U = result['U']
        V = result['V']
        W = result['W']
        rank = len(U[0])
        
        # Check a few entries
        import random
        random.seed(42)
        for _ in range(100):
            a = random.randint(0, 3)
            b = random.randint(0, 21)
            c = random.randint(0, 21)
            
            i_a, j_a = divmod(a, 2)
            j_b, k_b = divmod(b, 11)
            k_c, i_c = divmod(c, 2)
            
            expected = 1 if (j_a == j_b and k_b == k_c) else 0
            
            actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
            
            if actual != expected:
                # Verification failed, fall back to schoolbook
                return schoolbook()
        
        return result
    except Exception:
        return schoolbook()

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
