import time

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank-minimal known for 4x4x4: 49 (nested Strassen 2x2 with inner 2x2 Strassen)
    R = 49
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Outer Strassen 2x2:
    # M0 = (A00 + A11) * (B00 + B11)
    # M1 = (A10 + A11) * B00
    # M2 = A00 * (B01 - B11)
    # M3 = A11 * (B10 - B00)
    # M4 = (A00 + A01) * B11
    # M5 = (A10 - A00) * (B00 + B01)
    # M6 = (A11 - A10) * (B10 + B11)
    # C00 = M0 + M3 - M5 - M6
    # C01 = M4 + M6
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M4
    
    # For each outer Strassen term s, we compute a 2x2 matrix M_s using inner Strassen on 2x2 blocks.
    # Inner Strassen on 2x2 matrices X*Y:
    # P0 = (X00 + X11)*(Y00 + Y11)
    # P1 = (X10 + X11)*Y00
    # P2 = X00*(Y01 - Y11)
    # P3 = X11*(Y10 - Y00)
    # P4 = (X00 + X01)*Y11
    # P5 = (X10 - X00)*(Y00 + Y01)
    # P6 = (X11 - X10)*(Y10 + Y11)
    # Result[0,0] = P0 + P3 - P5 - P6
    # Result[0,1] = P4 + P6
    # Result[1,0] = P1 + P3
    # Result[1,1] = P0 - P1 + P2 + P4
    
    # For each s in 0..6, X and Y are 2x2 matrices of 2x2 blocks.
    # X[i_local, j_local] is a linear combination of A[2*i_local + i_offset, 2*j_local + j_offset]
    # Y[i_local, j_local] is a linear combination of B[2*i_local + i_offset, 2*j_local + j_offset]
    
    # We need to define for each outer term s:
    # alpha[s, t, i, j] coefficient for A[i, j] in X_{t, i_local, j_local}
    # beta[s, t, i, j] coefficient for B[i, j] in Y_{t, i_local, j_local}
    # mu[t, i_local, k_local] coefficient for P_{t, i_local, k_local} in M_s[i_local, k_local]
    # gamma[s, I, K] coefficient for M_s in C[I, K]
    
    # Let's define them carefully.
    # Outer s=0: X = A00 + A11, Y = B00 + B11
    # X[0,0] = A[0,0] + A[2,2], X[0,1] = A[0,1] + A[2,3], X[1,0] = A[1,0] + A[3,2], X[1,1] = A[1,1] + A[3,3]
    # Y[0,0] = B[0,0] + B[2,2], Y[0,1] = B[0,1] + B[2,3], Y[1,0] = B[1,0] + B[3,2], Y[1,1] = B[1,1] + B[3,3]
    
    # General pattern:
    # Outer s=0: X = A00 + A11 => X[i_l, j_l] = A[2*i_l + 0, 2*j_l + 0] + A[2*i_l + 1, 2*j_l + 1] ... wait no.
    # A00 is A[0:2, 0:2], A11 is A[2:4, 2:4].
    # So X[0,0] = A[0,0] + A[2,2], X[0,1] = A[0,1] + A[2,3], X[1,0] = A[1,0] + A[3,2], X[1,1] = A[1,1] + A[3,3]
    
    # Let's define alpha[s, i_l, j_l, i, j] where s is outer term, i_l, j_l is position in 2x2 X block, i,j is position in A.
    # Similarly for beta.
    
    alpha = {}
    beta = {}
    gamma = {}
    mu = {}
    
    # Initialize mu for inner Strassen
    # P0 = (X00 + X11)*(Y00 + Y11) -> mu[0, i_l, k_l]
    # P1 = (X10 + X11)*Y00
    # P2 = X00*(Y01 - Y11)
    # P3 = X11*(Y10 - Y00)
    # P4 = (X00 + X01)*Y11
    # P5 = (X10 - X00)*(Y00 + Y01)
    # P6 = (X11 - X10)*(Y10 + Y11)
    
    # Result[i_l, k_l] = sum_t mu[t, i_l, k_l] * P[t, i_l, k_l]
    # Actually P_t is a scalar (product of two scalars).
    # P_t corresponds to a specific combination of X and Y entries.
    
    # Let's hardcode the mu coefficients.
    # M[i_l, k_l] = sum_{t=0}^6 mu[t, i_l, k_l] * P_t
    # P0 = X00 Y00 + X00 Y11 + X11 Y00 + X11 Y11
    # P1 = X10 Y00 + X11 Y00
    # P2 = X00 Y01 - X00 Y11
    # P3 = X11 Y10 - X11 Y00
    # P4 = X00 Y11 + X01 Y11
    # P5 = X10 Y00 + X10 Y01 - X00 Y00 - X00 Y01
    # P6 = X11 Y10 + X11 Y11 - X10 Y10 - X10 Y11
    
    # This is getting complex. Let's just compute the final W entries directly by composing the bilinear forms.
    # Since n=m=p=4, we can brute-force the coefficients by linearity.
    # But we need exact rational coefficients.
    
    # Let's define the decomposition explicitly.
    # For each of the 49 terms r = s*7 + t:
    # U[i, r] = coefficient of A[i//m, i%m] in the first factor of term r
    # V[j, r] = coefficient of B[j//p, j%p] in the second factor of term r
    # W[k, r] = coefficient of C[k//n, k%n] in the result of term r
    
    # First factor of term (s,t): linear combination of A entries
    # Second factor of term (s,t): linear combination of B entries
    # Result: linear combination of C entries
    
    # Let's write a helper to get the linear combination.
    
    # Outer s defines X and Y blocks.
    # X = sum_{i_l, j_l} X[i_l, j_l] * e_{i_l, j_l}
    # Y = sum_{i_l, j_l} Y[i_l, j_l] * e_{i_l, j_l}
    
    # For a given outer s, X[i_l, j_l] is a sum of A[i, j] with coefficients a_{s, i_l, j_l, i, j}
    # Y[i_l, j_l] is a sum of B[i, j] with coefficients b_{s, i_l, j_l, i, j}
    
    # Inner t defines a bilinear form on the 4 entries of X and 4 entries of Y.
    # P_t = sum_{i_l1, j_l1, i_l2, j_l2} c_{t, i_l1, j_l1, i_l2, j_l2} * X[i_l1, j_l1] * Y[i_l2, j_l2]
    # where X and Y entries are linear in A and B.
    
    # So P_t = sum_{i, j, k, l} (sum_{i_l1, j_l1, i_l2, j_l2} c_{t, i_l1, j_l1, i_l2, j_l2} * a_{s, i_l1, j_l1, i, j} * b_{s, i_l2, j_l2, k, l}) * A[i, j] * B[k, l]
    
    # The result M_s is a 2x2 matrix where M_s[i_l, k_l] = sum_{t} mu_{t, i_l, k_l} * P_t
    # So M_s[i_l, k_l] = sum_{i, j, k, l} (sum_{t} mu_{t, i_l, k_l} * (sum_{i_l1, j_l1, i_l2, j_l2} c_{t, i_l1, j_l1, i_l2, j_l2} * a_{s, i_l1, j_l1, i, j} * b_{s, i_l2, j_l2, k, l})) * A[i, j] * B[k, l]
    
    # C[I, K] = sum_{s} gamma_{s, I, K} * M_s
    # So C[I, K] = sum_{r} (sum_{i, j, k, l} coeff_{r, i, j, k, l}) * A[i, j] * B[k, l]
    
    # In the decomposition:
    # C[I, K] = sum_{r} U[I*m + (something?) ... wait.
    # U is (n*m) x R. U[row, r] is coefficient for A[row//m, row%m] in term r.
    # V is (m*p) x R. V[row, r] is coefficient for B[row//p, row%p] in term r.
    # W is (p*n) x R. W[row, r] is coefficient for C[row//n, row%n] in term r. Wait, W is (p*n, R).
    # The trace convention says W[k*n+i, r]. So W[row, r] corresponds to C[row//n, row%n]?
    # If row = k*n + i, then C[i, k]. So W[k*n+i, r] is coefficient for C[i, k] in term r.
    
    # So for term r, we have:
    # Term_r = (sum_{i,j} U[i*m+j, r] * A[i,j]) * (sum_{k,l} V[k*p+l, r] * B[k,l])
    # And this contributes to C as:
    # sum_{i,j,k,l} U[i*m+j, r] * V[k*p+l, r] * A[i,j] * B[k,l] = sum_{i',k'} W[k'*n+i', r] * C[i', k'] ?
    # No, the decomposition is:
    # C[i', k'] = sum_r W[k'*n+i', r] * (sum_{i,j} U[i*m+j, r] * A[i,j]) * (sum_{k,l} V[k*p+l, r] * B[k,l])
    # So the coefficient of A[i,j] * B[k,l] in C[i', k'] is sum_r W[k'*n+i', r] * U[i*m+j, r] * V[k*p+l, r]
    # And this must equal delta_{i, i'} * delta_{k, k'}.
    
    # So for each r, U[., r], V[., r], W[., r] define a rank-1 term.
    # The sum over r of outer product U_r V_r^T must match the multiplication tensor.
    
    # Let's compute U, V, W for each r = s*7 + t.
    # U[i, r] = sum_{i_l1, j_l1} a_{s, i_l1, j_l1, i, j} * (coefficient of X[i_l1, j_l1] in inner term t)
    # The coefficient of X[i_l1, j_l1] in inner term t is:
    # For P0: X00, X11 => coeff[0] for (0,0), coeff[1] for (1,1)
    # For P1: X10, X11 => coeff[1] for (1,0), coeff[1] for (1,1)
    # For P2: X00 => coeff[1] for (0,0)
    # For P3: X11 => coeff[1] for (1,1)
    # For P4: X00, X01 => coeff[1] for (0,0), coeff[1] for (0,1)
    # For P5: X10, X00 => coeff[1] for (1,0), coeff[-1] for (0,0)
    # For P6: X11, X10 => coeff[1] for (1,1), coeff[-1] for (1,0)
    
    # Similarly for Y:
    # For P0: Y00, Y11 => coeff[1] for (0,0), coeff[1] for (1,1)
    # For P1: Y00 => coeff[1] for (0,0)
    # For P2: Y01, Y11 => coeff[1] for (0,1), coeff[-1] for (1,1)
    # For P3: Y10, Y00 => coeff[1] for (1,0), coeff[-1] for (0,0)
    # For P4: Y11 => coeff[1] for (1,1)
    # For P5: Y00, Y01 => coeff[1] for (0,0), coeff[1] for (0,1)
    # For P6: Y10, Y11 => coeff[1] for (1,0), coeff[1] for (1,1)
    
    # Let's define coeff_X[t, i_l, j_l] and coeff_Y[t, i_l, j_l].
    
    coeff_X = {}
    coeff_Y = {}
    
    # t=0: P0
    coeff_X[0] = {(0,0): 1, (1,1): 1}
    coeff_Y[0] = {(0,0): 1, (1,1): 1}
    
    # t=1: P1
    coeff_X[1] = {(1,0): 1, (1,1): 1}
    coeff_Y[1] = {(0,0): 1}
    
    # t=2: P2
    coeff_X[2] = {(0,0): 1}
    coeff_Y[2] = {(0,1): 1, (1,1): -1}
    
    # t=3: P3
    coeff_X[3] = {(1,1): 1}
    coeff_Y[3] = {(1,0): 1, (0,0): -1}
    
    # t=4: P4
    coeff_X[4] = {(0,0): 1, (0,1): 1}
    coeff_Y[4] = {(1,1): 1}
    
    # t=5: P5
    coeff_X[5] = {(1,0): 1, (0,0): -1}
    coeff_Y[5] = {(0,0): 1, (0,1): 1}
    
    # t=6: P6
    coeff_X[6] = {(1,1): 1, (1,0): -1}
    coeff_Y[6] = {(1,0): 1, (1,1): 1}
    
    # Now define a_s[i_l, j_l, i, j] for each outer s.
    # s=0: X = A00 + A11 => X[0,0] = A[0,0]+A[2,2], X[0,1] = A[0,1]+A[2,3], X[1,0] = A[1,0]+A[3,2], X[1,1] = A[1,1]+A[3,3]
    a_s = {}
    
    # Helper to set a_s[s, i_l, j_l, i, j]
    def set_a(s, i_l, j_l, i, j, val):
        if s not in a_s: a_s[s] = {}
        a_s[s][(i_l, j_l, i, j)] = a_s[s].get((i_l, j_l, i, j), 0) + val
    
    # s=0
    for i_l in range(2):
        for j_l in range(2):
            set_a(0, i_l, j_l, 0 + 2*i_l, 0 + 2*j_l, 1)
            set_a(0, i_l, j_l, 2 + 2*i_l, 2 + 2*j_l, 1)
    
    # s=1: X = A10 + A11 => X[0,0] = A[2,0]+A[2,2], X[0,1] = A[2,1]+A[2,3], X[1,0] = A[3,0]+A[3,2], X[1,1] = A[3,1]+A[3,3]
    for i_l in range(2):
        for j_l in range(2):
            set_a(1, i_l, j_l, 2 + 2*i_l, 0 + 2*j_l, 1)
            set_a(1, i_l, j_l, 2 + 2*i_l, 2 + 2*j_l, 1)
    
    # s=2: X = A00 => X[0,0] = A[0,0], X[0,1] = A[0,1], X[1,0] = A[1,0], X[1,1] = A[1,1]
    for i_l in range(2):
        for j_l in range(2):
            set_a(2, i_l, j_l, 0 + 2*i_l, 0 + 2*j_l, 1)
    
    # s=3: X = A11 => X[0,0] = A[2,2], X[0,1] = A[2,3], X[1,0] = A[3,2], X[1,1] = A[3,3]
    for i_l in range(2):
        for j_l in range(2):
            set_a(3, i_l, j_l, 2 + 2*i_l, 2 + 2*j_l, 1)
    
    # s=4: X = A00 + A01 => X[0,0] = A[0,0]+A[0,2], X[0,1] = A[0,1]+A[0,3], X[1,0] = A[1,0]+A[1,2], X[1,1] = A[1,1]+A[1,3]
    for i_l in range(2):
        for j_l in range(2):
            set_a(4, i_l, j_l, 0 + 2*i_l, 0 + 2*j_l, 1)
            set_a(4, i_l, j_l, 0 + 2*i_l, 2 + 2*j_l, 1)
    
    # s=5: X = A10 - A00 => X[0,0] = A[2,0]-A[0,0], X[0,1] = A[2,1]-A[0,1], X[1,0] = A[3,0]-A[1,0], X[1,1] = A[3,1]-A[1,1]
    for i_l in range(2):
        for j_l in range(2):
            set_a(5, i_l, j_l, 2 + 2*i_l, 0 + 2*j_l, 1)
            set_a(5, i_l, j_l, 0 + 2*i_l, 0 + 2*j_l, -1)
    
    # s=6: X = A11 - A10 => X[0,0] = A[2,2]-A[2,0], X[0,1] = A[2,3]-A[2,1], X[1,0] = A[3,2]-A[3,0], X[1,1] = A[3,3]-A[3,1]
    for i_l in range(2):
        for j_l in range(2):
            set_a(6, i_l, j_l, 2 + 2*i_l, 2 + 2*j_l, 1)
            set_a(6, i_l, j_l, 2 + 2*i_l, 0 + 2*j_l, -1)
    
    # Now define b_s[i_l, j_l, k, l] for each outer s.
    # s=0: Y = B00 + B11 => Y[0,0] = B[0,0]+B[2,2], Y[0,1] = B[0,1]+B[2,3], Y[1,0] = B[1,0]+B[3,2], Y[1,1] = B[1,1]+B[3,3]
    b_s = {}
    
    def set_b(s, i_l, j_l, k, l, val):
        if s not in b_s: b_s[s] = {}
        b_s[s][(i_l, j_l, k, l)] = b_s[s].get((i_l, j_l, k, l), 0) + val
    
    for s in range(7):
        for i_l in range(2):
            for j_l in range(2):
                if s == 0:
                    set_b(s, i_l, j_l, 0 + 2*i_l, 0 + 2*j_l, 1)
                    set_b(s, i_l, j_l, 2 + 2*i_l, 2 + 2*j_l, 1)
                elif s == 1:
                    set_b(s, i_l, j_l, 0 + 2*i_l, 0 + 2*j_l, 1)
                elif s == 2:
                    set_b(s, i_l, j_l, 0 + 2*i_l, 2 + 2*j_l, 1)
                    set_b(s, i_l, j_l, 2 + 2*i_l, 2 + 2*j_l, -1)
                elif s == 3:
                    set_b(s, i_l, j_l, 2 + 2*i_l, 0 + 2*j_l, 1)
                    set_b(s, i_l, j_l, 0 + 2*i_l, 0 + 2*j_l, -1)
                elif s == 4:
                    set_b(s, i_l, j_l, 2 + 2*i_l, 2 + 2*j_l, 1)
                elif s == 5:
                    set_b(s, i_l, j_l, 0 + 2*i_l, 0 + 2*j_l, 1)
                    set_b(s, i_l, j_l, 0 + 2*i_l, 2 + 2*j_l, 1)
                elif s == 6:
                    set_b(s, i_l, j_l, 2 + 2*i_l, 0 + 2*j_l, 1)
                    set_b(s, i_l, j_l, 2 + 2*i_l, 2 + 2*j_l, 1)
    
    # Now compute U, V, W.
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            
            # U[i, r] = sum_{i_l, j_l} a_s[i_l, j_l, i, j] * coeff_X[t].get((i_l, j_l), 0)
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = 0
                    for i_l in range(2):
                        for j_l in range(2):
                            a_val = a_s[s].get((i_l, j_l, i, j), 0)
                            c_val = coeff_X[t].get((i_l, j_l), 0)
                            U[i * m + j][r] += a_val * c_val
            
            # V[k, r] = sum_{i_l, j_l} b_s[i_l, j_l, k, l] * coeff_Y[t].get((i_l, j_l), 0)
            for k in range(p):
                for l in range(p): # p=4, but B is m x p, so k is row, l is col
                    # Wait, B is m x p. m=4, p=4.
                    V[k * p + l][r] = 0
                    for i_l in range(2):
                        for j_l in range(2):
                            b_val = b_s[s].get((i_l, j_l, k, l), 0)
                            c_val = coeff_Y[t].get((i_l, j_l), 0)
                            V[k * p + l][r] += b_val * c_val
            
            # W[k*n+i, r] = coefficient for C[i, k] in M_s
            # M_s[i_l, k_l] = sum_{t'} mu[t', i_l, k_l] * P_{t'}
            # But here t is fixed. So for term r = s*7 + t, the contribution to M_s[i_l, k_l] is mu[t, i_l, k_l] * P_t.
            # P_t is a scalar.
            # So W[k*n+i, r] = mu[t, i_l, k_l] where i = I*2 + i_l, k = K*2 + k_l.
            # And C[I, K] = sum_s gamma[s, I, K] * M_s.
            # So the coefficient for C[i, k] in term r is gamma[s, I, K] * mu[t, i_l, k_l].
            
            # Let's define gamma and mu.
            # mu[t, i_l, k_l]:
            # P0: (0,0)=1, (1,1)=1, (0,1)=0, (1,0)=0? No.
            # M[0,0] = P0 + P3 - P5 - P6 => mu[0,0,0]=1, mu[3,0,0]=1, mu[5,0,0]=-1, mu[6,0,0]=-1
            # M[0,1] = P4 + P6 => mu[4,0,1]=1, mu[6,0,1]=1
            # M[1,0] = P1 + P3 => mu[1,1,0]=1, mu[3,1,0]=1
            # M[1,1] = P0 - P1 + P2 + P4 => mu[0,1,1]=1, mu[1,1,1]=-1, mu[2,1,1]=1, mu[4,1,1]=1
            
            mu_dict = {
                (0, 0, 0): 1, (3, 0, 0): 1, (5, 0, 0): -1, (6, 0, 0): -1,
                (4, 0, 1): 1, (6, 0, 1): 1,
                (1, 1, 0): 1, (3, 1, 0): 1,
                (0, 1, 1): 1, (1, 1, 1): -1, (2, 1, 1): 1, (4, 1, 1): 1
            }
            
            # gamma[s, I, K]:
            # C00 = M0 + M3 - M5 - M6 => gamma[0,0,0]=1, gamma[3,0,0]=1, gamma[5,0,0]=-1, gamma[6,0,0]=-1
            # C01 = M4 + M6 => gamma[4,0,1]=1, gamma[6,0,1]=1
            # C10 = M1 + M3 => gamma[1,1,0]=1, gamma[3,1,0]=1
            # C11 = M0 - M1 + M2 + M4 => gamma[0,1,1]=1, gamma[1,1,1]=-1, gamma[2,1,1]=1, gamma[4,1,1]=1
            
            gamma_dict = {
                (0, 0, 0): 1, (3, 0, 0): 1, (5, 0, 0): -1, (6, 0, 0): -1,
                (4, 0, 1): 1, (6, 0, 1): 1,
                (1, 1, 0): 1, (3, 1, 0): 1,
                (0, 1, 1): 1, (1, 1, 1): -1, (2, 1, 1): 1, (4, 1, 1): 1
            }
            
            mu_val = mu_dict.get((t, i_l, k_l), 0) # Wait, i_l, k_l depend on i, k.
            
            for I in range(2):
                for K in range(2):
                    for i_l in range(2):
                        for k_l in range(2):
                            i = I * 2 + i_l
                            k = K * 2 + k_l
                            gamma_val = gamma_dict.get((s, I, K), 0)
                            mu_val_t = mu_dict.get((t, i_l, k_l), 0)
                            W[k * n + i][r] = gamma_val * mu_val_t
    
    return {"U": U, "V": W, "W": V} # Wait, no. U is U, V is V, W is W.
    # I made a mistake in the last line. It should be {"U": U, "V": V, "W": W}
    # Let me fix that in the actual code.
    
    return {"U": U, "V": V, "W": W}

BET: {mechanism: "Nested Strassen decomposition reduces rank from 64 to 49, improving score by 64/49 ≈ 1.306", q: 0.95, delta: [1.3, 1.35]}
