import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def _verify_exact(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Build the tensor T[i,j,k,l] = delta_{i,k} * delta_{j,l} for (A*B)[i,l]
    # Actually T[a,b,c] where a=(i,j), b=(j,k), c=(k,i)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Check: sum_r U[a,r] * V[b,r] * W[c,r] == T[a,b,c]
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    # Convert to Fractions for exact arithmetic
    U_frac = [[Fraction(str(x)) for x in row] for row in U]
    V_frac = [[Fraction(str(x)) for x in row] for row in V]
    W_frac = [[Fraction(str(x)) for x in row] for row in W]
    
    # Check all entries
    for a in range(n * m):
        i, j = a // m, a % m
        for b in range(m * p):
            j2, k = b // p, b % p
            for c in range(p * n):
                k2, i2 = c // n, c % n
                # Expected value
                if j == j2 and k == k2 and i == i2:
                    expected = 1
                else:
                    expected = 0
                
                # Computed value
                computed = 0
                for r in range(rank):
                    computed += U_frac[a][r] * V_frac[b][r] * W_frac[c][r]
                
                if computed != expected:
                    return False
    
    return True

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 50
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    # Try to find a lower rank decomposition
    best_result = None
    
    # Search ranks from 25 to 49
    for R in range(25, min(rank_school, 50)):
        if time.time() > deadline - 30:  # Leave 30s margin
            break
        
        # Try multiple random initializations
        max_trials = 5 if R <= 30 else 3
        for trial in range(max_trials):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(trial * 10000 + R * 100 + 42)
            
            # Random initialization
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            U = np.zeros((n * m, R))
            
            # Solve for U given V and W
            # For each row a of U, we need: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # This is linear in U[a,:] for fixed a
            # T[a, (b,c)] = sum_r U[a,r] * (V[b,r] * W[c,r])
            # So we can write this as a linear system
            
            success = True
            for a in range(n * m):
                i, j = a // m, a % m
                
                # Build the linear system for U[a,:]
                # For each (b, c) pair, we have one equation
                # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
                
                # Number of (b,c) pairs is (m*p) * (p*n) = 25 * 10 = 250
                # But many are zero, so we only need the non-zero ones
                # Non-zero when b's j-component == j and c's k-component == j's... 
                # Actually T[a,b,c] = 1 iff j_b == j and k_c == j... no
                # T[i*m+j, j2*p+k, k2*n+i2] = 1 iff j==j2, k==k2, i==i2
                
                # For fixed a=(i,j), non-zero when j2==j, k2==j... no
                # j2==j, and k2 can be anything, but i2==i
                # So b = j*p + k for k in 0..p-1, c = k*n + i for k in 0..p-1
                # That's p equations, but they're not independent necessarily
                
                # Let's build the full system
                # Variables: U[a, 0..R-1]
                # Equations: for each (b,c) where T[a,b,c] != 0
                #   sum_r U[a,r] * V[b,r] * W[c,r] = 1
                
                eqs = []
                rhs = []
                for b in range(m * p):
                    j2, k = b // p, b % p
                    for c in range(p * n):
                        k2, i2 = c // n, c % n
                        if j2 == j and k2 == k and i2 == i:
                            # This is the only non-zero entry for this (b,c)
                            # Wait, that's not right. Let me re-examine.
                            pass
                
                # Actually, for fixed a=(i,j), the non-zero entries of T are:
                # T[i*m+j, b, c] = 1 iff b's first index == j AND c's second index == i AND b's second index == c's first index
                # b = j2*p + k, c = k2*n + i2
                # Condition: j2 == j, i2 == i, k == k2
                # So for each k in 0..p-1: b = j*p+k, c = k*n+i
                # That gives p equations
                
                # But we might have more constraints if we consider all (b,c) pairs
                # The zero constraints are also important
                
                # Let's use all (b,c) pairs to get a better-determined system
                # But that's 250 equations for R unknowns, which is overdetermined
                # We want an exact solution, so we need consistency
                
                # Better approach: only use the p non-zero equations
                # But R might be > p, so we have underdetermined system
                # We want a solution that also satisfies the zero constraints
                
                # Let's set up the system with all (b,c) where T != 0
                # That's p equations
                A_mat = np.zeros((p, R))
                rhs_vec = np.zeros(p)
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    for r in range(R):
                        A_mat[k, r] = V[b, r] * W[c, r]
                    rhs_vec[k] = 1.0
                
                # Solve least squares
                try:
                    U[a, :], residuals, rank_sol, sv = np.linalg.lstsq(A_mat, rhs_vec, rcond=None)
                except:
                    success = False
                    break
                
                # Verify all entries for this a
                for b in range(m * p):
                    j2, k = b // p, b % p
                    for c in range(p * n):
                        k2, i2 = c // n, c % n
                        if j2 == j and k2 == k and i2 == i:
                            expected = 1.0
                        else:
                            expected = 0.0
                        
                        computed = sum(U[a, r] * V[b, r] * W[c, r] for r in range(R))
                        if abs(computed - expected) > 1e-6:
                            success = False
                            break
                    if not success:
                        break
                if not success:
                    break
            
            if not success:
                continue
            
            # Full verification
            all_ok = True
            for a in range(n * m):
                for b in range(m * p):
                    j2, k = b // p, b % p
                    for c in range(p * n):
                        k2, i2 = c // n, c % n
                        if j2 == a % m and k2 == k and i2 == a // m:
                            expected = 1.0
                        else:
                            expected = 0.0
                        
                        computed = sum(U[a, r] * V[b, r] * W[c, r] for r in range(R))
                        if abs(computed - expected) > 1e-6:
                            all_ok = False
                            break
                    if not all_ok:
                        break
                if not all_ok:
                    break
            
            if all_ok:
                # Check for half-integer entries
                half_int_ok = True
                for mat in [U, V, W]:
                    for row in mat:
                        for val in row:
                            nearest = round(val * 2) / 2
                            if abs(val - nearest) > 1e-4:
                                half_int_ok = False
                                break
                        if not half_int_ok:
                            break
                    if not half_int_ok:
                        break
                
                if half_int_ok:
                    # Round to half-integers
                    U_rounded = [[round(val * 2) / 2 for val in row] for row in U]
                    V_rounded = [[round(val * 2) / 2 for val in row] for row in V]
                    W_rounded = [[round(val * 2) / 2 for val in row] for row in W]
                    
                    # Final exact verification
                    if _verify_exact(U_rounded, V_rounded, W_rounded):
                        best_result = {
                            "U": U_rounded,
                            "V": V_rounded,
                            "W": W_rounded
                        }
                        break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Return schoolbook
    rank = rank_school
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
