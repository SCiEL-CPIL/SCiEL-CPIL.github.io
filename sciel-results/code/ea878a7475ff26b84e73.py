"""Combined program: Strassen-based decomposition with systematic rank reduction."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction
import numpy as np

DEADLINE = 135.0  # safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication, T[i,j,k] should be 1 if we're computing (AB)[i,k] = sum_j A[i,j]*B[j,k]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _strassen_decomposition(n, m, p):
    """Build the Strassen-based decomposition starting from rank 46."""
    rank = 46
    U = [[0]*rank for _ in range(n*m)]  # 10 x 46
    V = [[0]*rank for _ in range(m*p)]  # 25 x 46
    W = [[0]*rank for _ in range(p*n)]  # 10 x 46
    
    c = 0  # current column index
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal c
        # X = A[i_off:i_off+2, j_off:j_off+2]
        # Y = B[j_off:j_off+2, k_off:k_off+2]
        # Z = C[i_off:i_off+2, k_off:k_off+2]
        
        # Strassen factors:
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # Output:
        # Z00 = M1+M4-M5+M7
        # Z01 = M3+M5
        # Z10 = M2+M4
        # Z11 = M1-M2+M3+M6
        
        # U indices for X:
        x00 = i_off * m + j_off
        x01 = i_off * m + j_off + 1
        x10 = (i_off+1) * m + j_off
        x11 = (i_off+1) * m + j_off + 1
        
        # V indices for Y:
        y00 = j_off * p + k_off
        y01 = j_off * p + k_off + 1
        y10 = (j_off+1) * p + k_off
        y11 = (j_off+1) * p + k_off + 1
        
        # W indices for Z:
        z00 = k_off * n + i_off
        z01 = (k_off+1) * n + i_off
        z10 = k_off * n + (i_off+1)
        z11 = (k_off+1) * n + (i_off+1)
        
        # M1 = (X00+X11)(Y00+Y11)
        U[x00][c] += 1; U[x11][c] += 1
        V[y00][c] += 1; V[y11][c] += 1
        W[z00][c] += 1; W[z11][c] += 1  # Z00 gets +M1, Z11 gets +M1
        c += 1
        
        # M2 = (X10+X11)Y00
        U[x10][c] += 1; U[x11][c] += 1
        V[y00][c] += 1
        W[z10][c] += 1; W[z11][c] -= 1  # Z10 gets +M2, Z11 gets -M2
        c += 1
        
        # M3 = X00(Y01-Y11)
        U[x00][c] += 1
        V[y01][c] += 1; V[y11][c] -= 1
        W[z01][c] += 1; W[z11][c] += 1  # Z01 gets +M3, Z11 gets +M3
        c += 1
        
        # M4 = X11(Y10-Y00)
        U[x11][c] += 1
        V[y10][c] += 1; V[y00][c] -= 1
        W[z00][c] += 1; W[z10][c] += 1  # Z00 gets +M4, Z10 gets +M4
        c += 1
        
        # M5 = (X00+X01)Y11
        U[x00][c] += 1; U[x01][c] += 1
        V[y11][c] += 1
        W[z00][c] -= 1; W[z01][c] += 1  # Z00 gets -M5, Z01 gets +M5
        c += 1
        
        # M6 = (X10-X00)(Y00+Y01)
        U[x10][c] += 1; U[x00][c] -= 1
        V[y00][c] += 1; V[y01][c] += 1
        W[z11][c] += 1  # Z11 gets +M6
        c += 1
        
        # M7 = (X01-X11)(Y10+Y11)
        U[x01][c] += 1; U[x11][c] -= 1
        V[y10][c] += 1; V[y11][c] += 1
        W[z00][c] += 1  # Z00 gets +M7
        c += 1
    
    def add_schoolbook_2x2_x_2x1(i_off, j_off, k_off):
        """A[i_off:i_off+2, j_off:j_off+2] (2x2) × B[j_off:j_off+2, k_off] (2x1) → C[i_off:i_off+2, k_off] (2x1)"""
        nonlocal c
        for ii in range(2):
            for jj in range(2):
                U[(i_off+ii)*m + (j_off+jj)][c] += 1
                V[(j_off+jj)*p + k_off][c] += 1
                W[k_off*n + (i_off+ii)][c] += 1
                c += 1
    
    def add_schoolbook_2x1_x_1x2(i_off, j_off, k_off):
        """A[i_off:i_off+2, j_off] (2x1) × B[j_off, k_off:k_off+2] (1x2) → C[i_off:i_off+2, k_off:k_off+2] (2x2)"""
        nonlocal c
        for ii in range(2):
            for kk in range(2):
                U[(i_off+ii)*m + j_off][c] += 1
                V[j_off*p + (k_off+kk)][c] += 1
                W[(k_off+kk)*n + (i_off+ii)][c] += 1
                c += 1
    
    def add_schoolbook_2x1_x_1x1(i_off, j_off, k_off):
        """A[i_off:i_off+2, j_off] (2x1) × B[j_off, k_off] (1x1) → C[i_off:i_off+2, k_off] (2x1)"""
        nonlocal c
        for ii in range(2):
            U[(i_off+ii)*m + j_off][c] += 1
            V[j_off*p + k_off][c] += 1
            W[k_off*n + (i_off+ii)][c] += 1
            c += 1
    
    # Block 1: A[:,{0,1}] × B[{0,1},{0,1}] → C[{0,1},{0,1}]
    add_strassen_block(0, 0, 0)
    
    # Block 2: A[:,{0,1}] × B[{0,1},{2,3}] → C[{0,1},{2,3}]
    add_strassen_block(0, 0, 2)
    
    # Block 3: A[:,{0,1}] × B[{0,1},{4}] → C[{0,1},{4}]
    add_schoolbook_2x2_x_2x1(0, 0, 4)
    
    # Block 4: A[:,{2,3}] × B[{2,3},{0,1}] → C[{0,1},{0,1}]
    add_strassen_block(0, 2, 0)
    
    # Block 5: A[:,{2,3}] × B[{2,3},{2,3}] → C[{0,1},{2,3}]
    add_strassen_block(0, 2, 2)
    
    # Block 6: A[:,{2,3}] × B[{2,3},{4}] → C[{0,1},{4}]
    add_schoolbook_2x2_x_2x1(0, 2, 4)
    
    # Block 7: A[:,{4}] × B[{4},{0,1}] → C[{0,1},{0,1}]
    add_schoolbook_2x1_x_1x2(0, 4, 0)
    
    # Block 8: A[:,{4}] × B[{4},{2,3}] → C[{0,1},{2,3}]
    add_schoolbook_2x1_x_1x2(0, 4, 2)
    
    # Block 9: A[:,{4}] × B[{4},{4}] → C[{0,1},{4}]
    add_schoolbook_2x1_x_1x1(0, 4, 4)
    
    return U, V, W

def _find_merge_factors(u1, v1, w1, u2, v2, w2, n, m, p):
    """
    Find rational factors f such that merging column 2 into column 1 with factor f
    preserves the tensor.
    Condition: (u1 + f u2) o (v1 + f v2) o (w1 + f w2) == u1 o v1 o w1 + u2 o v2 o w2
    """
    # The difference is a polynomial in f of degree 3.
    # P(f) = sum_{i,j,k} ( (u1[i] + f u2[i]) (v1[j] + f v2[j]) (w1[k] + f w2[k]) - (u1[i]v1[j]w1[k] + u2[i]v2[j]w2[k]) )
    # We want P(f) == 0 for all i,j,k.
    # This is a system of linear equations in the coefficients of f^0, f^1, f^2, f^3.
    # Let P(f) = c0 + c1 f + c2 f^2 + c3 f^3.
    # We need c0=c1=c2=c3=0.
    
    # c3 = sum u2[i] v2[j] w2[k] * f^3? No.
    # Term: (u1 + f u2)(v1 + f v2)(w1 + f w2)
    # = u1v1w1 + f(u1v1w2 + u1v2w1 + u2v1w1) + f^2(u1v2w2 + u2v1w2 + u2v2w1) + f^3 u2v2w2
    
    # We want this to equal u1v1w1 + u2v2w2.
    # So:
    # c1 = sum (u1v1w2 + u1v2w1 + u2v1w1) == 0
    # c2 = sum (u1v2w2 + u2v1w2 + u2v2w1) == 0
    # c3 = sum (u2v2w2) - sum(u2v2w2) = 0? 
    # Wait, the RHS has u2v2w2. The LHS has f^3 u2v2w2.
    # So c3 = sum u2v2w2 - sum u2v2w2 = 0 is only if the coefficient of f^3 is 0.
    # But u2v2w2 is the tensor for column 2.
    # The equation is:
    # u1v1w1 + f(...)+ f^2(...)+ f^3 u2v2w2 = u1v1w1 + u2v2w2
    # f^3 u2v2w2 - u2v2w2 + f^2(...) + f(...) = 0
    # (f^3 - 1) u2v2w2 + f^2 C2 + f C1 = 0
    
    # This must hold for all i,j,k.
    # So we have a vector equation:
    # (f^3 - 1) T2 + f^2 C2_vec + f C1_vec = 0
    
    # Where T2 is the tensor of col 2, C2_vec and C1_vec are tensors derived from the cross terms.
    
    # We can solve this by checking if the vectors T2, C2_vec, C1_vec are linearly dependent with coefficients (f^3-1, f^2, f).
    
    # Let's compute the vectors.
    # Dimensions: n*m*p = 50.
    # We represent tensors as flat vectors of length 50.
    
    T2 = np.zeros(n*m*p)
    C2 = np.zeros(n*m*p)
    C1 = np.zeros(n*m*p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i*m + j)*p + k
                u1_i = u1[i*m+j]
                u2_i = u2[i*m+j]
                v1_j = v1[j*p+k]
                v2_j = v2[j*p+k]
                w1_k = w1[k*n+i]
                w2_k = w2[k*n+i]
                
                T2[idx] = u2_i * v2_j * w2_k
                
                # f^2 terms: u1 v2 w2 + u2 v1 w2 + u2 v2 w1
                C2[idx] = u1_i * v2_j * w2_k + u2_i * v1_j * w2_k + u2_i * v2_j * w1_k
                
                # f terms: u1 v1 w2 + u1 v2 w1 + u2 v1 w1
                C1[idx] = u1_i * v1_j * w2_k + u1_i * v2_j * w1_k + u2_i * v1_j * w1_k
                
    # We want to find f such that (f^3 - 1) T2 + f^2 C2 + f C1 = 0.
    # This is a system of 50 linear equations in the "unknowns" which are functions of f.
    # We can pick a few non-zero entries to form a smaller system.
    # Or we can use numpy to find the rank of the matrix [T2, C2, C1] and check for consistency.
    # Actually, we can just try to find f that satisfies the equation for a subset of indices.
    
    # Find indices where T2 is non-zero.
    non_zero_idx = np.nonzero(T2)[0]
    if len(non_zero_idx) == 0:
        # T2 is zero vector? Then we need f^2 C2 + f C1 = 0.
        # This implies f=0 or C2/f + C1 = 0.
        # If f=0, merge is just removing col 2? No, u1+0*u2 = u1.
        # If we remove col 2, we lose T2. If T2=0, it's fine.
        # But T2 is a rank 1 tensor, so it's rarely 0.
        pass
        
    # Let's try to solve for f using the first few non-zero equations.
    # We'll use a random subset of 4 equations to solve for f (cubic).
    # Then verify with all equations.
    
    candidates = []
    
    # If T2 is zero, we look for f such that f^2 C2 + f C1 = 0 => f(f C2 + C1) = 0.
    # So f=0 or f C2 = -C1.
    if np.all(T2 == 0):
        candidates.append(Fraction(0))
        # Check if C1 is proportional to C2.
        # If C1 = -f C2, then f = -C1[idx]/C2[idx].
        # We can try to find such f.
        nz_c2 = np.nonzero(C2)[0]
        if len(nz_c2) > 0:
            idx = nz_c2[0]
            if C2[idx] != 0:
                f_val = -C1[idx] / C2[idx]
                # Snap to rational
                f_frac = Fraction(f_val).limit_denominator(100)
                candidates.append(f_frac)
    else:
        # Select 4 random indices with non-zero T2
        if len(non_zero_idx) >= 4:
            import random
            random.seed(42)
            selected = random.sample(list(non_zero_idx), 4)
        else:
            selected = list(non_zero_idx)
            
        # Build system:
        # a3 f^3 + a2 f^2 + a1 f + a0 = 0
        # where a3 = T2[i], a2 = C2[i], a1 = C1[i], a0 = -T2[i]
        
        # We can solve this by finding roots of the cubic polynomial.
        # But we need to satisfy ALL equations.
        # So we pick 3 equations to determine f? No, cubic has 3 roots.
        # We can pick 4 equations and check consistency.
        
        # Let's just try to find roots of the polynomial formed by the first equation,
        # and then verify against all.
        idx0 = selected[0]
        a3 = T2[idx0]
        a2 = C2[idx0]
        a1 = C1[idx0]
        a0 = -T2[idx0]
        
        if a3 != 0:
            roots = np.roots([a3, a2, a1, a0])
            for r in roots:
                # Snap to rational
                f_real = r.real
                f_frac = Fraction(f_real).limit_denominator(100)
                # Also try nearby fractions
                for denom in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
                    num = round(f_real * denom)
                    candidates.append(Fraction(num, denom))
        
        # Also try simple rationals
        for denom in [1, 2, 3, 4, 5]:
            for num in range(-5, 6):
                candidates.append(Fraction(num, denom))
                
    # Verify candidates
    valid_factors = []
    for f in candidates:
        f_val = float(f)
        # Check (f^3 - 1) T2 + f^2 C2 + f C1 == 0
        lhs = (f_val**3 - 1) * T2 + (f_val**2) * C2 + f_val * C1
        if np.allclose(lhs, 0, atol=1e-9):
            # Exact verification with Fractions
            is_valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = (i*m + j)*p + k
                        val = (f**3 - 1) * T2[idx] + (f**2) * C2[idx] + f * C1[idx]
                        # T2, C2, C1 are integers (from inputs)
                        # f is Fraction
                        if val != 0:
                            is_valid = False
                            break
                if not is_valid:
                    break
            if is_valid:
                valid_factors.append(f)
                
    return valid_factors

def _try_merge(U, V, W, n, m, p, c1, c2):
    """Try to merge columns c1 and c2 by finding a valid factor."""
    rank = len(U[0])
    if c1 == c2:
        return None
    
    u1 = [row[c1] for row in U]
    v1 = [row[c1] for row in V]
    w1 = [row[c1] for row in W]
    u2 = [row[c2] for row in U]
    v2 = [row[c2] for row in V]
    w2 = [row[c2] for row in W]
    
    factors = _find_merge_factors(u1, v1, w1, u2, v2, w2, n, m, p)
    
    for factor in factors:
        new_rank = rank - 1
        new_U = [[0] * new_rank for _ in range(n * m)]
        new_V = [[0] * new_rank for _ in range(m * p)]
        new_W = [[0] * new_rank for _ in range(p * n)]
        
        new_col = 0
        for c in range(rank):
            if c == c2:
                continue
            if c == c1:
                for i in range(n * m):
                    new_U[i][new_col] = U[i][c1] + factor * U[i][c2]
                for j in range(m * p):
                    new_V[j][new_col] = V[j][c1] + factor * V[j][c2]
                for k in range(p * n):
                    new_W[k][new_col] = W[k][c1] + factor * W[k][c2]
            else:
                for i in range(n * m):
                    new_U[i][new_col] = U[i][c]
                for j in range(m * p):
                    new_V[j][new_col] = V[j][c]
                for k in range(p * n):
                    new_W[k][new_col] = W[k][c]
            new_col += 1
        
        # Verify
        if _verify(new_U, new_V, new_W, n, m, p):
            return new_U, new_V, new_W
            
    return None

def _search_lower_rank(U, V, W, n, m, p, target_rank, max_time=30.0):
    """Try to find a decomposition of rank target_rank starting from current rank."""
    current_rank = len(U[0])
    if current_rank < target_rank:
        return U, V, W
    
    start_time = time.time()
    
    while current_rank > target_rank:
        if time.time() - start_time > max_time:
            break
            
        merged = False
        for c1 in range(current_rank):
            for c2 in range(c1 + 1, current_rank):
                result = _try_merge(U, V, W, n, m, p, c1, c2)
                if result is not None:
                    U, V, W = result
                    current_rank -= 1
                    merged = True
                    break
            if merged:
                break
        
        if not merged:
            break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with Strassen-based decomposition (rank 46)
    U, V, W = _strassen_decomposition(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    start_time = time.time()
    
    # Try to find lower rank decompositions
    for target_rank in range(best_rank - 1, 0, -1):
        if time.time() > DEADLINE:
            break
        
        # Give more time for higher targets (closer to current rank)
        time_budget = min(20.0, (DEADLINE - time.time()) * 0.5)
        if time_budget < 5.0:
            break
            
        result = _search_lower_rank(U, V, W, n, m, p, target_rank, max_time=time_budget)
        if result is not None:
            new_rank = len(result[0][0])
            if new_rank < best_rank:
                best_U, best_V, best_W = result
                best_rank = new_rank
                U, V, W = result
                # Continue trying to go lower
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
