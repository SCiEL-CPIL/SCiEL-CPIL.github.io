"""Structured Strassen + linear solve for 2x3 * 3x5 matrix multiplication tensor."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition using the trace convention W[k*n+i, r]."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return the 7-term Strassen decomposition for 2x2 * 2x2.
    Each term is (U_vals, V_vals, W_vals) where:
      U_vals: 4 values for A[i,j] (i,j in 0..1)
      V_vals: 4 values for B[j,k] (j,k in 0..1)
      W_vals: 4 values for C[i,k] (i,k in 0..1)
    """
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M7
    
    terms = []
    # M1: (A00+A11)(B00+B11) -> contributes to C00(+1), C11(-1)
    # U: A00=1, A01=0, A10=0, A11=1
    # V: B00=1, B01=0, B10=0, B11=1
    # W: C00=1, C01=0, C10=0, C11=-1
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,-1]))
    # M2: (A10+A11)B00 -> C10(+1)
    # U: A00=0, A01=0, A10=1, A11=1
    # V: B00=1, B01=0, B10=0, B11=0
    # W: C00=0, C01=0, C10=1, C11=0
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    # M3: A00(B10-B11) -> C01(+1), C11(+1)
    # U: A00=1, A01=0, A10=0, A11=0
    # V: B00=0, B01=0, B10=1, B11=-1
    # W: C00=0, C01=1, C10=0, C11=1
    terms.append(([1,0,0,0], [0,0,1,-1], [0,1,0,1]))
    # M4: A11(B01-B00) -> C00(+1), C10(+1)
    # U: A00=0, A01=0, A10=0, A11=1
    # V: B00=-1, B01=1, B10=0, B11=0
    # W: C00=1, C01=0, C10=1, C11=0
    terms.append(([0,0,0,1], [-1,1,0,0], [1,0,1,0]))
    # M5: (A00+A01)B11 -> C00(-1), C01(+1)
    # U: A00=1, A01=1, A10=0, A11=0
    # V: B00=0, B01=0, B10=0, B11=1
    # W: C00=-1, C01=1, C10=0, C11=0
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    # M6: (A10-A00)(B00+B10) -> C10(+1)  [wait, let me recheck]
    # Actually standard Strassen M6 = (A10-A00)(B00+B10), contributes to C10
    # U: A00=-1, A01=0, A10=1, A11=0
    # V: B00=1, B01=0, B10=1, B11=0
    # W: C00=0, C01=0, C10=1, C11=0
    terms.append(([-1,0,1,0], [1,0,1,0], [0,0,1,0]))
    # M7: (A01-A11)(B01+B11) -> C00(+1), C11(+1)
    # U: A00=0, A01=1, A10=0, A11=-1
    # V: B00=0, B01=1, B10=0, B11=1
    # W: C00=1, C01=0, C10=0, C11=1
    terms.append(([0,1,0,-1], [0,1,0,1], [1,0,0,1]))
    
    return terms

def _strassen_verify():
    """Verify Strassen terms correctly decompose 2x2x2."""
    terms = _strassen_2x2x2()
    # Build tensor: T[i,j,k] = sum_c U_c[i,j]*V_c[j,k]*W_c[i,k]
    T = [[[0,0],[0,0]] for _ in range(2)]  # T[i][k] for i,k in 0..1, but we need T[i,j,k]
    # Actually the tensor is T[i,j,k] where i=0..1 (row of A), j=0..1 (col of A / row of B), k=0..1 (col of B)
    # T[i,j,k] should be 1 if this is the multiplication tensor
    # Wait, the multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # Because C[i,k] = sum_j A[i,j] B[j,k], so the tensor entry T[i,j,k] = 1
    
    # Let's verify: sum_c U_c[i,j] * V_c[j,k] * W_c[i,k] should equal 1 for all i,j,k in 0..1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for (Uv, Vv, Wv) in terms:
                    # Uv indexed by [i*2+j] -> Uv[i*2+j]
                    # Vv indexed by [j*2+k] -> Vv[j*2+k]
                    # Wv indexed by [k*2+i] -> Wv[k*2+i]
                    s += Uv[i*2+j] * Vv[j*2+k] * Wv[k*2+i]
                if s != 1:
                    return False
    return True

def _try_structured_rank(n, m, p, target_rank, deadline):
    """
    Structured approach:
    - Use Strassen (7 terms) for the 2x2x2 sub-tensor (i in 0..1, j in 0..1, k in 0..1)
    - Solve for remaining terms to cover all (i,j,k) with j=2 or k>=2
    
    The full tensor has entries T[i,j,k]=1 for i in 0..1, j in 0..2, k in 0..4.
    
    After placing Strassen terms for the (i,j,k) with j in 0..1, k in 0..1,
    we have residual entries that need to be covered.
    
    Actually, let's think more carefully. The Strassen terms cover:
    - For each term c=0..6, it contributes U_c[i,j]*V_c[j,k]*W_c[i,k] to T[i,j,k]
    - The Strassen terms are designed so that sum_c U_c[i,j]*V_c[j,k]*W_c[i,k] = 1 for i,j,k in 0..1
    
    But our tensor needs T[i,j,k]=1 for ALL i in 0..1, j in 0..2, k in 0..4.
    The Strassen terms only naturally handle j in 0..1, k in 0..1.
    
    For j=2 or k>=2, the Strassen terms will contribute some values (possibly 0 if
    we set the relevant entries to 0).
    
    Strategy: 
    1. Extend Strassen terms to have zeros for j=2 and k>=2 entries.
    2. Compute the residual: for each (i,j,k), residual = 1 - sum_strassen_contrib
    3. Solve a linear system for additional terms to cover the residual.
    
    But this is complex. Let me try a different approach:
    
    Alternative: Directly construct the decomposition with a known structure.
    
    For 2x3 * 3x5, think of it as:
    C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]
    
    We can use:
    - 7 Strassen terms for the 2x2 part (i in 0..1, j in 0..1) applied to B[:,0:2]
    - Additional terms for B[:,2] and for k=2,3,4
    
    Let's try a more direct constructive approach with rational solving.
    """
    if time.time() > deadline:
        return None
    
    # Let's try to build a decomposition with the following structure:
    # Rank = 7 (Strassen for j=0,1 and k=0,1) + R_extra
    # 
    # Actually, let's try a completely different structured approach:
    # Use the fact that we can share structure across k values.
    
    # For fixed i, the equation is:
    # C[i,k] = sum_j A[i,j] * B[j,k] for k=0..4
    # This is a 2x3 times 3x5 multiplication.
    
    # Let's try to find a decomposition using bilinear forms with shared structure.
    # 
    # Key insight: if we fix the U and V matrices, the W matrix is determined
    # by a linear system. So we can:
    # 1. Choose U and V with a structured pattern
    # 2. Solve for W
    # 3. Check if W has the required properties
    
    # Let's try: for each rank term c, define:
    # U_c[i,j] and V_c[j,k] as sparse patterns
    # Then W_c[i,k] is determined by: sum_c U_c[i,j] V_c[j,k] W_c[i,k] = 1 for all (i,j,k)
    
    # This is a system of 30 equations (2*3*5) in 10*rank unknowns (W has 10*rank entries)
    # But it's not linear in W because... wait, it IS linear in W!
    # For fixed U and V: sum_c U_c[i,j] V_c[j,k] W_c[i,k] = 1
    # This is linear in W_c[i,k].
    
    # So: choose U and V, then solve for W linearly!
    
    # Let's try random structured U and V, then solve for W.
    
    import random
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        random.seed(attempt * 104729 + target_rank * 7919)
        
        rank = target_rank
        nU = n * m  # 6
        nV = m * p  # 15
        nW = p * n  # 10
        nT = n * m * p  # 30
        
        # Generate structured U and V
        # Use small integers with some zeros for sparsity
        U = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(nU)]
        V = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(nV)]
        
        # Ensure some structure: make first few columns sparse
        for c in range(min(3, rank)):
            for i in range(nU):
                if random.random() < 0.5:
                    U[i][c] = 0
            for j in range(nV):
                if random.random() < 0.5:
                    V[j][c] = 0
        
        # Now solve for W: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is a linear system in the W variables.
        # Variables: W[k*n+i, c] for k in 0..4, i in 0..1, c in 0..rank-1
        # Total unknowns: 10 * rank
        # Equations: 30
        
        # Build the linear system A * w = b where b = ones(30)
        num_unknowns = nW * rank
        num_equations = nT
        
        # A is num_equations x num_unknowns
        # Row index: eq = ((i * m) + j) * p + k
        # Col index: var = (k * n + i) * rank + c
        
        A = [[Fraction(0) for _ in range(num_unknowns)] for _ in range(num_equations)]
        b = [Fraction(1) for _ in range(num_equations)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq = (i * m + j) * p + k
                    for c in range(rank):
                        coeff = Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c])
                        var = (k * n + i) * rank + c
                        A[eq][var] = coeff
        
        # Solve using Gaussian elimination over rationals
        # This is a 30 x (10*rank) system. For rank=27, that's 30 x 270.
        # We need to find a solution. Since 30 < 270, there are many solutions.
        # We want a solution where W entries are "nice" (integers or half-integers).
        
        # Use Gaussian elimination to find one solution.
        # We'll use the first 30 columns as the pivot set (if possible).
        
        # Actually, let's use a simpler approach: set most W variables to 0 and solve for a subset.
        # Pick 30 specific W variables to solve for, set the rest to 0.
        
        # Choose which W variables to solve for: pick one per equation if possible.
        # For each equation (i,j,k), pick a variable that appears in it.
        
        # Let's pick: for equation (i,j,k), use variable W[k*n+i, c] where c is chosen to maximize |coeff|.
        
        pivot_vars = []  # list of (eq, var) pairs
        used_vars = set()
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq = (i * m + j) * p + k
                    # Find best variable for this equation
                    best_var = -1
                    best_coeff = Fraction(0)
                    for c in range(rank):
                        var = (k * n + i) * rank + c
                        if var in used_vars:
                            continue
                        coeff = A[eq][var]
                        if coeff != 0 and abs(coeff) > abs(best_coeff):
                            best_coeff = coeff
                            best_var = var
                    if best_var >= 0:
                        pivot_vars.append((eq, best_var))
                        used_vars.add(best_var)
                    else:
                        # No good variable found, use any non-zero
                        for c in range(rank):
                            var = (k * n + i) * rank + c
                            if A[eq][var] != 0:
                                pivot_vars.append((eq, var))
                                used_vars.add(var)
                                break
        
        if len(pivot_vars) < num_equations:
            continue
        
        # Now solve: for each pivot equation, the pivot variable is determined by the other variables.
        # Set non-pivot variables to 0, then back-substitute.
        
        # Build reduced system: only pivot variables are unknowns.
        # A_reduced is 30 x 30, b_reduced is 30
        
        pivot_var_list = [pv[1] for pv in pivot_vars]
        pivot_eq_list = [pv[0] for pv in pivot_vars]
        
        A_red = [[Fraction(0) for _ in range(num_equations)] for _ in range(num_equations)]
        for row_idx, eq in enumerate(pivot_eq_list):
            for col_idx, var in enumerate(pivot_var_list):
                A_red[row_idx][col_idx] = A[eq][var]
        
        # Solve A_red * w_pivot = b
        # Gaussian elimination
        try:
            # Make a copy
            M = [row[:] for row in A_red]
            rhs = b[:]
            
            for col in range(num_equations):
                # Find pivot
                pivot_row = -1
                for row in range(col, num_equations):
                    if M[row][col] != 0:
                        pivot_row = row
                        break
                if pivot_row == -1:
                    continue  # Singular, skip
                
                # Swap
                M[col], M[pivot_row] = M[pivot_row], M[col]
                rhs[col], rhs[pivot_row] = rhs[pivot_row], rhs[col]
                
                # Normalize
                pivot_val = M[col][col]
                for j in range(col, num_equations):
                    M[col][j] = M[col][j] / pivot_val
                rhs[col] = rhs[col] / pivot_val
                
                # Eliminate
                for row in range(num_equations):
                    if row == col:
                        continue
                    factor = M[row][col]
                    if factor == 0:
                        continue
                    for j in range(col, num_equations):
                        M[row][j] -= factor * M[col][j]
                    rhs[row] -= factor * rhs[col]
            
            # Solution
            w_pivot = rhs  # After full elimination, rhs contains the solution
            
            # Check if solution is "nice" (small integers or half-integers)
            w_solution = {}
            nice = True
            for idx, var in enumerate(pivot_var_list):
                val = w_pivot[idx]
                w_solution[var] = val
                # Check if it's a small integer or half-integer
                if val.denominator > 2:
                    nice = False
                    continue
                num = val.numerator
                if abs(num) > 10:
                    nice = False
            
            if not nice:
                continue
            
            # Build full W
            W = [[Fraction(0) for _ in range(rank)] for _ in range(nW)]
            for var, val in w_solution.items():
                row = var // rank
                col = var % rank
                W[row][col] = val
            
            # Convert to integers or floats
            # Check if all are integers
            all_int = True
            for row in range(nW):
                for col in range(rank):
                    if W[row][col].denominator != 1:
                        all_int = False
                        break
            
            if all_int:
                W_int = [[int(W[row][col]) for col in range(rank)] for row in range(nW)]
                U_int = [[int(U[row][col]) for col in range(rank)] for row in range(nU)]
                V_int = [[int(V[row][col]) for col in range(rank)] for row in range(nV)]
                
                if _verify(U_int, V_int, W_int, n, m, p):
                    return {"U": U_int, "V": V_int, "W": W_int}
            else:
                # Try half-integers
                W_half = [[float(W[row][col]) for col in range(rank)] for row in range(nW)]
                U_half = [[float(U[row][col]) for col in range(rank)] for row in range(nU)]
                V_half = [[float(V[row][col]) for col in range(rank)] for row in range(nV)]
                
                # Verify with floats (with tolerance)
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = sum(U_half[i*m+j][c] * V_half[j*p+k][c] * W_half[k*n+i][c] for c in range(rank))
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return {"U": U_half, "V": V_half, "W": W_half}
        
        except Exception:
            continue
    
    return None

def _try_direct_construct(n, m, p, deadline):
    """
    Try to directly construct a low-rank decomposition using known algebraic identities.
    
    For 2x3 * 3x5, one approach:
    - View as 2x(2+1) * (2+1)x5
    - Use Strassen for the 2x2 part (7 terms)
    - Use additional terms for the cross and 1x1 parts
    
    Total: 7 + 2*5 + 2*5 + 5 = 7 + 10 + 10 + 5 = 32? No, that's not right.
    
    Let me think differently. The rank of 2x3*3x5 is at most:
    - rank(2x2*2x2) + rank(2x1*1x5) + rank(1x2*2x5) + rank(1x1*1x5)
    - = 7 + 2 + 2 + 1 = 12? No, that's not how it works.
    
    Actually, the correct upper bound using block decomposition:
    C = [A1 A2] * [B1; B2] where A1 is 2x2, A2 is 2x1, B1 is 2x5, B2 is 1x5
    C = A1*B1 + A2*B2
    
    rank(A1*B1) <= rank(2x2*2x5) 
    rank(A2*B2) <= rank(2x1*1x5)
    
    For 2x2*2x5: we can use Strassen-like structure.
    Actually, 2x2*2x5 can be done in rank 7*5/2? No.
    
    Let's just try the linear system approach more aggressively.
    """
    if time.time() > deadline:
        return None
    
    # Try various ranks with the linear system approach
    for rank in range(20, 31):
        if time.time() > deadline:
            break
        result = _try_structured_rank(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 30
    
    # Get schoolbook as guaranteed fallback
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # First, verify Strassen works (sanity check)
    # _strassen_verify()  # Uncomment for debugging
    
    # Try structured approach
    result = _try_direct_construct(n, m, p, deadline)
    if result is not None:
        best = result
        best_rank = len(result['U'][0])
    
    # If we didn't find anything better, try a few more ranks with more attempts
    if best_rank >= 28 and time.time() < deadline - 10:
        for rank in range(25, min(28, best_rank)):
            if time.time() > deadline - 5:
                break
            result = _try_structured_rank(n, m, p, rank, deadline - 5)
            if result is not None:
                best = result
                best_rank = rank
                break
    
    return best
# EVOLVE-BLOCK-END
