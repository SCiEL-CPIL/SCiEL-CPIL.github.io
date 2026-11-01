import numpy as np
import time
from itertools import permutations, product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _all_strassen_2x2():
    """
    Enumerate all rank-7 decompositions of 2x2*2x2 that are "Strassen-like":
    each bilinear product M_r uses a non-trivial linear combination of A entries
    and B entries (not just a single A[i,j]*B[j,k]).
    
    We represent each Strassen decomposition as (U, V, W) where:
    U is 4x7 (rows indexed by i*m+j), V is 4x7 (rows indexed by j*p+k), W is 4x7 (rows indexed by k*n+i).
    
    The key Strassen structure: 7 products where each involves sums/differences.
    We enumerate by trying all assignments of the 7 "slots" to the 7 Strassen patterns,
    with sign flips and permutations of A/B indices.
    """
    n, m, p = 2, 2, 2
    
    # The 7 Strassen patterns (A-support, B-support, C-support)
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # Standard Strassen patterns:
    # M1: A{0,3} * B{0,3} -> C{0,1,2,3} with signs +,-,+,+
    # M2: A{2,3} * B{0} -> C{2,3} with signs +,+
    # M3: A{0} * B{2,3} -> C{0,2} with signs +,-
    # M4: A{3} * B{0,1} -> C{1,3} with signs +,-
    # M5: A{0,1} * B{3} -> C{1,3} with signs +,+
    # M6: A{2,0} * B{0,2} -> C{2,3} with signs +,+  (A: 1,0 with signs +,-)
    # M7: A{1,3} * B{2,3} -> C{1,3} with signs +,+  (A: 1,3 with signs +,-)
    
    # Define all possible "Strassen-like" patterns
    # Each pattern is: (A_support, A_signs, B_support, B_signs, C_support, C_signs)
    # where support is a tuple of indices, signs is a tuple of +1/-1
    
    patterns = []
    
    # Pattern type 1: 2 A's * 2 B's -> 4 C's (like M1)
    for a_support in permutations(range(4), 2):
        for a_signs in product([1, -1], repeat=2):
            for b_support in permutations(range(4), 2):
                for b_signs in product([1, -1], repeat=2):
                    for c_support in permutations(range(4), 4):
                        for c_signs in product([1, -1], repeat=4):
                            patterns.append((a_support, a_signs, b_support, b_signs, c_support, c_signs))
    
    # Pattern type 2: 2 A's * 1 B -> 2 C's (like M2, M5)
    for a_support in permutations(range(4), 2):
        for a_signs in product([1, -1], repeat=2):
            for b_idx in range(4):
                for c_support in permutations(range(4), 2):
                    for c_signs in product([1, -1], repeat=2):
                        patterns.append((a_support, a_signs, (b_idx,), (1,), c_support, c_signs))
    
    # Pattern type 3: 1 A * 2 B's -> 2 C's (like M3, M4)
    for a_idx in range(4):
        for b_support in permutations(range(4), 2):
            for b_signs in product([1, -1], repeat=2):
                for c_support in permutations(range(4), 2):
                    for c_signs in product([1, -1], repeat=2):
                        patterns.append(((a_idx,), (1,), b_support, b_signs, c_support, c_signs))
    
    # Pattern type 4: 2 A's * 2 B's -> 2 C's (like M6, M7)
    for a_support in permutations(range(4), 2):
        for a_signs in product([1, -1], repeat=2):
            for b_support in permutations(range(4), 2):
                for b_signs in product([1, -1], repeat=2):
                    for c_support in permutations(range(4), 2):
                        for c_signs in product([1, -1], repeat=2):
                            patterns.append((a_support, a_signs, b_support, b_signs, c_support, c_signs))
    
    # This is too many. Let's be smarter: just use the known 7 Strassen patterns
    # and try permutations of which pattern goes in which slot, plus sign flips.
    
    # Actually, let's just use the standard 7 Strassen patterns and try:
    # 1. Permuting the 7 slots
    # 2. Sign flips on each pattern's A and B entries
    # 3. Permuting the A indices within each pattern's support
    # 4. Permuting the B indices within each pattern's support
    
    # The 7 standard Strassen patterns:
    standard_patterns = [
        # (A_support, A_signs, B_support, B_signs, C_support, C_signs)
        ((0, 3), (1, 1), (0, 3), (1, 1), (0, 1, 2, 3), (1, -1, 1, 1)),  # M1
        ((2, 3), (1, 1), (0,), (1,), (2, 3), (1, 1)),  # M2
        ((0,), (1,), (2, 3), (1, -1), (0, 2), (1, -1)),  # M3
        ((3,), (1,), (1, 0), (1, -1), (1, 3), (1, -1)),  # M4
        ((0, 1), (1, 1), (3,), (1,), (1, 3), (1, 1)),  # M5
        ((2, 0), (1, -1), (0, 2), (1, 1), (2, 3), (1, 1)),  # M6
        ((1, 3), (1, -1), (2, 3), (1, 1), (1, 3), (1, 1)),  # M7
    ]
    
    decompositions = []
    
    # Try all permutations of the 7 patterns
    for perm in permutations(range(7)):
        # For each pattern, try sign flips on A and B
        # To keep it tractable, just try the standard patterns in permuted order
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        for slot, pat_idx in enumerate(perm):
            a_sup, a_sgn, b_sup, b_sgn, c_sup, c_sgn = standard_patterns[pat_idx]
            
            for idx, a_idx in enumerate(a_sup):
                U[a_idx][slot] = a_sgn[idx]
            for idx, b_idx in enumerate(b_sup):
                V[b_idx][slot] = b_sgn[idx]
            for idx, c_idx in enumerate(c_sup):
                W[c_idx][slot] = c_sgn[idx]
        
        # Verify
        if _verify_decomposition(U, V, W, 2, 2, 2):
            decompositions.append((U, V, W))
        
        if len(decompositions) > 50:
            break
    
    if not decompositions:
        # Fallback: just the standard one
        U, V, W = _strassen_standard()
        decompositions = [(U, V, W)]
    
    return decompositions

def _strassen_standard():
    """Standard Strassen 2x2x2 decomposition."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[2][0] = 1; W[1][0] = -1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = 1
    
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[2][2] = -1
    
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[1][3] = 1; W[3][3] = -1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[3][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = 1; W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[2][6] = 1
    
    return U, V, W

def _build_rank14_from_strassen(U_s, V_s, W_s):
    """Build a rank-14 decomposition from a Strassen 2x2x2 decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # First 2x2 block (columns 0-1)
    for r in range(7):
        for i in range(n*m):
            U[i][r] = U_s[i][r]
        for j in range(m):
            for k in range(2):
                V[j*p+k][r] = V_s[j*2+k][r]
        for k in range(2):
            for i in range(n):
                W[k*n+i][r] = W_s[k*2+i][r]
    
    # Second 2x2 block (columns 2-3)
    for r in range(7):
        for i in range(n*m):
            U[i][r+7] = U_s[i][r]
        for j in range(m):
            for k in range(2):
                V[j*p+k+2][r+7] = V_s[j*2+k][r]
        for k in range(2):
            for i in range(n):
                W[(k+2)*n+i][r+7] = W_s[k*2+i][r]
    
    return U, V, W

def _try_merge_columns(U14, V14, W14, n, m, p):
    """
    Try to merge two columns from the rank-14 decomposition to get rank 13.
    
    The idea: if two columns r1 and r2 of (U, V, W) are "compatible" (their
    outer products sum to zero in some sense), we can merge them.
    
    More precisely: we want to find a linear combination a*r1 + b*r2 that
    is a valid column, and then remove one of them.
    
    Actually, the correct approach: we want to find if there's a rank-13
    decomposition by checking if any two columns can be replaced by a single
    column that preserves the tensor equation.
    
    For a rank-r decomposition, the constraint is:
    sum_{r'} U[i,j,r'] * V[j,k,r'] * W[k,i,r'] = 1 for all (i,j,k)
    
    If we have a rank-14 decomposition and want rank 13, we need to find
    a linear dependency among the 14 "rank-1 tensors" U_r ⊗ V_r ⊗ W_r
    that allows us to express one as a combination of the others.
    
    But this is nonlinear. Instead, let's try: for each pair of columns (r1, r2),
    try to find scalars a, b such that replacing columns r1 and r2 with a single
    column (a*U_r1 + b*U_r2, a*V_r1 + b*V_r2, a*W_r1 + b*W_r2) still works.
    
    This is equivalent to: the tensor T = sum_r U_r ⊗ V_r ⊗ W_r must equal
    sum_{r≠r1,r2} U_r ⊗ V_r ⊗ W_r + (a*U_r1 + b*U_r2) ⊗ (a*V_r1 + b*V_r2) ⊗ (a*W_r1 + b*W_r2)
    
    This is a system of 16 equations in 2 unknowns (a, b). Let's try it.
    """
    rank = 14
    U = np.array(U14, dtype=float)  # (4, 14)
    V = np.array(V14, dtype=float)  # (8, 14)
    W = np.array(W14, dtype=float)  # (8, 14)
    
    n, m, p = 2, 2, 4
    
    # Compute the current tensor contribution
    def tensor_value(U_arr, V_arr, W_arr):
        """Compute the 4x8x8 tensor from the decomposition."""
        result = np.zeros((n, m, p))
        for r in range(U_arr.shape[1]):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        result[i, j, k] += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
        return result
    
    # Current tensor should be all 1s
    # T = sum_r U_r ⊗ V_r ⊗ W_r
    
    # For each pair (r1, r2), try to find a, b such that:
    # sum_{r≠r1,r2} U_r ⊗ V_r ⊗ W_r + (aU_r1+bU_r2) ⊗ (aV_r1+bV_r2) ⊗ (aW_r1+bW_r2) = all 1s
    
    # Let T_base = sum_{r≠r1,r2} U_r ⊗ V_r ⊗ W_r
    # We need: T_base + (aU_r1+bU_r2) ⊗ (aV_r1+bV_r2) ⊗ (aW_r1+bW_r2) = all 1s
    # i.e., (aU_r1+bU_r2) ⊗ (aV_r1+bV_r2) ⊗ (aW_r1+bW_r2) = all 1s - T_base
    
    # The RHS is a fixed tensor. The LHS is a rank-1 tensor parameterized by (a,b).
    # A rank-1 tensor X ⊗ Y ⊗ Z satisfies X_{ij} Y_{jk} Z_{ki} = C_{ijk} for all i,j,k.
    # This is a system of 16 equations in 2 unknowns (a, b).
    
    for r1 in range(rank):
        for r2 in range(r1+1, rank):
            # Compute T_base
            T_base = np.zeros((n, m, p))
            for r in range(rank):
                if r == r1 or r == r2:
                    continue
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_base[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Target: all 1s - T_base
            target = np.ones((n, m, p)) - T_base
            
            # Check if target is a rank-1 tensor of the form (aU_r1+bU_r2) ⊗ (aV_r1+bV_r2) ⊗ (aW_r1+bW_r2)
            # For each (i,j,k): target[i,j,k] = (a*U[i,j,r1]+b*U[i,j,r2]) * (a*V[j,k,r1]+b*V[j,k,r2]) * (a*W[k,i,r1]+b*W[k,i,r2])
            
            # This is a system of 16 equations in 2 unknowns (a, b).
            # Let's set up the system and solve.
            
            # For simplicity, try a grid of (a, b) values in {-2, -1, -0.5, 0, 0.5, 1, 2}
            found = False
            for a in [-2, -1, -0.5, 0, 0.5, 1, 2]:
                for b in [-2, -1, -0.5, 0, 0.5, 1, 2]:
                    if a == 0 and b == 0:
                        continue
                    
                    # Compute the merged column
                    U_merged = a * U[:, r1] + b * U[:, r2]
                    V_merged = a * V[:, r1] + b * V[:, r2]
                    W_merged = a * W[:, r1] + b * W[:, r2]
                    
                    # Check if this gives the correct tensor
                    ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = U_merged[i*m+j] * V_merged[j*p+k] * W_merged[k*n+i]
                                if abs(val - target[i, j, k]) > 1e-9:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    
                    if ok:
                        # Found a valid merge! Build the rank-13 decomposition.
                        new_rank = rank - 1
                        U_new = [[0]*new_rank for _ in range(n*m)]
                        V_new = [[0]*new_rank for _ in range(m*p)]
                        W_new = [[0]*new_rank for _ in range(p*n)]
                        
                        col = 0
                        for r in range(rank):
                            if r == r1 or r == r2:
                                continue
                            for i in range(n*m):
                                U_new[i][col] = U[i, r]
                            for j in range(m*p):
                                V_new[j][col] = V[j, r]
                            for k in range(p*n):
                                W_new[k][col] = W[k, r]
                            col += 1
                        
                        # Add the merged column
                        for i in range(n*m):
                            U_new[i][col] = U_merged[i]
                        for j in range(m*p):
                            V_new[j][col] = V_merged[j]
                        for k in range(p*n):
                            W_new[k][col] = W_merged[k]
                        
                        # Verify
                        if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                            return U_new, V_new, W_new
                        
                        found = True
                        break
                if found:
                    break
            if found:
                break
    
    return None

def _search_rank13_systematic(n, m, p, deadline):
    """
    Systematically search for a rank-13 decomposition by:
    1. Trying different Strassen-based rank-14 constructions
    2. Attempting to merge columns in each
    """
    # Get the standard Strassen decomposition
    U_s, V_s, W_s = _strassen_standard()
    U14, V14, W14 = _build_rank14_from_strassen(U_s, V_s, W_s)
    
    # Try merging columns
    result = _try_merge_columns(U14, V14, W14, n, m, p)
    if result is not None:
        U_new, V_new, W_new = result
        if _verify_decomposition(U_new, V_new, W_new, n, m, p):
            return {"U": U_new, "V": V_new, "W": W_new}
    
    # Try with sign-flipped Strassen patterns
    # Flip signs of some patterns
    for flip_mask in range(128):  # 2^7
        if time.time() > deadline:
            break
        
        U_s_flip = [row[:] for row in U_s]
        V_s_flip = [row[:] for row in V_s]
        W_s_flip = [row[:] for row in W_s]
        
        for r in range(7):
            if flip_mask & (1 << r):
                for i in range(4):
                    U_s_flip[i][r] *= -1
                    V_s_flip[i][r] *= -1
        
        U14_f, V14_f, W14_f = _build_rank14_from_strassen(U_s_flip, V_s_flip, W_s_flip)
        result = _try_merge_columns(U14_f, V14_f, W14_f, n, m, p)
        if result is not None:
            U_new, V_new, W_new = result
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                return {"U": U_new, "V": V_new, "W": W_new}
    
    return None

def _search_rank13_ilp(n, m, p, deadline):
    """
    Use integer linear programming to search for a rank-13 decomposition.
    
    The constraint is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k).
    
    This is a trilinear system. We can fix U and V to small integer values
    and solve for W, then check if W is integer/half-integer.
    
    Strategy: fix U to have entries in {-1, 0, 1} with exactly 2 non-zeros per column,
    and V to have entries in {-1, 0, 1} with exactly 2-3 non-zeros per column.
    Then solve the linear system for W.
    """
    rank = 13
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    
    # Try many random sparse patterns
    np.random.seed(42)
    
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Generate sparse U
        for r in range(rank):
            rows = np.random.choice(n*m, size=2, replace=False)
            for row in rows:
                U[row, r] = np.random.choice([-1, 1])
        
        # Generate sparse V
        for r in range(rank):
            num_nonzero = np.random.choice([2, 3])
            rows = np.random.choice(m*p, size=num_nonzero, replace=False)
            for row in rows:
                V[row, r] = np.random.choice([-1, 1])
        
        # Build the linear system for W
        A = np.zeros((n*m*p, p*n*rank), dtype=float)
        b = np.ones(n*m*p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        coeff = U[i*m+j, r] * V[j*p+k, r]
                        if coeff != 0:
                            A[eq_idx, (k*n+i)*rank + r] = coeff
                    eq_idx += 1
        
        # Solve least squares
        try:
            W_flat, residuals, rank_sv, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        W = W_flat.reshape(p*n, rank)
        
        # Check residual
        if np.linalg.norm(A @ W_flat - b) > 1e-6:
            continue
        
        # Check if W is close to half-integers
        W_2x = W * 2
        W_rounded = np.round(W_2x)
        error = np.linalg.norm(W_2x - W_rounded)
        
        if error < 0.1:
            W_int = W_rounded / 2.0
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W_int.tolist()
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS = [2, 2, 4]
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Guaranteed fallback: rank-14 Strassen construction
    U_s, V_s, W_s = _strassen_standard()
    U14, V14, W14 = _build_rank14_from_strassen(U_s, V_s, W_s)
    best_result = {"U": U14, "V": V14, "W": W14}
    best_rank = 14
    
    # Verify fallback
    if not _verify_decomposition(U14, V14, W14, n, m, p):
        # Fall back to trivial rank-16
        rank = n * m * p
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
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    # Try systematic rank-13 search
    if time.time() < deadline - 10:
        result = _search_rank13_systematic(n, m, p, deadline)
        if result is not None:
            U, V, W = result["U"], result["V"], result["W"]
            if _verify_decomposition(U, V, W, n, m, p):
                best_result = result
                best_rank = 13
    
    # Try ILP-based rank-13 search
    if time.time() < deadline - 5 and best_rank > 13:
        result = _search_rank13_ilp(n, m, p, deadline)
        if result is not None:
            U, V, W = result["U"], result["V"], result["W"]
            if _verify_decomposition(U, V, W, n, m, p):
                best_result = result
                best_rank = 13
    
    return best_result

DIMENSIONS = [2, 2, 4]
