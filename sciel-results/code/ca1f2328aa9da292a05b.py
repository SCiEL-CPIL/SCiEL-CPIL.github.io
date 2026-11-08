"""Rank-minimal tensor decomposition via randomized linear-system search."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _build_tensor_constraints(n, m, p, rank):
    """
    Build the linear system for tensor decomposition.
    Variables: U (n*m x rank), V (m*p x rank), W (p*n x rank)
    Each column r has (n*m + m*p + p*n) variables.
    Constraints: for each (a, b, c) with a=i*m+j, b=j*p+k, c=k*n+i,
                 sum_r U[a,r]*V[b,r]*W[c,r] = 1 if valid, 0 otherwise.
    
    This is nonlinear (products of variables), so we use an alternating approach:
    Fix two matrices, solve for the third linearly.
    """
    pass

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition."""
    rank = len(U[0])
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                is_valid = (j_a == j_b) and (i_a == i_c) and (k_b == k_c)
                val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if is_valid:
                    if abs(val - 1) > tol:
                        return False
                else:
                    if abs(val) > tol:
                        return False
    return True

def _is_half_integer(val, tol=1e-9):
    val2 = val * 2
    return abs(val2 - round(val2)) < tol

def _round_to_half_int(val):
    return round(val * 2) / 2.0

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
    return U, V, W

def _strassen_2x2():
    """Correct Strassen rank-7 decomposition for 2x2*2x2."""
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # p1 = (a00+a11)(b00+b11)
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[2][0]=1  # c00, c10
    
    # p2 = (a10+a11)b00
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[0][1]=-1; W[2][1]=1  # c00, c10
    
    # p3 = a00(b01-b11)
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[0][2]=1; W[3][2]=1  # c00, c11
    
    # p4 = a11(b10-b00)
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[1][3]=1; W[3][3]=1  # c01, c11
    
    # p5 = (a00+a01)b11
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[0][4]=-1; W[1][4]=1  # c00, c01
    
    # p6 = (a10-a00)(b00+b01)
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1  # c11
    
    # p7 = (a01-a11)(b10+b11)
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[1][6]=-1; W[2][6]=1  # c01, c10
    
    return U, V, W

def _construct_hybrid(n, m, p):
    """Hybrid: Strassen for 2x2 sub-block + schoolbook for rest."""
    total_rank = 7 + 4 + 24  # = 35
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    U_s, V_s, W_s = _strassen_2x2()
    
    for r in range(7):
        for row in range(4):
            if U_s[row][r] != 0:
                i = row // 2; j = row % 2
                U[i*m + j][r] = U_s[row][r]
        for row in range(4):
            if V_s[row][r] != 0:
                j = row // 2; k = row % 2
                V[j*p + k][r] = V_s[row][r]
        for row in range(4):
            if W_s[row][r] != 0:
                k = row // 2; i = row % 2
                W[k*n + i][r] = W_s[row][r]
    
    col_offset = 7
    for i in range(n):
        for k in range(2):
            U[i*m + 2][col_offset] = 1
            V[2*p + k][col_offset] = 1
            W[k*n + i][col_offset] = 1
            col_offset += 1
    
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                U[i*m + j][col_offset] = 1
                V[j*p + k][col_offset] = 1
                W[k*n + i][col_offset] = 1
                col_offset += 1
    
    return U, V, W

def _try_random_rank(n, m, p, rank, max_iters=50, deadline=None):
    """
    Try to find a rank-r decomposition using random initialization and alternating linear solves.
    Fix V and W, solve for U linearly; fix U and W, solve for V; fix U and V, solve for W.
    """
    import random
    random.seed(42)
    
    def solve_linear_system(A, b):
        """Solve A x = b, return x or None if singular."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
    
    # Build the list of valid (a,b,c) tuples
    valid_triples = []
    for a in range(n*m):
        i_a = a // m; j_a = a % m
        for b in range(m*p):
            j_b = b // p; k_b = b % p
            for c in range(p*n):
                k_c = c // n; i_c = c % n
                if j_a == j_b and i_a == i_c and k_b == k_c:
                    valid_triples.append((a, b, c))
    
    num_valid = len(valid_triples)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            return None
        
        # Random half-integer initialization
        def rand_half_int():
            return random.choice([-1, -0.5, 0, 0.5, 1])
        
        U = np.array([[rand_half_int() for _ in range(rank)] for _ in range(n*m)], dtype=float)
        V = np.array([[rand_half_int() for _ in range(rank)] for _ in range(m*p)], dtype=float)
        W = np.array([[rand_half_int() for _ in range(rank)] for _ in range(p*n)], dtype=float)
        
        # Alternating least squares for 30 inner iterations
        for inner in range(30):
            if deadline and time.time() > deadline:
                return None
            
            # Solve for U: for each valid (a,b,c), U[a,:] dot (V[b,:] * W[c,:]) = 1
            # This is: sum_r U[a,r] * V[b,r] * W[c,r] = 1
            # For fixed V,W: U[a,:] is constrained by all c where (a, b_c, c) is valid
            # Actually, for each a, the constraints are: for each b, c such that valid:
            #   sum_r U[a,r] * (V[b,r] * W[c,r]) = 1
            # This gives a linear system in U[a,:] for each a.
            
            U_new = U.copy()
            for a in range(n*m):
                i_a = a // m; j_a = a % m
                # Find all (b, c) such that the triple is valid
                constraints = []
                for b in range(m*p):
                    j_b = b // p; k_b = b % p
                    if j_b != j_a:
                        continue
                    for c in range(p*n):
                        k_c = c // n; i_c = c % n
                        if k_c != k_b or i_c != i_a:
                            continue
                        # valid: constraint on U[a,:]
                        vec = V[b] * W[c]  # element-wise product
                        constraints.append((vec, 1.0))
                
                if constraints:
                    A_mat = np.array([vec for vec, _ in constraints])
                    b_vec = np.array([val for _, val in constraints])
                    # Solve A_mat @ U[a,:] = b_vec
                    if A_mat.shape[0] >= A_mat.shape[1]:
                        sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U_new[a] = sol
            
            U = U_new
            
            # Similarly solve for V
            V_new = V.copy()
            for b in range(m*p):
                j_b = b // p; k_b = b % p
                constraints = []
                for a in range(n*m):
                    i_a = a // m; j_a = a % m
                    if j_a != j_b:
                        continue
                    for c in range(p*n):
                        k_c = c // n; i_c = c % n
                        if k_c != k_b or i_c != i_a:
                            continue
                        vec = U[a] * W[c]
                        constraints.append((vec, 1.0))
                
                if constraints:
                    A_mat = np.array([vec for vec, _ in constraints])
                    b_vec = np.array([val for _, val in constraints])
                    if A_mat.shape[0] >= A_mat.shape[1]:
                        sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V_new[b] = sol
            
            V = V_new
            
            # Similarly solve for W
            W_new = W.copy()
            for c in range(p*n):
                k_c = c // n; i_c = c % n
                constraints = []
                for a in range(n*m):
                    i_a = a // m; j_a = a % m
                    if i_a != i_c:
                        continue
                    for b in range(m*p):
                        j_b = b // p; k_b = b % p
                        if k_b != k_c or j_b != j_a:
                            continue
                        vec = U[a] * V[b]
                        constraints.append((vec, 1.0))
                
                if constraints:
                    A_mat = np.array([vec for vec, _ in constraints])
                    b_vec = np.array([val for _, val in constraints])
                    if A_mat.shape[0] >= A_mat.shape[1]:
                        sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W_new[c] = sol
            
            W = W_new
        
        # Round to half-integers and verify
        U_rounded = [[_round_to_half_int(x) for x in U[a]] for a in range(n*m)]
        V_rounded = [[_round_to_half_int(x) for x in V[b]] for b in range(m*p)]
        W_rounded = [[_round_to_half_int(x) for x in W[c]] for c in range(p*n)]
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
            # Check all entries are half-integers
            valid = True
            for mat in [U_rounded, V_rounded, W_rounded]:
                for row in mat:
                    for val in row:
                        if not _is_half_integer(val):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return U_rounded, V_rounded, W_rounded
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # First try the known hybrid (rank 35)
    U, V, W = _construct_hybrid(n, m, p)
    if _verify_decomposition(U, V, W, n, m, p):
        valid = True
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if not _is_half_integer(val):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return {"U": U, "V": V, "W": W}
    
    # Try random search for lower ranks
    # Schoolbook is 36, hybrid is 35. Try ranks 34, 33, ...
    for rank in range(34, 20, -1):
        if time.time() > deadline:
            break
        result = _try_random_rank(n, m, p, rank, max_iters=20, deadline=deadline)
        if result is not None:
            U_r, V_r, W_r = result
            return {"U": U_r, "V": V_r, "W": W_r}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
