import numpy as np
import time
import sympy as sp
from sympy import symbols, Matrix, simplify, solve, Poly, GroebnerBasis, Rational

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Schoolbook decomposition (rank 12)
    rank_school = n * m * p
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
    best_rank = rank_school

    def get_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T_target = get_target_tensor()

    def compute_tensor(U, V, W):
        """Compute tensor from factor matrices using einsum for speed"""
        return np.einsum('ir,jr,kr->ijk', U, V, W)

    def verify(U, V, W):
        """Check if decomposition is exact"""
        T = compute_tensor(U, V, W)
        return np.allclose(T, T_target, atol=1e-9)

    # Strategy: Use known Strassen decomposition for 2x2 block (rank 7)
    # and extend to 2x3 by adding 4 more terms for the third column.
    # Total rank = 7 + 4 = 11.
    
    # Let's try to construct a rank 11 decomposition analytically.
    # A 2x2x2 matrix multiplication tensor has rank 7 (Strassen).
    # Our tensor is 2x2x3. We can view it as two 2x2x2 tensors sharing the same A and B blocks?
    # No, A is 2x2, B is 2x3.
    # C = A * B.
    # Let A = [[a11, a12], [a21, a22]]
    # Let B = [[b11, b12, b13], [b21, b22, b23]]
    # C = [[c11, c12, c13], [c21, c22, c23]]
    # c11 = a11*b11 + a12*b21
    # c12 = a11*b12 + a12*b22
    # c13 = a11*b13 + a12*b23
    # c21 = a21*b11 + a22*b21
    # c22 = a21*b12 + a22*b22
    # c23 = a21*b13 + a22*b23
    
    # We can use Strassen's 7 multiplications for the first two columns (2x2x2 part)
    # and add 4 more for the third column? Or can we reuse some?
    
    # Strassen for 2x2:
    # m1 = (a11 + a22)(b11 + b22)
    # m2 = (a21 + a22)b11
    # m3 = a11(b12 - b22)
    # m4 = a22(b21 - b11)
    # m5 = (a11 + a12)b22
    # m6 = (a21 - a11)(b11 + b12)
    # m7 = (a12 - a22)(b21 + b22)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # For the third column, we need:
    # c13 = a11*b13 + a12*b23
    # c23 = a21*b13 + a22*b23
    
    # If we just use schoolbook for the third column, that's 4 more terms. Total 11.
    # Can we do better? Maybe 10?
    
    # Let's try to solve for rank 10 and 11 using sympy for a structured ansatz.
    # Given the complexity, let's first ensure we can return a valid rank 11 solution.
    
    # Construct Rank 11 Solution:
    # First 7 terms from Strassen for columns 1 and 2.
    # Next 4 terms from Schoolbook for column 3.
    
    U_11 = np.zeros((4, 11))
    V_11 = np.zeros((6, 11))
    W_11 = np.zeros((6, 11))
    
    # Strassen terms (indices 0-6)
    # m1 = (a11 + a22)(b11 + b22)
    U_11[0, 0] = 1; U_11[3, 0] = 1
    V_11[0, 0] = 1; V_11[3, 0] = 1
    # c11 += m1, c22 += m1
    W_11[0, 0] = 1; W_11[4, 0] = 1
    
    # m2 = (a21 + a22)b11
    U_11[2, 1] = 1; U_11[3, 1] = 1
    V_11[0, 1] = 1
    # c21 += m2, c22 -= m2
    W_11[2, 1] = 1; W_11[4, 1] = -1
    
    # m3 = a11(b12 - b22)
    U_11[0, 2] = 1
    V_11[1, 2] = 1; V_11[3, 2] = -1
    # c12 += m3, c22 += m3
    W_11[1, 2] = 1; W_11[4, 2] = 1
    
    # m4 = a22(b21 - b11)
    U_11[3, 3] = 1
    V_11[2, 3] = 1; V_11[0, 3] = -1
    # c11 += m4, c21 += m4
    W_11[0, 3] = 1; W_11[2, 3] = 1
    
    # m5 = (a11 + a12)b22
    U_11[0, 4] = 1; U_11[1, 4] = 1
    V_11[3, 4] = 1
    # c11 -= m5, c12 += m5
    W_11[0, 4] = -1; W_11[1, 4] = 1
    
    # m6 = (a21 - a11)(b11 + b12)
    U_11[2, 5] = 1; U_11[0, 5] = -1
    V_11[0, 5] = 1; V_11[1, 5] = 1
    # c22 += m6
    W_11[4, 5] = 1
    
    # m7 = (a12 - a22)(b21 + b22)
    U_11[1, 6] = 1; U_11[3, 6] = -1
    V_11[2, 6] = 1; V_11[3, 6] = 1
    # c11 += m7
    W_11[0, 6] = 1
    
    # Schoolbook terms for column 3 (indices 7-10)
    # c13 = a11*b13 + a12*b23
    # c23 = a21*b13 + a22*b23
    
    # t7 = a11*b13
    U_11[0, 7] = 1
    V_11[4, 7] = 1  # b13 is at index 1*3 + 2 = 5? Wait.
    # B indices: j*p + k.
    # j=0, k=2 -> 0*3+2 = 2.
    # j=1, k=2 -> 1*3+2 = 5.
    # Let's re-verify indices.
    # A indices: i*m + j.
    # i=0, j=0 -> 0. a11
    # i=0, j=1 -> 1. a12
    # i=1, j=0 -> 2. a21
    # i=1, j=1 -> 3. a22
    
    # B indices: j*p + k.
    # j=0, k=0 -> 0. b11
    # j=0, k=1 -> 1. b12
    # j=0, k=2 -> 2. b13
    # j=1, k=0 -> 3. b21
    # j=1, k=1 -> 4. b22
    # j=1, k=2 -> 5. b23
    
    # C indices: k*n + i.
    # k=0, i=0 -> 0. c11
    # k=1, i=0 -> 1. c12
    # k=2, i=0 -> 2. c13
    # k=0, i=1 -> 3. c21
    # k=1, i=1 -> 4. c22
    # k=2, i=1 -> 5. c23
    
    # Let's re-check Strassen indices with this mapping.
    # m1 = (a11 + a22)(b11 + b22)
    # A: 0, 3. B: 0, 3.
    # C: c11 (0) += m1, c22 (4) += m1.
    # My previous code:
    # U_11[0, 0] = 1; U_11[3, 0] = 1 -> a11, a22. Correct.
    # V_11[0, 0] = 1; V_11[3, 0] = 1 -> b11, b21? NO.
    # b11 is index 0. b22 is index 4.
    # So V_11[0, 0] = 1 (b11), V_11[4, 0] = 1 (b22).
    # My previous code had V_11[3, 0] = 1 which is b21. That was wrong.
    
    # Let's restart the construction carefully.
    
    U_11 = np.zeros((4, 11))
    V_11 = np.zeros((6, 11))
    W_11 = np.zeros((6, 11))
    
    # m1 = (a11 + a22)(b11 + b22)
    U_11[0, 0] = 1; U_11[3, 0] = 1
    V_11[0, 0] = 1; V_11[4, 0] = 1
    W_11[0, 0] = 1; W_11[4, 0] = 1
    
    # m2 = (a21 + a22)b11
    U_11[2, 1] = 1; U_11[3, 1] = 1
    V_11[0, 1] = 1
    W_11[3, 1] = 1; W_11[4, 1] = -1
    
    # m3 = a11(b12 - b22)
    U_11[0, 2] = 1
    V_11[1, 2] = 1; V_11[4, 2] = -1
    W_11[1, 2] = 1; W_11[4, 2] = 1
    
    # m4 = a22(b21 - b11)
    U_11[3, 3] = 1
    V_11[3, 3] = 1; V_11[0, 3] = -1
    W_11[0, 3] = 1; W_11[3, 3] = 1
    
    # m5 = (a11 + a12)b22
    U_11[0, 4] = 1; U_11[1, 4] = 1
    V_11[4, 4] = 1
    W_11[0, 4] = -1; W_11[1, 4] = 1
    
    # m6 = (a21 - a11)(b11 + b12)
    U_11[2, 5] = 1; U_11[0, 5] = -1
    V_11[0, 5] = 1; V_11[1, 5] = 1
    W_11[4, 5] = 1
    
    # m7 = (a12 - a22)(b21 + b22)
    U_11[1, 6] = 1; U_11[3, 6] = -1
    V_11[3, 6] = 1; V_11[4, 6] = 1
    W_11[0, 6] = 1
    
    # Check if this part is correct for cols 1,2.
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # Now add 4 terms for col 3.
    # c13 = a11*b13 + a12*b23
    # c23 = a21*b13 + a22*b23
    
    # t7 = a11*b13
    U_11[0, 7] = 1
    V_11[2, 7] = 1
    W_11[2, 7] = 1
    
    # t8 = a12*b23
    U_11[1, 8] = 1
    V_11[5, 8] = 1
    W_11[2, 8] = 1
    
    # t9 = a21*b13
    U_11[2, 9] = 1
    V_11[2, 9] = 1
    W_11[5, 9] = 1
    
    # t10 = a22*b23
    U_11[3, 10] = 1
    V_11[5, 10] = 1
    W_11[5, 10] = 1
    
    if verify(U_11, V_11, W_11):
        best_U, best_V, best_W = U_11.tolist(), V_11.tolist(), W_11.tolist()
        best_rank = 11
    
    # Now try to find rank 10 using Sympy.
    # This is a system of polynomial equations.
    # We can try to solve for the entries of U, V, W for rank 10.
    # Due to symmetry, we can fix some entries to reduce the search space.
    
    if time.time() < deadline - 10:
        # Try rank 10
        # Variables: 4*10 + 6*10 + 6*10 = 160 variables.
        # Equations: 2*2*3 = 12 equations (from tensor components).
        # Actually, the tensor has 4*6*6 = 144 components, but most are 0.
        # Non-zero components:
        # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i.
        # For each (i,j,k), there is one non-zero entry.
        # There are 2*2*3 = 12 such entries.
        # So we have 12 equations of degree 3 (product of 3 variables).
        # Plus 132 equations that T[a,b,c] = 0 for other indices.
        # Total 144 equations.
        
        # Solving this directly with Sympy might be too slow.
        # Let's use a smarter approach: fix the structure.
        # We know rank 11 is possible. Can we merge two terms?
        
        # Let's try a random search for rank 10 with a tighter constraint set.
        # Allowed values: -1, 0, 1, 2, -2, 1/2, -1/2.
        allowed_vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        
        def try_rank_10(max_attempts=200):
            nonlocal best_U, best_V, best_W, best_rank
            for attempt in range(max_attempts):
                if time.time() > deadline - 5.0:
                    return False
                
                U = np.random.choice(allowed_vals, size=(4, 10))
                V = np.random.choice(allowed_vals, size=(6, 10))
                W = np.random.choice(allowed_vals, size=(6, 10))
                
                if verify(U, V, W):
                    best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
                    best_rank = 10
                    return True
            return False
        
        try_rank_10()
    
    return {"U": best_U, "V": best_V, "W": best_W}
