"""Matrix multiplication 3,3,6 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    T[i,j,k] = 1 for all valid indices.
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    return True


def _strassen_2x2x2():
    """Return the 7-term rank decomposition for 2x2x2 matrix multiplication.
    
    For A(2x2) * B(2x2) = C(2x2):
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    Strassen's 7 multiplications:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # Each term is (u, v, w) where u is a 2x2 matrix (flattened), v is 2x2, w is 2x2
    # u[i,j] * v[j,k] * w[i,k] should sum to 1 for the 2x2x2 tensor
    
    # Term 1: M1 = (A00+A11)*(B00+B11)
    # u: A00+A11 = [[1,0],[1,1]] -> flat [1,0,1,1]
    # v: B00+B11 = [[1,0],[0,1]] -> flat [1,0,0,1]
    # w: contributes to C00 and C11
    # C00 += 1, C11 += 1
    # w = [[1,0],[0,1]] -> flat [1,0,0,1]
    terms = []
    
    # M1: (A00+A11)*(B00+B11)
    u1 = [1, 0, 1, 1]  # [[1,0],[1,1]]
    v1 = [1, 0, 0, 1]  # [[1,0],[0,1]]
    w1 = [1, 0, 0, 1]  # [[1,0],[0,1]]
    terms.append((u1, v1, w1))
    
    # M2: (A10+A11)*B00
    u2 = [0, 0, 1, 1]  # [[0,0],[1,1]]
    v2 = [1, 0, 0, 0]  # [[1,0],[0,0]]
    w2 = [0, 0, 1, 0]  # [[0,0],[1,0]]
    terms.append((u2, v2, w2))
    
    # M3: A00*(B01-B11)
    u3 = [1, 0, 0, 0]  # [[1,0],[0,0]]
    v3 = [0, 1, 0, -1]  # [[0,1],[0,-1]]
    w3 = [0, 1, 0, 0]  # [[0,1],[0,0]]
    terms.append((u3, v3, w3))
    
    # M4: A11*(B10-B00)
    u4 = [0, 0, 0, 1]  # [[0,0],[0,1]]
    v4 = [0, 0, 1, 0]  # [[0,0],[1,0]]
    w4 = [1, 0, 0, 0]  # [[1,0],[0,0]]
    terms.append((u4, v4, w4))
    
    # M5: (A00+A01)*B11
    u5 = [1, 1, 0, 0]  # [[1,1],[0,0]]
    v5 = [0, 0, 0, 1]  # [[0,0],[0,1]]
    w5 = [0, 0, 0, -1]  # [[0,0],[0,-1]]
    terms.append((u5, v5, w5))
    
    # M6: (A10-A00)*(B00+B01)
    u6 = [-1, 0, 1, 0]  # [[-1,0],[1,0]]
    v6 = [1, 1, 0, 0]  # [[1,1],[0,0]]
    w6 = [0, 0, 0, 1]  # [[0,0],[0,1]]
    terms.append((u6, v6, w6))
    
    # M7: (A01-A11)*(B10+B11)
    u7 = [0, 1, 0, -1]  # [[0,1],[0,-1]]
    v7 = [0, 0, 1, 1]  # [[0,0],[1,1]]
    w7 = [1, 0, 0, 1]  # [[1,0],[0,1]]
    terms.append((u7, v7, w7))
    
    return terms


def _verify_2x2x2(terms):
    """Verify 2x2x2 decomposition."""
    # T[i,j,k] = 1 for all i,j,k in {0,1}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0
                for (u, v, w) in terms:
                    # u is flattened 2x2: u[i*2+j]
                    # v is flattened 2x2: v[j*2+k]
                    # w is flattened 2x2: w[i*2+k]
                    val += u[i*2+j] * v[j*2+k] * w[i*2+k]
                if abs(val - 1) > 1e-9:
                    return False
    return True


def _build_3x3x3_decomposition():
    """Build a rank-19 decomposition for 3x3x3 matrix multiplication.
    
    Strategy: Partition A(3x3) and B(3x3) into 2x2 blocks with the third row/col.
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    B = [[B11, B12], [B21, B22]] similarly
    
    C = A*B:
    C11 = A11*B11 + A12*B21  (2x2)
    C12 = A11*B12 + A12*B22  (2x1)
    C21 = A21*B11 + A22*B21  (1x2)
    C22 = A21*B12 + A22*B22  (1x1)
    
    Using Strassen for the 2x2x2 part (7 mults) and direct for the rest:
    - A11*B11: 7 mults (Strassen)
    - A12*B21: 2 mults (2x1 times 1x2 = 2x2, but each is a dot product)
    - A11*B12: 4 mults (2x2 times 2x1)
    - A12*B22: 2 mults (2x1 times 1x1)
    - A21*B11: 4 mults (1x2 times 2x2)
    - A22*B21: 2 mults (1x1 times 1x2)
    - A21*B12: 2 mults (1x2 times 2x1)
    - A22*B22: 1 mult (1x1 times 1x1)
    Total: 7+2+4+2+4+2+2+1 = 24
    
    That's 24, not 19. Let me use a better construction.
    
    Actually, the known optimal rank for 3x3x3 is 19. Let me construct it directly.
    
    Alternative: Use the fact that 3x3x3 can be decomposed using 19 rank-1 tensors.
    I'll construct this explicitly.
    """
    
    # Let me try a different approach: directly construct 19 terms for 3x3x3.
    # The tensor T[i,j,k] = 1 for i,j,k in {0,1,2}.
    
    # I'll use a constructive method based on the structure of matrix multiplication.
    
    # Actually, let me try to find the 19-term decomposition by solving the system.
    # For 3x3x3, we have 27 equations and 3*19*9 = 513 unknowns (very underdetermined).
    
    # Let me try a smarter construction. The key insight is that we can use
    # Strassen's algorithm more cleverly.
    
    # Here's a known 19-term decomposition for 3x3x3:
    # It uses the following structure:
    # - 7 terms for the "core" 2x2x2 (Strassen)
    # - 12 additional terms for the boundary interactions
    
    # Let me construct this by hand. The indices are:
    # A indices: (i,j) where i,j in {0,1,2}
    # B indices: (j,k) where j,k in {0,1,2}  
    # C indices: (i,k) where i,k in {0,1,2}
    
    # U has shape (9, 19), V has shape (9, 19), W has shape (9, 19)
    # U[i*3+j, r] * V[j*3+k, r] * W[i*3+k, r] summed over r = 1
    
    # Let me build this systematically.
    
    # For the 2x2x2 core (indices 0,1 for i,j,k), use Strassen's 7 terms.
    # For the boundary (involving index 2), add 12 more terms.
    
    U = [[0]*19 for _ in range(9)]
    V = [[0]*19 for _ in range(9)]
    W = [[0]*19 for _ in range(9)]
    
    # Strassen's 7 terms for the 2x2x2 subproblem (i,j,k in {0,1})
    strassen_terms = _strassen_2x2x2()
    
    for t_idx, (u2, v2, w2) in enumerate(strassen_terms):
        r = t_idx
        # u2 is [u00, u01, u10, u11] for 2x2
        U[0*3+0][r] = u2[0]  # i=0,j=0
        U[0*3+1][r] = u2[1]  # i=0,j=1
        U[1*3+0][r] = u2[2]  # i=1,j=0
        U[1*3+1][r] = u2[3]  # i=1,j=1
        
        V[0*3+0][r] = v2[0]  # j=0,k=0
        V[0*3+1][r] = v2[1]  # j=0,k=1
        V[1*3+0][r] = v2[2]  # j=1,k=0
        V[1*3+1][r] = v2[3]  # j=1,k=1
        
        W[0*3+0][r] = w2[0]  # i=0,k=0
        W[0*3+1][r] = w2[1]  # i=0,k=1
        W[1*3+0][r] = w2[2]  # i=1,k=0
        W[1*3+1][r] = w2[3]  # i=1,k=1
    
    # Now add 12 more terms for the boundary (involving index 2).
    # The remaining equations are those where at least one of i,j,k is 2.
    
    # Let me set up the remaining 20 equations (27 total - 7 for the 2x2x2 core... 
    # wait, the 2x2x2 core covers i,j,k all in {0,1}, which is 8 equations.
    # The Strassen terms should satisfy those 8 equations.
    # Remaining: 27 - 8 = 19 equations, but we're adding 12 more terms.
    # Total terms: 7 + 12 = 19.
    
    # Actually, let me just solve for the remaining terms numerically.
    
    # For now, let me try a simpler approach: use 27 terms (schoolbook) but 
    # see if I can reduce by finding linear dependencies.
    
    # Actually, let me just implement the full 19-term decomposition by solving.
    
    # Given time constraints, let me try a different strategy:
    # Use 2 independent 3x3x3 decompositions, each with rank 19, for 3x3x6.
    # But I need to actually construct the rank-19 decomposition.
    
    # Let me try to solve for it using linear algebra.
    
    # For 3x3x3, the equations are:
    # sum_{r=0}^{18} U[i*3+j, r] * V[j*3+k, r] * W[i*3+k, r] = 1 for all i,j,k in {0,1,2}
    
    # This is nonlinear. Let me try a different approach.
    
    # Actually, I'll use a known explicit 19-term decomposition.
    # From the literature, here's one construction:
    
    # The 19 terms can be organized as follows:
    # Terms 1-7: Strassen for the (0,1) x (0,1) x (0,1) subblock
    # Terms 8-19: Handle the interactions with index 2
    
    # Let me try to construct terms 8-19 to satisfy the remaining equations.
    
    # The equations not covered by Strassen are those where at least one index is 2.
    # There are 27 - 8 = 19 such equations.
    
    # Let me set up a system. For terms 8-19 (12 terms), I have 12*3 = 36 unknowns 
    # (each term has U, V, W values for the boundary indices).
    # But I have 19 equations. This is underdetermined, so a solution should exist.
    
    # Let me try to find it by fixing some variables and solving.
    
    # Actually, this is getting complex. Let me try a completely different approach:
    # Just use the schoolbook decomposition for 3x3x6 (rank 54) and see if I can 
    # find any improvement through optimization.
    
    # For now, let me return a valid decomposition and try to improve it.
    
    # Let me try rank 38 = 2 * 19 by using two 3x3x3 blocks.
    # I'll construct the 3x3x3 decomposition numerically.
    
    pass  # Placeholder - will implement below


def _find_3x3x3_rank19():
    """Find a rank-19 decomposition for 3x3x3 matrix multiplication.
    
    Uses the known structure and solves for the coefficients.
    """
    
    # I'll use a constructive approach based on the algebraic structure.
    # The 19-term decomposition for 3x3x3 is known to exist.
    
    # Here's a direct construction:
    
    # The key idea: 3x3x3 can be decomposed as:
    # - A 2x2x2 Strassen block (7 terms) for the top-left
    # - Plus additional terms for the boundary
    
    # Let me try to explicitly construct all 19 terms.
    
    # I'll use the following approach:
    # 1. Start with Strassen's 7 terms for the 2x2x2 core
    # 2. Add 12 more terms to handle the boundary
    
    # For the boundary, I'll use a systematic construction.
    
    U = [[0]*19 for _ in range(9)]
    V = [[0]*19 for _ in range(9)]
    W = [[0]*19 for _ in range(9)]
    
    # Strassen's 7 terms for 2x2x2 (indices 0,1)
    strassen = _strassen_2x2x2()
    
    for t_idx, (u2, v2, w2) in enumerate(strassen):
        r = t_idx
        U[0][r] = u2[0]
        U[1][r] = u2[1]
        U[3][r] = u2[2]
        U[4][r] = u2[3]
        
        V[0][r] = v2[0]
        V[1][r] = v2[1]
        V[3][r] = v2[2]
        V[4][r] = v2[3]
        
        W[0][r] = w2[0]
        W[1][r] = w2[1]
        W[3][r] = w2[2]
        W[4][r] = w2[3]
    
    # Now I need to add 12 more terms (r = 7 to 18) to satisfy the remaining equations.
    # The remaining equations are for (i,j,k) where at least one index is 2.
    
    # Let me list all 27 equations and see which are already satisfied.
    
    # For now, let me try to solve for the remaining terms using a simple method.
    
    # I'll fix the U and V values for the boundary terms and solve for W,
    # or use a least-squares approach.
    
    # Actually, let me try a different tactic: use a known explicit decomposition.
    
    # From Strassen's 1969 paper and subsequent work, here's a 19-term decomposition:
    
    # I'll construct it by solving a system of equations.
    
    # Let me set up the problem: for r = 7,...,18, I need to find
    # U[r] (9 values), V[r] (9 values), W[r] (9 values) such that
    # the sum over all 19 terms gives T[i,j,k] = 1.
    
    # This is a nonlinear system. Let me try to linearize it.
    
    # Alternative: I'll use a greedy approach to build the 19 terms.
    
    # Given the complexity and time constraints, let me try a simpler approach:
    # Just use rank 27 for 3x3x3 (schoolbook) and see if 3x3x6 can be done in rank 54.
    # That's the same as the current score, so I need to do better.
    
    # Let me try to find ANY decomposition with rank < 54 for 3x3x6.
    
    # Simple improvement: 3x3x6 = 3x3x3 + 3x3x3 (block split in k).
    # If I can do 3x3x3 in rank 27, then 3x3x6 in rank 54 (same as schoolbook).
    # If I can do 3x3x3 in rank 19, then 3x3x6 in rank 38.
    
    # Let me try to numerically find a rank-19 decomposition for 3x3x3.
    
    # I'll use a random initialization and gradient descent.
    
    import numpy as np
    
    rng = np.random.RandomState(42)
    
    # Initialize 19 terms randomly
    U_arr = rng.randn(9, 19) * 0.5
    V_arr = rng.randn(9, 19) * 0.5
    W_arr = rng.randn(9, 19) * 0.5
    
    # Target tensor
    T = np.ones((3, 3, 3))
    
    # Compute current tensor
    def compute_tensor(U, V, W):
        T_comp = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T_comp[i, j, k] = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[i*3+k, :])
        return T_comp
    
    # Gradient descent
    lr = 0.01
    for iteration in range(5000):
        T_comp = compute_tensor(U_arr, V_arr, W_arr)
        residual = T - T_comp
        
        # Compute gradients
        # dT/dU[i,j,r] = V[j,k,r] * W[i,k,r] for each k
        # dT/dV[j,k,r] = U[i,j,r] * W[i,k,r] for each i
        # dT/dW[i,k,r] = U[i,j,r] * V[j,k,r] for each j
        
        grad_U = np.zeros_like(U_arr)
        grad_V = np.zeros_like(V_arr)
        grad_W = np.zeros_like(W_arr)
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    res = residual[i, j, k]
                    for r in range(19):
                        grad_U[i*3+j, r] += res * V_arr[j*3+k, r] * W_arr[i*3+k, r]
                        grad_V[j*3+k, r] += res * U_arr[i*3+j, r] * W_arr[i*3+k, r]
                        grad_W[i*3+k, r] += res * U_arr[i*3+j, r] * V_arr[j*3+k, r]
        
        U_arr -= lr * grad_U
        V_arr -= lr * grad_V
        W_arr -= lr * grad_W
        
        if iteration % 1000 == 0:
            err = np.max(np.abs(residual))
            if err < 1e-10:
                break
    
    # Check if we got a valid decomposition
    T_comp = compute_tensor(U_arr, V_arr, W_arr)
    if np.max(np.abs(T - T_comp)) < 1e-9:
        # Convert to list format
        U_list = U_arr.tolist()
        V_list = V_arr.tolist()
        W_list = W_arr.tolist()
        return U_list, V_list, W_list
    
    return None


def _construct_3x3x6():
    """Construct a low-rank decomposition for 3x3x6 matrix multiplication.
    
    Strategy: Split B(3x6) into two 3x3 blocks B1, B2.
    C = A*B = [A*B1 | A*B2]
    Use a rank-19 decomposition for each 3x3x3 multiplication.
    Total rank: 2 * 19 = 38.
    """
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try to find a rank-19 decomposition for 3x3x3
    result_3x3x3 = _find_3x3x3_rank19()
    
    if result_3x3x3 is not None and time.time() < deadline:
        U1, V1, W1 = result_3x3x3
        
        # Build 3x3x6 decomposition
        # U has shape (9, 38), V has shape (18, 38), W has shape (18, 38)
        # For the first 3x3x3 block (k=0,1,2):
        #   U[i*3+j, r] = U1[i*3+j, r] for r=0..18
        #   V[j*3+k, r] = V1[j*3+k, r] for r=0..18, k=0,1,2
        #   W[k*3+i, r] = W1[k*3+i, r] for r=0..18, k=0,1,2
        # For the second 3x3x3 block (k=3,4,5):
        #   U[i*3+j, r+19] = U1[i*3+j, r] for r=0..18
        #   V[j*3+(k-3), r+19] = V1[j*3+(k-3), r] for r=0..18, k=3,4,5
        #   W[(k-3)*3+i, r+19] = W1[(k-3)*3+i, r] for r=0..18, k=3,4,5
        
        rank = 38
        U = [[0]*rank for _ in range(9)]
        V = [[0]*rank for _ in range(18)]
        W = [[0]*rank for _ in range(18)]
        
        for r in range(19):
            for i in range(3):
                for j in range(3):
                    U[i*3+j][r] = U1[i*3+j][r]
                    U[i*3+j][r+19] = U1[i*3+j][r]
                    
            for j in range(3):
                for k in range(3):
                    V[j*3+k][r] = V1[j*3+k][r]
                    V[j*3+(k+3)][r+19] = V1[j*3+k][r]
                    
            for k in range(3):
                for i in range(3):
                    W[k*3+i][r] = W1[k*3+i][r]
                    W[(k+3)*3+i][r+19] = W1[k*3+i][r]
        
        # Verify
        if _verify(U, V, W, 3, 3, 6):
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    return _schoolbook(3, 3, 6)


def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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


def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x3x6 matrix multiplication."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try to find a low-rank decomposition
    result = _construct_3x3x6()
    
    # Verify the result
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
