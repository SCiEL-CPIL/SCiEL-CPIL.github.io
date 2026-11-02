import numpy as np
import time
from itertools import product

def construct():
    """
    Finds a low-rank exact tensor decomposition for 3x3x6 matrix multiplication.
    Strategy: Exploit block structure. 3x3x6 can be seen as 3 blocks of 3x3x2.
    We search for a rank-13 decomposition for 3x3x2 (using known optimal results from literature/Strassen variants)
    and combine them. If a lower rank exists for the whole 3x3x6, we might find it via direct search,
    but the block structure is a strong prior.
    
    Actually, the known optimal rank for 3x3x2 is 13.
    Total rank would be 3 * 13 = 39.
    Schoolbook is 54.
    
    However, we can also try to find a direct decomposition of 3x3x6 with rank < 39.
    Let's first try to construct the rank-39 solution using the known rank-13 for 3x3x2.
    Then, if time permits, search for lower ranks.
    
    Known rank-13 decomposition for 3x3x2:
    This is non-trivial to generate from scratch. I will implement a search for it or use a hardcoded one if I can recall/derive it.
    Since I cannot reliably recall the specific 13-term coefficients for 3x3x2, I will attempt to find them via a constrained search or use a simpler construction.
    
    Alternative: Use the standard Strassen-like recursion or known small matrix multiplication algorithms.
    For 3x3x2, the tensor is T[i,j,k] where i in {0,1,2} (rows of A), j in {0,1,2} (cols of A / rows of B), k in {0,1} (cols of B).
    Wait, the problem is 3x3 * 3x6.
    Let A be 3x3, B be 3x6. C = A*B is 3x6.
    We can split B into B1 (3x2) and B2 (3x2) and B3 (3x2).
    Then C = [A*B1, A*B2, A*B3].
    Each A*Bt is 3x2.
    So we need 3 decompositions of 3x3x2.
    If we use the same decomposition for all three (which we can, since it's just a direct sum), the rank adds up.
    So Rank(3x3x6) <= 3 * Rank(3x3x2).
    
    What is Rank(3x3x2)?
    Lower bound: max(3*2, 3*2, 3*3) is not right.
    Lower bound is max(n*p, m*p, n*m) ? No.
    For 3x3x2, the tensor size is 3*3*2 = 18.
    Schoolbook rank is 18.
    Can we do better?
    Strassen is 2x2x2 rank 7.
    3x3x2 is related to 2x2x2?
    Actually, the optimal rank for 3x3x2 is known to be 13.
    (See "The rank of 3x3x2 matrix multiplication" or similar papers).
    
    If I cannot hardcode the 13-term decomposition, I can try to find it via search.
    However, searching for rank 13 in 3x3x2 space (9*13 + 6*13 + 6*13 variables) is hard.
    
    Let's try a different approach: Search for a direct decomposition of 3x3x6 with rank < 54.
    Maybe rank 39 is achievable.
    Or maybe even lower?
    
    Let's first generate the rank-39 solution using a rank-13 3x3x2 decomposition.
    Since I don't have the 13-term decomposition hardcoded, I will try to find a rank-14 or rank-15 decomposition for 3x3x2 via search, which would give total rank 42 or 45.
    Or, I can use the standard Strassen algorithm extended?
    
    Actually, there is a known rank-13 algorithm for 3x3x2.
    Let me try to construct it or find it via a quick search in the code.
    Given the time limit, searching for rank 13 is risky.
    
    Let's try to find a rank-14 decomposition for 3x3x2.
    U: 9x14, V: 6x14, W: 6x14.
    Tensor equation: sum_r U[i,j,r] * V[j,k,r] * W[i',k',r] = delta...
    Wait, for 3x3x2:
    A is 3x3, B is 3x2. C is 3x2.
    Indices:
    A[i,j] i in 0..2, j in 0..2.
    B[j,k] j in 0..2, k in 0..1.
    C[i,k] i in 0..2, k in 0..1.
    T[(i,j),(j',k),(i',k')] = 1 iff i=i', j=j', k=k'.
    
    Equation: sum_r U[i*3+j, r] * V[j'*2+k, r] * W[i'*2+k', r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    This is a system of 9*6*6 = 324 equations.
    Variables: 9*R + 6*R + 6*R = 21R.
    For R=14, 294 variables.
    
    We can use a numerical solver to find a solution, then round to integers/half-integers.
    
    Let's try to find a rank-13 or 14 decomposition for 3x3x2.
    """
    start_time = time.time()
    deadline = start_time + 100.0
    
    n, m, p = 3, 3, 6
    
    def generate_schoolbook():
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
    
    best_result = generate_schoolbook()
    best_rank = n * m * p
    
    # Try to find a rank-R decomposition for 3x3x2
    # If found, extend to 3x3x6 with rank 3*R
    # Target R: 13, 14, 15...
    
    def find_3x3x2_decomposition(R, max_attempts=200):
        """
        Search for rank-R decomposition of 3x3x2.
        Returns (U, V, W) or None.
        U: 9xR, V: 6xR, W: 6xR
        """
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Random initialization
            U_r = np.random.randint(-2, 3, size=(9, R))
            V_r = np.random.randint(-2, 3, size=(6, R))
            W_r = np.zeros((6, R))
            
            # Solve for W given U, V
            # For each (i,k), we have equations for j, j', k'
            # sum_r U[i*3+j, r] * V[j'*2+k, r] * W[i'*2+k', r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
            
            # Fix i, k.
            # The terms involving W[i*2+k, r] are when i'=i and k'=k.
            # Then sum_r U[i*3+j, r] * V[j'*2+k, r] * W[i*2+k, r] = delta_{j,j'}
            # This is a system of 9 equations (j, j' in 0..2) for R unknowns W[i*2+k, r].
            # Matrix M: 9xR, M[j*3+j', r] = U[i*3+j, r] * V[j'*2+k, r]
            # RHS: delta_{j,j'}
            
            valid = True
            for i in range(3):
                for k in range(2):
                    M = np.zeros((9, R))
                    for j in range(3):
                        for jp in range(3):
                            M[j * 3 + jp, :] = U_r[i * 3 + j, :] * V_r[jp * 2 + k, :]
                    
                    rhs = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
                    
                    try:
                        w_vec, residuals, rank_m, s = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if w_vec is close to integers or half-integers
                    w_doubled = w_vec * 2
                    closest_ints = np.round(w_doubled)
                    error = np.max(np.abs(w_doubled - closest_ints))
                    
                    if error < 1e-6:
                        W_r[i * 2 + k, :] = w_vec
                    else:
                        valid = False
                        break
            
            if valid:
                # Verify full tensor
                U_np = U_r.astype(float)
                V_np = V_r.astype(float)
                W_np = W_r.astype(float)
                
                correct = True
                for i in range(3):
                    for ip in range(3):
                        for k in range(2):
                            for kp in range(2):
                                for j in range(3):
                                    for jp in range(3):
                                        expected = 1.0 if (j == jp and i == ip and k == kp) else 0.0
                                        actual = 0.0
                                        for r in range(R):
                                            actual += U_np[i * 3 + j, r] * V_np[jp * 2 + k, r] * W_np[ip * 2 + kp, r]
                                        if abs(actual - expected) > 1e-6:
                                            correct = False
                                            break
                                    if not correct:
                                        break
                                if not correct:
                                    break
                            if not correct:
                                break
                        if not correct:
                            break
                    if not correct:
                        break
                if correct:
                    return (U_r.tolist(), V_r.tolist(), W_r.tolist())
        
        return None
    
    # Try to find 3x3x2 decomposition
    for R in [13, 14, 15]:
        if time.time() > deadline - 20:
            break
        decomp = find_3x3x2_decomposition(R)
        if decomp is not None:
            U_core, V_core, W_core = decomp
            total_rank = 3 * R
            
            if total_rank < best_rank:
                # Extend to 3x3x6
                U_tot = [[0] * total_rank for _ in range(n * m)]
                V_tot = [[0] * total_rank for _ in range(m * p)]
                W_tot = [[0] * total_rank for _ in range(p * n)]
                
                # 3 blocks of 3x3x2
                for block in range(3):
                    offset = block * R
                    for i in range(3):
                        for j in range(3):
                            U_tot[i * 3 + j][offset + r] = U_core[i * 3 + j][r] # Wait, r is local index
                            # Actually, U is the same for all blocks?
                            # No, U is A (3x3). It is shared.
                            # But in the decomposition, U has columns corresponding to the rank.
                            # If we use the same decomposition for each block, we can reuse the same U?
                            # No, the tensor is a direct sum.
                            # T_total = T1 + T2 + T3.
                            # Decomposition of T1: sum_r u1_r v1_r w1_r.
                            # Decomposition of T2: sum_r u2_r v2_r w2_r.
                            # If we use the same algorithm, u1_r = u2_r = u_r.
                            # But the rank adds up.
                            # So U_tot has 3*R columns.
                            # For block b, the columns offset..offset+R-1 correspond to the decomposition.
                            # U_tot[i*3+j, offset+r] = U_core[i*3+j, r]
                            
                # Let's fill it properly
                for block in range(3):
                    offset = block * R
                    for i in range(3):
                        for j in range(3):
                            for r in range(R):
                                U_tot[i * 3 + j][offset + r] = U_core[i * 3 + j][r]
                    
                    for j in range(3):
                        for k in range(2):
                            for r in range(R):
                                V_tot[j * 6 + block * 2 + k][offset + r] = V_core[j * 2 + k][r]
                    
                    for i in range(3):
                        for k in range(2):
                            for r in range(R):
                                W_tot[(block * 2 + k) * 3 + i][offset + r] = W_core[k * 3 + i][r]
                
                best_result = {"U": U_tot, "V": V_tot, "W": W_tot}
                best_rank = total_rank
                break
    
    return best_result
