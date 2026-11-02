import numpy as np
import time
import itertools
from collections import deque

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Target tensor: T[i, k, j, j'] = 1 if j == j' else 0
    # This corresponds to C = A @ B where C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T_{ikjj'} = delta_{jj'}
    T = np.zeros((n, p, m, m))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                T[i, k, j, j] = 1.0
    
    # We work in integers by scaling by 2 (half-integers become integers)
    # The decomposition is M[i,k,j,j'] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r]
    # If U,V,W are half-integers, let U' = 2U, etc.
    # Then 8 * M = sum_r U' V' W'
    # So we need sum_r U' V' W' = 8 * T
    T_int = (T * 8).astype(int)

    def compute_residual_int(U_int, V_int, W_int, R):
        """
        Compute M_int = sum_r W[:,r] U[:,r] V[:,r] and return M_int - T_int.
        U_int shape: (n*m, R)
        V_int shape: (m*p, R)
        W_int shape: (p*n, R)
        """
        M_int = np.zeros((n, p, m, m), dtype=int)
        for r in range(R):
            u_col = U_int[:, r]
            v_col = V_int[:, r]
            w_col = W_int[:, r]
            
            # u_col is (n*m,), reshape to (n, m)
            # v_col is (m*p,), reshape to (m, p)
            # w_col is (p*n,), reshape to (p, n)
            
            # M[i, k, j, j'] += w[k, i] * u[i, j] * v[j', k]
            # Note indices:
            # U index: i*m + j
            # V index: j'*p + k
            # W index: k*n + i
            
            for i in range(n):
                for k in range(p):
                    w_val = w_col[k * n + i]
                    if w_val == 0:
                        continue
                    for j in range(m):
                        u_val = u_col[i * m + j]
                        if u_val == 0:
                            continue
                        for jp in range(m):
                            v_val = v_col[jp * p + k]
                            if v_val == 0:
                                continue
                            M_int[i, k, j, jp] += w_val * u_val * v_val
        
        return M_int - T_int

    def check_exact(U_int, V_int, W_int, R):
        diff = compute_residual_int(U_int, V_int, W_int, R)
        return np.all(diff == 0)

    # Known optimal rank for 2x3x3 is 17.
    # We will try to find a solution of rank 17 using backtracking.
    # To make search feasible, we restrict entries to small integers.
    # Since we scaled by 2, entries are integers.
    # Likely values are -2, -1, 0, 1, 2 (corresponding to -1, -0.5, 0, 0.5, 1)
    # Sometimes 3 or 4 might appear, but let's start with {-2, -1, 0, 1, 2}
    
    # Strategy:
    # 1. Try to construct a solution using a structured approach based on Strassen.
    # 2. If that fails, use randomized backtracking.
    
    # First, let's try a specific known decomposition structure for 2x3x3.
    # The 2x3x3 multiplication can be seen as two 2x3x2 multiplications plus one 2x3x1?
    # Or more simply, we can just search.
    
    # Given the time limit and complexity, a pure random search over 5^300 is impossible.
    # We need a smarter approach.
    # However, the "knowledge" says: "Replaces the rank-18 schoolbook decomposition with a rank-17 decomposition constructed by combining Strassen's algorithm for the 2x2x2 subproblem with schoolbook algorithms for the remaining 2x1x2 and 2x3x1 subproblems".
    
    # Let's try to implement this specific construction.
    # 2x3x3 = 2x(3x3).
    # Split B into B1 (2x3) and B2 (1x3)? No, B is 3x3.
    # Split A (2x3) into A1 (2x2) and A2 (2x1).
    # C = A1 B1 + A2 B2.
    # A1 is 2x2, B1 is 2x3. Rank of 2x2x3 is 7 (Strassen-like).
    # A2 is 2x1, B2 is 1x3. Rank of 2x1x3 is 2 (outer product).
    # Total rank = 7 + 2 = 9? No, that's not right.
    # 2x2x2 is rank 7.
    # 2x2x3: We can view it as 2x2x2 + 2x2x1.
    # 2x2x1 is rank 4? No, 2x1x1 is rank 2. 2x2x1 is sum of two 2x1x1?
    # Actually, rank(2x2x1) = 2*1*1 = 2? No.
    # Rank(2x2x1) is 2. (C = A B, A 2x2, B 2x1. C[i] = sum_j A[i,j] B[j]. This is just matrix-vector mult. Rank is 2? No, it's 2x2x1 tensor. It's decomposable into 2 rank-1 tensors?
    # C[i,0,j] = A[i,j]. This is just A. Rank is 2? No, the tensor is T[i, j, k] where k=0.
    # Wait, matrix multiplication tensor T[i, k, j] for A(nxm) B(mxk).
    # For 2x2x1: A is 2x2, B is 2x1. C is 2x1.
    # T[i, 0, j] = delta_{j,?} No.
    # C[i, 0] = sum_j A[i, j] B[j, 0].
    # The tensor is T[i, k, j]. Here k=0.
    # T[i, 0, j] = 1 if we are looking at the coefficient of A[i,j] B[j,0].
    # This is just the identity tensor for the contraction.
    # The rank of the 2x2x1 multiplication tensor is 2?
    # Let's check: C = A B. A is 2x2, B is 2x1.
    # C1 = A11 B1 + A21 B2
    # C2 = A12 B1 + A22 B2
    # This can be done with 4 multiplications (schoolbook).
    # Can it be done with less?
    # Strassen is for 2x2x2.
    # For 2x2x1, it's just matrix-vector multiplication.
    # It is known that rank(2x2x1) = 2? No, it's 2*1*1 = 2?
    # Actually, rank(2xn x 1) is n.
    # So rank(2x2x1) = 2.
    # Wait, if rank is 2, then C1 = U1 V1, C2 = U2 V2?
    # C1 = A11 B1 + A21 B2. This is a linear combination.
    # The tensor T[i, k, j] where k is fixed.
    # T[i, 0, j] = 1.
    # This is a 2x1x2 tensor.
    # Rank of 2x1x2 tensor:
    # T[i, 0, j] = 1.
    # This is the identity matrix reshaped.
    # Rank of identity matrix 2x2 is 2.
    # So rank(2x2x1) = 2.
    
    # So, 2x3x3 = 2x(2+1)x3.
    # Split A into A1 (2x2) and A2 (2x1).
    # C = A1 B1 + A2 B2.
    # B1 is 2x3, B2 is 1x3.
    # Term 1: 2x2x3 multiplication.
    # Term 2: 2x1x3 multiplication.
    
    # Rank(2x1x3):
    # A is 2x1, B is 1x3. C is 2x3.
    # C[i, k] = A[i, 0] B[0, k].
    # This is an outer product.
    # Tensor T[i, k, j] where j=0.
    # T[i, k, 0] = 1.
    # This is a 2x3x1 tensor.
    # Rank is 1? No, it's 2x3.
    # T[i, k, 0] = 1 for all i, k.
    # This is a rank-1 tensor (all ones).
    # So rank(2x1x3) = 1?
    # Wait, the tensor is T[i, k, j] = delta_{j,0} * 1.
    # Yes, it's a single rank-1 term: U[i]=1, V[k]=1, W[j]=delta_{j,0}.
    # So rank(2x1x3) = 1.
    
    # Rank(2x2x3):
    # A is 2x2, B is 2x3.
    # C = A B.
    # This can be viewed as two 2x2x2 multiplications? No.
    # B is 2x3. Split B into B1 (2x2) and B2 (2x1).
    # C = A B1 + A B2.
    # Rank(2x2x2) = 7 (Strassen).
    # Rank(2x2x1) = 2.
    # Total = 9.
    # But we can do better?
    # It is known that rank(2x2x3) is 7? Or 8?
    # Actually, rank(2x2xn) is 2n for n>=2? No.
    # Let's look up rank(2x2x3).
    # It is 7? No, 2x2x2 is 7.
    # 2x2x3 involves 6 multiplications in schoolbook.
    # Strassen reduces 8 to 7 for 2x2x2.
    # For 2x2x3, we have 3 columns.
    # Maybe we can share computations?
    
    # Let's try a different split.
    # 2x3x3.
    # Schoolbook: 18.
    # Strassen on 2x2x2 part:
    # Let A = [A11 A12 A13], B = [[B11 B12 B13], [B21 B22 B23], [B31 B32 B33]].
    # This is getting complicated.
    
    # Let's stick to the "randomized algebraic approach" suggested in the prompt, but implemented as a **constrained optimization with integer programming** or **random restarts with local search on the discrete lattice**.
    
    # Given the difficulty of constructing the exact Strassen-based solution from scratch in code without looking it up, and the time limit, I will implement a **robust local search** that starts from a good initialization (e.g., sparse random) and refines it.
    
    # However, the previous program failed to find rank 17.
    # The key is to find the *right* structure.
    
    # Let's try a **Genetic Algorithm** approach on the discrete lattice.
    # Population of (U, V, W) tuples.
    # Fitness: number of non-zero entries in residual.
    # Crossover: mix columns.
    # Mutation: change an entry.
    
    def fitness(U, V, W, R):
        diff = compute_residual_int(U, V, W, R)
        return np.count_nonzero(diff)

    def random_init(R, max_val=2):
        # Sparse initialization
        U = np.zeros((n*m, R), dtype=int)
        V = np.zeros((m*p, R), dtype=int)
        W = np.zeros((p*n, R), dtype=int)
        
        # Initialize with random sparse patterns
        for r in range(R):
            # Pick random non-zeros
            num_nz_u = np.random.randint(1, 4)
            idx_u = np.random.choice(n*m, num_nz_u, replace=False)
            U[idx_u, r] = np.random.choice([-max_val, -max_val+1, 1, max_val], num_nz_u)
            
            num_nz_v = np.random.randint(1, 4)
            idx_v = np.random.choice(m*p, num_nz_v, replace=False)
            V[idx_v, r] = np.random.choice([-max_val, -max_val+1, 1, max_val], num_nz_v)
            
            num_nz_w = np.random.randint(1, 4)
            idx_w = np.random.choice(p*n, num_nz_w, replace=False)
            W[idx_w, r] = np.random.choice([-max_val, -max_val+1, 1, max_val], num_nz_w)
            
        return U, V, W

    def local_search(U, V, W, R, max_steps=1000):
        """
        Perform local search on the discrete lattice.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        current_fit = fitness(U, V, W, R)
        if current_fit == 0:
            return U, V, W, True
        
        step = 0
        while step < max_steps and time.time() < deadline - 5:
            improved = False
            step += 1
            
            # Try perturbing U
            for _ in range(10):
                idx = np.random.randint(0, n*m*R)
                row, col = idx // R, idx % R
                old_val = U[row, col]
                new_val = old_val + np.random.choice([-2, -1, 1, 2])
                if abs(new_val) > 4: # Bound values
                    continue
                U[row, col] = new_val
                new_fit = fitness(U, V, W, R)
                if new_fit < current_fit:
                    current_fit = new_fit
                    improved = True
                    if current_fit == 0:
                        return U, V, W, True
                    break
                else:
                    U[row, col] = old_val
            
            # Try perturbing V
            if not improved:
                for _ in range(10):
                    idx = np.random.randint(0, m*p*R)
                    row, col = idx // R, idx % R
                    old_val = V[row, col]
                    new_val = old_val + np.random.choice([-2, -1, 1, 2])
                    if abs(new_val) > 4:
                        continue
                    V[row, col] = new_val
                    new_fit = fitness(U, V, W, R)
                    if new_fit < current_fit:
                        current_fit = new_fit
                        improved = True
                        if current_fit == 0:
                            return U, V, W, True
                        break
                    else:
                        V[row, col] = old_val
            
            # Try perturbing W
            if not improved:
                for _ in range(10):
                    idx = np.random.randint(0, p*n*R)
                    row, col = idx // R, idx % R
                    old_val = W[row, col]
                    new_val = old_val + np.random.choice([-2, -1, 1, 2])
                    if abs(new_val) > 4:
                        continue
                    W[row, col] = new_val
                    new_fit = fitness(U, V, W, R)
                    if new_fit < current_fit:
                        current_fit = new_fit
                        improved = True
                        if current_fit == 0:
                            return U, V, W, True
                        break
                    else:
                        W[row, col] = old_val
            
            if not improved:
                # Stuck, break out to try a new initialization
                break
        
        return U, V, W, current_fit == 0

    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    # Initialize with schoolbook
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1

    # Try ranks 17, 16, 15...
    # We know 17 is possible.
    for R in range(17, 13, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        attempts = 0
        max_attempts = 50 if R == 17 else 20
        
        while attempts < max_attempts and time.time() < deadline - 5 and not found:
            attempts += 1
            U0, V0, W0 = random_init(R)
            U_opt, V_opt, W_opt, success = local_search(U0, V0, W0, R)
            
            if success:
                best_rank = R
                best_U = (U_opt / 2.0).tolist()
                best_V = (V_opt / 2.0).tolist()
                best_W = (W_opt / 2.0).tolist()
                found = True
                break

        if found:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
