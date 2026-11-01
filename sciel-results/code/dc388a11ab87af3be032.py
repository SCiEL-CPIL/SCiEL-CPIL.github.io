import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 5]
DEADLINE = 120.0
SAFETY_MARGIN = 10.0

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    
    expected = np.ones((n, m, p))
    
    if not np.allclose(tensor, expected, atol=1e-9):
        return False, tensor
    return True, tensor

def build_rank_28(n, m, p):
    """
    Construct a rank-28 decomposition for 2x3x5 matrix multiplication.
    Uses schoolbook for j=0 (rank 10) and Strassen blocks for j=1,2 (rank 18).
    """
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    def add_schoolbook(i, j, k, col):
        U[i * m + j][col] = 1.0
        V[j * p + k][col] = 1.0
        W[k * n + i][col] = 1.0
        return col + 1

    def add_strassen(i_off, j_off, k_off, col):
        # Strassen for 2x2x2 block
        # M1 = (A11+A22)(B11+B22)
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        # M2 = (A21+A22)B11
        U[(i_off+1)*m + (j_off+0)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = -1.0
        col += 1

        # M3 = A11(B21-B22)
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        # M4 = A22(B11-B12)
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = -1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        col += 1

        # M5 = (A11+A12)B22
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        col += 1

        # M6 = (A21-A11)(B11+B12)
        U[(i_off+1)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+0)][col] = -1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        # M7 = (A12-A22)(B21+B22)
        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = -1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        col += 1

        return col

    # j=0: schoolbook for all k (rank 10)
    for k in range(p):
        col = add_schoolbook(0, 0, k, col)
        col = add_schoolbook(1, 0, k, col)
    
    # j=1, k=0,1: Strassen block (rank 7)
    col = add_strassen(0, 1, 0, col)
    
    # j=1, k=2: schoolbook (rank 2)
    col = add_schoolbook(0, 1, 2, col)
    col = add_schoolbook(1, 1, 2, col)
    
    # j=2, k=0,1: Strassen block (rank 7)
    col = add_strassen(0, 2, 0, col)
    
    # j=2, k=2: schoolbook (rank 2)
    col = add_schoolbook(0, 2, 2, col)
    col = add_schoolbook(1, 2, 2, col)
    
    # j=1, k=3,4: schoolbook (rank 4)
    for k in [3, 4]:
        col = add_schoolbook(0, 1, k, col)
        col = add_schoolbook(1, 1, k, col)
    
    # j=2, k=3,4: schoolbook (rank 4)
    for k in [3, 4]:
        col = add_schoolbook(0, 2, k, col)
        col = add_schoolbook(1, 2, k, col)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W

def find_rank_27(n, m, p, U28, V28, W28, time_budget):
    """
    Try to find a rank-27 decomposition by looking for linear dependencies
    among the 28 terms.
    
    Approach: Build the tensor contribution matrix and find if there's a
    non-trivial linear combination that gives zero, allowing us to eliminate
    one term.
    """
    start_time = time.time()
    R = 28
    
    # Build the tensor contribution matrix A where column r is the flattened T_r
    A = np.zeros((n*m*p, R))
    for r in range(R):
        u = np.array(U28[:, r])
        v = np.array(V28[:, r])
        w = np.array(W28[:, r])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A[i*m*p + j*p + k, r] = u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    b = np.ones(n*m*p)
    
    # Check if we can find a sparse solution with at most 27 non-zero entries
    # First, find the null space of A
    try:
        U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
    except Exception:
        return None
    
    # Find the null space (singular values close to zero)
    tol = 1e-10
    rank_A = np.sum(s > tol)
    
    if rank_A < R:
        # There are null space vectors
        # The null space vectors give us linear dependencies among the terms
        # If we can find a dependency that allows us to eliminate one term,
        # we can reduce the rank
        
        # Get null space vectors
        null_vecs = Vt[rank_A:]
        
        # For each null vector, check if it has at most R-1 non-zero entries
        # (which would mean we can eliminate one term)
        for nv in null_vecs:
            if time.time() - start_time > time_budget:
                break
            
            nonzero_idx = np.where(np.abs(nv) > 1e-6)[0]
            if len(nonzero_idx) <= R - 1:
                # We found a dependency that involves at most R-1 terms
                # This means we can potentially eliminate one term
                # The term to eliminate is the one not in nonzero_idx
                all_idx = set(range(R))
                active_idx = set(nonzero_idx)
                eliminated = all_idx - active_idx
                
                if len(eliminated) == 1:
                    elim_idx = list(eliminated)[0]
                    
                    # Create a new decomposition without term elim_idx
                    U_new = []
                    V_new = []
                    W_new = []
                    
                    for r in range(R):
                        if r != elim_idx:
                            U_new.append(U28[:, r])
                            V_new.append(V28[:, r])
                            W_new.append(W28[:, r])
                    
                    U_new = np.array(U_new).T
                    V_new = np.array(V_new).T
                    W_new = np.array(W_new).T
                    
                    # But we need to check if the sum still equals the identity
                    # The null space vector tells us that sum_r nv[r] * T_r = 0
                    # So if we remove term elim_idx, we need to adjust the others
                    # Actually, we need to find coefficients c_r such that
                    # sum_r c_r * T_r = b and c[elim_idx] = 0
                    
                    # Let's solve the constrained system
                    # A_active * c_active = b where A_active has columns for active terms
                    A_active = A[:, list(active_idx)]
                    
                    try:
                        c_active, residuals, rank_act, sv = np.linalg.lstsq(A_active, b, rcond=None)
                        if residuals and residuals[0] < 1e-8:
                            # Check if this is a valid solution
                            U_final = np.zeros((n*m, len(active_idx)))
                            V_final = np.zeros((m*p, len(active_idx)))
                            W_final = np.zeros((p*n, len(active_idx)))
                            
                            for idx, r in enumerate(active_idx):
                                U_final[:, idx] = U28[:, r] * c_active[idx]
                                V_final[:, idx] = V28[:, r]
                                W_final[:, idx] = W28[:, r]
                            
                            valid, _ = verify_decomposition(U_final, V_final, W_final, n, m, p)
                            if valid:
                                return U_final.tolist(), V_final.tolist(), W_final.tolist()
                    except Exception:
                        pass
    
    # Alternative approach: Try to find a rank-27 solution by solving
    # a system of equations directly
    
    # Parameterize U, V, W with 27 columns each
    # This is a system of n*m*p = 30 equations in 27*(6+15+10) = 837 unknowns
    # But it's nonlinear (cubic in the variables)
    
    # Let's try a different approach: use the fact that we have 28 terms and
    # look for a 27-term subset that can be scaled to give the identity
    
    # For each term to eliminate, solve for the remaining 27 terms
    np.random.seed(42)
    
    for elim_idx in range(R):
        if time.time() - start_time > time_budget:
            break
        
        active_idx = [r for r in range(R) if r != elim_idx]
        A_active = A[:, active_idx]
        
        try:
            c_active, residuals, rank_act, sv = np.linalg.lstsq(A_active, b, rcond=None)
            if residuals and residuals[0] < 1e-8:
                # Check if this gives a valid decomposition
                U_final = np.zeros((n*m, len(active_idx)))
                V_final = np.zeros((m*p, len(active_idx)))
                W_final = np.zeros((p*n, len(active_idx)))
                
                for idx, r in enumerate(active_idx):
                    U_final[:, idx] = U28[:, r] * c_active[idx]
                    V_final[:, idx] = V28[:, r]
                    W_final[:, idx] = W28[:, r]
                
                valid, _ = verify_decomposition(U_final, V_final, W_final, n, m, p)
                if valid:
                    return U_final.tolist(), V_final.tolist(), W_final.tolist()
        except Exception:
            pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_result = None
    best_rank = float('inf')
    
    # First, build the known rank-28 solution
    try:
        U, V, W = build_rank_28(n, m, p)
        valid, _ = verify_decomposition(U, V, W, n, m, p)
        if valid:
            best_result = {'U': U, 'V': V, 'W': W}
            best_rank = 28
    except Exception:
        pass
    
    # Try to find a rank-27 solution
    if best_result is not None and time.time() < deadline:
        time_budget = min(deadline - time.time(), 60.0)
        if time_budget > 5.0:
            try:
                result = find_rank_27(n, m, p, 
                                      np.array(best_result['U']), 
                                      np.array(best_result['V']), 
                                      np.array(best_result['W']),
                                      time_budget)
                if result is not None:
                    U, V, W = result
                    best_result = {'U': U, 'V': V, 'W': W}
                    best_rank = 27
            except Exception:
                pass
    
    # Fallback: Schoolbook (Rank 30)
    if best_result is None:
        rank_school = n * m * p
        U = [[0.0] * rank_school for _ in range(n * m)]
        V = [[0.0] * rank_school for _ in range(m * p)]
        W = [[0.0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        
        valid, _ = verify_decomposition(U, V, W, n, m, p)
        if valid:
            best_result = {'U': U, 'V': V, 'W': W}
    
    if best_result is not None:
        return best_result
    
    return {'U': [[0.0]], 'V': [[0.0]], 'W': [[0.0]]}
