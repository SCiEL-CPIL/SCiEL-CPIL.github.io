"""Find rank-minimal exact tensor decomposition for 2x2 * 2x4 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 4]

def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U row: i*m+j, V row: j'*p+k', W row: k'*n+i'
    
    # Build the multiplication tensor T as a 3D array
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try rank-13 search first
    if time.time() < deadline - 5:
        success = _search_rank(13, n, m, p, T, deadline)
        if success is not None:
            U, V, W = success
            if _verify_solution(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Try rank-14 Strassen-based solution
    rank14 = _build_rank14_solution()
    if rank14 is not None:
        U14, V14, W14 = rank14
        if _verify_solution(U14, V14, W14, n, m, p):
            return {"U": U14, "V": V14, "W": W14}
    
    # Fall back to schoolbook rank 16
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb))
    V_fb = np.zeros((m*p, rank_fb))
    W_fb = np.zeros((p*n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}


def _build_rank14_solution():
    """Build the rank-14 Strassen-based solution."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(0,3), 4=(1,0), 5=(1,1), 6=(1,2), 7=(1,3)
    # W indices (k*n+i): C[0,0]=0, C[0,1]=2, C[0,2]=4, C[0,3]=6, C[1,0]=1, C[1,1]=3, C[1,2]=5, C[1,3]=7
    
    # Left Strassen block (B columns 0,1)
    # Col 0: L1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0], C[1,1]
    
    # Col 1: L2 = (A10+A11)(B00)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1  # C[1,0], C[1,1]
    
    # Col 2: L3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[5, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1  # C[0,1], C[1,1]
    
    # Col 3: L4 = A11(B10-B00)
    U[3, 3] = 1
    V[4, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1  # C[0,0], C[1,0]
    
    # Col 4: L5 = (A00+A01)(B11)
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1  # C[0,0], C[0,1]
    
    # Col 5: L6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C[1,1]
    
    # Col 6: L7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[4, 6] = 1; V[5, 6] = 1
    W[0, 6] = 1  # C[0,0]
    
    # Right Strassen block (B columns 2,3)
    # B indices: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    
    # Col 7: R1 = (A00+A11)(B02+B13)
    U[0, 7] = 1; U[3, 7] = 1
    V[2, 7] = 1; V[7, 7] = 1
    W[4, 7] = 1; W[7, 7] = 1  # C[0,2], C[1,3]
    
    # Col 8: R2 = (A10+A11)(B02)
    U[2, 8] = 1; U[3, 8] = 1
    V[2, 8] = 1
    W[5, 8] = 1; W[7, 8] = -1  # C[1,2], C[1,3]
    
    # Col 9: R3 = A00(B03-B13)
    U[0, 9] = 1
    V[3, 9] = 1; V[7, 9] = -1
    W[6, 9] = 1; W[7, 9] = 1  # C[0,3], C[1,3]
    
    # Col 10: R4 = A11(B12-B02)
    U[3, 10] = 1
    V[6, 10] = 1; V[2, 10] = -1
    W[4, 10] = 1; W[5, 10] = 1  # C[0,2], C[1,2]
    
    # Col 11: R5 = (A00+A01)(B13)
    U[0, 11] = 1; U[1, 11] = 1
    V[7, 11] = 1
    W[4, 11] = -1; W[6, 11] = 1  # C[0,2], C[0,3]
    
    # Col 12: R6 = (A10-A00)(B02+B03)
    U[2, 12] = 1; U[0, 12] = -1
    V[2, 12] = 1; V[3, 12] = 1
    W[7, 12] = 1  # C[1,3]
    
    # Col 13: R7 = (A01-A11)(B12+B13)
    U[1, 13] = 1; U[3, 13] = -1
    V[6, 13] = 1; V[7, 13] = 1
    W[4, 13] = 1  # C[0,2]
    
    return U.tolist(), V.tolist(), W.tolist()


def _verify_solution(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            a = i * m + j
                            b = jp * p + k
                            c = kp * n + ip
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            actual = np.sum(U[a] * V[b] * W[c])
                            if abs(actual - expected) > 1e-8:
                                return False
    return True


def _search_rank(target_rank, n, m, p, T, deadline):
    """Search for a rank-r decomposition using alternating least squares."""
    n, m, p = 2, 2, 4
    dim_a = n * m  # 4
    dim_b = m * p  # 8
    dim_c = p * n  # 8
    rank = target_rank
    
    # Precompute the tensor as a list of (a,b,c) tuples with T[a,b,c]=1
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))
    
    # Also precompute all (a,b,c) combinations for verification
    all_entries = [(a, b, c) for a in range(dim_a) for b in range(dim_b) for c in range(dim_c)]
    
    def compute_error(U, V, W):
        """Compute the squared error of the decomposition."""
        err = 0.0
        for a in range(dim_a):
            for b in range(dim_b):
                for c in range(dim_c):
                    val = np.dot(U[a] * V[b], W[c])
                    expected = 1.0 if (a, b, c) in tensor_set else 0.0
                    err += (val - expected) ** 2
        return err
    
    tensor_set = set(tensor_entries)
    
    def solve_V(U, W):
        """Given U and W, solve for V via least squares."""
        # For each b, we need V[b] such that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        # For fixed b, this is: sum_r (U[:,r] * W[c,r]) * V[b,r] = T[:,b,c] for all a,c
        # Actually, let's think of it differently.
        # The constraint is: for all (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        # For fixed b, define M_b[a,c] = T[a,b,c]
        # We need: sum_r (U[:,r] ⊗ W[c,r])^T V[b] = M_b.flatten()
        # This is a linear system in V[b] for each b.
        
        V_sol = np.zeros((dim_b, rank))
        for b in range(dim_b):
            # Build the linear system: A_b * V[b] = T[:,b,:]
            # A_b has shape (dim_a * dim_c, rank)
            # A_b[a*dim_c+c, r] = U[a,r] * W[c,r]
            A_b = np.zeros((dim_a * dim_c, rank))
            for a in range(dim_a):
                for c in range(dim_c):
                    A_b[a * dim_c + c, :] = U[a, :] * W[c, :]
            rhs = T[:, b, :].flatten()
            # Solve least squares
            try:
                V_sol[b, :], residuals, rank_sv, sv = np.linalg.lstsq(A_b, rhs, rcond=None)
            except:
                V_sol[b, :] = 0
        return V_sol
    
    def solve_U(V, W):
        """Given V and W, solve for U via least squares."""
        U_sol = np.zeros((dim_a, rank))
        for a in range(dim_a):
            # A_a[b*dim_c+c, r] = V[b,r] * W[c,r]
            A_a = np.zeros((dim_b * dim_c, rank))
            for b in range(dim_b):
                for c in range(dim_c):
                    A_a[b * dim_c + c, :] = V[b, :] * W[c, :]
            rhs = T[a, :, :].flatten()
            try:
                U_sol[a, :], residuals, rank_sv, sv = np.linalg.lstsq(A_a, rhs, rcond=None)
            except:
                U_sol[a, :] = 0
        return U_sol
    
    def solve_W(U, V):
        """Given U and V, solve for W via least squares."""
        W_sol = np.zeros((dim_c, rank))
        for c in range(dim_c):
            # A_c[a*dim_b+r_idx, r] = U[a,r] * V[b,r]
            # For fixed c: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            # A_c[a*dim_b+b, r] = U[a,r] * V[b,r]
            A_c = np.zeros((dim_a * dim_b, rank))
            for a in range(dim_a):
                for b in range(dim_b):
                    A_c[a * dim_b + b, :] = U[a, :] * V[b, :]
            rhs = T[:, :, c].flatten()
            try:
                W_sol[c, :], residuals, rank_sv, sv = np.linalg.lstsq(A_c, rhs, rcond=None)
            except:
                W_sol[c, :] = 0
        return W_sol
    
    # Try multiple random initializations
    np.random.seed(42)
    best_error = float('inf')
    best_solution = None
    
    max_restarts = 20
    max_iters = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize with random values
        U = np.random.randn(dim_a, rank)
        V = np.random.randn(dim_b, rank)
        W = np.random.randn(dim_c, rank)
        
        # Normalize
        U /= np.linalg.norm(U) + 1e-10
        V /= np.linalg.norm(V) + 1e-10
        W /= np.linalg.norm(W) + 1e-10
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Alternating least squares
            V = solve_V(U, W)
            U = solve_U(V, W)
            W = solve_W(U, V)
            
            # Compute error
            err = compute_error(U, V, W)
            if err < best_error:
                best_error = err
                best_solution = (U.copy(), V.copy(), W.copy())
            
            # Check for convergence
            if err < 1e-10:
                break
        
        # Also try structured initializations based on Strassen
        if restart == 0:
            # Use Strassen-like initialization
            U_s, V_s, W_s = _build_rank14_solution()
            U_s = np.array(U_s)[:, :rank]
            V_s = np.array(V_s)[:, :rank]
            W_s = np.array(W_s)[:, :rank]
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                V_s = solve_V(U_s, W_s)
                U_s = solve_U(V_s, W_s)
                W_s = solve_W(U_s, V_s)
                err = compute_error(U_s, V_s, W_s)
                if err < best_error:
                    best_error = err
                    best_solution = (U_s.copy(), V_s.copy(), W_s.copy())
                if err < 1e-10:
                    break
    
    if best_solution is not None and best_error < 1e-8:
        U, V, W = best_solution
        return U.tolist(), V.tolist(), W.tolist()
    
    return None
