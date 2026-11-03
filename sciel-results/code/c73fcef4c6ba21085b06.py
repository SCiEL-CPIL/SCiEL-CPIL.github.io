import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    dim_U, dim_V, dim_W = n * m, m * p, p * n

    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit

    # Build target tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Baseline: rank-17 solution
    U17, V17, W17 = _build_rank17()

    best_rank = 17
    best_U, best_V, best_W = U17, V17, W17

    # Try to find rank-16 solution
    if time.time() < deadline - 30:
        r16 = _search_rank16(T, U17, V17, W17, deadline)
        if r16 is not None:
            best_rank = 16
            best_U, best_V, best_W = r16

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


def _build_rank17():
    n, m, p = 2, 3, 3
    dim_U, dim_V, dim_W = 6, 9, 6
    rank = 17
    U = np.zeros((dim_U, rank), dtype=np.int64)
    V = np.zeros((dim_V, rank), dtype=np.int64)
    W = np.zeros((dim_W, rank), dtype=np.int64)

    # Strassen for A[0:2,0:2] * B[0:2,0:2]
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A10+A11)*B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # M4 = A11*(B10-B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[2, 4] = 1; W[0, 4] = -1

    # M6 = (A10-A00)*(B00+B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # Schoolbook: A[:,2]*B[2,:] -> C[:,0:2]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1

    # Schoolbook: A[:,0:2]*B[:,2] -> C[:,2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1

    # Schoolbook: A[:,2]*B[2,2] -> C[:,2]
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1

    return U, V, W

def _search_rank16(T, U17, V17, W17, deadline):
    """
    Search for rank-16 by fixing the first 7 terms to Strassen
    and optimizing the remaining 9 terms. Uses vectorized error computation.
    """
    dim_U, dim_V, dim_W = 6, 9, 6
    rank = 16
    num_strassen = 7
    num_free = rank - num_strassen  # 9 terms

    U_strassen = U17[:, :num_strassen].copy()
    V_strassen = V17[:, :num_strassen].copy()
    W_strassen = W17[:, :num_strassen].copy()

    T_strassen = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for r in range(num_strassen):
        T_strassen += np.einsum('a,b,c->abc', U_strassen[:, r], V_strassen[:, r], W_strassen[:, r])
    
    T_free = T - T_strassen

    # Precompute T_free as a matrix for faster comparison
    T_free_flat = T_free.reshape(-1)

    def compute_error_vec(U_f, V_f, W_f):
        """Vectorized error computation: sum over rank-1 terms."""
        # S = sum_r U_f[:,r] outer V_f[:,r] outer W_f[:,r]
        # Reshape to flat and compare with T_free_flat
        # S_flat[a*dim_V*dim_W + b*dim_W + c] = sum_r U_f[a,r]*V_f[b,r]*W_f[c,r]
        # Use einsum for speed
        S = np.einsum('ar,br,cr->abc', U_f, V_f, W_f)
        return np.sum((S - T_free) ** 2)

    # Initialize: try multiple starting points
    np.random.seed(42)
    
    # Starting point 1: first 9 schoolbook terms
    U_init1 = U17[:, 7:7+num_free].astype(np.float64).copy()
    V_init1 = V17[:, 7:7+num_free].astype(np.float64).copy()
    W_init1 = W17[:, 7:7+num_free].astype(np.float64).copy()
    
    # Starting point 2: random sparse init
    U_init2 = np.zeros((dim_U, num_free), dtype=np.float64)
    V_init2 = np.zeros((dim_V, num_free), dtype=np.float64)
    W_init2 = np.zeros((dim_W, num_free), dtype=np.float64)
    for r in range(num_free):
        # Each term should hit specific (i,j,k) patterns
        U_init2[np.random.randint(dim_U), r] = np.random.choice([-1, 1])
        V_init2[np.random.randint(dim_V), r] = np.random.choice([-1, 1])
        W_init2[np.random.randint(dim_W), r] = np.random.choice([-1, 1])
    
    # Starting point 3: another random
    U_init3 = np.zeros((dim_U, num_free), dtype=np.float64)
    V_init3 = np.zeros((dim_V, num_free), dtype=np.float64)
    W_init3 = np.zeros((dim_W, num_free), dtype=np.float64)
    for r in range(num_free):
        U_init3[np.random.randint(dim_U), r] = np.random.choice([-1, 1])
        V_init3[np.random.randint(dim_V), r] = np.random.choice([-1, 1])
        W_init3[np.random.randint(dim_W), r] = np.random.choice([-1, 1])

    initializations = [
        (U_init1, V_init1, W_init1),
        (U_init2, V_init2, W_init2),
        (U_init3, V_init3, W_init3),
    ]

    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None

    for U_init, V_init, W_init in initializations:
        if time.time() > deadline - 15:
            break
        
        current_U = U_init.copy()
        current_V = V_init.copy()
        current_W = W_init.copy()
        current_err = compute_error_vec(current_U, current_V, current_W)
        
        if current_err < best_err:
            best_err = current_err
            best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
        
        # Coordinate descent with integer steps
        no_improve = 0
        max_no_improve = 2000
        steps = 0
        max_steps = 50000
        
        while time.time() < deadline - 15 and steps < max_steps and no_improve < max_no_improve:
            steps += 1
            
            r = np.random.randint(num_free)
            mat = np.random.choice(3)
            
            if mat == 0:
                idx = np.random.randint(dim_U)
                old_val = current_U[idx, r]
                delta = np.random.choice([-2, -1, 1, 2])
                new_val = old_val + delta
                if abs(new_val) > 3:
                    no_improve += 1
                    continue
                current_U[idx, r] = new_val
                new_err = compute_error_vec(current_U, current_V, current_W)
                if new_err < current_err - 1e-12:
                    current_err = new_err
                    no_improve = 0
                else:
                    current_U[idx, r] = old_val
                    no_improve += 1
            elif mat == 1:
                idx = np.random.randint(dim_V)
                old_val = current_V[idx, r]
                delta = np.random.choice([-2, -1, 1, 2])
                new_val = old_val + delta
                if abs(new_val) > 3:
                    no_improve += 1
                    continue
                current_V[idx, r] = new_val
                new_err = compute_error_vec(current_U, current_V, current_W)
                if new_err < current_err - 1e-12:
                    current_err = new_err
                    no_improve = 0
                else:
                    current_V[idx, r] = old_val
                    no_improve += 1
            else:
                idx = np.random.randint(dim_W)
                old_val = current_W[idx, r]
                delta = np.random.choice([-2, -1, 1, 2])
                new_val = old_val + delta
                if abs(new_val) > 3:
                    no_improve += 1
                    continue
                current_W[idx, r] = new_val
                new_err = compute_error_vec(current_U, current_V, current_W)
                if new_err < current_err - 1e-12:
                    current_err = new_err
                    no_improve = 0
                else:
                    current_W[idx, r] = old_val
                    no_improve += 1
            
            if current_err < best_err:
                best_err = current_err
                best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
            
            # Check for exact solution
            if current_err < 1e-9:
                U_int = np.round(np.hstack([U_strassen, current_U])).astype(np.int64)
                V_int = np.round(np.hstack([V_strassen, current_V])).astype(np.int64)
                W_int = np.round(np.hstack([W_strassen, current_W])).astype(np.int64)
                
                S = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
                for r in range(rank):
                    S += np.einsum('a,b,c->abc', U_int[:, r], V_int[:, r], W_int[:, r])
                if np.allclose(S, T, atol=1e-9):
                    return U_int, V_int, W_int
    
    # Final verification of best found
    if best_err < 1e-9 and best_U is not None:
        U_int = np.round(np.hstack([U_strassen, best_U])).astype(np.int64)
        V_int = np.round(np.hstack([V_strassen, best_V])).astype(np.int64)
        W_int = np.round(np.hstack([W_strassen, best_W])).astype(np.int64)
        
        S = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
        for r in range(rank):
            S += np.einsum('a,b,c->abc', U_int[:, r], V_int[:, r], W_int[:, r])
        if np.allclose(S, T, atol=1e-9):
            return U_int, V_int, W_int

    return None
