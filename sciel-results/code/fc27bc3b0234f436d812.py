import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5
_TENSOR_SIZE = N * M * P  # 100


def _validate(U, V, W, tol=1e-9):
    """Check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2(i0, j0, k0, U, V, W, r_start):
    """Add 7 rank-1 terms for 2x2x2 Strassen decomposition.
    Block A rows [i0:i0+2], B cols [j0:j0+2], C rows [k0:k0+2].
    Returns r_start + 7.
    """
    r = r_start
    # M1 = (A[i0,:2] + A[i0+1,:2]) * (B[:2,k0] - B[:2,k0+1])
    #    -> C[i0,k0] += M1, C[i0+1,k0] += M1, C[i0,k0+1] -= M1
    U[i0*(M)+j0, r] = 1.0
    U[(i0+1)*M+j0, r] = 1.0
    U[i0*(M)+j0+1, r] = 1.0
    U[(i0+1)*M+j0+1, r] = 1.0
    V[j0*(P)+k0, r] = 1.0
    V[j0*(P)+k0+1, r] = -1.0
    V[(j0+1)*(P)+k0, r] = 1.0
    V[(j0+1)*(P)+k0+1, r] = -1.0
    W[k0*(N)+i0, r] = 1.0
    W[k0*(N)+i0+1, r] = 1.0
    W[(k0+1)*(N)+i0, r] = -1.0
    r += 1

    # M2 = A[i0+1,:2] * (B[:2,k0] + B[:2,k0+1])
    #    -> C[i0+1,k0+1] += M2
    U[i0*(M)+j0, r] = 0.0
    U[(i0+1)*M+j0, r] = 1.0
    U[i0*(M)+j0+1, r] = 0.0
    U[(i0+1)*M+j0+1, r] = 1.0
    V[j0*(P)+k0, r] = 1.0
    V[j0*(P)+k0+1, r] = 1.0
    V[(j0+1)*(P)+k0, r] = 1.0
    V[(j0+1)*(P)+k0+1, r] = 1.0
    W[(k0+1)*(N)+i0+1, r] = 1.0
    r += 1

    # M3 = (A[i0,:2] - A[i0+1,:2]) * B[:2,k0+1]
    #    -> C[i0,k0+1] += M3, C[i0+1,k0+1] -= M3
    U[i0*(M)+j0, r] = 1.0
    U[(i0+1)*M+j0, r] = -1.0
    U[i0*(M)+j0+1, r] = 1.0
    U[(i0+1)*M+j0+1, r] = -1.0
    V[j0*(P)+k0, r] = 0.0
    V[j0*(P)+k0+1, r] = 1.0
    V[(j0+1)*(P)+k0, r] = 0.0
    V[(j0+1)*(P)+k0+1, r] = 1.0
    W[k0*(N)+i0, r] = 0.0
    W[k0*(N)+i0+1, r] = 0.0
    W[(k0+1)*(N)+i0, r] = 1.0
    W[(k0+1)*(N)+i0+1, r] = -1.0
    r += 1

    # M4 = A[i0,:2] * B[:2,k0]
    #    -> C[i0,k0] += M4
    U[i0*(M)+j0, r] = 1.0
    U[(i0+1)*M+j0, r] = 0.0
    U[i0*(M)+j0+1, r] = 1.0
    U[(i0+1)*M+j0+1, r] = 0.0
    V[j0*(P)+k0, r] = 1.0
    V[j0*(P)+k0+1, r] = 0.0
    V[(j0+1)*(P)+k0, r] = 1.0
    V[(j0+1)*(P)+k0+1, r] = 0.0
    W[k0*(N)+i0, r] = 1.0
    r += 1

    # M5 = (A[i0+1,:2] + A[i0,:2]) * B[:2,k0+1]
    #    -> C[i0+1,k0] += M5
    U[i0*(M)+j0, r] = 1.0
    U[(i0+1)*M+j0, r] = 1.0
    U[i0*(M)+j0+1, r] = 1.0
    U[(i0+1)*M+j0+1, r] = 1.0
    V[j0*(P)+k0, r] = 0.0
    V[j0*(P)+k0+1, r] = 1.0
    V[(j0+1)*(P)+k0, r] = 0.0
    V[(j0+1)*(P)+k0+1, r] = 1.0
    W[k0*(N)+i0+1, r] = 1.0
    r += 1

    # M6 = (A[i0+1,:2] - A[i0,:2]) * (B[:2,k0] + B[:2,k0+1])
    #    -> C[i0+1,k0] += M6
    U[i0*(M)+j0, r] = -1.0
    U[(i0+1)*M+j0, r] = 1.0
    U[i0*(M)+j0+1, r] = -1.0
    U[(i0+1)*M+j0+1, r] = 1.0
    V[j0*(P)+k0, r] = 1.0
    V[j0*(P)+k0+1, r] = 1.0
    V[(j0+1)*(P)+k0, r] = 1.0
    V[(j0+1)*(P)+k0+1, r] = 1.0
    W[k0*(N)+i0+1, r] = 1.0
    r += 1

    # M7 = (A[i0,:2] - A[i0+1,:2]) * (B[:2,k0+1] - B[:2,k0])
    #    -> C[i0,k0] += M7
    U[i0*(M)+j0, r] = 1.0
    U[(i0+1)*M+j0, r] = -1.0
    U[i0*(M)+j0+1, r] = 1.0
    U[(i0+1)*M+j0+1, r] = -1.0
    V[j0*(P)+k0, r] = -1.0
    V[j0*(P)+k0+1, r] = 1.0
    V[(j0+1)*(P)+k0, r] = -1.0
    V[(j0+1)*(P)+k0+1, r] = 1.0
    W[k0*(N)+i0, r] = 1.0
    r += 1

    return r


def _build_full_tensor():
    """Build the full matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    n, m, p = N, M, P
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T


def _tensor_to_2d(T):
    """Reshape 3D tensor to 2D for rank analysis."""
    return T.reshape(-1, T.shape[2])


def _find_rank1_decomposition(T, max_rank, tol=1e-9, max_iters=5000):
    """
    Try to find a low-rank decomposition of tensor T by iteratively
    peeling off rank-1 components using a greedy approach.
    
    Returns (U, V, W, rank) if successful, else None.
    """
    n, m, p = N, M, P
    T_copy = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    for rank in range(max_rank):
        if time.time() > DEADLINE_S - 5:
            return None
        
        # Find best rank-1 approximation via SVD on a random projection
        # or use a heuristic: find the element with largest magnitude
        # and try to fit a rank-1 tensor through it
        
        # Greedy approach: find (i,j,k) with largest |T[i,j,k]|
        idx = np.unravel_index(np.argmax(np.abs(T_copy)), T_copy.shape)
        i, j, k = idx
        
        if abs(T_copy[i, j, k]) < tol:
            # Check if remaining tensor is essentially zero
            if np.max(np.abs(T_copy)) < tol:
                if rank == 0:
                    return None  # Zero tensor, no decomposition needed
                break
        
        # Try to find u, v, w such that u⊗v⊗w approximates T_copy
        # Use the constraint that u[i]*v[j]*w[k] ≈ T_copy[i,j,k]
        
        # Heuristic: set u[i]=1, v[j]=1, w[k]=T_copy[i,j,k]
        # Then solve for remaining entries
        
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        
        u[i] = 1.0
        v[j] = 1.0
        w[k] = T_copy[i, j, k]
        
        # For each (i',j'), if T_copy[i',j',k] is nonzero and u[i'] and v[j'] are set,
        # we can determine w[k] more precisely
        # Actually, let's use a simpler approach: assume u and v have sparse support
        
        # Better heuristic: try small integer/half-integer values
        # For now, use a simple rank-1 fit via alternating least squares
        
        # Initialize with the dominant entry
        best_u, best_v, best_w = None, None, None
        best_err = float('inf')
        
        # Try a few random initializations
        np.random.seed(42)
        for attempt in range(50):
            if time.time() > DEADLINE_S - 5:
                return None
            
            # Random sparse initialization
            u_init = np.zeros(n*m)
            v_init = np.zeros(m*p)
            w_init = np.zeros(p*n)
            
            # Set a few random entries
            for _ in range(3):
                u_init[np.random.randint(n*m)] = np.random.choice([-1, 1])
            for _ in range(3):
                v_init[np.random.randint(m*p)] = np.random.choice([-1, 1])
            for _ in range(3):
                w_init[np.random.randint(p*n)] = np.random.choice([-1, 1])
            
            # Ensure the dominant entry is captured
            if u_init[i] == 0:
                u_init[i] = 1.0
            if v_init[j] == 0:
                v_init[j] = 1.0
            if w_init[k] == 0:
                w_init[k] = 1.0
            
            # Alternating least squares
            u_curr, v_curr, w_curr = u_init.copy(), v_init.copy(), w_init.copy()
            
            for iteration in range(20):
                # Update u: for each i', u[i'] = sum_{j',k'} T[i',j',k'] * v[j'] * w[k'] / (sum_{j',k'} v[j']^2 * w[k']^2 + eps)
                for i_prime in range(n*m):
                    num = 0.0
                    den = 0.0
                    for j_prime in range(m*p):
                        if abs(v_curr[j_prime]) > tol:
                            for k_prime in range(p*n):
                                if abs(w_curr[k_prime]) > tol:
                                    num += T_copy[i_prime, j_prime, k_prime] * v_curr[j_prime] * w_curr[k_prime]
                                    den += v_curr[j_prime]**2 * w_curr[k_prime]**2
                    if den > tol:
                        u_curr[i_prime] = num / den
                
                # Update v
                for j_prime in range(m*p):
                    num = 0.0
                    den = 0.0
                    for i_prime in range(n*m):
                        if abs(u_curr[i_prime]) > tol:
                            for k_prime in range(p*n):
                                if abs(w_curr[k_prime]) > tol:
                                    num += T_copy[i_prime, j_prime, k_prime] * u_curr[i_prime] * w_curr[k_prime]
                                    den += u_curr[i_prime]**2 * w_curr[k_prime]**2
                    if den > tol:
                        v_curr[j_prime] = num / den
                
                # Update w
                for k_prime in range(p*n):
                    num = 0.0
                    den = 0.0
                    for i_prime in range(n*m):
                        if abs(u_curr[i_prime]) > tol:
                            for j_prime in range(m*p):
                                if abs(v_curr[j_prime]) > tol:
                                    num += T_copy[i_prime, j_prime, k_prime] * u_curr[i_prime] * v_curr[j_prime]
                                    den += u_curr[i_prime]**2 * v_curr[j_prime]**2
                    if den > tol:
                        w_curr[k_prime] = num / den
                
                # Check convergence
                outer = np.einsum('i,j,k->ijk', u_curr, v_curr, w_curr)
                err = np.max(np.abs(T_copy - outer))
                if err < best_err:
                    best_err = err
                    best_u, best_v, best_w = u_curr.copy(), v_curr.copy(), w_curr.copy()
                
                if err < tol:
                    break
            
            if best_err < tol:
                break
        
        if best_err >= tol:
            # This rank-1 component couldn't be fit well
            # Try a different approach: just use the dominant entry as a rank-1 term
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            u[i] = 1.0
            v[j] = 1.0
            w[k] = T_copy[i, j, k]
            best_err = np.max(np.abs(T_copy - np.einsum('i,j,k->ijk', u, v, w)))
        
        if best_u is not None and best_err < tol:
            u, v, w = best_u, best_v, best_w
        else:
            u, v, w = np.zeros(n*m), np.zeros(m*p), np.zeros(p*n)
            u[i] = 1.0
            v[j] = 1.0
            w[k] = T_copy[i, j, k]
        
        # Subtract this rank-1 component
        T_copy -= np.einsum('i,j,k->ijk', u, v, w)
        
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
        # Check if we've reached zero
        if np.max(np.abs(T_copy)) < tol:
            break
    
    if np.max(np.abs(T_copy)) > tol:
        return None
    
    if not U_list:
        return None
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    return U, V, W, len(U_list)


def construct():
    t0 = time.time()
    n, m, p = N, M, P
    
    # Start with the known rank-98 decomposition
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    r = 0
    r = _strassen_2x2x2(0, 0, 0, U, V, W, r)
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
    r = _strassen_2x2x2(2, 0, 0, U, V, W, r)
    for i in range(2, 4):
        for j in range(2, 5):
            for k in range(2):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
    for i in range(2, 4):
        for j in range(2):
            for k in range(2, 5):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
    for i in range(2, 4):
        for j in range(2, 5):
            for k in range(2, 5):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
    
    if r != rank:
        pass

    if not _validate(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = (U.copy(), V.copy(), W.copy(), rank)
    
    # Try to find a lower-rank decomposition
    T = _build_full_tensor()
    
    # Try various target ranks
    for target_rank in range(80, 98):
        if time.time() > DEADLINE_S - 10:
            break
        
        result = _find_rank1_decomposition(T, target_rank)
        if result is not None:
            U_new, V_new, W_new, actual_rank = result
            if actual_rank < best_result[3]:
                if _validate(U_new, V_new, W_new):
                    best_result = (U_new, V_new, W_new, actual_rank)
    
    U, V, W, _ = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
