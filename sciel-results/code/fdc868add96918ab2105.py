import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Condition: sum_c W[k, i, c] * U[i', j', c] * V[j', k', c] == delta[i', i] * delta[k', k]
    # We check for all i, i', k, k'
    for i in range(n):
        for k in range(p):
            for ip in range(n):
                for kp in range(p):
                    for jp in range(m):
                        # Calculate S(ip, jp, kp; i, k)
                        # S = sum_c W_r[kp, i, c] * U_r[ip, jp, c] * V_r[jp, kp, c]
                        # Note: V index is j, k. Here j=jp, k=kp.
                        # W index is k, i. Here k=kp, i=i.
                        # U index is i, j. Here i=ip, j=jp.
                        
                        # Wait, the condition was:
                        # sum_c W[k, i, c] U[i', j', c] V[j', k', c]
                        # Here k is the output col, i is output row.
                        # i', j' are A indices. j', k' are B indices.
                        # The term in C[i,k] comes from A[i',j'] B[j',k'] with weight S.
                        # We want C[i,k] = sum_j A[i,j] B[j,k].
                        # So S(ip, jp, kp; i, k) should be 1 if ip==i and kp==k and jp==j?
                        # No, sum_j A[i,j] B[j,k].
                        # The term A[i,j] B[j,k] has ip=i, jp=j, kp=k.
                        # So S(i, j, k; i, k) should be 1.
                        # And S(ip, jp, kp; i, k) should be 0 otherwise?
                        # Yes, because any other term A[ip,jp] B[jp,kp] should not contribute to C[i,k].
                        
                        val = np.sum(W_r[k, i, :] * U_r[ip, jp, :] * V_r[jp, kp, :])
                        
                        expected = 1.0 if (ip == i and kp == k) else 0.0
                        # Note: jp is summed in the target, so for fixed i,k, 
                        # we need sum_j S(i,j,k; i,k) A[i,j]B[j,k] = sum_j A[i,j]B[j,k].
                        # This requires S(i,j,k; i,k) = 1 for all j.
                        # And S(ip,jp,kp; i,k) = 0 for ip!=i or kp!=k.
                        
                        if abs(val - expected) > tol:
                            return False
    return True

def _solve_for_W(U, V, n, m, p):
    R = U.shape[1]
    W = np.zeros((p*n, R))
    for i in range(n):
        for k in range(p):
            # We need to solve for W[k*n+i, :]
            # Constraints:
            # For all ip, kp, jp:
            # sum_c W[k,i,c] U[ip,jp,c] V[jp,kp,c] = delta[ip,i] delta[kp,k]
            
            # This is a linear system for W[k,i,:].
            # Let w = W[k,i,:].
            # For each (ip, kp, jp), we have an equation:
            # w . (U[ip,jp,:] * V[jp,kp,:]) = delta[ip,i] delta[kp,k]
            
            A = []
            b = []
            for ip in range(n):
                for kp in range(p):
                    for jp in range(m):
                        A.append(U[ip*m+jp, :] * V[jp*p+kp, :])
                        b.append(1.0 if (ip == i and kp == k) else 0.0)
            
            A = np.array(A)
            b = np.array(b)
            w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            W[k*n+i, :] = w
    return W

def _solve_for_V(U, W, n, m, p):
    R = U.shape[1]
    V = np.zeros((m*p, R))
    for j in range(m):
        for k in range(p):
            # Solve for V[j*p+k, :]
            # Constraints:
            # For all i, ip, kp:
            # sum_c W[k,i,c] U[ip,j,c] V[j,k,c] = delta[ip,i] delta[kp,k]
            # Note: V depends on j, k.
            # The term is W[k,i,:] * U[ip,j,:] * V[j,k,:]
            
            A = []
            b = []
            for i in range(n):
                for ip in range(n):
                    for kp in range(p):
                        # We need to be careful. The equation is for a specific (i,k) output.
                        # But V[j,k] is shared across all i?
                        # No, V[j,k,c] is a single variable.
                        # The equations are coupled.
                        # However, in alternating minimization, we fix U and W, and solve for V.
                        # The equations for V[j,k,:] are:
                        # For each i, ip, kp:
                        # (W[k,i,:] * U[ip,j,:]) . V[j,k,:] = delta[ip,i] delta[kp,k]
                        
                        # This is a system of n * n * p equations for R variables.
                        # A row is W[k,i,:] * U[ip,j,:]
                        # b is delta[ip,i] delta[kp,k]
                        
                        A.append(W[k*n+i, :] * U[ip*m+j, :])
                        b.append(1.0 if (ip == i and kp == k) else 0.0)
                        
            A = np.array(A)
            b = np.array(b)
            v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            V[j*p+k, :] = v
    return V

def _solve_for_U(V, W, n, m, p):
    R = V.shape[1]
    U = np.zeros((n*m, R))
    for i in range(n):
        for j in range(m):
            # Solve for U[i*m+j, :]
            # Constraints:
            # For all k, kp, ip:
            # sum_c W[k,i,c] U[i,j,c] V[j,kp,c] = delta[ip,i] delta[kp,k]
            # Wait, the index in U is i,j.
            # The equation involves U[ip,j,:].
            # If we are solving for U[i,j,:], it appears in equations where ip==i.
            
            A = []
            b = []
            for k in range(p):
                for kp in range(p):
                    for ip in range(n):
                        if ip == i:
                            # Equation: W[k,i,:] * U[i,j,:] * V[j,kp,:] = delta[i,i] delta[kp,k] = 1 * delta[kp,k]
                            A.append(W[k*n+i, :] * V[j*p+kp, :])
                            b.append(1.0 if kp == k else 0.0)
                        # If ip != i, U[i,j,:] is not involved in that specific term?
                        # The term is W[k,i,c] U[ip,j,c] V[j,kp,c].
                        # We are solving for U[i,j,:]. It only appears when ip=i.
                        
            A = np.array(A)
            b = np.array(b)
            u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            U[i*m+j, :] = u
    return U

def _round_to_half(x):
    return np.round(x * 2) / 2

def _alternating_minimize(U, V, W, n, m, p, max_iter=50, tol=1e-9, deadline=None):
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = float('inf')
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        W = _solve_for_W(U, V, n, m, p)
        V = _solve_for_V(U, W, n, m, p)
        U = _solve_for_U(V, W, n, m, p)
        
        U_r = _round_to_half(U)
        V_r = _round_to_half(V)
        W_r = _round_to_half(W)
        
        # Compute error
        U_arr = np.array(U_r, dtype=float).reshape(n, m, best_rank)
        V_arr = np.array(V_r, dtype=float).reshape(m, p, best_rank)
        W_arr = np.array(W_r, dtype=float).reshape(p, n, best_rank)
        
        err = 0.0
        for i in range(n):
            for k in range(p):
                for ip in range(n):
                    for kp in range(p):
                        for jp in range(m):
                            val = np.sum(W_arr[k, i, :] * U_arr[ip, jp, :] * V_arr[jp, kp, :])
                            expected = 1.0 if (ip == i and kp == k) else 0.0
                            err += (val - expected) ** 2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
        
        if _validate(U_r, V_r, W_r, n, m, p, tol):
            return U_r, V_r, W_r, U_r.shape[1]
    
    if _validate(best_U, best_V, best_W, n, m, p, tol):
        return best_U, best_V, best_W, best_U.shape[1]
    
    return None

def _try_rank(R, n, m, p, n_restarts=5, deadline=None):
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        np.random.seed(42 + restart * 1000)
        U = np.random.randn(n*m, R) * 0.3
        V = np.random.randn(m*p, R) * 0.3
        W = np.random.randn(p*n, R) * 0.3
        
        result = _alternating_minimize(U, V, W, n, m, p, max_iter=30, deadline=deadline)
        if result is not None:
            return result
    
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    U_school, V_school, W_school, rank_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try ranks from low to high
    for R in range(20, best_rank + 1, 1):
        if time.time() > deadline:
            break
        result = _try_rank(R, n, m, p, n_restarts=3, deadline=deadline)
        if result is not None:
            U_r, V_r, W_r, r = result
            best_U, best_V, best_W = U_r.tolist(), V_r.tolist(), W_r.tolist()
            best_rank = r
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
