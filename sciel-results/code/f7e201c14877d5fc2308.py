import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor T[i,j,jp,k,kp,ip] = delta(i,ip)*delta(j,jp)*delta(k,kp)."""
    # We only need the "diagonal" entries where the indices match
    # T is implicitly defined; we'll work with the constraint equations directly
    pass

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i, jp, kp): the matrix product contribution should be
    # sum_c U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+i', c] = delta(i,i')*delta(j,jp)*delta(k,kp)
    # More efficiently: for fixed (i, jp), define M_jp[i, k, c] = U[i*m+jp, c] * V[jp*p+k, c]
    # Then sum over c and check against W
    
    # Direct check for all combinations
    for i in range(n):
        for jp in range(m):
            for kp in range(p):
                for ip in range(n):
                    for j in range(m):
                        for k in range(p):
                            lhs = 0.0
                            for c in range(rank):
                                lhs += U[i*m+j][c] * V[jp*p+k][c] * W[kp*n+ip][c]
                            rhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            if abs(lhs - rhs) > 1e-9:
                                return False
    return True

def _alternating_minimize(U, V, W, n, m, p, max_iters=200, tol=1e-12):
    """Alternating least squares for CP decomposition of the multiplication tensor."""
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    r = U.shape[1]
    
    prev_loss = float('inf')
    
    for it in range(max_iters):
        # Fix U, V: solve for W
        # For each output index (kp, ip):
        # sum_c U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
        # For fixed (kp, ip), and for each (i, j, jp, k):
        # This is a linear system in W[kp*n+ip, :]
        # 
        # Actually, let's think of it differently. The tensor equation is:
        # T = sum_c u_c ⊗ v_c ⊗ w_c
        # where u_c is indexed by (i,j), v_c by (jp,k), w_c by (kp,ip)
        #
        # For fixed U, V: we need to find W such that
        # sum_c W[kp*n+ip, c] * (sum over i,j,jp,k of U[i*m+j,c]*V[jp*p+k,c]*delta(i,ip)*delta(j,jp)*delta(k,kp)) = 1
        # But delta(i,ip)*delta(j,jp)*delta(k,kp) means i=ip, j=jp, k=kp
        # So: sum_c W[kp*n+ip, c] * U[ip*m+jp,c] * V[jp*p+kp,c] = 1
        # This is ONE equation per (kp,ip) with r unknowns. Underdetermined!
        #
        # I need to use ALL the equations. Let me reformulate.
        # The full set of equations for fixed (i, jp, kp, ip, j, k) is:
        # sum_c U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
        #
        # For fixed (kp, ip), the unknowns are W[kp*n+ip, 0..r-1].
        # The equations involve all (i, j, jp, k).
        # But delta(i,ip)*delta(j,jp)*delta(k,kp) is nonzero only when i=ip, j=jp, k=kp.
        # So for fixed (kp, ip), only the equation with i=ip, j=jp, k=kp is nonzero... 
        # No wait, that's not right either. The left side sums over c, and for each c,
        # U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c] contributes.
        #
        # Let me just build the linear system properly.
        # For W: unknowns are W[0..p*n-1, 0..r-1], total p*n*r unknowns.
        # Equations: for each (i,j,jp,k,kp,ip): one equation.
        # Total: n*m*m*p*p*n = (n*m)*(m*p)*(p*n) equations.
        # For n=2,m=4,p=6: 8*16*12 = 1536 equations, 12*r unknowns.
        #
        # This is a large linear system. Let me use a different approach.
        #
        # Alternative: use the fact that the tensor has a specific structure.
        # T[(i,j),(jp,k),(kp,ip)] = delta(i,ip)*delta(j,jp)*delta(k,kp)
        #
        # This can be written as: T = I_n ⊗ I_m ⊗ I_p (in some sense)
        #
        # Actually, let me just use gradient descent on the loss function.
        
        # Compute loss
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                lhs = 0.0
                                for c in range(r):
                                    lhs += U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c]
                                rhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                loss += (lhs - rhs) ** 2
        
        if loss < tol:
            break
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
        
        # Update W: for each (kp, ip), minimize sum over (i,j,jp,k) of
        # (sum_c U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c] - delta(i,ip)*delta(j,jp)*delta(k,kp))^2
        # w.r.t. W[kp*n+ip, :]
        # This is a linear least squares problem.
        # Let A_w be a matrix where row (i,j,jp,k) has entry A_w[(i,j,jp,k), c] = U[i*m+j,c]*V[jp*p+k,c]
        # But we only care about entries where the RHS is nonzero, i.e., i=ip, j=jp, k=kp
        # For fixed (kp, ip), the nonzero RHS equations are those with i=ip, j=jp, k=kp
        # So: A_w[jp, c] = U[ip*m+jp, c] * V[jp*p+kp, c] for jp=0..m-1
        # b_w[jp] = 1
        # Solve: A_w @ W[kp*n+ip, :] = b_w
        # But this is m equations with r unknowns. If r > m, underdetermined.
        # Use least squares: W[kp*n+ip, :] = A_w^T @ (A_w @ A_w^T)^{-1} @ b_w
        
        for kp in range(p):
            for ip in range(n):
                # Build A_w: m x r
                A_w = np.zeros((m, r))
                b_w = np.zeros(m)
                for jp in range(m):
                    A_w[jp, :] = U[ip*m+jp, :] * V[jp*p+kp, :]
                    b_w[jp] = 1.0
                # Solve A_w @ x = b_w in least squares sense
                # x = A_w^T @ inv(A_w @ A_w^T) @ b_w
                AAT = A_w @ A_w.T  # m x m
                try:
                    x = np.linalg.solve(AAT, A_w.T @ b_w)
                    W[kp*n+ip, :] = x
                except np.linalg.LinAlgError:
                    W[kp*n+ip, :] = np.linalg.lstsq(A_w, b_w, rcond=None)[0]
        
        # Update U: for each (i, j), minimize sum over (jp,k,kp,ip) of
        # (sum_c U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c] - delta(i,ip)*delta(j,jp)*delta(k,kp))^2
        # For fixed (i, j), nonzero RHS when ip=i, jp=j, k=kp
        # So: A_u[kp, c] = V[j*p+kp, c] * W[kp*n+i, c] for kp=0..p-1
        # b_u[kp] = 1
        for i in range(n):
            for j in range(m):
                A_u = np.zeros((p, r))
                b_u = np.zeros(p)
                for kp in range(p):
                    A_u[kp, :] = V[j*p+kp, :] * W[kp*n+i, :]
                    b_u[kp] = 1.0
                AAT = A_u @ A_u.T
                try:
                    x = np.linalg.solve(AAT, A_u.T @ b_u)
                    U[i*m+j, :] = x
                except np.linalg.LinAlgError:
                    U[i*m+j, :] = np.linalg.lstsq(A_u, b_u, rcond=None)[0]
        
        # Update V: for each (jp, k), minimize sum over (i,j,kp,ip) of
        # (sum_c U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c] - delta(i,ip)*delta(j,jp)*delta(k,kp))^2
        # For fixed (jp, k), nonzero RHS when i=ip, j=jp, k=kp=k
        # So: A_v[i, c] = U[i*m+jp, c] * W[k*n+i, c] for i=0..n-1
        # b_v[i] = 1
        for jp in range(m):
            for k in range(p):
                A_v = np.zeros((n, r))
                b_v = np.zeros(n)
                for i in range(n):
                    A_v[i, :] = U[i*m+jp, :] * W[k*n+i, :]
                    b_v[i] = 1.0
                AAT = A_v @ A_v.T
                try:
                    x = np.linalg.solve(AAT, A_v.T @ b_v)
                    V[jp*p+k, :] = x
                except np.linalg.LinAlgError:
                    V[jp*p+k, :] = np.linalg.lstsq(A_v, b_v, rcond=None)[0]
    
    return U, V, W, prev_loss

def _round_to_half_integer(arr):
    """Round array entries to nearest half-integer."""
    return np.round(arr * 2) / 2

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Based on known results, try ranks from 28 down to 20
    for target_rank in range(28, 19, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for restart in range(3):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(restart * 1000 + target_rank)
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            U, V, W, loss = _alternating_minimize(U, V, W, n, m, p, max_iters=100, tol=1e-10)
            
            if loss < 1e-8:
                # Round to half-integers and verify
                U_r = _round_to_half_integer(U)
                V_r = _round_to_half_integer(V)
                W_r = _round_to_half_integer(W)
                
                if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                    best_rank = target_rank
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    found = True
                    break
            
            # Also try without rounding (in case exact solution has integers)
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if _verify_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
                best_rank = target_rank
                best_result = {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                found = True
                break
        
        if found:
            break
    
    return best_result
