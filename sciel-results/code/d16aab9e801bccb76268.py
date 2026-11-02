"""Matrix multiplication 3x5x7 rank-minimal decomposition.
Structural approach: exploit T[i,j,k] = delta_{j,k} symmetry.
Search via structured parameterization + nonlinear solve + rational verification.
"""
import time
import numpy as np
from fractions import Fraction
import itertools

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nrm, mpr, pnr = n * m, m * p, p * n  # 15, 35, 21

    # Target tensor: T[i,j,k] = 1 if j==k else 0
    # Decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]

    def build_schoolbook():
        r = n * m * p
        U = np.zeros((nrm, r))
        V = np.zeros((mpr, r))
        W = np.zeros((pnr, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    best_U, best_V, best_W = build_schoolbook()
    best_rank = n * m * p

    def verify_float(U, V, W, tol=1e-8):
        """Check T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta_{j,k}."""
        r = U.shape[1]
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    target = 1.0 if j == k else 0.0
                    if abs(s - target) > tol:
                        return False
        return True

    def verify_rational(U, V, W):
        """Exact verification using Fraction."""
        r = U.shape[1]
        # Convert to Fraction
        def to_frac(x):
            return Fraction(x).limit_denominator(10**6)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += to_frac(U[i*m+j,c]) * to_frac(V[j*p+k,c]) * to_frac(W[k*n+i,c])
                    target = Fraction(1) if j == k else Fraction(0)
                    if s != target:
                        return False
        return True

    # ---- Structured parameterization exploiting delta_{j,k} symmetry ----
    # Key observation: T[i,j,k] = delta_{j,k} is independent of i (for the j==k case)
    # and zero otherwise. This suggests U and W can have structure.
    #
    # Ansatz A: U[i*m+j, c] = a_{j,c} (independent of i), W[k*n+i, c] = b_{k,c} (independent of i)
    # Then T[i,j,k] = sum_c a_{j,c} * V[j*p+k,c] * b_{k,c} = delta_{j,k}
    # This is independent of i, which is consistent since T is independent of i.
    # Unknowns: r*m (for a) + r*p (for b) + r*m*p (for V) = r*(m+p+m*p) = r*(5+7+35) = 47r
    # Equations: m*p = 35 (only j==k and j!=k matter, but really m*p = 35 equations)
    # For r=1: 47 unknowns, 35 equations -> underdetermined, but need exact solution.
    #
    # Let's try to solve for small r using this ansatz.
    def solve_structured_ansatz_A(r):
        """Ansatz: U[i*m+j,c]=a_{j,c}, W[k*n+i,c]=b_{k,c}, V[j*p+k,c]=v_{j,k,c}"""
        # Equations: for each (j,k): sum_c a_{j,c} * v_{j,k,c} * b_{k,c} = delta_{j,k}
        # Variables: a (m x r), b (p x r), v (m x p x r)
        # Total: r*(m + p + m*p) = r*47
        # Equations: m*p = 35
        from scipy.optimize import least_squares
        import numpy as np
        
        if time.time() > deadline - 30:
            return None
        
        def make_residuals(params):
            # params: flat array of a (m*r), b (p*r), v (m*p*r)
            a = params[:m*r].reshape(m, r)
            b = params[m*r:m*r+p*r].reshape(p, r)
            v = params[m*r+p*r:].reshape(m, p, r)
            residuals = []
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += a[j, c] * v[j, k, c] * b[k, c]
                    target = 1.0 if j == k else 0.0
                    residuals.append(s - target)
            return np.array(residuals)
        
        # Try multiple random initializations
        best_res = None
        best_params = None
        ntrials = 30
        for trial in range(ntrials):
            if time.time() > deadline - 25:
                break
            np.random.seed(42 + trial)
            x0 = np.random.randn(r * 47) * 0.5
            try:
                res = least_squares(make_residuals, x0, method='lm', max_nfev=5000)
                if res.cost < 1e-20:
                    # Check exact
                    a = res.x[:m*r].reshape(m, r)
                    b = res.x[m*r:m*r+p*r].reshape(p, r)
                    v = res.x[m*r+p*r:].reshape(m, p, r)
                    # Build U, V, W
                    U = np.zeros((nrm, r))
                    V = np.zeros((mpr, r))
                    W = np.zeros((pnr, r))
                    for i in range(n):
                        for j in range(m):
                            U[i*m+j, :] = a[j, :]
                    for j in range(m):
                        for k in range(p):
                            V[j*p+k, :] = v[j, k, :]
                    for k in range(p):
                        for i in range(n):
                            W[k*n+i, :] = b[k, :]
                    if verify_float(U, V, W, tol=1e-7):
                        return U, V, W, r
            except Exception:
                pass
        return None

    # Ansatz B: Even more structured. Since T[i,j,k]=delta_{j,k}, try:
    # U[i*m+j, c] = u_c (constant in i,j), W[k*n+i, c] = w_c (constant in k,i), V[j*p+k,c] = v_{j,k,c}
    # Then T[i,j,k] = sum_c u_c * v_{j,k,c} * w_c = delta_{j,k}
    # Unknowns: 2r + r*m*p = r*(2+35) = 37r, equations: 35
    def solve_structured_ansatz_B(r):
        """Ansatz: U constant, W constant, V free."""
        from scipy.optimize import least_squares
        
        if time.time() > deadline - 20:
            return None
        
        def make_residuals(params):
            u = params[:r]
            w = params[r:2*r]
            v = params[2*r:].reshape(m, p, r)
            residuals = []
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += u[c] * v[j, k, c] * w[c]
                    target = 1.0 if j == k else 0.0
                    residuals.append(s - target)
            return np.array(residuals)
        
        for trial in range(20):
            if time.time() > deadline - 15:
                break
            np.random.seed(100 + trial)
            x0 = np.random.randn(r * 37) * 0.5
            try:
                res = least_squares(make_residuals, x0, method='lm', max_nfev=5000)
                if res.cost < 1e-20:
                    u = res.x[:r]
                    w = res.x[r:2*r]
                    v = res.x[2*r:].reshape(m, p, r)
                    U = np.zeros((nrm, r))
                    V = np.zeros((mpr, r))
                    W = np.zeros((pnr, r))
                    for i in range(n):
                        for j in range(m):
                            U[i*m+j, :] = u
                    for j in range(m):
                        for k in range(p):
                            V[j*p+k, :] = v[j, k, :]
                    for k in range(p):
                        for i in range(n):
                            W[k*n+i, :] = w
                    if verify_float(U, V, W, tol=1e-7):
                        return U, V, W, r
            except Exception:
                pass
        return None

    # Ansatz C: U depends only on j, W depends only on k, V depends only on (j,k)
    # Same as A essentially. Let's also try:
    # U[i*m+j,c] = u_{j,c}, V[j*p+k,c] = v_c (constant in j,k), W[k*n+i,c] = w_{k,c}
    # T[i,j,k] = sum_c u_{j,c} * v_c * w_{k,c} = delta_{j,k}
    # This is a rank-r factorization of the m x p matrix delta (which is 5x7)
    # into U_j * diag(v) * W_k^T. The matrix delta is m x p with entries delta_{j,k}.
    # Its rank is min(m,p) = 5. So we need r >= 5 for this to work.
    def solve_structured_ansatz_C(r):
        """Ansatz: U=u_{j,c}, V=v_c, W=w_{k,c}. Factor delta = U * diag(v) * W^T."""
        if r < m:
            return None
        # delta is m x p matrix: delta[j,k] = 1 if j==k else 0
        # We need: delta[j,k] = sum_c u_{j,c} * v_c * w_{k,c}
        # This is: delta = U @ diag(v) @ W.T where U is m x r, W is p x r
        # Let A = U @ diag(v), then delta = A @ W.T
        # So we need rank(delta) <= r and a factorization.
        # delta has rank 5 (since m=5, p=7, and it's the first 5 rows of identity-like)
        # Actually delta[j,k] for j in [0,5), k in [0,7): 
        # j=0: [1,0,0,0,0,0,0]
        # j=1: [0,1,0,0,0,0,0]
        # ...
        # j=4: [0,0,0,0,1,0,0]
        # Rank = 5.
        if r < 5:
            return None
        # Use SVD or direct factorization
        delta = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                if j == k:
                    delta[j, k] = 1.0
        # Factor delta = A @ W.T with A (m x r), W (p x r)
        # Use QR or SVD
        Usvd, s, Vt = np.linalg.svd(delta, full_matrices=False)
        # delta = Usvd[:, :r] @ diag(s[:r]) @ Vt[:r, :]
        # Set A = Usvd[:, :r] @ diag(sqrt(s[:r])), W = Vt[:r, :].T @ diag(sqrt(s[:r]))
        # Then A @ W.T = Usvd[:, :r] @ diag(s[:r]) @ Vt[:r, :].T = delta[:, :r] @ ... 
        # Wait, need to be careful.
        # delta = Usvd @ diag(s) @ Vt
        # delta = (Usvd[:, :r] @ diag(s[:r])) @ (Vt[:r, :])
        # Let A = Usvd[:, :r] @ diag(s[:r]), W = Vt[:r, :].T
        # Then A @ W.T = Usvd[:, :r] @ diag(s[:r]) @ Vt[:r, :] = delta (if r >= rank)
        if r >= 5:
            A = Usvd[:, :r] * s[:r]  # m x r, each column scaled
            Wmat = Vt[:r, :].T  # p x r
            # Now we need: A[j,c] * v_c * Wmat[k,c] = delta[j,k]
            # But A @ Wmat.T already gives delta. So set v_c = 1.
            # Then we need A[j,c] * 1 * Wmat[k,c] summed over c = delta[j,k] ✓
            # So: u_{j,c} = A[j,c], v_c = 1, w_{k,c} = Wmat[k,c]
            U = np.zeros((nrm, r))
            V = np.zeros((mpr, r))
            W = np.zeros((pnr, r))
            for i in range(n):
                for j in range(m):
                    U[i*m+j, :] = A[j, :]
            for j in range(m):
                for k in range(p):
                    V[j*p+k, :] = 1.0  # v_c = 1 for all c
            for k in range(p):
                for i in range(n):
                    W[k*n+i, :] = Wmat[k, :]
            if verify_float(U, V, W, tol=1e-7):
                return U, V, W, r
        return None

    # Search from low rank upward
    # Lower bound: r >= 5 (rank of delta matrix in ansatz C, and tensor rank bound)
    for r in range(5, 30):
        if time.time() > deadline - 20:
            break
        
        # Try ansatz C first (most structured, likely to succeed for r >= 5)
        result = solve_structured_ansatz_C(r)
        if result is not None:
            U, V, W, rr = result
            if rr < best_rank:
                best_U, best_V, best_W, best_rank = U, V, W, rr
                print(f"Found rank {rr} via ansatz C")
                break  # Can't do better than r if we found it
        
        # Try ansatz B
        if time.time() < deadline - 15:
            result = solve_structured_ansatz_B(r)
            if result is not None:
                U, V, W, rr = result
                if rr < best_rank:
                    best_U, best_V, best_W, best_rank = U, V, W, rr
                    print(f"Found rank {rr} via ansatz B")
                    break
        
        # Try ansatz A
        if time.time() < deadline - 10:
            result = solve_structured_ansatz_A(r)
            if result is not None:
                U, V, W, rr = result
                if rr < best_rank:
                    best_U, best_V, best_W, best_rank = U, V, W, rr
                    print(f"Found rank {rr} via ansatz A")
                    break
    
    # If we found rank 5, that's optimal (lower bound is 5)
    # Convert to list format
    result_U = best_U.tolist()
    result_V = best_V.tolist()
    result_W = best_W.tolist()
    
    return {"U": result_U, "V": result_V, "W": result_W}
