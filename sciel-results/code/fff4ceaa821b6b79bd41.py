import numpy as np
import time
from scipy.linalg import expm, eig

try:
    import mpmath
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

try:
    from decimal import Decimal, getcontext
    HAS_DECIMAL = True
except ImportError:
    HAS_DECIMAL = False

def integrate(f, t_span, y0):
    deadline = time.time() + 290
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)

    # Extract A
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        col = f(t0, e)
        A[:, j] = np.asarray(col, dtype=float)
        if time.time() > deadline:
            break

    # Compute expm(dt * A) @ y0
    
    # Try mpmath first
    if HAS_MPMATH:
        try:
            mpmath.mp.dps = 50
            A_mp = mpmath.matrix(n, n)
            for i in range(n):
                for j in range(n):
                    A_mp[i, j] = mpmath.mpf(A[i, j])
            
            dt_mp = mpmath.mpf(str(dt)) # Use string to avoid float repr issues if dt is not exact
            M_mp = mpmath.expm(dt_mp * A_mp)
            
            M = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(n):
                    M[i, j] = float(M_mp[i, j])
            
            return M @ y0
        except Exception:
            pass

    # Try decimal (standard library)
    if HAS_DECIMAL:
        try:
            getcontext().prec = 50
            A_dec = [[Decimal(0) for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    A_dec[i][j] = Decimal(str(A[i, j]))
            
            dt_dec = Decimal(str(dt))
            
            # Taylor series for expm
            # expm(X) = I + X + X^2/2! + ...
            X = [[A_dec[i][j] * dt_dec for j in range(n)] for i in range(n)]
            M_dec = [[Decimal(0) for _ in range(n)] for _ in range(n)]
            for i in range(n):
                M_dec[i][i] = Decimal(1)
            
            term = [[Decimal(1) for _ in range(n)] for _ in range(n)]
            # Identity matrix
            for i in range(n):
                term[i][i] = Decimal(1)
            
            # We need to add terms until they are small
            # term_k = X^k / k!
            # Start with k=0 (Identity)
            M_dec = [row[:] for row in term]
            
            k = 1
            while time.time() < deadline:
                # term = term * X / k
                # X is constant
                # New term = (term * X) / k
                new_term = [[Decimal(0) for _ in range(n)] for _ in range(n)]
                for i in range(n):
                    for j in range(n):
                        s = Decimal(0)
                        for m in range(n):
                            s += term[i][m] * X[m][j]
                        new_term[i][j] = s / k
                
                # Check if term is small
                max_val = Decimal(0)
                for i in range(n):
                    for j in range(n):
                        if abs(new_term[i][j]) > max_val:
                            max_val = abs(new_term[i][j])
                
                if max_val < Decimal('1e-45'):
                    break
                
                for i in range(n):
                    for j in range(n):
                        M_dec[i][j] += new_term[i][j]
                
                term = new_term
                k += 1
                
                if k > 1000: # Safety break
                    break

            M = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(n):
                    M[i, j] = float(M_dec[i][j])
            
            return M @ y0
        except Exception:
            pass

    # Fallback to scipy
    use_eig = False
    try:
        vals, vecs = eig(dt * A)
        cond = np.linalg.cond(vecs)
        if cond < 1e8:
            use_eig = True
    except Exception:
        use_eig = False

    if use_eig:
        try:
            V_inv = np.linalg.inv(vecs)
            M = (vecs @ np.diag(np.exp(vals))) @ V_inv
        except Exception:
            M = expm(dt * A)
    else:
        M = expm(dt * A)
    
    return M @ y0
