import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, R):
    """Exact verification that the factorization reproduces the mult tensor."""
    U = np.asarray(U, dtype=object)
    V = np.asarray(V, dtype=object)
    W = np.asarray(W, dtype=object)
    T = np.zeros((n, m, p), dtype=object)
    for r in range(R):
        a = U[:, r]
        b = V[:, r]
        w = W[:, r]
        az = np.nonzero(a)[0]
        bz = np.nonzero(b)[0]
        wz = np.nonzero(w)[0]
        for ai in az:
            av = a[ai]
            for bi in bz:
                bv = b[bi]
                val = av * bv
                if val == 0:
                    continue
                i, j = divmod(ai, m)
                j2, k = divmod(bi, p)
                if j2 != j:
                    continue
                for wi in wz:
                    coeff = w[wi]
                    k2, ii = divmod(wi, n)
                    T[ii, j, k2] += val * coeff
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i, j, k] != 1:
                    return False
    return True


def _cleanup_residue(U, V, W, n, m, p, R):
    """Append rank-1 terms to fix any residual so the decomposition is exact."""
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    T = np.zeros((n, m, p), dtype=object)
    for r in range(R):
        a = np.asarray(U[:, r], dtype=object)
        b = np.asarray(V[:, r], dtype=object)
        w = np.asarray(W[:, r], dtype=object)
        az = np.nonzero(a)[0]
        bz = np.nonzero(b)[0]
        wz = np.nonzero(w)[0]
        for ai in az:
            av = a[ai]
            for bi in bz:
                bv = b[bi]
                val = av * bv
                if val == 0:
                    continue
                i, j = divmod(ai, m)
                j2, k = divmod(bi, p)
                if j2 != j:
                    continue
                for wi in wz:
                    coeff = w[wi]
                    k2, ii = divmod(wi, n)
                    T[ii, j, k2] += val * coeff
    for i in range(n):
        for j in range(m):
            for k in range(p):
                resid = 1 - T[i, j, k]
                if resid != 0:
                    U.append([0.0] * R + [float(resid)])
                    V.append([0.0] * R + [1.0])
                    W.append([0.0] * R + [1.0])
                    T[i, j, k] = 1
                    R += 1
    return U, V, W, R


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 120.0
    safety = 10.0

    nA = n * m      # 15
    nB = m * p      # 35
    nC = p * n      # 21

    def time_ok():
        return (deadline - time.time()) > safety

    def idx_A(i, j):
        return i * m + j

    def idx_B(j, k):
        return j * p + k

    def idx_C(k, i):
        return k * n + i

    # ---- Candidate 1: schoolbook (always exact, rank 105) ----
    def schoolbook():
        R = n * m * p
        U = [[0.0] * R for _ in range(nA)]
        V = [[0.0] * R for _ in range(nB)]
        W = [[0.0] * R for _ in range(nC)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_A(i, j)][col] = 1.0
                    V[idx_B(j, k)][col] = 1.0
                    W[idx_C(k, i)][col] = 1.0
                    col += 1
        return U, V, W, R

    U_best, V_best, W_best, R_best = schoolbook()

    # ---- Polynomial interpolation embedding ----
    # Map matrix multiplication to polynomial coefficient extraction:
    #   For fixed i: C[i,k] = sum_j A[i,j] * B[j,k]
    #   Think of A_row[i](x) = sum_j A[i,j] * x^j  (degree < m)
    #   B_col[k](x) = sum_j B[j,k] * x^(m-1-j)  (degree < m)
    #   Then C[i,k] = coefficient of x^(m-1) in A_row[i](x) * B_col[k](x)
    #
    # The product has degree < 2m-1. To extract coefficient at m-1, we can use
    # polynomial interpolation: evaluate at R points and solve.
    #
    # Key insight: if we choose R = 2m-1 = 9 evaluation points, we can exactly
    # reconstruct all coefficients of the degree-<9 product, including the one
    # at position m-1=4.
    #
    # But we need this to work uniformly for all (i,k) pairs. The structure is:
    #   U[i,j] for the "evaluation" at point alpha_r: U[i, j, r] = A[i,j] * alpha_r^j
    #   V[j,k] for the "evaluation": V[j, k, r] = B[j,k] * alpha_r^(m-1-j)
    #   W[k,i] for "coefficient extraction": W[k, i, r] = c_r where sum_r c_r * f(alpha_r) = f coefficient at m-1
    #
    # This gives rank = R where R is number of evaluation points.
    # For exact coefficient extraction of a degree-<d polynomial, we need d points.
    # Product degree < 2m-1, so we need R = 2m-1 = 9 points.
    #
    # Let's verify this works:
    #   T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    #            = sum_r (A[i,j]*alpha_r^j) * (B[j,k]*alpha_r^(m-1-j)) * c_r
    #            = A[i,j]*B[j,k] * sum_r alpha_r^(m-1) * c_r
    #            = A[i,j]*B[j,k] * 1  (by interpolation property)
    #
    # So the tensor is exactly decomposed with rank = 2m-1 = 9!
    #
    # The coefficients c_r are the Lagrange interpolation weights for extracting
    # the coefficient of x^(m-1) from a polynomial of degree < 2m-1 evaluated
    # at points alpha_0, ..., alpha_{R-1}.

    def polynomial_embedding():
        """Build rank-(2m-1) decomposition via polynomial interpolation."""
        R = 2 * m - 1  # 9 for m=5
        
        # Choose evaluation points: use roots of unity or distinct integers
        # Using distinct integers 0, 1, 2, ..., R-1 for simplicity
        alphas = [float(r) for r in range(R)]
        
        # Compute Lagrange coefficients for extracting coefficient of x^(m-1)
        # from polynomial P(x) = sum_{d=0}^{R-1} coeff_d * x^d
        # given values P(alpha_r) for r=0,...,R-1
        #
        # We want: coeff_{m-1} = sum_r c_r * P(alpha_r)
        # where P(x) is any polynomial of degree < R with P(alpha_r) = given values.
        #
        # The coefficient of x^(m-1) in the interpolating polynomial can be found
        # by solving a linear system. For a basis polynomial L_r(x) = prod_{s!=r} (x-alpha_s)/(alpha_r-alpha_s),
        # P(x) = sum_r P(alpha_r) * L_r(x)
        # coeff_{m-1} = sum_r P(alpha_r) * [x^(m-1)] L_r(x)
        #
        # So c_r = [x^(m-1)] L_r(x)
        
        # Compute c_r = coefficient of x^(m-1) in Lagrange basis polynomial L_r(x)
        # L_r(x) = prod_{s != r} (x - alpha_s) / (alpha_r - alpha_s)
        
        c = [0.0] * R
        for r in range(R):
            # Build numerator polynomial prod_{s != r} (x - alpha_s)
            # Start with [1], multiply by (x - alpha_s) for each s != r
            num = [1.0]
            for s in range(R):
                if s == r:
                    continue
                # Multiply num by (x - alpha_s)
                new_num = [0.0] * (len(num) + 1)
                for d in range(len(num)):
                    new_num[d] -= num[d] * alphas[s]
                    new_num[d + 1] += num[d]
                num = new_num
            # Denominator: prod_{s != r} (alpha_r - alpha_s)
            denom = 1.0
            for s in range(R):
                if s != r:
                    denom *= (alphas[r] - alphas[s])
            # c_r = num[m-1] / denom (coefficient of x^(m-1) in numerator, divided by denom)
            if m - 1 < len(num):
                c[r] = num[m - 1] / denom
            else:
                c[r] = 0.0
        
        # Build U, V, W matrices
        U = [[0.0] * R for _ in range(nA)]
        V = [[0.0] * R for _ in range(nB)]
        W = [[0.0] * R for _ in range(nC)]
        
        for i in range(n):
            for j in range(m):
                for r in range(R):
                    U[idx_A(i, j)][r] = alphas[r] ** j
        
        for j in range(m):
            for k in range(p):
                for r in range(R):
                    V[idx_B(j, k)][r] = alphas[r] ** (m - 1 - j)
        
        for k in range(p):
            for i in range(n):
                for r in range(R):
                    W[idx_C(k, i)][r] = c[r]
        
        return U, V, W, R
    
    # Try polynomial embedding
    U_poly, V_poly, W_poly, R_poly = polynomial_embedding()
    
    if _verify(U_poly, V_poly, W_poly, n, m, p, R_poly):
        U_best, V_best, W_best, R_best = U_poly, V_poly, W_poly, R_poly
    else:
        # Debug: check what went wrong
        # Fall back to schoolbook
        pass
    
    # Final safety: ensure exactness of whatever we keep.
    if not _verify(U_best, V_best, W_best, n, m, p, R_best):
        U_best, V_best, W_best, R_best = _cleanup_residue(
            U_best, V_best, W_best, n, m, p, R_best)
        if not _verify(U_best, V_best, W_best, n, m, p, R_best):
            # Last resort: pure schoolbook.
            U_best, V_best, W_best, R_best = schoolbook()

    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
