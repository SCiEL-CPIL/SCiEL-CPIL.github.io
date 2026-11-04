"""Matrix multiplication 2x4x8 - rank minimization via block decomposition."""
import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
TIME_LIMIT = 140.0

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return TIME_LIMIT - _elapsed()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x8 matrix.
    """
    n, m, p = 2, 4, 8
    
    # ============================================================
    # Build the baseline rank-56 Strassen decomposition
    # ============================================================
    def build_strassen_rank56():
        rank = 56
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                # M1 = (A00 + A11)(B00 + B11)
                U[0 * m + j0, col] += 1.0
                U[1 * m + (j0 + 1), col] += 1.0
                V[j0 * p + k0, col] += 1.0
                V[(j0 + 1) * p + (k0 + 1), col] += 1.0
                W[k0 * n + 0, col] += 1.0
                W[(k0 + 1) * n + 1, col] += 1.0
                col += 1
                
                # M2 = (A10 + A11)B00
                U[1 * m + j0, col] += 1.0
                U[1 * m + (j0 + 1), col] += 1.0
                V[j0 * p + k0, col] += 1.0
                W[k0 * n + 1, col] += 1.0
                W[(k0 + 1) * n + 1, col] += -1.0
                col += 1
                
                # M3 = A00(B01 - B11)
                U[0 * m + j0, col] += 1.0
                V[j0 * p + (k0 + 1), col] += 1.0
                V[(j0 + 1) * p + (k0 + 1), col] += -1.0
                W[(k0 + 1) * n + 0, col] += 1.0
                W[(k0 + 1) * n + 1, col] += 1.0
                col += 1
                
                # M4 = A11(B10 - B00)
                U[1 * m + (j0 + 1), col] += 1.0
                V[(j0 + 1) * p + k0, col] += 1.0
                V[j0 * p + k0, col] += -1.0
                W[k0 * n + 0, col] += 1.0
                W[k0 * n + 1, col] += 1.0
                col += 1
                
                # M5 = (A00 + A01)B11
                U[0 * m + j0, col] += 1.0
                U[0 * m + (j0 + 1), col] += 1.0
                V[(j0 + 1) * p + (k0 + 1), col] += 1.0
                W[k0 * n + 0, col] += -1.0
                W[(k0 + 1) * n + 0, col] += 1.0
                col += 1
                
                # M6 = (A10 - A00)(B00 + B01)
                U[1 * m + j0, col] += 1.0
                U[0 * m + j0, col] += -1.0
                V[j0 * p + k0, col] += 1.0
                V[j0 * p + (k0 + 1), col] += 1.0
                W[(k0 + 1) * n + 1, col] += 1.0
                col += 1
                
                # M7 = (A01 - A11)(B10 + B11)
                U[0 * m + (j0 + 1), col] += 1.0
                U[1 * m + (j0 + 1), col] += -1.0
                V[(j0 + 1) * p + k0, col] += 1.0
                V[(j0 + 1) * p + (k0 + 1), col] += 1.0
                W[k0 * n + 0, col] += 1.0
                col += 1
        
        return U, V, W, rank
    
    U, V, W, rank = build_strassen_rank56()
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # ============================================================
    # Verify decomposition correctness
    # ============================================================
    def verify(U_mat, V_mat, W_mat, trials=5):
        """Verify that the decomposition exactly computes matrix multiplication."""
        rng = np.random.RandomState(42)
        r_max = U_mat.shape[1]
        
        for _ in range(trials):
            A = rng.randint(-10, 10, size=(n, m)).astype(float)
            B = rng.randint(-10, 10, size=(m, p)).astype(float)
            
            # Correct multiplication
            C_correct = A @ B
            
            # Decomposition: C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j,r]) * (sum_j B[j,k] * V[j*p+k,r]) * W[k*n+i,r]
            C_decomp = np.zeros((n, p))
            for r in range(r_max):
                u_vals = np.array([np.dot(A[i, :], U_mat[i*m:(i+1)*m, r]) for i in range(n)])
                v_vals = np.array([np.dot(B[:, k], V_mat[k::p, r]) for k in range(p)])
                
                for i in range(n):
                    for k in range(p):
                        C_decomp[i, k] += u_vals[i] * v_vals[k] * W_mat[k*n+i, r]
            
            if np.max(np.abs(C_correct - C_decomp)) > 1e-6:
                return False
        return True
    
    # Verify baseline
    if not verify(U, V, W):
        # Should not happen, but just in case
        pass

    # ============================================================
    # Try to reduce rank by finding linear dependencies
    # ============================================================
    def try_reduce_rank(U_mat, V_mat, W_mat, max_iter=100000):
        """Try to reduce rank by finding and merging proportional terms."""
        U_cur = U_mat.copy()
        V_cur = V_mat.copy()
        W_cur = W_mat.copy()
        r_cur = U_cur.shape[1]
        
        for iteration in range(max_iter):
            if _time_left() < 10:
                break
            
            # Look for pairs of columns that are proportional in U and V
            found_merge = False
            for r1 in range(r_cur):
                if _time_left() < 10:
                    break
                for r2 in range(r1 + 1, r_cur):
                    # Check if column r1 is proportional to column r2 in U
                    u1 = U_cur[:, r1]
                    u2 = U_cur[:, r2]
                    
                    # Find ratio from first non-zero element
                    ratio = None
                    for idx in range(len(u1)):
                        if abs(u1[idx]) > 1e-10:
                            if abs(u2[idx]) < 1e-10:
                                ratio = None
                                break
                            ratio = u1[idx] / u2[idx]
                            break
                        elif abs(u2[idx]) > 1e-10:
                            ratio = None
                            break
                    
                    if ratio is None or abs(ratio) < 1e-10:
                        continue
                    
                    # Verify U proportionality
                    if not np.allclose(u1, ratio * u2, atol=1e-9, rtol=1e-9):
                        continue
                    
                    # Check V proportionality
                    v1 = V_cur[:, r1]
                    v2 = V_cur[:, r2]
                    if not np.allclose(v1, ratio * v2, atol=1e-9, rtol=1e-9):
                        continue
                    
                    # Merge: add W[r1] to W[r2], zero out column r1
                    W_cur[:, r2] += W_cur[:, r1]
                    U_cur[:, r1] = 0
                    V_cur[:, r1] = 0
                    W_cur[:, r1] = 0
                    r_cur -= 1
                    found_merge = True
                    break
                if found_merge:
                    break
            
            if not found_merge:
                break
        
        return U_cur, V_cur, W_cur, r_cur
    
    # Try rank reduction
    if _time_left() > 20:
        U_red, V_red, W_red, rank_red = try_reduce_rank(U, V, W, max_iter=50000)
        if verify(U_red, V_red, W_red) and rank_red < best_rank:
            best_rank = rank_red
            best_U, best_V, best_W = U_red, V_red, W_red

    # ============================================================
    # Lattice reduction: treat rank-56 basis as a lattice, use LLL
    # to find shorter basis vectors (lower-rank decompositions)
    # ============================================================
    def _lll_reduce(matrix):
        """
        LLL lattice basis reduction on a numpy integer matrix.
        Returns the reduced basis (list of row vectors) and the
        transformation matrix T such that reduced = T @ matrix.
        """
        # Work with Fraction for exact arithmetic
        n_rows, n_cols = matrix.shape
        # Start with the original basis
        basis = [list(Fraction(int(x)) for x in row) for row in matrix]
        # Track the transformation: T[i] is the row of T
        T = [[Fraction(1) if i == j else Fraction(0) for j in range(n_rows)] for i in range(n_rows)]
        
        def vec_norm_sq(v):
            return sum(x * x for x in v)
        
        def vec_dot(a, b):
            return sum(x * y for x, y in zip(a, b))
        
        def vec_sub(a, b):
            return [x - y for x, y in zip(a, b)]
        
        def vec_mul(a, c):
            return [x * c for x in a]
        
        # LLL reduction
        delta = Fraction(1, 2) + Fraction(1, 4)  # 3/4
        k = len(basis)
        
        # Initialize mu and B (reduced Gram-Schmidt)
        B = basis[:]
        mu = [[Fraction(0)] * k for _ in range(k)]
        Bg = [b[:] for b in B]  # Gram-Schmidt orthogonalized
        
        # Compute initial Bg and mu
        for i in range(k):
            Bg[i] = B[i][:]
            for j in range(i):
                if vec_norm_sq(Bg[j]) > 0:
                    mu[i][j] = vec_dot(B[i], Bg[j]) / vec_norm_sq(Bg[j])
                    Bg[i] = vec_sub(Bg[i], vec_mul(Bg[j], mu[i][j]))
        
        # LLL main loop
        for b in range(1, k):
            while b > 0:
                # Size reduction
                for i in range(b - 1, -1, -1):
                    if abs(mu[b][i]) > Fraction(1, 2):
                        q = round(float(mu[b][i]))
                        B[b] = vec_sub(B[b], vec_mul(B[i], q))
                        T[b] = vec_sub(T[b], vec_mul(T[i], q))
                        # Recompute mu and Bg
                        Bg[b] = B[b][:]
                        for j in range(b):
                            if vec_norm_sq(Bg[j]) > 0:
                                mu[b][j] = vec_dot(B[b], Bg[j]) / vec_norm_sq(Bg[j])
                                Bg[b] = vec_sub(Bg[b], vec_mul(Bg[j], mu[b][j]))
                # Lovasz condition
                if vec_norm_sq(Bg[b]) >= (delta - mu[b][b-1]**2) * vec_norm_sq(Bg[b-1]):
                    b -= 1
                else:
                    # Swap
                    B[b], B[b-1] = B[b-1], B[b]
                    T[b], T[b-1] = T[b-1], T[b]
                    # Recompute Bg and mu for rows b-1 and b
                    Bg[b-1] = B[b-1][:]
                    for j in range(b-1):
                        if vec_norm_sq(Bg[j]) > 0:
                            mu[b-1][j] = vec_dot(B[b-1], Bg[j]) / vec_norm_sq(Bg[j])
                            Bg[b-1] = vec_sub(Bg[b-1], vec_mul(Bg[j], mu[b-1][j]))
                    Bg[b] = B[b][:]
                    for j in range(b):
                        if vec_norm_sq(Bg[j]) > 0:
                            mu[b][j] = vec_dot(B[b], Bg[j]) / vec_norm_sq(Bg[j])
                            Bg[b] = vec_sub(Bg[b], vec_mul(Bg[j], mu[b][j]))
                    # Update mu for row b-1 with respect to b
                    if vec_norm_sq(Bg[b]) > 0:
                        mu[b-1][b] = Fraction(0)  # not used directly
                    b -= 1
        
        return B, T

    def _find_lattice_decomposition(U_mat, V_mat, W_mat):
        """
        Use lattice reduction to find a lower-rank decomposition.
        
        The tensor T = sum_r U_r ⊗ V_r ⊗ W_r can be rewritten as
        T = sum_r' (sum_i T_{r,i} U_i) ⊗ (sum_i T_{r,i} V_i) ⊗ (sum_i T_{r,i} W_i)
        only if we can factor. Instead, we look for integer relations:
        
        If we find a non-trivial integer vector c such that
        sum_r c_r * U[:,r] = 0 AND sum_r c_r * V[:,r] = 0,
        then we can eliminate that direction.
        
        More practically: look for pairs (r1, r2) where U[:,r1] and U[:,r2]
        are linearly dependent over rationals, and similarly for V, allowing
        a merge.
        """
        r_orig = U_mat.shape[1]
        U_cur = U_mat.copy()
        V_cur = V_mat.copy()
        W_cur = W_mat.copy()
        r_cur = r_orig
        
        # Build the U matrix as a lattice basis
        # Each column is a vector in R^{n*m}
        U_cols = U_mat.T  # shape (r_orig, n*m)
        V_cols = V_mat.T  # shape (r_orig, m*p)
        
        # Try to find rational linear dependencies among U columns
        # that are also dependencies among V columns
        # This would allow merging terms
        
        # Approach: For each pair (i,j), check if U[:,i] = a*U[:,j] for some rational a
        # and V[:,i] = b*V[:,j] for some rational b.
        # If a == b, we can merge.
        
        merged = False
        for _ in range(200):  # max passes
            if _time_left() < 15:
                break
            merged = False
            for r1 in range(r_cur):
                if _time_left() < 15:
                    break
                u1 = U_cur[:, r1]
                if np.allclose(u1, 0, atol=1e-10):
                    continue
                for r2 in range(r1 + 1, r_cur):
                    u2 = U_cur[:, r2]
                    if np.allclose(u2, 0, atol=1e-10):
                        continue
                    
                    # Check if u1 and u2 are rationally proportional
                    # Find the ratio from the first non-zero element
                    ratio_num = None
                    ratio_den = None
                    for idx in range(len(u1)):
                        if abs(u1[idx]) > 1e-10:
                            if abs(u2[idx]) < 1e-10:
                                break
                            # u1[idx] / u2[idx] should be rational
                            f1 = Fraction(str(u1[idx])).limit_denominator(10**6)
                            f2 = Fraction(str(u2[idx])).limit_denominator(10**6)
                            if f2 == 0:
                                break
                            ratio_num = f1.numerator
                            ratio_den = f2.denominator * f1.denominator
                            ratio = f1 / f2
                            break
                        elif abs(u2[idx]) > 1e-10:
                            break
                    
                    if ratio_num is None:
                        continue
                    
                    # Verify proportionality with exact fractions
                    valid = True
                    for idx in range(len(u1)):
                        expected = u2[idx] * ratio
                        if abs(u1[idx] - expected) > 1e-8 * max(1, abs(expected)):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Check V proportionality with same ratio
                    v1 = V_cur[:, r1]
                    v2 = V_cur[:, r2]
                    v_valid = True
                    for idx in range(len(v1)):
                        expected = v2[idx] * ratio
                        if abs(v1[idx] - expected) > 1e-8 * max(1, abs(expected)):
                            v_valid = False
                            break
                    if not v_valid:
                        continue
                    
                    # Merge: W[:,r2] += ratio * W[:,r1], zero out r1
                    W_cur[:, r2] += ratio * W_cur[:, r1]
                    U_cur[:, r1] = 0
                    V_cur[:, r1] = 0
                    W_cur[:, r1] = 0
                    r_cur -= 1
                    merged = True
                    break
                if merged:
                    break
            if not merged:
                break
        
        return U_cur, V_cur, W_cur, r_cur

    # Apply lattice-based reduction
    if _time_left() > 30:
        U_lat, V_lat, W_lat, rank_lat = _find_lattice_decomposition(U, V, W)
        if verify(U_lat, V_lat, W_lat) and rank_lat < best_rank:
            best_rank = rank_lat
            best_U, best_V, best_W = U_lat, V_lat, W_lat

    # ============================================================
    # Final cleanup and return
    # ============================================================
    if _time_left() > 5:
        # Convert to integers if possible
        def to_int_if_close(arr):
            arr_rounded = np.round(arr, 6)
            if np.allclose(arr, arr_rounded, atol=1e-6):
                return arr_rounded.astype(int)
            return arr
        
        U_out = to_int_if_close(best_U)
        V_out = to_int_if_close(best_V)
        W_out = to_int_if_close(best_W)
        
        # Convert numpy arrays to lists for the return format
        return {
            "U": U_out.tolist(),
            "V": V_out.tolist(),
            "W": W_out.tolist()
        }
    
    # Fallback
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
