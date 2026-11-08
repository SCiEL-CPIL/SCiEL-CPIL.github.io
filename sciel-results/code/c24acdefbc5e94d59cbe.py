"""Matrix multiplication 2x4 by 4x7 rank-minimal decomposition search."""
import numpy as np
import itertools
from fractions import Fraction
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build tensor from factors
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for r in range(rank):
        T += np.outer(np.outer(U_arr[:, r], V_arr[:, r]), W_arr[:, r])
    # Compare with true tensor
    T_true = _build_tensor(n, m, p).astype(np.float64)
    return np.allclose(T, T_true, atol=tol)

def _check_half_integer(mat):
    """Check if all entries are integer or half-integer (i.e., 2*x is integer)."""
    for row in mat:
        for val in row:
            if isinstance(val, (int, np.integer)):
                continue
            v = float(val)
            if abs(2 * v - round(2 * v)) > 1e-9:
                return False
    return True

def _to_rational_list(mat):
    """Convert a matrix of floats (that are half-integers) to list of Fractions."""
    result = []
    for row in mat:
        r = []
        for val in row:
            v = float(val)
            # Convert to fraction: multiply by 2, round, divide by 2
            r.append(Fraction(round(2 * v), 2))
        result.append(r)
    return result

def _find_rank_14_2x2x7(time_limit=5.0):
    """
    Attempt to find a rank 14 decomposition for 2x2x7 matrix multiplication.
    Returns (U, V, W) as numpy arrays or None.
    """
    n, m, p = 2, 2, 7
    T = _build_tensor(n, m, p)
    support = np.argwhere(T != 0)
    
    start_time = time.time()
    # Try rank 14
    r = 14
    
    for attempt in range(500):
        if time.time() - start_time > time_limit:
            break
        
        # Random W
        W_rand = np.random.choice([-1, 0, 1], size=(p * n, r))
        if np.linalg.matrix_rank(W_rand) < r:
            continue
            
        try:
            WtW = W_rand.T @ W_rand
            WtW_inv = np.linalg.inv(WtW)
        except:
            continue
            
        # Compute X_vec
        TW = np.einsum('abc,cr->abr', T.astype(np.float64), W_rand)
        X_vec = np.einsum('abr,rs->abs', TW, WtW_inv)
        
        ok_all_rank1 = True
        U_sol = np.zeros((n * m, r))
        V_sol = np.zeros((m * p, r))
        
        for r_idx in range(r):
            C_r = X_vec[:, :, r_idx]
            svals = np.linalg.svd(C_r, compute_uv=False)
            if len(svals) > 1 and svals[1] > 1e-8 * max(svals[0], 1e-12):
                ok_all_rank1 = False
                break
            if svals[0] < 1e-12:
                U_sol[:, r_idx] = 0
                V_sol[:, r_idx] = 0
            else:
                U_sv, S_sv, Vt_sv = np.linalg.svd(C_r)
                U_sol[:, r_idx] = U_sv[:, 0] * np.sqrt(S_sv[0])
                V_sol[:, r_idx] = Vt_sv[0, :] * np.sqrt(S_sv[0])
                
        if not ok_all_rank1:
            continue
            
        # Verify support
        T_recon_support = np.zeros(len(support))
        for idx, (a, b, c) in enumerate(support):
            val = 0.0
            for r_idx in range(r):
                val += U_sol[a, r_idx] * V_sol[b, r_idx] * W_rand[c, r_idx]
            T_recon_support[idx] = val
            
        if not np.allclose(T_recon_support, 1.0, atol=1e-6):
            continue
            
        # Verify zeros (sample)
        # For 2x2x7, total entries 4*14*14 = 784. Support 28. Zeros 756.
        # Check a few
        ok_zero = True
        for _ in range(20):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            if T[a,b,c] != 0:
                continue
            val = 0.0
            for r_idx in range(r):
                val += U_sol[a, r_idx] * V_sol[b, r_idx] * W_rand[c, r_idx]
            if abs(val) > 1e-6:
                ok_zero = False
                break
        if not ok_zero:
            continue
            
        # Check half-integer
        if all_half_int_arr(U_sol) and all_half_int_arr(V_sol) and all_half_int_arr(W_rand):
            return U_sol, V_sol, W_rand
        else:
            # Try rounding
            U_round = np.round(U_sol * 2) / 2
            V_round = np.round(V_sol * 2) / 2
            W_round = np.round(W_rand * 2) / 2
            # Re-verify rounded
            T_recon = np.zeros((n*m, m*p, p*n))
            for r_idx in range(r):
                T_recon += np.outer(np.outer(U_round[:, r_idx], V_round[:, r_idx]), W_round[:, r_idx])
            if np.allclose(T_recon, T.astype(np.float64), atol=1e-6):
                return U_round, V_round, W_round
                
    return None

def _construct_rank_28_from_14(U1, V1, W1, U2, V2, W2):
    """
    Construct 2x4x7 decomposition from two 2x2x7 decompositions.
    A = [A_L, A_R], B = [B_T; B_B].
    AB = A_L B_T + A_R B_B.
    """
    n, m, p = 2, 4, 7
    rank = 28
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # First part: A_L (rows 0,1 of A) * B_T (cols 0,1 of B? No, rows 0,1 of B)
    # A_L is A[0:2, 0:2] (since m=4, split into 2+2)
    # B_T is B[0:2, :] (rows 0,1 of B)
    # U1 corresponds to A_L part (size 4)
    # V1 corresponds to B_T part (size 14)
    # W1 corresponds to C part (size 14)
    
    # Map indices:
    # A index a = i*m + j. A_L uses j=0,1. So a in {0,1,2,3} (i=0,1; j=0,1).
    # B index b = j*p + k. B_T uses j=0,1. So b in {0..13} (j=0,1; k=0..6).
    # C index c = k*n + i. C uses all k, i. c in {0..13}.
    
    # U1 is 4x14. U is 8x28.
    # For r in 0..13:
    #   U[0:4, r] = U1[:, r]
    #   V[0:14, r] = V1[:, r]
    #   W[:, r] = W1[:, r]
    
    for r in range(14):
        U[0:4, r] = U1[:, r]
        V[0:14, r] = V1[:, r]
        W[:, r] = W1[:, r]
        
    # Second part: A_R (rows 0,1 of A, cols 2,3) * B_B (rows 2,3 of B)
    # A_R uses j=2,3. So a in {4,5,6,7}.
    # B_B uses j=2,3. So b in {14..27}.
    # C is same.
    
    for r in range(14):
        U[4:8, 14+r] = U2[:, r]
        V[14:28, 14+r] = V2[:, r]
        W[:, 14+r] = W2[:, r]
        
    return U, V, W

def _try_structural_search(n, m, p, time_limit=100.0):
    """
    Try to find a rank-reduced decomposition using structural knowledge.
    """
    start_time = time.time()
    
    # Try the 2x2x7 split strategy
    if time.time() - start_time < time_limit * 0.5:
        U1, V1, W1 = _find_rank_14_2x2x7(time_limit=time_limit * 0.2)
        if U1 is not None:
            U2, V2, W2 = _find_rank_14_2x2x7(time_limit=time_limit * 0.2)
            if U2 is not None:
                U, V, W = _construct_rank_28_from_14(U1, V1, W1, U2, V2, W2)
                # Verify
                if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return U.tolist(), V.tolist(), W.tolist()
                
    # Fallback to original random search if time remains
    # (Omitted for brevity, as it was failing)
    return None

def all_half_int_arr(arr, tol=1e-6):
    """Check if all entries in array are integer or half-integer."""
    for val in arr.flat:
        v = float(val)
        if abs(2 * v - round(2 * v)) > tol:
            return False
    return True

def _to_list(arr):
    """Convert numpy array to list of lists with rounded half-integer values."""
    result = []
    for row in arr:
        r = []
        for val in row:
            v = float(val)
            # Round to nearest half-integer
            rv = round(2 * v) / 2.0
            r.append(rv)
        result.append(r)
    return result

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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
    return U, V, W

# EVOLVE-BLOCK-END

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Try to find a better decomposition
    try:
        remaining = deadline - time.time()
        if remaining > 5.0:
            result = _try_structural_search(n, m, p, time_limit=remaining)
            if result is not None:
                U, V, W = result
                # Verify
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    except:
        pass
    
    # Check if we're still within deadline
    if time.time() > deadline - 2.0:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
