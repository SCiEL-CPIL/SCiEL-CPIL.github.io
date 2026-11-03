import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq
from itertools import product
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
DEADLINE_SECONDS = 135.0  # safety margin before 150s kill

def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1
    All other entries are 0.
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _is_exact_solution(U, V, W, T, tol=1e-10):
    """Check if the decomposition is exact within tolerance."""
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    diff = reconstructed - T
    return np.max(np.abs(diff)) < tol

def _tensor_error(U, V, W, T):
    """Compute the Frobenius norm squared of the error."""
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.sum((reconstructed - T) ** 2)

def _schoolbook_solution(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _berlekamp_massey(sequence):
    """
    Berlekamp-Massey algorithm to find the minimal linear recurrence.
    Returns the recurrence coefficients [c1, c2, ..., ck] such that
    s[n] = c1*s[n-1] + c2*s[n-2] + ... + ck*s[n-k]
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Initialize
    C = [Fraction(1)]  # C(x) = 1
    B = [Fraction(1)]  # B(x) = 1
    L = 0  # current length
    m = 1  # number of iterations since last update
    b = Fraction(1)  # last discrepancy
    
    for N in range(n):
        # Compute discrepancy
        d = Fraction(0)
        for i in range(L + 1):
            if i <= N:
                d += C[i] * sequence[N - i]
        
        if d == 0:
            m += 1
        else:
            T = C[:]
            # C(x) = C(x) - (d/b) * x^m * B(x)
            factor = d / b
            for i in range(len(B)):
                if i + m < len(C):
                    C[i + m] -= factor * B[i]
                else:
                    C.append(0)
                    while len(C) <= i + m:
                        C.append(0)
                    C[i + m] -= factor * B[i]
            
            if 2 * L <= N:
                L = N + 1 - L
                B = T
                b = d
                m = 1
            else:
                m += 1
    
    # The recurrence is: s[n] + C[1]*s[n-1] + ... + C[L]*s[n-L] = 0
    # So s[n] = -C[1]*s[n-1] - ... - C[L]*s[n-L]
    coeffs = [-C[i] for i in range(1, L + 1)] if L > 0 else []
    return coeffs

def _find_rational_structure(T, n, m, p, max_rank=45):
    """
    Use Berlekamp-Massey on various slices of the tensor to find
    linear recurrences that suggest a lower-rank structure.
    Returns candidate rank and structure hints.
    """
    A, B, C = n*m, m*p, p*n
    
    # Extract various 1D sequences from the tensor
    # For each fixed pair of indices, extract the third dimension
    sequences = []
    
    # Sequences along each dimension
    for a in range(A):
        for b in range(B):
            seq = [Fraction(int(T[a, b, c]), 1) for c in range(C)]
            if any(x != 0 for x in seq):
                sequences.append(seq)
    
    for a in range(A):
        for c in range(C):
            seq = [Fraction(int(T[a, b, c]), 1) for b in range(B)]
            if any(x != 0 for x in seq):
                sequences.append(seq)
    
    for b in range(B):
        for c in range(C):
            seq = [Fraction(int(T[a, b, c]), 1) for a in range(A)]
            if any(x != 0 for x in seq):
                sequences.append(seq)
    
    # Find minimal recurrence lengths
    min_lengths = []
    for seq in sequences[:200]:  # Limit for performance
        coeffs = _berlekamp_massey(seq)
        min_lengths.append(len(coeffs))
    
    if min_lengths:
        # The minimal rank is related to the maximum minimal recurrence length
        max_rec_len = max(min_lengths) if min_lengths else 1
        # The rank should be at least max_rec_len
        return max_rec_len
    return 1

def _structured_rational_init(n, m, p, rank, seed):
    """
    Create structured rational initializations using 0, ±1, ±1/2 values.
    """
    rng = random.Random(seed)
    values = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)]
    
    U = [[Fraction(0) for _ in range(rank)] for _ in range(n*m)]
    V = [[Fraction(0) for _ in range(rank)] for _ in range(m*p)]
    W = [[Fraction(0) for _ in range(rank)] for _ in range(p*n)]
    
    strategy = seed % 5
    
    if strategy == 0:
        # Pure random from rational values
        for a in range(n*m):
            for r in range(rank):
                U[a][r] = rng.choice(values)
        for b in range(m*p):
            for r in range(rank):
                V[b][r] = rng.choice(values)
        for c in range(p*n):
            for r in range(rank):
                W[c][r] = rng.choice(values)
    elif strategy == 1:
        # Structured: based on matrix multiplication pattern
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    # Assign to the corresponding positions
                    U[i*m + j][col] = Fraction(1)
                    V[j*p + k][col] = Fraction(1)
                    W[k*n + i][col] = Fraction(1)
                    col += 1
        # Add some random structure to remaining columns
        for r in range(col, rank):
            for a in range(n*m):
                if rng.random() < 0.3:
                    U[a][r] = rng.choice([Fraction(1, 2), Fraction(-1, 2)])
            for b in range(m*p):
                if rng.random() < 0.3:
                    V[b][r] = rng.choice([Fraction(1, 2), Fraction(-1, 2)])
            for c in range(p*n):
                if rng.random() < 0.3:
                    W[c][r] = rng.choice([Fraction(1, 2), Fraction(-1, 2)])
    elif strategy == 2:
        # Sparse random
        for r in range(rank):
            nz_u = rng.sample(range(n*m), min(3, n*m))
            nz_v = rng.sample(range(m*p), min(5, m*p))
            nz_w = rng.sample(range(p*n), min(5, p*n))
            for a in nz_u:
                U[a][r] = rng.choice([Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)])
            for b in nz_v:
                V[b][r] = rng.choice([Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)])
            for c in nz_w:
                W[c][r] = rng.choice([Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)])
    elif strategy == 3:
        # Block-structured: group indices
        for r in range(rank):
            # Each factor column corresponds to a block
            i_block = r // 5
            j_block = (r % 5) // 2
            k_block = r % 2
            
            for i in range(n):
                for j in range(m):
                    if i == i_block and j == j_block:
                        U[i*m + j][r] = Fraction(1)
            for j in range(m):
                for k in range(p):
                    if j == j_block and k == k_block:
                        V[j*p + k][r] = Fraction(1)
            for k in range(p):
                for i in range(n):
                    if k == k_block and i == i_block:
                        W[k*n + i][r] = Fraction(1)
    else:
        # Hybrid with symmetry
        for r in range(rank):
            # Use symmetric patterns
            base = r % 10
            for a in range(n*m):
                if (a // m) % 2 == base % 2:
                    U[a][r] = Fraction(1) if (a % m) % 2 == 0 else Fraction(-1)
            for b in range(m*p):
                if (b // p) % 2 == (base // 2) % 2:
                    V[b][r] = Fraction(1) if (b % p) % 2 == 0 else Fraction(-1)
            for c in range(p*n):
                if (c // n) % 2 == (base // 4) % 2:
                    W[c][r] = Fraction(1) if (c % n) % 2 == 0 else Fraction(-1)
    
    return U, V, W

def _rational_tensor_error(U, V, W, T, n, m, p):
    """
    Compute the exact rational tensor error.
    T[i*m+j, j*p+k, k*n+i] = 1, all others 0.
    """
    A, B, C = n*m, m*p, p*n
    rank = len(U[0])
    
    error = Fraction(0)
    for a in range(A):
        for b in range(B):
            for c in range(C):
                # Compute sum_r U[a][r] * V[b][r] * W[c][r]
                val = Fraction(0)
                for r in range(rank):
                    val += U[a][r] * V[b][r] * W[c][r]
                
                # Expected value
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(c, n)
                
                if i == i2 and j == j2 and k == k2:
                    expected = Fraction(1)
                else:
                    expected = Fraction(0)
                
                diff = val - expected
                error += diff * diff
    
    return error

def _verify_rational_solution(U, V, W, T, n, m, p, tol=Fraction(1, 10**10)):
    """Verify exact solution using rational arithmetic."""
    A, B, C = n*m, m*p, p*n
    rank = len(U[0])
    
    for a in range(A):
        for b in range(B):
            for c in range(C):
                val = Fraction(0)
                for r in range(rank):
                    val += U[a][r] * V[b][r] * W[c][r]
                
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(c, n)
                
                if i == i2 and j == j2 and k == k2:
                    expected = Fraction(1)
                else:
                    expected = Fraction(0)
                
                if abs(val - expected) > tol:
                    return False
    
    return True

def _als_step(U, V, W, T, n, m, p):
    """
    Perform one ALS iteration with improved numerical stability.
    """
    A, B, C = n*m, m*p, p*n
    rank = U.shape[1]
    
    # Update U
    VW = np.einsum('br,cr->bcr', V, W)  # (B, C, rank)
    VW_flat = VW.transpose(2, 0, 1).reshape(rank, B*C)  # (rank, B*C)
    T_flat = T.reshape(A, B*C)  # (A, B*C)
    
    if rank > 0:
        U_new, _, _, _ = lstsq(VW_flat.T, T_flat.T, cond=None)
        U = U_new.T  # (A, rank)
    
    # Update V
    UW = np.einsum('ar,cr->acr', U, W)  # (A, C, rank)
    UW_flat = UW.transpose(2, 0, 1).reshape(rank, A*C)  # (rank, A*C)
    T_flat_V = T.transpose(1, 0, 2).reshape(B, A*C)  # (B, A*C)
    
    if rank > 0:
        V_new, _, _, _ = lstsq(UW_flat.T, T_flat_V.T, cond=None)
        V = V_new.T  # (B, rank)
    
    # Update W
    UV = np.einsum('ar,br->abr', U, V)  # (A, B, rank)
    UV_flat = UV.transpose(2, 0, 1).reshape(rank, A*B)  # (rank, A*B)
    T_flat_W = T.transpose(2, 0, 1).reshape(C, A*B)  # (C, A*B)
    
    if rank > 0:
        W_new, _, _, _ = lstsq(UV_flat.T, T_flat_W.T, cond=None)
        W = W_new.T  # (C, rank)
    
    return U, V, W

def _try_rank_rational(n, m, p, rank, T, seed=0, time_limit=10.0):
    """
    Attempt to find a rational decomposition of the given rank.
    Returns (U, V, W, error) if successful, or None.
    """
    start_time = time.time()
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    # Try multiple rational initializations
    for init_seed in range(6):
        if time.time() - start_time > time_limit:
            break
        
        current_seed = seed * 100 + init_seed
        U_rat, V_rat, W_rat = _structured_rational_init(n, m, p, rank, current_seed)
        
        # Convert to numpy for error computation
        U_np = np.array([[float(x) for x in row] for row in U_rat], dtype=np.float64)
        V_np = np.array([[float(x) for x in row] for row in V_rat], dtype=np.float64)
        W_np = np.array([[float(x) for x in row] for row in W_rat], dtype=np.float64)
        
        err = _tensor_error(U_np, V_np, W_np, T)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
        
        # Run ALS iterations to polish
        for iteration in range(200):
            if time.time() - start_time > time_limit:
                break
            
            U_np, V_np, W_np = _als_step(U_np, V_np, W_np, T, n, m, p)
            
            err = _tensor_error(U_np, V_np, W_np, T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
            
            if best_err < 1e-14:
                break
    
    if best_U is None:
        return None, None, None, float('inf')
    
    return best_U, best_V, best_W, best_err

def _polish_with_rational(U, V, W, T, n, m, p, rank, time_limit=5.0):
    """
    Polish a near-converged solution and try to snap to rational values.
    """
    start_time = time.time()
    best_err = _tensor_error(U, V, W, T)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for _ in range(500):
        if time.time() - start_time > time_limit:
            break
        
        U, V, W = _als_step(U, V, W, T, n, m, p)
        err = _tensor_error(U, V, W, T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-14:
            break
    
    # Try to snap to rational values (0, ±1, ±1/2)
    rational_values = [0.0, 1.0, -1.0, 0.5, -0.5]
    for mat in [best_U, best_V, best_W]:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                # Find closest rational value
                best_val = min(rational_values, key=lambda v: abs(mat[i, j] - v))
                if abs(mat[i, j] - best_val) < 0.1:
                    mat[i, j] = best_val
    
    final_err = _tensor_error(best_U, best_V, best_W, T)
    if final_err < best_err:
        best_err = final_err
    
    return best_U, best_V, best_W, best_err

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook_solution(n, m, p)
    T = _build_tensor(n, m, p)
    
    found_rank = n * m * p
    found_U, found_V, found_W = U, V, W
    
    # Use Berlekamp-Massey to estimate minimal rank
    estimated_min_rank = _find_rational_structure(T, n, m, p)
    print(f"Estimated minimal rank from BM: {estimated_min_rank}")
    
    # Try ranks in a smart order
    # Start from estimated minimum and go up, also try known good ranks
    ranks_to_try = []
    for r in range(max(estimated_min_rank, 20), 50):
        ranks_to_try.append(r)
    
    # Also include some specific ranks
    for r in [25, 26, 27, 28, 30, 35, 40, 45]:
        if r not in ranks_to_try:
            ranks_to_try.append(r)
    
    ranks_to_try = sorted(set(ranks_to_try))
    
    for rank in ranks_to_try:
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        
        if rank >= found_rank:
            continue
        
        # Try multiple seeds for this rank
        success = False
        max_seeds = 4 if rank <= 35 else 3
        for seed in range(max_seeds):
            if time.time() - start_time > DEADLINE_SECONDS - 3:
                break
            
            remaining_time = DEADLINE_SECONDS - 5 - (time.time() - start_time)
            time_limit = min(12.0, max(4.0, remaining_time / (max_seeds - seed + 1)))
            
            try:
                U_new, V_new, W_new, err = _try_rank_rational(n, m, p, rank, T, seed=seed, time_limit=time_limit)
                if U_new is None:
                    continue
                
                if err < 1e-10:
                    # Polish and try to snap to rationals
                    U_pol, V_pol, W_pol, err_pol = _polish_with_rational(
                        U_new, V_new, W_new, T, n, m, p, rank, time_limit=4.0
                    )
                    
                    if err_pol < 1e-10 and _is_exact_solution(U_pol, V_pol, W_pol, T, tol=1e-10):
                        found_rank = rank
                        found_U, found_V, found_W = U_pol, V_pol, W_pol
                        success = True
                        break
            except Exception:
                continue
        
        if success:
            continue
    
    # Return the best found
    U_list = found_U.tolist()
    V_list = found_V.tolist()
    W_list = found_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
