import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [3, 3, 5]
DEADLINE_MARGIN = 15.0  # seconds to spare

def _build_schoolbook(n, m, p):
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _is_half_integer(x, tol=1e-9):
    """Check if x is an integer or half-integer."""
    r = round(2 * x)
    return abs(2 * x - r) < tol


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using tensor reconstruction via einsum."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    R = U.shape[1]
    # Check half-integer constraint
    for arr in [U, V, W]:
        if not np.all(np.abs(2 * arr - np.round(2 * arr)) < tol):
            return False
    # Reconstruct full tensor T[u,v,w] = sum_c U[u,c] V[v,c] W[w,c]
    T = np.einsum('ur,vr,wr->uvw', U, V, W)
    # Check 45 positive entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if abs(T[u, v, w] - 1.0) > tol:
                    return False
    # Check all negative entries
    for u in range(n * m):
        i_u = u // m
        j_u = u % m
        for v in range(m * p):
            j_v = v // p
            k_v = v % p
            for w in range(p * n):
                k_w = w // n
                i_w = w % n
                if i_u == i_w and j_u == j_v and k_v == k_w:
                    continue
                if abs(T[u, v, w]) > tol:
                    return False
    return True


def _try_direct_sum_decomposition(n, m, p):
    """
    Try to decompose M_{n,m,p} by splitting the p-dimension.
    For 3x3x5: split into 3x3x3 (rank 9) + 3x3x2 (try to find low rank).
    
    The key idea: if we can find a rank-r decomposition for M_{3,3,2},
    then M_{3,3,5} has rank <= 9 + r.
    
    We'll try to construct M_{3,3,2} decompositions using structured
    approaches with half-integer coefficients.
    """
    # First, build the 3x3x3 schoolbook (rank 9)
    n1, m1, p1 = 3, 3, 3
    R1 = n1 * m1 * p1  # 9
    U1 = np.zeros((n1 * m1, R1))
    V1 = np.zeros((m1 * p1, R1))
    W1 = np.zeros((p1 * n1, R1))
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j, col] = 1.0
                V1[j * p1 + k, col] = 1.0
                W1[k * n1 + i, col] = 1.0
                col += 1
    
    # Now try to build M_{3,3,2} with various ranks
    n2, m2, p2 = 3, 3, 2
    R2_school = n2 * m2 * p2  # 18
    
    # Try different strategies for M_{3,3,2}
    best_2 = None
    best_rank_2 = R2_school
    
    # Strategy 1: Schoolbook for 3x3x2 (rank 18)
    U2_sb = np.zeros((n2 * m2, R2_school))
    V2_sb = np.zeros((m2 * p2, R2_school))
    W2_sb = np.zeros((p2 * n2, R2_school))
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2_sb[i * m2 + j, col] = 1.0
                V2_sb[j * p2 + k, col] = 1.0
                W2_sb[k * n2 + i, col] = 1.0
                col += 1
    best_2 = (U2_sb, V2_sb, W2_sb)
    best_rank_2 = R2_school
    
    # Strategy 2: Try to reduce rank using structured linear combinations
    # For M_{3,3,2}, we can try to merge terms that share structure.
    # The 18 terms are (i,j,k) for i,j in {0,1,2}, k in {0,1}
    # For fixed (i,j), we have 2 terms (k=0,1). We can try to combine them.
    
    # Try rank 15: merge some pairs
    # For each (i,j) pair, we have terms for k=0 and k=1.
    # If we use the same U and V entries for both k values with different W,
    # we can't simply merge. But we can try linear combinations.
    
    # Let's try a more systematic approach: use integer programming-like search
    # over small candidate decompositions.
    
    # Actually, let's try a known result: M_{3,3,2} might have rank 15 or 16.
    # Let me try to construct rank 15.
    
    # For rank 15, we need to eliminate 3 terms from schoolbook.
    # This means finding 3 linear dependencies.
    
    # Alternative: try rank 12, 14, etc. with random search restricted to
    # half-integer coefficients.
    
    rng = np.random.default_rng(123)
    start = time.time()
    
    for target_rank in range(17, 8, -1):  # Try 17 down to 9
        if time.time() - start > 60:
            break
        found = _try_rank_structured(n2, m2, p2, target_rank, rng, time_budget=30)
        if found is not None:
            U2, V2, W2 = found
            best_2 = (U2, V2, W2)
            best_rank_2 = target_rank
            break
    
    if best_2 is None:
        return None
    
    U2, V2, W2 = best_2
    
    # Combine: the full 3x3x5 tensor
    # The U matrix has rows indexed by (i,j) for i,j in 0..2
    # The V matrix has rows indexed by (j,k) for j in 0..2, k in 0..4
    # The W matrix has rows indexed by (k,i) for k in 0..4, i in 0..2
    
    # For the first 3 columns of B (k=0,1,2), use the 3x3x3 decomposition
    # For the last 2 columns of B (k=3,4), use the 3x3x2 decomposition
    
    # U: shape (9, R_total)
    # V: shape (15, R_total) 
    # W: shape (15, R_total)
    
    R_total = R1 + best_rank_2
    U_total = np.zeros((9, R_total))
    V_total = np.zeros((15, R_total))
    W_total = np.zeros((15, R_total))
    
    # Fill in the 3x3x3 part (columns 0..8)
    U_total[:, :R1] = U1
    # V1 has shape (9, 9) for (j,k) with j in 0..2, k in 0..2
    # We need to place this in V_total where rows are (j,k) with j in 0..2, k in 0..4
    # V_total row index = j * 5 + k
    for j in range(3):
        for k in range(3):
            V_total[j * 5 + k, :R1] = V1[j * 3 + k, :]
    # W1 has shape (9, 9) for (k,i) with k in 0..2, i in 0..2
    # W_total row index = k * 3 + i
    for k in range(3):
        for i in range(3):
            W_total[k * 3 + i, :R1] = W1[k * 3 + i, :]
    
    # Fill in the 3x3x2 part (columns R1..R1+best_rank_2-1)
    U_total[:, R1:] = U2
    # V2 has shape (9, best_rank_2) for (j,k) with j in 0..2, k in 0..1
    # We need to place this in V_total where rows are (j,k) with j in 0..2, k in 3..4
    # So V_total row index = j * 5 + (k+3)
    for j in range(3):
        for k in range(2):
            V_total[j * 5 + (k + 3), R1:] = V2[j * 2 + k, :]
    # W2 has shape (6, best_rank_2) for (k,i) with k in 0..1, i in 0..2
    # We need to place this in W_total where rows are (k,i) with k in 3..4, i in 0..2
    # So W_total row index = (k+3) * 3 + i
    for k in range(2):
        for i in range(3):
            W_total[(k + 3) * 3 + i, R1:] = W2[k * 3 + i, :]
    
    # Verify the full decomposition
    if _verify_fast(U_total, V_total, W_total, n, m, p):
        return U_total, V_total, W_total, R_total
    
    return None


def _try_rank_structured(n, m, p, target_rank, rng, time_budget=30):
    """
    Try to find a rank-target_rank decomposition for M_{n,m,p}
    using structured search with half-integer coefficients.
    """
    start = time.time()
    R = target_rank
    total_terms = n * m * p
    
    # List all (i,j,k) triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    if R >= total_terms:
        # Just use schoolbook
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        for c, (i, j, k) in enumerate(triples):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W
    
    # Specific case for 3x3x2 with rank 15
    if n == 3 and m == 3 and p == 2 and R == 15:
        # Construct 3x3x2 decomposition with rank 15
        # Use 9 terms for k=0 (schoolbook) and 6 terms for k=1 (reduced)
        # Actually, let's try to construct a valid rank 15 decomposition.
        # We know schoolbook is 18.
        # We can try to merge 3 pairs.
        
        U = np.zeros((9, 15))
        V = np.zeros((6, 15))
        W = np.zeros((6, 15))
        
        # First 9 terms: standard for k=0
        for i in range(3):
            for j in range(3):
                c = i * 3 + j
                U[i * 3 + j, c] = 1.0
                V[j * 2 + 0, c] = 1.0
                W[0 * 3 + i, c] = 1.0
        
        # Next 6 terms: for k=1
        # We need to compute sum_j A_ij B_j1
        # Let's try to use a rank-3 decomposition for each row i?
        # Rank 3 for 1x3x1 is schoolbook.
        # So 3 rows * 3 terms = 9 terms. Total 9+9=18.
        # To get 15, we need to save 3 terms.
        # This implies using a rank-2 decomposition for some rows? Impossible.
        # Or sharing terms between rows?
        
        # Let's try a different split: 5 terms for k=0, 10 for k=1? No.
        
        # Fallback to random search if specific construction fails
        pass

    # General random search
    coeff_set = [-1.0, -0.5, 0.0, 0.5, 1.0]
    max_attempts = 200
    for attempt in range(max_attempts):
        if time.time() - start > time_budget:
            break
        
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Random assignment of triples to columns
        assignment = rng.integers(R, size=total_terms)
        
        for t, (i, j, k) in enumerate(triples):
            c = assignment[t]
            coeff = coeff_set[rng.integers(len(coeff_set))]
            if coeff == 0.0:
                coeff = 0.5
            U[i * m + j, c] += coeff
            V[j * p + k, c] += coeff
            W[k * n + i, c] += coeff
        
        if _verify_fast(U, V, W, n, m, p):
            return U, V, W
            
    return None


def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = 150.0 - DEADLINE_MARGIN

    # Schoolbook baseline (rank 45)
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try direct sum decomposition: 3x3x3 + 3x3x2
    if time.time() < deadline:
        result = _try_direct_sum_decomposition(n, m, p)
        if result is not None:
            U_ds, V_ds, W_ds, R_ds = result
            if R_ds < best_rank:
                best_rank = R_ds
                best_result = {"U": U_ds.tolist(), "V": V_ds.tolist(), "W": W_ds.tolist()}

    # Additional search: try to improve further if time permits
    if time.time() < deadline - 30:
        # Try other structured approaches
        rng = np.random.default_rng(42)
        for R in range(best_rank - 1, max(14, best_rank - 10), -1):
            if time.time() > deadline - 5:
                break
            found = _try_rank_structured(n, m, p, R, rng, time_budget=10)
            if found is not None:
                U_f, V_f, W_f = found
                best_rank = R
                best_result = {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
                break

    return best_result
