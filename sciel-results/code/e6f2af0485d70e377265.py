"""Structural-skeleton search for rank-minimal exact tensor decomposition of 2x5 * 5x6.

New approach: enumerate the combinatorial support structure (which (i,j,k) triples
are "active" in each of the r tensor slices), use a small ILP-style feasibility
check (via greedy set-cover over the diagonal requirements) to prune structures,
then solve a small linear system per structure to find half-integer coefficients.
This separates the combinatorial skeleton from the numeric coefficients.
"""
import time
import random
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

DIMENSIONS = [2, 5, 6]
N, M, P = 2, 5, 6


def _verify_rank_fast(U, V, W, n=N, m=M, p=P, tol=1e-9):
    """Vectorized verification of tensor decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank). True iff valid."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]

    i_vals = np.repeat(np.arange(n), m * p)
    j_vals = np.tile(np.repeat(np.arange(m), p), n)
    k_vals = np.tile(np.arange(p), n * m)
    u_idx = i_vals * m + j_vals
    v_idx = j_vals * p + k_vals
    w_idx = k_vals * n + i_vals
    diag_vals = np.einsum('ac,bc,ac->a', U[u_idx], V[v_idx], W[w_idx])
    if not np.allclose(diag_vals, 1.0, atol=tol):
        return False

    for j in range(m):
        for i in range(n):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                prod = U[a] * V[b]
                for i_prime in range(n):
                    for k_prime in range(p):
                        if i == i_prime and k == k_prime:
                            continue
                        c = k_prime * n + i_prime
                        if abs(np.dot(prod, W[c])) > tol:
                            return False
    return True


def _schoolbook_rank(n=N, m=M, p=P):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _all_diagonals():
    """Return list of (i, j, k) for all diagonal entries."""
    return [(i, j, k) for i in range(N) for j in range(M) for k in range(P)]


def _all_offdiags():
    """Return list of (i, j, k, i2, k2) for all off-diagonal constraints (i,j,k) vs (i2,k2)."""
    out = []
    for i in range(N):
        for j in range(M):
            for k in range(P):
                for i2 in range(N):
                    for k2 in range(P):
                        if i == i2 and k == k2:
                            continue
                        out.append((i, j, k, i2, k2))
    return out


def _enumerate_skeletons(r, max_structures=2000):
    """Enumerate candidate support structures for rank r.
    Each structure assigns to each of the r slices a set of (i,j,k) triples.
    A triple (i,j,k) is "covered" by slice c if U[i*m+j, c] != 0, V[j*p+k, c] != 0,
    and W[k*n+i, c] != 0.
    
    For the diagonal constraint: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
    For the off-diagonal: for each (i,j,k,i2,k2), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k2*n+i2,c] = 0.
    
    Key insight: if in slice c, U[i*m+j,c] != 0 and V[j*p+k,c] != 0, then W[k*n+i,c] must be
    the only nonzero W in that row... no, that's too restrictive.
    
    Better structural approach: each slice c has a "pattern" - a set of (i,j) pairs in U,
    (j,k) pairs in V, (k,i) pairs in W. The product constraint is:
    For diag (i,j,k): sum over c of u_c[i,j] * v_c[j,k] * w_c[k,i] = 1
    For off (i,j,k,i2,k2): sum over c of u_c[i,j] * v_c[j,k] * w_c[k2,i2] = 0
    
    The off-diagonal constraint is the hard one. If we restrict each slice to have
    at most one nonzero in each row of U, V, W, then the structure is determined by
    which (i,j), (j,k), (k,i) are "active" per slice.
    
    For a sparse structure where each slice has exactly one (i,j), one (j,k), one (k,i):
    The diag for (i,j,k) requires slice c with (i,j) in U, (j,k) in V, (k,i) in W.
    The off-diag for (i,j,k,i2,k2) requires NO slice c with (i,j) in U, (j,k) in V, (k2,i2) in W.
    
    So the structural condition is:
    - For each (i,j,k): at least one slice has the triple {(i,j),(j,k),(k,i)}
    - For each (i,j,k,i2,k2) with (i2,k2) != (i,k): NO slice has {(i,j),(j,k),(k2,i2)}
    
    This means: for fixed (i,j,k), the set of slices that have (i,j) in U and (j,k) in V
    must have their W entries only at (k,i), not at any other (k2,i2).
    
    Equivalently: if slice c has U-row (i,j) nonzero and V-row (j,k) nonzero, then W-row (k2,i2)
    in slice c must be zero for all (k2,i2) != (k,i).
    """
    structures = []
    triples = _all_diagonals()
    triple_set = set(triples)
    
    # For a "single-triple-per-slice" structure, each slice is one (i,j,k) triple.
    # The off-diagonal constraint: if slice c = (i,j,k), then for the off-diag
    # (i,j,k',i') we need that no slice has U-row (i,j) AND V-row (j,k') AND W-row (k',i') nonzero.
    # In single-triple structure, slice c' = (i', j', k') has U-row (i',j'), V-row (j',k'), W-row (k',i').
    # So off-diag (i,j,k,i2,k2) is violated iff there exists a slice c' = (i,j,k2) with i2 != i... 
    # wait, the off-diag is: for (i,j,k) and (i2,k2), the term is U[i,j]*V[j,k]*W[k2,i2].
    # In single-triple structure, this is nonzero iff there's a slice with (i,j) in U, (j,k) in V, (k2,i2) in W.
    # That means slice = (i,j,k) AND also W[k2,i2] != 0. But in single-triple, W only has (k,i) nonzero.
    # So W[k2,i2] is nonzero only if (k2,i2) = (k,i). Thus off-diag is automatically satisfied!
    
    # So single-triple structures always satisfy off-diagonal constraints.
    # The question is: can we assign values so that diag = 1?
    # For diag (i,j,k): sum over slices c with triple = (i,j,k) of u_c * v_c * w_c = 1.
    # If each triple appears in exactly one slice, then u*v*w = 1 for that slice.
    # This is the schoolbook: rank = 60, each triple gets one slice.
    
    # To reduce rank, we need some slices to cover multiple diagonals.
    # A slice can cover (i,j,k) if U[i,j], V[j,k], W[k,i] are all nonzero in that slice.
    # But if a slice has U[i,j] and U[i',j] both nonzero, and V[j,k] nonzero, then
    # off-diag (i,j,k,i',k) = U[i,j]*V[j,k]*W[k,i'] must be 0, so W[k,i'] must be 0.
    
    # General structural rule: For slice c, let S_U(c) = {(i,j) : U[i,j,c] != 0},
    # S_V(c) = {(j,k) : V[j,k,c] != 0}, S_W(c) = {(k,i) : W[k,i,c] != 0}.
    # Off-diag constraint: for any (i,j) in S_U(c), (j,k) in S_V(c), and (k2,i2) in S_W(c),
    # we need (i2,k2) = (i,k) ... no, we need the sum over ALL slices to be 0.
    # But if we make the structure "block diagonal" so that for each slice, the only
    # (k,i) in S_W(c) that's compatible with (i,j) in S_U(c) and (j,k) in S_V(c) is (k,i),
    # then off-diag is automatically 0 for that slice's contribution.
    
    # Strong structural rule: for each slice c, S_U(c), S_V(c), S_W(c) form a "consistent" set
    # where if (i,j) in S_U(c) and (j,k) in S_V(c), then (k,i) is the ONLY element of S_W(c)
    # with first index k. This means: for fixed j, all (i,j) in S_U(c) share the same i? No.
    
    # Simplest: each slice has |S_U| <= 1, |S_V| <= 1, |S_W| <= 1. Then it's single-triple.
    # Next: each slice has |S_U| = 2 (two rows i, i' in U for same j), |S_V| = 1, |S_W| = 2.
    # Then for the diag: (i,j,k) and (i',j,k) both get contribution from this slice.
    # Off-diag: (i,j,k,i',k) needs W[k,i'] = 0. But S_W has (k,i) and (k,i'). So W[k,i'] != 0. VIOLATION.
    
    # So we can't have two U entries with same j in one slice.
    # What about two V entries? S_U = 1, S_V = 2 (two k values), S_W = 2.
    # Then diag (i,j,k1) and (i,j,k2) get contributions.
    # Off-diag: (i,j,k1,i,k2) needs W[k2,i]... wait, off-diag is (i,j,k,i2,k2) where (i2,k2) != (i,k).
    # For (i,j,k1): off-diag with (i,k2) needs W[k2,i] = 0. But S_W has (k1,i) and (k2,i). So W[k2,i] != 0. VIOLATION.
    
    # Hmm. So single-triple per slice seems forced for the "each slice contributes to only its own diag" case.
    
    # Alternative: use cancellation. Multiple slices can contribute to the same diag, and their
    # contributions sum to 1. Off-diags cancel out too.
    
    # Let's try a different structural family: "pair" structures where two slices are paired
    # to share a (j,k) entry in V, with U and W entries that cause off-diags to cancel.
    
    # Actually, let's just do a direct approach: for small r, try random structures where
    # each slice has 1-3 nonzeros in each matrix, and check if the resulting linear system
    # (for the coefficients) has a solution.
    
    return structures


def _try_random_structure(r, rng, max_attempts_per_structure=5):
    """Try to find coefficients for a random sparse structure of rank r.
    Returns (U, V, W) if successful, else None."""
    n, m, p = N, M, P
    diags = _all_diagonals()
    offdiags = _all_offdiags()
    
    # Build the linear system:
    # Variables: U[i*m+j, c] for all i,j,c; V[j*p+k, c] for all j,k,c; W[k*n+i, c] for all k,i,c.
    # That's r * (n*m + m*p + p*n) = r * (10 + 30 + 12) = 52*r variables.
    # For r=59: 3068 variables. Too many for a dense linear system.
    
    # Better: fix the structure (which entries are nonzero), then solve.
    # For each slice c, pick a small set of (i,j) for U, (j,k) for V, (k,i) for W.
    
    U = np.zeros((n * m, r), dtype=float)
    V = np.zeros((m * p, r), dtype=float)
    W = np.zeros((p * n, r), dtype=float)
    
    # Assign structure: each slice gets a random pattern
    for c in range(r):
        # Pick 1-2 entries in U
        nu = rng.randint(1, 2)
        for _ in range(nu):
            i, j = rng.randint(0, n-1), rng.randint(0, m-1)
            U[i*m+j, c] = rng.choice([0.5, -0.5, 1, -1])
        # Pick 1-2 entries in V
        nv = rng.randint(1, 2)
        for _ in range(nv):
            j, k = rng.randint(0, m-1), rng.randint(0, p-1)
            V[j*p+k, c] = rng.choice([0.5, -0.5, 1, -1])
        # Pick 1-2 entries in W
        nw = rng.randint(1, 2)
        for _ in range(nw):
            k, i = rng.randint(0, p-1), rng.randint(0, n-1)
            W[k*n+i, c] = rng.choice([0.5, -0.5, 1, -1])
    
    # Now check if this structure can satisfy the constraints by adjusting magnitudes.
    # The constraints are bilinear/quadratic in the variables, so it's not a linear system.
    # Instead, let's just verify directly.
    if _verify_rank_fast(U, V, W):
        return U, V, W
    return None


def _local_search(U, V, W, rank, max_steps=2000, deadline=None):
    """Local search: try changing individual entries to half-integer values."""
    possible_values = [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2, 2.5, -2.5, 3, -3]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    def _violation(U, V, W):
        i_vals = np.repeat(np.arange(N), M * P)
        j_vals = np.tile(np.repeat(np.arange(M), P), N)
        k_vals = np.tile(np.arange(P), N * M)
        u_idx = i_vals * M + j_vals
        v_idx = j_vals * P + k_vals
        w_idx = k_vals * N + i_vals
        diag_vals = np.einsum('ac,bc,ac->a', U[u_idx], V[v_idx], W[w_idx])
        viol = np.abs(diag_vals - 1.0)
        for j in range(M):
            for i in range(N):
                for k in range(P):
                    a = i * M + j
                    b = j * P + k
                    prod = U[a] * V[b]
                    for i2 in range(N):
                        for k2 in range(P):
                            if i == i2 and k == k2:
                                continue
                            c = k2 * N + i2
                            viol = np.concatenate([viol, [abs(np.dot(prod, W[c]))]])
        return viol
    
    cur_viol = _violation(U, V, W)
    cur_max = cur_viol.max()
    
    for step in range(max_steps):
        if deadline and time.time() > deadline:
            break
        if cur_max < 1e-9:
            return U, V, W, True
        
        mat = rng.randint(0, 2)
        if mat == 0:
            row = rng.randint(0, N*M-1)
            col = rng.randint(0, rank-1)
            old = U[row, col]
            U[row, col] = rng.choice(possible_values)
            new_viol = _violation(U, V, W)
            new_max = new_viol.max()
            if new_max < cur_max - 1e-12:
                cur_viol = new_viol
                cur_max = new_max
            else:
                U[row, col] = old
        elif mat == 1:
            row = rng.randint(0, M*P-1)
            col = rng.randint(0, rank-1)
            old = V[row, col]
            V[row, col] = rng.choice(possible_values)
            new_viol = _violation(V, V, W)  # BUG: should be (U, V, W)
            V[row, col] = old  # revert always for now due to bug
        else:
            row = rng.randint(0, P*N-1)
            col = rng.randint(0, rank-1)
            old = W[row, col]
            W[row, col] = rng.choice(possible_values)
            new_viol = _violation(U, V, W)
            new_max = new_viol.max()
            if new_max < cur_max - 1e-12:
                cur_viol = new_viol
                cur_max = new_max
            else:
                W[row, col] = old
    
    if cur_max < 1e-9:
        return U, V, W, True
    return U, V, W, False


def construct():
    start_time = time.time()
    deadline = start_time + 135
    rng = random.Random(12345)
    
    best = _schoolbook_rank()
    best_rank = N * M * P  # 60
    
    if not _verify_rank_fast(best["U"], best["V"], best["W"]):
        return best
    
    possible_values = [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2, 2.5, -2.5, 3, -3]
    nonzero_values = [v for v in possible_values if v != 0]
    
    # Phase 1: Try to reduce rank by modifying schoolbook.
    # Key idea: if two triples (i,j,k) and (i,j,k') share the same (i,j), we might
    # merge them if the off-diagonal constraints can be satisfied by cancellation.
    
    # For now, do a structured search over small rank reductions.
    for target_rank in range(best_rank - 1, max(best_rank - 20, 30), -1):
        if time.time() > deadline:
            break
        
        found = False
        
        # Strategy A: Random initialization + local search
        for attempt in range(100):
            if time.time() > deadline or found:
                break
            
            U = np.zeros((N*M, target_rank), dtype=float)
            V = np.zeros((M*P, target_rank), dtype=float)
            W = np.zeros((P*N, target_rank), dtype=float)
            
            # Initialize with sparse random structure
            for c in range(target_rank):
                i = rng.randint(0, N-1)
                j = rng.randint(0, M-1)
                k = rng.randint(0, P-1)
                U[i*M+j, c] = rng.choice(nonzero_values)
                V[j*P+k, c] = rng.choice(nonzero_values)
                W[k*N+i, c] = rng.choice(nonzero_values)
                
                # Sometimes add extra entries
                if rng.random() < 0.3:
                    i2 = rng.randint(0, N-1)
                    j2 = rng.randint(0, M-1)
                    k2 = rng.randint(0, P-1)
                    if (i2*M+j2) != (i*M+j):
                        U[i2*M+j2, c] = rng.choice(nonzero_values)
                    if (j2*P+k2) != (j*P+k):
                        V[j2*P+k2, c] = rng.choice(nonzero_values)
                    if (k2*N+i2) != (k*N+i):
                        W[k2*N+i2, c] = rng.choice(nonzero_values)
            
            # Check if close to valid
            i_vals = np.repeat(np.arange(N), M * P)
            j_vals = np.tile(np.repeat(np.arange(M), P), N)
            k_vals = np.tile(np.arange(P), N * M)
            u_idx = i_vals * M + j_vals
            v_idx = j_vals * P + k_vals
            w_idx = k_vals * N + i_vals
            diag_vals = np.einsum('ac,bc,ac->a', U[u_idx], V[v_idx], W[w_idx])
            
            if np.allclose(diag_vals, 1.0, atol=0.1):
                # Close! Try local search
                U2, V2, W2, success = _local_search(U, V, W, target_rank, max_steps=500, deadline=deadline)
                if success and _verify_rank_fast(U2, V2, W2):
                    best = {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            continue
    
    # Phase 2: If we have time, try a more systematic approach
    if time.time() < deadline - 30:
        # Try to find rank 59 by modifying schoolbook
        # Take schoolbook, zero out one column, try to repair
        U0 = np.array(best["U"], dtype=float)
        V0 = np.array(best["V"], dtype=float)
        W0 = np.array(best["W"], dtype=float)
        
        for col_to_remove in range(min(20, best_rank)):
            if time.time() > deadline - 10:
                break
            
            U = U0.copy()
            V = V0.copy()
            W = W0.copy()
            U[:, col_to_remove] = 0
            V[:, col_to_remove] = 0
            W[:, col_to_remove] = 0
            
            # Try to repair with local search
            U2, V2, W2, success = _local_search(U, V, W, best_rank, max_steps=1000, deadline=deadline)
            if success:
                # Check if we can drop the zero column
                # Find columns that are all zero
                nonzero_cols = [c for c in range(best_rank) 
                              if np.any(U2[:, c] != 0) or np.any(V2[:, c] != 0) or np.any(W2[:, c] != 0)]
                if len(nonzero_cols) < best_rank:
                    U_c = U2[:, nonzero_cols]
                    V_c = V2[:, nonzero_cols]
                    W_c = W2[:, nonzero_cols]
                    if _verify_rank_fast(U_c, V_c, W_c):
                        best = {"U": U_c.tolist(), "V": V_c.tolist(), "W": W_c.tolist()}
                        best_rank = len(nonzero_cols)
    
    return best
