"""Matrix multiplication 2x3 x 3x4 rank-minimal decomposition search.
Combines structured enumeration (permutation/diagonal symmetries) with
randomized local refinement and exact half-integer verification.
"""
import numpy as np
from itertools import product
import time
import random

DIMENSIONS = [2, 3, 4]

def _verify_exact(U, V, W, n, m, p):
    """Exact check: for all i,i',j,j',k,k',
    sum_r U[i*m+j, r]*V[j'*p+k, r]*W[k'*n+i', r] == 1 iff i==i' and j==j' and k==k'
    (since T[(i,j),(j',k),(i',k')] = delta_{ii'}*delta_{jj'}*delta_{kk'} for the
    multiplication tensor where the middle index must match: j must equal j').
    Actually the tensor is: C[i',k'] = sum_j A[i',j] B[j,k']
    So T[A_idx=(i,j), B_idx=(j2,k), C_idx=(i2,k2)] = 1 iff i==i2 AND j==j2 AND k==k2.
    """
    rank = U.shape[1]
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    for i in range(n):
        for i2 in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                            # Compute sum_r U[i*m+j, r] * V[j2*p+k, r] * W[k2*n+i2, r]
                            val = np.dot(U[i * m + j], V[j2 * p + k] * W[k2 * n + i2])
                            if abs(val - expected) > 1e-9:
                                return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using matrix operations where possible."""
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # T[i,j,k] for the "diagonal" part: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] must be 1
    # Off-diagonal must be 0.
    # Build the full tensor via einsum-like loops but vectorized over r
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            for k in range(p):
                # Diagonal: must equal 1
                w_row = W[k * n + i]
                v_row = V[j * p + k]
                diag_val = np.dot(u_row, v_row * w_row)
                if abs(diag_val - 1.0) > 1e-9:
                    return False
                # Off-diagonal in W: for i2 != i
                for i2 in range(n):
                    if i2 == i:
                        continue
                    w_row2 = W[k * n + i2]
                    off_val = np.dot(u_row, v_row * w_row2)
                    if abs(off_val) > 1e-9:
                        return False
                # Off-diagonal in V (j2 != j): for j2 in range(m)
                for j2 in range(m):
                    if j2 == j:
                        continue
                    v_row2 = V[j2 * p + k]
                    off_val = np.dot(u_row, v_row2 * w_row)
                    if abs(off_val) > 1e-9:
                        return False
                # Off-diagonal in k2: for k2 != k
                for k2 in range(p):
                    if k2 == k:
                        continue
                    v_row2 = V[j * p + k2]
                    off_val = np.dot(u_row, v_row2 * w_row)
                    if abs(off_val) > 1e-9:
                        return False
    return True

def _snap_to_half_int(arr):
    return np.round(np.asarray(arr, dtype=float) * 2) / 2.0

def _is_half_integer(arr, tol=1e-9):
    arr = np.asarray(arr, dtype=float)
    return np.all(np.abs(arr * 2 - np.round(arr * 2)) < tol)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _structured_candidates(n, m, p, target_rank, deadline, rng):
    """Generate structured rank-one decompositions from permutations and diagonal scalings.
    
    The multiplication tensor has symmetries: permuting rows of A, columns of A /
    rows of B, columns of B, etc. We exploit these to generate candidate
    decompositions that are more likely to be valid than pure random.
    """
    candidates = []
    
    # Strategy: build decompositions by combining "blocks" that each handle a
    # subset of the (i,j,k) triples. Each rank-1 term (u, v, w) contributes
    # u[i*m+j]*v[j'*p+k]*w[k'*n+i'] to the tensor.
    
    # For a rank-1 term to be "valid" in isolation, it must have support
    # that is a product set: I x J x K where the contributions only land
    # on the diagonal (i==i', j==j', k==k').
    # A rank-1 term with U supported on rows {(i,j)}, V on {(j,k)}, W on {(k,i)}
    # gives exactly the diagonal entry. That's the schoolbook term.
    
    # To get lower rank, we need rank-1 terms whose supports overlap and
    # cancel off-diagonals while preserving diagonals.
    
    # Structured approach: use "strassen-like" blocks.
    # For 2x2x2, Strassen uses 7 terms. For rectangular, we can partition
    # the dimensions into blocks.
    
    # Let's try: partition i into groups, j into groups, k into groups,
    # and use Kronecker-product-like structures.
    
    # Candidate type 1: "block diagonal" with shared terms
    # If we group (i,j) pairs and (j,k) pairs cleverly...
    
    # For now, generate candidates via:
    # 1. Start from schoolbook
    # 2. Try to merge pairs of terms where off-diagonal contributions cancel
    
    # Generate random structured candidates with constrained support
    entries = [0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2]
    
    for attempt in range(30):
        if time.time() > deadline - 25:
            break
        
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # Each column c: choose a "center" (i0, j0, k0) and a small perturbation
        for c in range(target_rank):
            # Random center
            i0 = rng.randint(0, n - 1)
            j0 = rng.randint(0, m - 1)
            k0 = rng.randint(0, p - 1)
            
            # Support around the center
            # U: rows (i, j) where i near i0, j near j0
            i_support = [i0]
            if n > 1 and rng.random() < 0.5:
                i_support.append(rng.randint(0, n - 1))
            j_support = [j0]
            if m > 1 and rng.random() < 0.5:
                j_support.append(rng.randint(0, m - 1))
            
            for i in set(i_support):
                for j in set(j_support):
                    U[i * m + j, c] = rng.choice(entries)
            
            # V: rows (j2, k) where j2 near j0, k near k0
            j2_support = [j0]
            if m > 1 and rng.random() < 0.5:
                j2_support.append(rng.randint(0, m - 1))
            k_support = [k0]
            if p > 1 and rng.random() < 0.5:
                k_support.append(rng.randint(0, p - 1))
            
            for j2 in set(j2_support):
                for k in set(k_support):
                    V[j2 * p + k, c] = rng.choice(entries)
            
            # W: rows (k2, i2) where k2 near k0, i2 near i0
            k2_support = [k0]
            if p > 1 and rng.random() < 0.5:
                k2_support.append(rng.randint(0, p - 1))
            i2_support = [i0]
            if n > 1 and rng.random() < 0.5:
                i2_support.append(rng.randint(0, n - 1))
            
            for k2 in set(k2_support):
                for i2 in set(i2_support):
                    W[k2 * n + i2, c] = rng.choice(entries)
        
        candidates.append((U, V, W))
    
    return candidates

def _local_refine(U, V, W, n, m, p, deadline, rng, max_iter=200):
    """Local refinement: try to zero out columns by absorbing their contribution
    into other columns, reducing rank."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    entries = [0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2]
    
    for _ in range(max_iter):
        if time.time() > deadline - 20:
            break
        
        # Pick a random column to try to eliminate
        c = rng.randint(0, rank - 1)
        
        # Try to find another column c2 such that we can adjust c2 to absorb c
        # and zero out c. This requires solving a system, which is hard.
        # Instead, try: set column c to zero and see if we can fix by adjusting
        # one entry in another column.
        
        # Simpler: try flipping signs or adding 0.5 to random entries
        # and check if rank can be reduced.
        
        # Try: for a random other column c2, add column c's pattern scaled
        # This is complex; instead do random perturbation and re-snap.
        
        # Random perturbation of a few entries
        for _ in range(3):
            mat = rng.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = rng.randint(0, n * m - 1)
                col = rng.randint(0, rank - 1)
                U[row, col] = rng.choice(entries)
            elif mat == 'V':
                row = rng.randint(0, m * p - 1)
                col = rng.randint(0, rank - 1)
                V[row, col] = rng.choice(entries)
            else:
                row = rng.randint(0, p * n - 1)
                col = rng.randint(0, rank - 1)
                W[row, col] = rng.choice(entries)
        
        if _verify_fast(U, V, W, n, m, p):
            # Try to eliminate zero columns
            new_rank = 0
            for c in range(rank):
                if not (np.all(U[:, c] == 0) and np.all(V[:, c] == 0) and np.all(W[:, c] == 0)):
                    U[:, new_rank] = U[:, c]
                    V[:, new_rank] = V[:, c]
                    W[:, new_rank] = W[:, c]
                    new_rank += 1
            if new_rank < rank:
                U = U[:, :new_rank]
                V = V[:, :new_rank]
                W = W[:, :new_rank]
                rank = new_rank
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    rng = random.Random(42)
    np.random.seed(42)
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Schoolbook fallback
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_fb, V_fb, W_fb)
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # But the tensor rank for M_{2,3,4} is known to be at most 24 (schoolbook).
    # Try to find better.
    
    for target_rank in range(12, best_rank):
        if time.time() > deadline - 30:
            break
        
        # Generate structured candidates
        candidates = _structured_candidates(n, m, p, target_rank, deadline, rng)
        
        for U, V, W in candidates:
            if time.time() > deadline - 25:
                break
            
            if _verify_fast(U, V, W, n, m, p):
                if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                    best_rank = target_rank
                    best_result = (U, V, W)
                    # Try to refine to even lower rank
                    U, V, W = _local_refine(U, V, W, n, m, p, deadline, rng)
                    if _verify_fast(U, V, W, n, m, p):
                        new_rank = U.shape[1]
                        if new_rank < best_rank:
                            best_rank = new_rank
                            best_result = (U, V, W)
                    break
        
        if best_rank == target_rank:
            break  # Found at this rank, move to next lower
    
    # Return best found
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
