"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly represents matrix multiplication."""
    # Build the tensor from the decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This should equal 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Compute T[a,b,c] for all a,b,c
    # T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # This is expensive for large dimensions, so check sparsely
    nm, mp, pn = n*m, m*p, p*n
    # Check all positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1) > tol:
                    return False
    # Check some random non-active positions
    rng = np.random.RandomState(42)
    for _ in range(200):
        a = rng.randint(nm)
        b = rng.randint(mp)
        c = rng.randint(pn)
        # Check if this is an active position
        i1, j1 = a // m, a % m
        j2, k2 = b // p, b % p
        k3, i3 = c // n, c % n
        is_active = (i1 == i3) and (j1 == j2) and (k2 == k3)
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if is_active:
            if abs(val - 1) > tol:
                return False
        else:
            if abs(val) > tol:
                return False
    return True

def _check_integer_half(U, V, W, tol=1e-9):
    """Check if all entries are integer or half-integer."""
    for mat in [U, V, W]:
        mat = np.array(mat, dtype=float)
        doubled = mat * 2
        rounded = np.round(doubled)
        if np.max(np.abs(doubled - rounded)) > tol:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = nm * mp * pn // (mp // (m*p)) # just n*m*p
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    # Strategy: random search with pruning
    rng = np.random.RandomState(123)
    
    # Try ranks from best_rank-1 downward
    for target_rank in range(best_rank - 1, 20, -1):
        if time.time() > deadline:
            break
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            # Random sparse initialization
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            # Initialize with random ±1, ±0.5 values at random positions
            for r in range(target_rank):
                # Choose random non-zero positions
                nz_u = rng.choice(nm, size=min(nm, 3), replace=False)
                nz_v = rng.choice(mp, size=min(mp, 3), replace=False)
                nz_w = rng.choice(pn, size=min(pn, 3), replace=False)
                for idx in nz_u:
                    U[idx, r] = rng.choice([0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2])
                for idx in nz_v:
                    V[idx, r] = rng.choice([0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2])
                for idx in nz_w:
                    W[idx, r] = rng.choice([0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2])
            
            if _verify(U, V, W, n, m, p):
                if _check_integer_half(U, V, W):
                    best_rank = target_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break  # Found a valid decomposition at this rank
    
    # Convert to list of lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
