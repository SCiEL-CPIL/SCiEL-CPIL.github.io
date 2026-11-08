"""Matrix multiplication (2,3,9) rank search with Kronecker-product structure."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START

def _schoolbook_full():
    """Return the schoolbook rank-54 decomposition for (2,3,9)."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify_full(U, V, W, n=2, m=3, p=9, r=None):
    """Verify that U,V,W give an exact rank-r decomposition of the (n,m,p) matmul tensor."""
    if r is None:
        r = U.shape[1]
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    T = np.einsum('imc,jpc,knc->ijk', Ur, Vr, Wr)
    return np.allclose(T, 1.0, atol=1e-9)


def _verify_sub(U, V, W, n, m, p, r):
    """Verify a sub-decomposition for (n,m,p) matmul."""
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    T = np.einsum('imc,jpc,knc->ijk', Ur, Vr, Wr)
    return np.allclose(T, 1.0, atol=1e-9)


def _search_sub(n, m, p, r, deadline, val_sets, trials_per_iter=2000, rng=None):
    """
    Search for a rank-r decomposition of the (n,m,p) matmul tensor.
    Returns (U, V, W) or None.
    """
    if rng is None:
        rng = random.Random()
    best = None
    total_dim = n * m + m * p + p * n
    while time.time() < deadline:
        # For each trial batch, try different sparsity patterns
        for _ in range(trials_per_iter):
            if time.time() > deadline:
                break
            vs = rng.choice(val_sets)
            # Random sparsity level between 0.3 and 0.7
            sparsity = rng.uniform(0.3, 0.7)
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for c in range(r):
                for idx in range(n * m):
                    if rng.random() < sparsity:
                        U[idx, c] = rng.choice(vs)
                for idx in range(m * p):
                    if rng.random() < sparsity:
                        V[idx, c] = rng.choice(vs)
                for idx in range(p * n):
                    if rng.random() < sparsity:
                        W[idx, c] = rng.choice(vs)
            if _verify_sub(U, V, W, n, m, p, r):
                return U, V, W
    return None


def _kron_full_from_sub(U_sub, V_sub, W_sub, r1, r2, n1=2, m1=3, p1=3, n2=1, m2=1, p2=3):
    """
    Given a rank-r1 sub-decomposition for (n1,m1,p1) matmul and a rank-r2 sub-decomposition
    for (n2,m2,p2) matmul, construct the rank-(r1*r2) decomposition for
    (n1*n2, m1*m2, p1*p2) matmul.
    
    Here we fix n2=1, m2=1, p2=3, so the sub problem is 1x1x3 matmul (rank 3 schoolbook).
    """
    # Schoolbook for 1x1x3: rank 3
    # U2: (1*1, 3) = (1,3), V2: (1*3, 3) = (3,3), W2: (3*1, 3) = (3,3)
    U2 = np.eye(1, 3)  # shape (1,3): [1,0,0]
    # Actually for 1x1x3: T[i,j,k] = 1 for all i,j,k (since n=1,m=1,p=3, only one i,j)
    # Schoolbook: for i in range(1), j in range(1), k in range(3):
    #   U[i*1+j, col] = 1, V[j*3+k, col] = 1, W[k*1+i, col] = 1
    # So U = [[1,0,0]], V = [[1,0,0],[0,1,0],[0,0,1]], W = [[1,0,0],[0,1,0],[0,0,1]]
    U2 = np.array([[1.0, 0.0, 0.0]])  # shape (1, 3)
    V2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # shape (3, 3)
    W2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # shape (3, 3)
    
    r = r1 * r2
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    # U_full: (n*m, r) = (n1*n2*m1*m2, r1*r2)
    # From Kronecker: U_full = U1 (x) U2 with appropriate index mapping
    # U1: (n1*m1, r1), U2: (n2*m2, r2) = (1, 3)
    # The Kronecker product of U1 and U2 gives (n1*m1 * n2*m2, r1*r2)
    # But we need to map indices correctly.
    
    # For matrix multiplication tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Kronecker: if T = T1 (x) T2, then for indices (i1,i2, j1,j2, k1,k2):
    # T[i1,i2, j1,j2, k1,k2] = T1[i1,j1,k1] * T2[i2,j2,k2]
    # And T1[i1,j1,k1] = sum_{c1} U1[i1,j1,c1] * V1[j1,k1,c1] * W1[k1,i1,c1]
    # T2[i2,j2,k2] = sum_{c2} U2[i2,j2,c2] * V2[j2,k2,c2] * W2[k2,i2,c2]
    # So T = sum_{c1,c2} (U1[i1,j1,c1]*U2[i2,j2,c2]) * (V1[j1,k1,c1]*V2[j2,k2,c2]) * (W1[k1,i1,c1]*W2[k2,i2,c2])
    
    # For our case: n2=1, m2=1, so i2=0, j2=0 always.
    # i = i1 * n2 + i2 = i1 (since n2=1)
    # j = j1 * m2 + j2 = j1 (since m2=1)
    # k = k1 * p2 + k2 = k1 * 3 + k2
    
    # U_full[i, j, c1*r2+c2] = U1[i1, j1, c1] * U2[i2, j2, c2] = U1[i1, j1, c1] * U2[0, 0, c2]
    # U2[0,0,c2] = 1 if c2==0, 0 otherwise (since U2 = [[1,0,0]])
    
    # Wait, that's not right. Let me reconsider.
    # U2 shape is (n2*m2, r2) = (1, 3). U2[0, c2] = 1 if c2==0 else 0.
    # So U_full[i1, j1, c1*3+c2] = U1[i1, j1, c1] * U2[0, c2]
    # This means only c2=0 contributes: U_full[i1, j1, c1*3] = U1[i1, j1, c1]
    # All other entries are 0.
    
    # Similarly for V:
    # V_full[j, k, c1*r2+c2] = V1[j1, k1, c1] * V2[j2, k2, c2]
    # j = j1, k = k1*3 + k2
    # V_full[j1, k1*3+k2, c1*3+c2] = V1[j1, k1, c1] * V2[0, k2, c2]
    # V2[0, k2, c2]: V2 is (3,3) identity, so V2[0,k2,c2] = 1 if k2==0 and c2==0, else 0.
    # So V_full[j1, k1*3, c1*3] = V1[j1, k1, c1]
    
    # For W:
    # W_full[k, i, c1*r2+c2] = W1[k1, i1, c1] * W2[k2, i2, c2]
    # k = k1*3+k2, i = i1
    # W_full[k1*3+k2, i1, c1*3+c2] = W1[k1, i1, c1] * W2[k2, 0, c2]
    # W2 is (3,3) identity, so W2[k2, 0, c2] = 1 if k2==0 and c2==0, else 0.
    # So W_full[k1*3, i1, c1*3] = W1[k1, i1, c1]
    
    # This gives a very sparse structure! Let's construct it properly.
    U_full = np.zeros((n * m, r))
    V_full = np.zeros((m * p, r))
    W_full = np.zeros((p * n, r))
    
    for i1 in range(n1):
        for j1 in range(m1):
            for c1 in range(r1):
                u_val = U_sub[i1 * m1 + j1, c1]
                if u_val != 0:
                    # c2 = 0
                    U_full[i1 * m1 + j1, c1 * r2 + 0] = u_val
        
    for j1 in range(m1):
        for k1 in range(p1):
            for c1 in range(r1):
                v_val = V_sub[j1 * p1 + k1, c1]
                if v_val != 0:
                    # k2 = 0, c2 = 0
                    V_full[j1 * (p1 * p2) + k1 * p2 + 0, c1 * r2 + 0] = v_val
    
    for k1 in range(p1):
        for i1 in range(n1):
            for c1 in range(r1):
                w_val = W_sub[k1 * n1 + i1, c1]
                if w_val != 0:
                    # k2 = 0, c2 = 0
                    W_full[k1 * p2 * n1 + 0 * n1 + i1, c1 * r2 + 0] = w_val
    
    return U_full, V_full, W_full


def _kron_full_general(U1, V1, W1, U2, V2, W2, r1, r2, n1, m1, p1, n2, m2, p2):
    """
    General Kronecker product construction for matmul tensor decomposition.
    T = T1 (x) T2 where T1 is (n1,m1,p1) and T2 is (n2,m2,p2).
    Result is (n1*n2, m1*m2, p1*p2) with rank r1*r2.
    """
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    r = r1 * r2
    
    U_full = np.zeros((n * m, r))
    V_full = np.zeros((m * p, r))
    W_full = np.zeros((p * n, r))
    
    # U_full[i, j, c] where i = i1*n2 + i2, j = j1*m2 + j2, c = c1*r2 + c2
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    for c1 in range(r1):
                        for c2 in range(r2):
                            val = U1[i1 * m1 + j1, c1] * U2[i2 * m2 + j2, c2]
                            if val != 0:
                                U_full[i * m + j, c1 * r2 + c2] += val
    
    # V_full[j, k, c] where j = j1*m2 + j2, k = k1*p2 + k2, c = c1*r2 + c2
    for j1 in range(m1):
        for k1 in range(p1):
            for j2 in range(m2):
                for k2 in range(p2):
                    for c1 in range(r1):
                        for c2 in range(r2):
                            val = V1[j1 * p1 + k1, c1] * V2[j2 * p2 + k2, c2]
                            if val != 0:
                                V_full[j * p + k, c1 * r2 + c2] += val
    
    # W_full[k, i, c] where k = k1*p2 + k2, i = i1*n2 + i2, c = c1*r2 + c2
    for k1 in range(p1):
        for i1 in range(n1):
            for k2 in range(p2):
                for i2 in range(n2):
                    for c1 in range(r1):
                        for c2 in range(r2):
                            val = W1[k1 * n1 + i1, c1] * W2[k2 * n2 + i2, c2]
                            if val != 0:
                                W_full[k * n + i, c1 * r2 + c2] += val
    
    return U_full, V_full, W_full


def _search_direct(n, m, p, r, deadline, val_sets, rng=None, trials_per_iter=1500):
    """
    Direct random search for rank-r decomposition of (n,m,p) matmul.
    """
    if rng is None:
        rng = random.Random()
    while time.time() < deadline:
        for _ in range(trials_per_iter):
            if time.time() > deadline:
                break
            vs = rng.choice(val_sets)
            sparsity = rng.uniform(0.3, 0.7)
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for c in range(r):
                for idx in range(n * m):
                    if rng.random() < sparsity:
                        U[idx, c] = rng.choice(vs)
                for idx in range(m * p):
                    if rng.random() < sparsity:
                        V[idx, c] = rng.choice(vs)
                for idx in range(p * n):
                    if rng.random() < sparsity:
                        W[idx, c] = rng.choice(vs)
            if _verify_full(U, V, W, n, m, p, r):
                return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    deadline = time.time() + 110.0  # safety margin under 120s
    rng = random.Random(42)
    
    # Value sets for random search
    val_sets = [
        [0, 1, -1],
        [0, 1, -1, 0.5, -0.5],
        [0, 1, -1, 2, -2],
        [0, 1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5],
    ]
    
    # Lower bound on rank: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    low_bound = max(n * m, m * p, p * n)
    best = _schoolbook_full()
    best_rank = n * m * p  # 54
    
    # Strategy 1: Kronecker product decomposition
    # (2,3,9) = (2,3,3) (x) (1,1,3)
    # Rank = r1 * r2 where r2 = 3 (schoolbook for 1x1x3)
    # So we search for r1 for (2,3,3) and get rank 3*r1 for (2,3,9)
    
    # Search for (2,3,3) sub-decompositions with increasing rank
    # Lower bound for (2,3,3): max(6, 9, 6) = 9
    # Try r1 = 9, 10, 11, ..., up to some reasonable limit
    
    found_sub = False
    for r1 in range(9, 19):  # Search r1 from 9 to 18
        if time.time() > deadline:
            break
        target_rank = r1 * 3
        if target_rank >= best_rank:
            break
        
        # Try to find rank-r1 decomposition for (2,3,3)
        U_sub, V_sub, W_sub = _search_sub(2, 3, 3, r1, deadline, val_sets, 
                                           trials_per_iter=3000, rng=rng)
        if U_sub is not None:
            # Construct full (2,3,9) decomposition via Kronecker
            U_full, V_full, W_full = _kron_full_general(
                U_sub, V_sub, W_sub,
                np.array([[1.0, 0.0, 0.0]]),
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                r1, 3, 2, 3, 3, 1, 1, 3
            )
            if _verify_full(U_full, V_full, W_full, n, m, p, target_rank):
                best = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
                best_rank = target_rank
                found_sub = True
                # Continue searching for lower rank if possible
                break
    
    # Strategy 2: Direct search on full (2,3,9) problem
    # Try ranks from low_bound to best_rank-1
    for r in range(low_bound, best_rank):
        if time.time() > deadline:
            break
        result = _search_direct(n, m, p, r, deadline, val_sets, rng=rng,
                                 trials_per_iter=2000)
        if result is not None:
            U, V, W = result
            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = r
            # Can continue searching for even lower rank, but we're iterating ascending
            # so this is the minimum found at this stage
            # Don't break - we want to see if we can go lower, but we're already
            # at low_bound+something, and we iterate from low_bound up
    
    # If we haven't found anything better, return schoolbook
    return best

# EVOLVE-BLOCK-END
