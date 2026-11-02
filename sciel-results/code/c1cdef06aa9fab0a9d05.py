import time
import numpy as np

DIMENSIONS = [3, 3, 3]
_DEADLINE_S = 150.0
_START = time.time()


def _time_left():
    return _DEADLINE_S - (time.time() - _START)


def _verify(U, V, W, num_tests=20):
    """
    Verify tensor decomposition.
    U: (n*m, rank) - U[i*m + j, r]
    V: (m*p, rank) - V[j*p + k, r]
    W: (p*n, rank) - W[k*n + i, r]  (official trace convention)
    
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    n, m, p = DIMENSIONS
    R = len(U[0]) if isinstance(U[0], list) else U.shape[1]
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check tensor identity: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k)
    # T is 1 exactly when the indices are consistent
    
    # Build expected tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Compute reconstructed tensor
    T_recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        T_recon += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    if not np.allclose(T_recon, T, atol=1e-9):
        return False
    
    # Also do random matrix tests
    for _ in range(num_tests):
        A = np.random.randint(-10, 11, (3, 3)).astype(np.float64)
        B = np.random.randint(-10, 11, (3, 3)).astype(np.float64)
        C_expected = A @ B
        A_flat = A.flatten()  # A[i*m+j]
        B_flat = B.flatten()  # B[j*p+k]
        U_proj = U_arr.T @ A_flat  # (rank,)
        V_proj = V_arr.T @ B_flat  # (rank,)
        C_flat = np.zeros(9, dtype=np.float64)
        for r in range(R):
            C_flat += U_proj[r] * V_proj[r] * W_arr[:, r]
        C_computed = C_flat.reshape(3, 3)  # C_flat[k*n+i] -> reshape gives C[i,k]? No.
        # C_flat is indexed as C_flat[k*n + i], so to get C[i,k] we need to reshape carefully
        # C_flat.reshape(3,3) gives element [i,k] = C_flat[i*3+k] = C[k'=i, i'=k] which is wrong
        # We need C[i,k] = C_flat[k*3 + i], so C_computed = C_flat.reshape(3,3).T
        C_computed = C_flat.reshape(3, 3).T
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def _rank26_construct():
    """
    Rank-26 decomposition for 3x3 matrix multiplication.
    
    Partition: A = [[A1, A2], [A3, A4]] where A1 is 2x2, A2 is 2x1, A3 is 1x2, A4 is 1x1
    B = [[B1, B2], [B3, B4]] similarly
    
    C = A*B:
    C11 = A1*B1 + A2*B3  (2x2)
    C12 = A1*B2 + A2*B4  (2x1)
    C21 = A3*B1 + A4*B3  (1x2)
    C22 = A3*B2 + A4*B4  (1x1)
    
    Indexing:
    A indices: A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5, A[2,0]=6, A[2,1]=7, A[2,2]=8
    B indices: B[0,0]=0, B[0,1]=1, B[0,2]=2, B[1,0]=3, B[1,1]=4, B[1,2]=5, B[2,0]=6, B[2,1]=7, B[2,2]=8
    C indices (W convention W[k*n+i]): C[0,0]=0, C[0,1]=1, C[0,2]=2, C[1,0]=3, C[1,1]=4, C[1,2]=5, C[2,0]=6, C[2,1]=7, C[2,2]=8
    
    Wait - W[k*n+i] means W is indexed as [k, i] flattened with k as outer.
    So W[0] = C[0,0], W[1] = C[0,1], W[2] = C[0,2], W[3] = C[1,0], W[4] = C[1,1], W[5] = C[1,2],
    W[6] = C[2,0], W[7] = C[2,1], W[8] = C[2,2]
    
    So W index = k*3 + i where C[i,k] is the output element.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For C[0,0] (i=0,k=0): W index = 0*3+0 = 0
    For C[0,1] (i=0,k=1): W index = 1*3+0 = 3
    For C[0,2] (i=0,k=2): W index = 2*3+0 = 6
    For C[1,0] (i=1,k=0): W index = 0*3+1 = 1
    For C[1,1] (i=1,k=1): W index = 1*3+1 = 4
    For C[1,2] (i=1,k=2): W index = 2*3+1 = 7
    For C[2,0] (i=2,k=0): W index = 0*3+2 = 2
    For C[2,1] (i=2,k=1): W index = 1*3+2 = 5
    For C[2,2] (i=2,k=2): W index = 2*3+2 = 8
    """
    terms = []
    
    def add(u, v, w):
        terms.append((u, v, w))
    
    # --- Strassen 2x2x2 on top-left block (indices 0,1,3,4) ---
    # A1 = A[0:2,0:2] = indices {0,1,3,4}
    # B1 = B[0:2,0:2] = indices {0,1,3,4}
    # C11 = C[0:2,0:2] = C[0,0],C[0,1],C[1,0],C[1,1]
    # W indices: C[0,0]=0, C[0,1]=3, C[1,0]=1, C[1,1]=4
    
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # Contributes to C[0,0] and C[1,1]
    add({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 4: 1})
    
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # Contributes to C[1,0] and C[1,1]
    add({3: 1, 4: 1}, {0: 1}, {1: 1, 4: -1})
    
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # Contributes to C[0,1] and C[1,1]
    add({0: 1}, {1: 1, 4: -1}, {3: 1, 4: 1})
    
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # Contributes to C[0,0] and C[1,0]
    add({4: 1}, {3: 1, 0: -1}, {0: 1, 1: 1})
    
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # Contributes to C[0,0] and C[0,1]
    add({0: 1, 1: 1}, {4: 1}, {0: 1, 3: 1})
    
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # Contributes to C[1,0] and C[1,1]
    add({3: 1, 0: -1}, {0: 1, 1: 1}, {1: 1, 4: -1})
    
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # Contributes to C[0,1] and C[1,1]
    add({1: 1, 4: -1}, {3: 1, 4: 1}, {3: 1, 4: 1})
    
    # --- 1x1 block: C[2,2] = A[2,2]*B[2,2] ---
    add({8: 1}, {8: 1}, {8: 1})
    
    # --- Cross terms ---
    # C12 = A1*B2 + A2*B4 (2x1 output)
    # C[0,2] (W idx 6) = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] (W idx 7) = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    # C21 = A3*B1 + A4*B3 (1x2 output)
    # C[2,0] (W idx 2) = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] (W idx 5) = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    
    # For C[0,2]: terms A[0,0]*B[0,2], A[0,1]*B[1,2], A[0,2]*B[2,2]
    add({0: 1}, {2: 1}, {6: 1})
    add({1: 1}, {5: 1}, {6: 1})
    add({2: 1}, {8: 1}, {6: 1})
    
    # For C[1,2]: terms A[1,0]*B[0,2], A[1,1]*B[1,2], A[1,2]*B[2,2]
    add({3: 1}, {2: 1}, {7: 1})
    add({4: 1}, {5: 1}, {7: 1})
    add({5: 1}, {8: 1}, {7: 1})
    
    # For C[2,0]: terms A[2,0]*B[0,0], A[2,1]*B[1,0], A[2,2]*B[2,0]
    add({6: 1}, {0: 1}, {2: 1})
    add({7: 1}, {3: 1}, {2: 1})
    add({8: 1}, {6: 1}, {2: 1})
    
    # For C[2,1]: terms A[2,0]*B[0,1], A[2,1]*B[1,1], A[2,2]*B[2,1]
    add({6: 1}, {1: 1}, {5: 1})
    add({7: 1}, {4: 1}, {5: 1})
    add({8: 1}, {7: 1}, {5: 1})
    
    rank = len(terms)
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][r] = val
        for idx, val in v.items():
            V[idx][r] = val
        for idx, val in w.items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _try_rank25():
    """
    Attempt to find a rank-25 decomposition by merging some terms.
    This is a heuristic search within the block-partitioning family.
    """
    if _time_left() < 10.0:
        return None
    
    # Start from rank-26 and try to see if any two terms can be merged
    # This is a complex nonlinear problem, so we'll use a simple approach:
    # Try random perturbations and check if we can reduce rank
    
    # For now, return None as a safe fallback
    return None


def _try_rank24():
    """
    Attempt rank-24 via more aggressive block decomposition.
    """
    if _time_left() < 15.0:
        return None
    
    # This would require a more sophisticated approach
    # For now, return None
    return None


def construct():
    best = _schoolbook()
    best_rank = 27
    
    # Try rank-26 first
    if _time_left() > 5.0:
        cand26 = _rank26_construct()
        if _verify(cand26["U"], cand26["V"], cand26["W"], num_tests=20):
            best = cand26
            best_rank = 26
            # Print for debugging (won't affect output)
            pass
    
    # Try rank-25
    if _time_left() > 10.0:
        cand25 = _try_rank25()
        if cand25 is not None:
            if _verify(cand25["U"], cand25["V"], cand25["W"], num_tests=20):
                best = cand25
                best_rank = 25
    
    # Try rank-24
    if _time_left() > 15.0:
        cand24 = _try_rank24()
        if cand24 is not None:
            if _verify(cand24["U"], cand24["V"], cand24["W"], num_tests=20):
                best = cand24
                best_rank = 24
    
    return best
