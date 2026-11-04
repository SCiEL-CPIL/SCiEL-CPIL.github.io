import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]
N, M, P = 4, 4, 5
DEADLINE = 140.0  # safety margin before 150s kill


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition. U:(nm,rank) V:(mp,rank) W:(pn,rank).
    W index convention: W[k*n + i, c]."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64).reshape(n, m, rank)
    Va = np.array(V, dtype=np.float64).reshape(m, p, rank)
    Wa = np.array(W, dtype=np.float64).reshape(p, n, rank)
    rec = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    return np.allclose(rec, np.ones((n, m, p)), atol=tol)


def _verify_fast(U, V, W, rank, tol=1e-9):
    """Verify assuming arrays already reshaped: U:(n,m,rank) V:(m,p,rank) W:(p,n,rank)."""
    rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    return np.allclose(rec, np.ones((N, M, P)), atol=tol)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W


def _strassen_2x2():
    """Rank-7 tensors for 2x2 x 2x2 -> 2x2.
    U[i,j,c], V[j,k,c], W[k,i,c] with i,j,k in {0,1}.
    Returns U,V,W each shape (4,7) flattened row-major (i,j),(j,k),(k,i)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    def setU(i, j, c, v): U[i * 2 + j, c] += v
    def setV(j, k, c, v): V[j * 2 + k, c] += v
    def setW(k, i, c, v): W[k * 2 + i, c] += v

    # m0 = (a00+a11)(b00+b11)
    setU(0, 0, 0, 1); setU(1, 1, 0, 1)
    setV(0, 0, 0, 1); setV(1, 1, 0, 1)
    setW(0, 0, 0, 1); setW(1, 1, 0, 1)
    # m1 = (a10+a11)b00
    setU(1, 0, 1, 1); setU(1, 1, 1, 1)
    setV(0, 0, 1, 1)
    setW(0, 1, 1, 1); setW(1, 1, 1, -1)
    # m2 = a00(b01-b11)
    setU(0, 0, 2, 1)
    setV(0, 1, 2, 1); setV(1, 1, 2, -1)
    setW(1, 0, 2, 1); setW(1, 1, 2, 1)
    # m3 = a11(b10-b00)
    setU(1, 1, 3, 1)
    setV(1, 0, 3, 1); setV(0, 0, 3, -1)
    setW(0, 1, 3, 1); setW(1, 1, 3, 1)
    # m4 = (a00+a01)b11
    setU(0, 0, 4, 1); setU(0, 1, 4, 1)
    setV(1, 1, 4, 1)
    setW(1, 0, 4, -1); setW(1, 1, 4, 1)
    # m5 = (a10-a00)(b00+b01)
    setU(1, 0, 5, 1); setU(0, 0, 5, -1)
    setV(0, 0, 5, 1); setV(0, 1, 5, 1)
    setW(0, 1, 5, 1)
    # m6 = (a01-a11)(b10+b11)
    setU(0, 1, 6, 1); setU(1, 1, 6, -1)
    setV(1, 0, 6, 1); setV(1, 1, 6, 1)
    setW(1, 0, 6, 1)

    # sanity self-check
    rec = np.einsum('ijc,jkc,kic->ijk', U.reshape(2, 2, 7),
                    V.reshape(2, 2, 7), W.reshape(2, 2, 7))
    assert np.allclose(rec, np.ones((2, 2, 2)), atol=1e-12), "Strassen 2x2 broken"
    return U, V, W


def _build_4x4x4_strassen():
    """Rank-28 decomposition for 4x4 x 4x4 -> 4x4 via block Strassen.
    4x4 split into four 2x2 blocks. Block-level Strassen (7 terms),
    each term multiplies two 2x2 blocks (schoolbook rank 4). Total 7*4=28.
    Returns U:(16,28) V:(16,28) W:(16,28)."""
    Su, Sv, Sw = _strassen_2x2()
    U = np.zeros((16, 28))
    V = np.zeros((16, 28))
    W = np.zeros((16, 28))

    # Strassen linear forms over block coordinates (br,bc) in {0,1}^2
    # A-blocks: (br, bc) ; B-blocks: (br, bc) ; C-blocks: (br, bc)
    Uforms = [
        {('A', 0, 0): 1, ('A', 1, 1): 1},
        {('A', 1, 0): 1, ('A', 1, 1): 1},
        {('A', 0, 0): 1},
        {('A', 1, 1): 1},
        {('A', 0, 0): 1, ('A', 0, 1): 1},
        {('A', 1, 0): 1, ('A', 0, 0): -1},
        {('A', 0, 1): 1, ('A', 1, 1): -1},
    ]
    Vforms = [
        {('B', 0, 0): 1, ('B', 1, 1): 1},
        {('B', 0, 0): 1},
        {('B', 0, 1): 1, ('B', 1, 1): -1},
        {('B', 1, 0): 1, ('B', 0, 0): -1},
        {('B', 1, 1): 1},
        {('B', 0, 0): 1, ('B', 0, 1): 1},
        {('B', 1, 0): 1, ('B', 1, 1): 1},
    ]
    Wforms = [
        {('C', 0, 0): 1, ('C', 1, 1): 1},
        {('C', 1, 0): 1, ('C', 1, 1): -1},
        {('C', 0, 1): 1, ('C', 1, 1): 1},
        {('C', 1, 0): 1, ('C', 0, 0): 1},
        {('C', 0, 1): -1, ('C', 1, 1): 1},
        {('C', 1, 1): 1},
        {('C', 0, 0): 1},
    ]

    # For each Strassen term s and each inner schoolbook (lr,lc) in 2x2,
    # add the rank-1 contribution.
    for s in range(7):
        for lr in range(2):
            for lc in range(2):
                c = s * 4 + lr * 2 + lc
                # A block: for each (br,bc) in Uforms[s], add coeff to all 4 entries
                for (tag, br, bc), coeff in Uforms[s].items():
                    for i2 in range(2):
                        for j2 in range(2):
                            i = br * 2 + i2
                            j = bc * 2 + j2
                            U[i * 4 + j, c] += coeff
                # B block
                for (tag, br, bc), coeff in Vforms[s].items():
                    for j2 in range(2):
                        for k2 in range(2):
                            j = br * 2 + j2
                            k = bc * 2 + k2
                            V[j * 4 + k, c] += coeff
                # C block
                for (tag, br, bc), coeff in Wforms[s].items():
                    for k2 in range(2):
                        for i2 in range(2):
                            k = br * 2 + k2
                            i = bc * 2 + i2
                            W[k * 4 + i, c] += coeff

    # verify
    assert _verify(U.tolist(), V.tolist(), W.tolist(), 4, 4, 4), "4x4x4 Strassen broken"
    return U, V, W


def _build_4x4x5():
    """Combine 4x4x4 Strassen (rank 28) + boundary 4x4 x 4x1 (rank 16) = rank 44.
    U:(16,44) V:(20,44) W:(20,44)."""
    U44, V44, W44 = _build_4x4x4_strassen()
    rank_core = 28
    rank_total = 44
    U = np.zeros((16, rank_total))
    V = np.zeros((20, rank_total))
    W = np.zeros((20, rank_total))

    # core: A(4x4) * B_left(4x4) -> C_left(4x4)
    U[:, :rank_core] = U44
    # need to map V44 (j*4+k) into V (j*5+k)
    for j in range(4):
        for k in range(4):
            V[j * 5 + k, :rank_core] = V44[j * 4 + k, :rank_core]
    # W44 (k*4+i) into W (k*4+i) for k in 0..3
    W[:16, :rank_core] = W44

    # boundary: A(4x4) * B_right(4x1) -> C_right(4x1)
    # C[i,4] = sum_j A[i,j] B[j,4]
    for i in range(4):
        for j in range(4):
            c = rank_core + i * 4 + j
            U[i * 4 + j, c] = 1.0
            V[j * 5 + 4, c] = 1.0
            W[4 * 4 + i, c] = 1.0  # k=4, i

    assert _verify(U.tolist(), V.tolist(), W.tolist(), N, M, P), "4x4x5 broken"
    return U, V, W


def _build_4x4x5_strassen_boundary():
    """Try a more structured decomposition:
    Use Strassen for the 4x4 core (rank 28) and a structured rank-8 boundary
    by splitting the 4x1 boundary into two 2x1 blocks and using rank-2 for each.
    Total rank = 28 + 8 = 36.
    
    The boundary is: C[i,4] = sum_{j=0}^{3} A[i,j] B[j,4]
    Split j into {0,1} and {2,3}:
    C[i,4] = sum_{j=0}^{1} A[i,j] B[j,4] + sum_{j=2}^{3} A[i,j] B[j,4]
    
    For the first part (j in {0,1}), we need to compute 4x2 * 2x1 -> 4x1.
    This can be done with rank 2 if we use the structure:
    C[i,4] += A[i,0]B[0,4] + A[i,1]B[1,4]
    
    Actually, the rank of 4x2 * 2x1 -> 4x1 is min(4*2, 2*1, 4*1) = 4? No.
    The tensor is sum_{i,j} A[i,j] B[j,4] C[i,4] where j in {0,1}.
    This is a bilinear form with 4*2*1 = 8 terms, but can be done in rank 4?
    
    Actually, for 4x2 * 2x1, the rank is 4 (since the output is 4x1 and we need 4 independent sums).
    So two such blocks would be rank 8.
    
    Let's just use the simple rank-16 boundary and try to reduce it.
    """
    # For now, fall back to the standard construction
    return _build_4x4x5()


def _reduce_linear(U, V, W, tol=1e-8, max_passes=40):
    """Detect columns that are linear combinations of others and merge.
    U:(nm,rank) V:(mp,rank) W:(pn,rank) as lists. Returns reduced lists or None."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]

    # Build combined matrix of all columns to test linear dependence
    stacked = np.vstack([Ua, Va, Wa])  # (nm+mp+pn, rank)

    changed = True
    passes = 0
    while changed and passes < max_passes:
        changed = False
        passes += 1
        r = stacked.shape[1]
        if r <= 1:
            break
        # Find a column that is a linear combo of the others.
        full_rank = np.linalg.matrix_rank(stacked, tol=tol)
        # To be fast, find one redundant column via null space.
        try:
            _, svals, Vt = np.linalg.svd(stacked, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        # null space dimension = r - full_rank
        null_dim = r - full_rank
        if null_dim <= 0:
            break
        # null vectors are rows Vt[full_rank:]
        null_vecs = Vt[full_rank:]
        # find a null vector with a large component
        for nv in null_vecs:
            idx = int(np.argmax(np.abs(nv)))
            if abs(nv[idx]) < 1e-10:
                continue
            # column idx is redundant
            keep = [c for c in range(r) if c != idx]
            stacked = stacked[:, keep]
            Ua = Ua[:, keep]
            Va = Va[:, keep]
            Wa = Wa[:, keep]
            changed = True
            break
        if not changed:
            break

    if Ua.shape[1] < rank:
        return Ua.tolist(), Va.tolist(), Wa.tolist()
    return None


def _reduce_columns(U, V, W, tol=1e-9, max_iter=200):
    """Remove columns that are exactly zero or make decomposition still valid."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    for _ in range(max_iter):
        removed = False
        for c in range(rank - 1, -1, -1):
            Ut = [row[:c] + row[c + 1:] for row in U]
            Vt = [row[:c] + row[c + 1:] for row in V]
            Wt = [row[:c] + row[c + 1:] for row in W]
            if _verify(Ut, Vt, Wt, N, M, P, tol):
                U, V, W = Ut, Vt, Wt
                rank -= 1
                removed = True
                break
        if not removed:
            break
    if rank < len(U[0]):
        return U, V, W
    return None


def _sparse_random_search(deadline, best_rank, seed=12345):
    """Structured random search with sparse support over {-1,0,1}."""
    rng = random.Random(seed)
    target_ranks = [40, 38, 36, 34, 32, 30, 28]
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        budget = deadline - time.time()
        if budget < 1.0:
            break
        attempts = max(5, int(budget / 0.05))
        for _ in range(attempts):
            if time.time() > deadline:
                break
            U = np.zeros((16, target_rank))
            V = np.zeros((20, target_rank))
            W = np.zeros((20, target_rank))
            for c in range(target_rank):
                nu = rng.randint(1, 3)
                nv = rng.randint(1, 3)
                nw = rng.randint(1, 3)
                for idx in rng.sample(range(16), nu):
                    U[idx, c] = rng.choice([-1, 1])
                for idx in rng.sample(range(20), nv):
                    V[idx, c] = rng.choice([-1, 1])
                for idx in rng.sample(range(20), nw):
                    W[idx, c] = rng.choice([-1, 1])
            if _verify(U.tolist(), V.tolist(), W.tolist(), N, M, P):
                return U.tolist(), V.tolist(), W.tolist(), target_rank
    return None


def _block_circulant_search(deadline, best_rank, seed=54321):
    """Search for block-circulant structured decompositions.
    
    The idea: embed the 4x4x5 multiplication into a cyclic convolution structure.
    We try to find a low-rank decomposition where the W factors have a circulant
    structure across the 5 output columns, which may allow sharing of computation.
    
    Specifically, we search over structured bases where:
    - U has block structure (4x4 blocks)
    - V has block structure (4x5 blocks)  
    - W has circulant-like structure across the 5 k-dimensions
    
    This is a heuristic search within the structured family.
    """
    rng = random.Random(seed)
    
    # Try various rank targets with structured patterns
    target_ranks = [42, 40, 38, 36, 34, 32, 30]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        budget = deadline - time.time()
        if budget < 0.5:
            break
        
        # Number of structured attempts
        n_attempts = max(3, int(budget / 0.1))
        
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break
            
            # Initialize with the known rank-44 solution as starting point
            U = np.zeros((16, target_rank))
            V = np.zeros((20, target_rank))
            W = np.zeros((20, target_rank))
            
            # Use a mixture of structured and random patterns
            # Pattern 1: Block-diagonal with shared structure
            # Pattern 2: Circulant-like W factors
            
            # Start with core Strassen structure for first 28 columns
            U44, V44, W44 = _build_4x4x4_strassen()
            n_core = min(28, target_rank)
            U[:, :n_core] = U44[:, :n_core]
            for j in range(4):
                for k in range(4):
                    V[j * 5 + k, :n_core] = V44[j * 4 + k, :n_core]
            W[:16, :n_core] = W44[:16, :n_core]
            
            # Fill remaining columns with structured boundary terms
            n_boundary = target_rank - n_core
            if n_boundary > 0:
                # Try different boundary structures
                # Structure A: Standard schoolbook boundary
                # Structure B: Shared linear forms across rows
                
                if attempt % 3 == 0:
                    # Standard boundary
                    for b in range(n_boundary):
                        i = b % 4
                        j = b // 4
                        if j < 4:
                            U[i * 4 + j, n_core + b] = 1.0
                            V[j * 5 + 4, n_core + b] = 1.0
                            W[4 * 4 + i, n_core + b] = 1.0
                elif attempt % 3 == 1:
                    # Shared structure: pair up boundary terms
                    for b in range(n_boundary):
                        i = b % 4
                        j = (b // 4) % 4
                        if b < 16:
                            # Use shared linear forms
                            if b % 2 == 0:
                                # Pair j and j+1
                                jp = j
                                U[i * 4 + jp, n_core + b] = 1.0
                                U[i * 4 + (jp + 1) % 4, n_core + b] = 1.0
                                V[jp * 5 + 4, n_core + b] = 1.0
                                V[(jp + 1) % 4 * 5 + 4, n_core + b] = 1.0
                                W[4 * 4 + i, n_core + b] = 1.0
                            else:
                                # Single term
                                U[i * 4 + j, n_core + b] = 1.0
                                V[j * 5 + 4, n_core + b] = 1.0
                                W[4 * 4 + i, n_core + b] = 1.0
                else:
                    # Random sparse boundary
                    for b in range(n_boundary):
                        i = rng.randint(0, 3)
                        j = rng.randint(0, 3)
                        U[i * 4 + j, n_core + b] = rng.choice([-1, 1])
                        V[j * 5 + 4, n_core + b] = rng.choice([-1, 1])
                        W[4 * 4 + i, n_core + b] = rng.choice([-1, 1])
            
            # Verify
            if _verify(U.tolist(), V.tolist(), W.tolist(), N, M, P):
                return U.tolist(), V.tolist(), W.tolist(), target_rank
            
            # Try local modifications: flip signs, add small perturbations
            for mod in range(min(5, int(budget / 0.02))):
                if time.time() > deadline:
                    break
                
                U_mod = U.copy()
                V_mod = V.copy()
                W_mod = W.copy()
                
                # Randomly flip a few entries
                for _ in range(rng.randint(1, 3)):
                    arr = rng.choice([U_mod, V_mod, W_mod])
                    r_idx = rng.randint(0, arr.shape[0] - 1)
                    c_idx = rng.randint(0, arr.shape[1] - 1)
                    if arr[r_idx, c_idx] != 0:
                        arr[r_idx, c_idx] *= -1
                
                if _verify(U_mod.tolist(), V_mod.tolist(), W_mod.tolist(), N, M, P):
                    return U_mod.tolist(), V_mod.tolist(), W_mod.tolist(), target_rank
    
    return None


def construct():
    start = time.time()
    deadline = start + DEADLINE

    # fallback: schoolbook
    Us, Vs, Ws = _schoolbook(N, M, P)
    assert _verify(Us, Vs, Ws, N, M, P)
    best_rank = N * M * P
    best = {"U": Us, "V": Vs, "W": Ws}

    # structured 4x4x5 (rank 44)
    if time.time() < deadline:
        try:
            U445, V445, W445 = _build_4x4x5()
            if _verify(U445.tolist(), V445.tolist(), W445.tolist(), N, M, P):
                best_rank = 44
                best = {"U": U445.tolist(), "V": V445.tolist(), "W": W445.tolist()}

                # reduce via column removal
                if time.time() < deadline:
                    red = _reduce_columns(best["U"], best["V"], best["W"], max_iter=150)
                    if red is not None:
                        Ur, Vr, Wr = red
                        rr = len(Ur[0])
                        if rr < best_rank and _verify(Ur, Vr, Wr, N, M, P):
                            best_rank = rr
                            best = {"U": Ur, "V": Vr, "W": Wr}

                # reduce via linear-dependence merging
                if time.time() < deadline:
                    red2 = _reduce_linear(best["U"], best["V"], best["W"])
                    if red2 is not None:
                        Ur, Vr, Wr = red2
                        rr = len(Ur[0])
                        if rr < best_rank and _verify(Ur, Vr, Wr, N, M, P):
                            best_rank = rr
                            best = {"U": Ur, "V": Vr, "W": Wr}
        except Exception:
            pass

    # Block-circulant structured search
    if time.time() < deadline and best_rank > 40:
        try:
            res = _block_circulant_search(deadline, best_rank, seed=54321)
            if res is not None:
                Ur, Vr, Wr, rr = res
                if rr < best_rank:
                    best_rank = rr
                    best = {"U": Ur, "V": Vr, "W": Wr}
        except Exception:
            pass

    # sparse random search for even lower rank
    if time.time() < deadline and best_rank > 40:
        try:
            res = _sparse_random_search(deadline, best_rank, seed=98765)
            if res is not None:
                Ur, Vr, Wr, rr = res
                if rr < best_rank:
                    best_rank = rr
                    best = {"U": Ur, "V": Vr, "W": Wr}
        except Exception:
            pass

    if time.time() > deadline:
        return best
    return best
