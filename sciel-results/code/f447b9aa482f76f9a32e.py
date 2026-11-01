"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def search_rank_27(deadline):
    """
    Attempts to find a rank 27 decomposition.
    Uses random initialization and local search.
    Returns None if not found or time expires.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = 27
    
    # Target tensor
    T = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    # Random initialization
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    num_restarts = 100
    for restart in range(num_restarts):
        if time.time() > deadline:
            return None
            
        # Initialize with small integers
        U = np.random.randint(-2, 3, size=(nm, rank))
        V = np.random.randint(-2, 3, size=(mp, rank))
        W = np.random.randint(-2, 3, size=(pn, rank))
        
        # Local search: perturb one element at a time
        for step in range(500):
            if time.time() > deadline:
                return None
                
            # Compute current error
            # T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
            # This is slow. Use incremental update?
            # For simplicity, just compute.
            T_decomp = np.zeros((nm, mp, pn), dtype=int)
            for r in range(rank):
                T_decomp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            error = np.sum((T_decomp - T) ** 2)
            
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
                if error == 0:
                    return {
                        "U": best_U.tolist(),
                        "V": best_V.tolist(),
                        "W": best_W.tolist()
                    }

            # Perturb
            if np.random.random() < 0.5:
                idx = np.random.randint(nm)
                r_idx = np.random.randint(rank)
                U[idx, r_idx] += np.random.choice([-1, 1])
            elif np.random.random() < 0.5:
                idx = np.random.randint(mp)
                r_idx = np.random.randint(rank)
                V[idx, r_idx] += np.random.choice([-1, 1])
            else:
                idx = np.random.randint(pn)
                r_idx = np.random.randint(rank)
                W[idx, r_idx] += np.random.choice([-1, 1])
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook fallback (rank 32)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Strassen decomposition for 2x2 x 2x2
    # Terms are (u, v, w) where u, v, w are length-4 vectors
    # u indexes A (flattened row-major: [A00, A01, A10, A11])
    # v indexes B (flattened row-major: [B00, B01, B10, B11])
    # w indexes C (flattened row-major: [C00, C01, C10, C11])
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])    # M7
    ]

    def build_strassen_228():
        """Build rank-28 decomposition using 4 Strassen blocks."""
        rank = 28
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

        num_blocks = p // 2  # 4 blocks

        for block_idx in range(num_blocks):
            for term_idx in range(7):
                col = block_idx * 7 + term_idx
                u, v, w = strassen_terms[term_idx]

                # Fill U: index is i (0..3) for A[i//2, i%2]
                for i in range(4):
                    U[i][col] = u[i]

                # Fill V: B is 2x8, block covers columns 2*block_idx and 2*block_idx+1
                # Global B index: row j (0..1), col k_global
                # Local v index: j*2 + k_local where k_local is 0 or 1
                for j in range(2):
                    for k_local in range(2):
                        k_global = 2 * block_idx + k_local
                        b_idx = j * p + k_global
                        v_idx = j * 2 + k_local
                        V[b_idx][col] = v[v_idx]

                # Fill W: C is 8x2 (p x n), index is k_global*n + i
                # Global C index: row k_global, col i
                # Local w index: i*2 + k_local where k_local is 0 or 1
                # Wait, the convention is W[k*n+i, r]
                # C is p x n. C[k, i] corresponds to index k*n + i.
                # The local w vector is for a 2x2 block of C.
                # The block of C is at rows 2*block_idx and 2*block_idx+1, cols 0 and 1.
                # So local index (i, k_local) maps to global index (2*block_idx + k_local)*n + i.
                for i in range(2):
                    for k_local in range(2):
                        k_global = 2 * block_idx + k_local
                        c_idx = k_global * n + i
                        w_idx = i * 2 + k_local
                        W[c_idx][col] = w[w_idx]

        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify the decomposition exactly."""
        rank = len(U[0])
        # Build target tensor T[a,b,c]
        T = np.zeros((nm, mp, pn), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1

        U_np = np.array(U, dtype=int)
        V_np = np.array(V, dtype=int)
        W_np = np.array(W, dtype=int)

        # Compute T_decomp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Efficient check: for each (a,b), check if sum_r U[a,r] V[b,r] W[c,r] matches T[a,b,c] for all c
        # Or just check the full tensor
        # T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        # This might be slow for rank 28, but nm=4, mp=16, pn=16 is small.
        # Let's just loop over r and add outer products
        T_decomp = np.zeros((nm, mp, pn), dtype=int)
        for r in range(rank):
            T_decomp += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])

        return np.array_equal(T_decomp, T)

    # Try Strassen first
    result_28 = build_strassen_228()
    if verify(result_28["U"], result_28["V"], result_28["W"]):
        # Structural Change: Attempt a search for a rank 27 decomposition
        # using a randomized local search with integer constraints.
        # If successful, return rank 27. Otherwise, return rank 28.
        result_27 = search_rank_27(deadline)
        if result_27 is not None:
            if verify(result_27["U"], result_27["V"], result_27["W"]):
                return result_27
        return result_28

    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
