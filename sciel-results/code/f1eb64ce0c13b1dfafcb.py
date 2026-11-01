"""Matrix multiplication 2x5x6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    def verify_decomp(U, V, W, n, m, p):
        """Verify exact tensor decomposition."""
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i * m + j, :]
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1.0) > 1e-9:
                        return False
        return True

    def build_strassen_block(U, V, W, col, i0, j0, k0):
        """Add Strassen's 7 multiplications for 2x2x2 block starting at (i0,j0,k0).
        Returns new col index."""
        # Indices for the 2x2x2 block
        u00 = i0 * m + j0
        u01 = i0 * m + (j0 + 1)
        u10 = (i0 + 1) * m + j0
        u11 = (i0 + 1) * m + (j0 + 1)
        
        v00 = j0 * p + k0
        v01 = j0 * p + (k0 + 1)
        v10 = (j0 + 1) * p + k0
        v11 = (j0 + 1) * p + (k0 + 1)
        
        w00 = k0 * n + i0
        w01 = k0 * n + (i0 + 1)
        w10 = (k0 + 1) * n + i0
        w11 = (k0 + 1) * n + (i0 + 1)
        
        def add(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, c in u_coeffs:
                U[idx][col] += c
            for idx, c in v_coeffs:
                V[idx][col] += c
            for idx, c in w_coeffs:
                W[idx][col] += c
            col += 1
        
        # M1 = (A00+A11)(B00+B11)
        add([(u00, 1), (u11, 1)], [(v00, 1), (v11, 1)], [(w00, 1), (w11, 1)])
        # M2 = (A10+A11)B00
        add([(u10, 1), (u11, 1)], [(v00, 1)], [(w10, 1), (w11, -1)])
        # M3 = A00(B01-B11)
        add([(u00, 1)], [(v01, 1), (v11, -1)], [(w01, 1), (w11, 1)])
        # M4 = A11(B10-B00)
        add([(u11, 1)], [(v10, 1), (v00, -1)], [(w00, 1), (w10, 1)])
        # M5 = (A00+A01)B11
        add([(u00, 1), (u01, 1)], [(v11, 1)], [(w00, -1), (w01, 1)])
        # M6 = (A10-A00)(B00+B01)
        add([(u10, 1), (u00, -1)], [(v00, 1), (v01, 1)], [(w11, 1)])
        # M7 = (A01-A11)(B10+B11)
        add([(u01, 1), (u11, -1)], [(v10, 1), (v11, 1)], [(w00, 1)])
        
        return col

    def build_decomp(rank, strassen_blocks):
        """Build decomposition with given rank and Strassen blocks.
        strassen_blocks: list of (i0, j0, k0) tuples for 2x2x2 Strassen blocks."""
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Determine which standard terms are covered by Strassen blocks
        covered = set()
        for (i0, j0, k0) in strassen_blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((i0 + di, j0 + dj, k0 + dk))
        
        col = 0
        # Add standard terms not in any Strassen block
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered:
                        continue
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
        
        # Add Strassen blocks
        for (i0, j0, k0) in strassen_blocks:
            col = build_strassen_block(U, V, W, col, i0, j0, k0)
        
        return U, V, W, col

    # Strategy: try different configurations of Strassen blocks
    # For 2x5x6, we can place 2x2x2 Strassen blocks at various positions
    # Each block saves 1 rank (8 standard -> 7 Strassen)
    
    # Maximum possible Strassen blocks: limited by dimensions
    # i: 0-1 (only 1 position: i0=0)
    # j: 0-4 (positions: j0=0,1,2,3)
    # k: 0-5 (positions: k0=0,1,2,3,4)
    
    # Try configurations with 2 non-overlapping Strassen blocks
    # Block 1 at (0, j0_1, k0_1), Block 2 at (0, j0_2, k0_2)
    # They must not overlap in the j-k plane
    
    best_rank = 60
    best_result = None
    
    # Generate candidate block placements
    # Two blocks must have non-overlapping j-ranges and k-ranges
    # Block at (i0=0, j0, k0) covers j in {j0, j0+1}, k in {k0, k0+1}
    
    candidate_configs = []
    
    # Try all pairs of non-overlapping 2x2 blocks in the 5x6 j-k grid
    for j0_1 in range(4):  # 0-3
        for k0_1 in range(5):  # 0-4
            for j0_2 in range(4):  # 0-3
                for k0_2 in range(5):  # 0-4
                    # Check non-overlap: j-ranges must be disjoint OR k-ranges must be disjoint
                    j_overlap = not (j0_1 + 1 < j0_2 or j0_2 + 1 < j0_1)
                    k_overlap = not (k0_1 + 1 < k0_2 or k0_2 + 1 < k0_1)
                    
                    if j_overlap and k_overlap:
                        continue  # Blocks overlap, skip
                    
                    # Both blocks have i0=0 (since n=2, only one position)
                    config = [(0, j0_1, k0_1), (0, j0_2, k0_2)]
                    if config not in candidate_configs:
                        candidate_configs.append(config)
    
    # Also try single Strassen block (rank 59) as fallback
    for j0 in range(4):
        for k0 in range(5):
            candidate_configs.append([(0, j0, k0)])
    
    # Sort by expected rank (more blocks = lower rank)
    candidate_configs.sort(key=lambda c: -len(c))
    
    for config in candidate_configs:
        if time.time() > deadline:
            break
        
        num_blocks = len(config)
        rank = 60 - num_blocks  # Each Strassen block saves 1 rank
        
        if rank >= best_rank:
            continue
        
        U, V, W, final_col = build_decomp(rank, config)
        
        if final_col != rank:
            continue
        
        if verify_decomp(U, V, W, n, m, p):
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
            
            # If we achieved rank 58, we can stop early
            if rank <= 58:
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: single Strassen block
    U, V, W, _ = build_decomp(59, [(0, 0, 0)])
    if verify_decomp(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Last resort: rank 60 (standard)
    U = [[0] * 60 for _ in range(n * m)]
    V = [[0] * 60 for _ in range(m * p)]
    W = [[0] * 60 for _ in range(p * n)]
    for idx, (i, j, k) in enumerate([(i, j, k) for i in range(n) for j in range(m) for k in range(p)]):
        U[i * m + j][idx] = 1
        V[j * p + k][idx] = 1
        W[k * n + i][idx] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
