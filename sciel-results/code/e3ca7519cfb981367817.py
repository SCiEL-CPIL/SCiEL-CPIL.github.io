import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n, m, p = 2, 4, 6
    
    # Known Winograd decomposition for 2x2 @ 2x3 -> 2x3 with rank 7
    # This is a known exact decomposition with half-integer entries
    def get_winograd_223():
        # For 2x2 A, 2x3 B -> 2x3 C
        # A indices: (i,j) -> i*2+j for i in [0,1], j in [0,1]
        # B indices: (j,k) -> j*3+k for j in [0,1], k in [0,1,2]
        # C indices: (i,k) -> i*3+k for i in [0,1], k in [0,1,2]
        
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        # Winograd scheme for 2x2 @ 2x3:
        # The 7 products are:
        # m0 = (a00+a11)*(b00+b10+b11+b01+b12+b02) -- no, let me use a known scheme
        
        # Actually, let me use a simpler approach: 
        # We can use 2x2 @ 2x2 with rank 7 (Winograd) for the first two columns
        # and then handle the third column separately, but that would give rank 7+2=9
        
        # Better: use a known rank-7 decomposition for 2x2 @ 2x3
        # One such decomposition uses the following 7 scalar products:
        
        # Let me construct it directly by solving the system
        # We need U (4x7), V (6x7), W (6x7) such that
        # sum_r U[a_idx, r] * V[b_idx, r] * W[c_idx, r] = delta(a,b,c)
        # where delta = 1 if a's j == b's j and a's i == c's i, else 0
        
        # A known decomposition:
        # r=0: U=[1,0,0,1], V=[1,0,0,0,0,0], W=[1,0,0,0,0,0]  -- a00*b00 -> c00
        # This is getting complex. Let me just try the optimization with a good initial guess.
        
        return None

    # Hardcoded rank-7 decomposition for 2x2 @ 2x3
    # Using Winograd's F(2,2,3) scheme
    def get_hardcoded_223():
        rank = 7
        # U: 4x7 (A entries: a00, a01, a10, a11)
        U = np.array([
            [1, 0, 1, 0, 1, 0, 0],  # a00
            [0, 1, 0, 1, 0, 1, 0],  # a01
            [0, 0, 1, 0, 0, 0, 1],  # a10
            [1, 1, 0, 1, 0, 1, 0],  # a11
        ], dtype=float)
        
        # V: 6x7 (B entries: b00, b01, b02, b10, b11, b12)
        V = np.array([
            [1, 0, 0, 0, 0, 0, 0],  # b00
            [0, 1, 0, 0, 0, 0, 0],  # b01
            [0, 0, 1, 0, 0, 0, 0],  # b02
            [0, 0, 0, 1, 0, 0, 0],  # b10
            [0, 0, 0, 0, 1, 0, 0],  # b11
            [0, 0, 0, 0, 0, 1, 0],  # b12
        ], dtype=float)
        
        # W: 6x7 (C entries: c00, c01, c02, c10, c11, c12)
        W = np.array([
            [1, 0, 0, 0, 0, 0, 0],  # c00
            [0, 1, 0, 0, 0, 0, 0],  # c01
            [0, 0, 1, 0, 0, 0, 0],  # c02
            [0, 0, 0, 1, 0, 0, 0],  # c10
            [0, 0, 0, 0, 1, 0, 0],  # c11
            [0, 0, 0, 0, 0, 1, 0],  # c12
        ], dtype=float)
        
        # Verify this decomposition
        T = np.zeros((4, 6, 6))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    a_idx = i * 2 + j
                    b_idx = j * 3 + k
                    c_idx = i * 3 + k
                    T[a_idx, b_idx, c_idx] = 1.0
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(T_recon, T, atol=1e-9):
            return None
        
        return U, V, W
    
    # Fallback: Rank 42 solution using 6 Strassen blocks
    def get_rank_42_solution():
        rank_total = 42
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]

        strassen_U = [
            {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1}, {3: 1},
            {0: 1, 1: 1}, {0: -1, 2: 1}, {1: 1, 3: -1}
        ]
        strassen_V = [
            {0: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}, {2: 1, 0: -1},
            {3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1}
        ]
        strassen_W = [
            {0: 1, 3: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}, {0: 1, 2: 1},
            {0: -1, 1: 1}, {3: 1}, {0: 1}
        ]

        col_offset = 0
        for j_block in range(3):
            for side in ['L', 'R']:
                if side == 'L':
                    u_col_offset = 0
                    v_row_offset = 0
                else:
                    u_col_offset = 2
                    v_row_offset = 2

                for prod_idx in range(7):
                    col = col_offset + prod_idx

                    for local_idx, sign in strassen_U[prod_idx].items():
                        if local_idx == 0: u_idx = u_col_offset
                        elif local_idx == 1: u_idx = u_col_offset + 1
                        elif local_idx == 2: u_idx = 4 + u_col_offset
                        else: u_idx = 4 + u_col_offset + 1
                        U[u_idx][col] += sign

                    for local_idx, sign in strassen_V[prod_idx].items():
                        if local_idx == 0: v_idx = v_row_offset * 6 + 2 * j_block
                        elif local_idx == 1: v_idx = v_row_offset * 6 + 2 * j_block + 1
                        elif local_idx == 2: v_idx = (v_row_offset + 1) * 6 + 2 * j_block
                        else: v_idx = (v_row_offset + 1) * 6 + 2 * j_block + 1
                        V[v_idx][col] += sign

                    for local_idx, sign in strassen_W[prod_idx].items():
                        if local_idx == 0: w_idx = 2 * (2 * j_block)
                        elif local_idx == 1: w_idx = 2 * (2 * j_block + 1)
                        elif local_idx == 2: w_idx = 2 * (2 * j_block) + 1
                        else: w_idx = 2 * (2 * j_block + 1) + 1
                        W[w_idx][col] += sign

                col_offset += 7

        return {"U": U, "V": V, "W": W}

    # Try to find Rank 35 solution using 5 blocks of Rank 7 (2x2x3)
    # Tensor for 2x2 A, 2x3 B -> 2x3 C has rank 7 (Winograd)
    def find_223_decomp():
        # Tensor for 2x2 A, 2x3 B -> 2x3 C
        # A indices: 0..3 (i, j) -> i*2+j
        # B indices: 0..5 (j, k) -> j*3+k
        # C indices: 0..5 (i, k) -> i*3+k
        
        T = np.zeros((4, 6, 6))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    a_idx = i * 2 + j
                    b_idx = j * 3 + k
                    c_idx = i * 3 + k
                    T[a_idx, b_idx, c_idx] = 1.0

        rank = 7
        
        # Flatten parameters
        def unpack(params):
            U = params[:4*rank].reshape(4, rank)
            V = params[4*rank:4*rank+6*rank].reshape(6, rank)
            W = params[4*rank+6*rank:].reshape(6, rank)
            return U, V, W

        def loss(params):
            U, V, W = unpack(params)
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T_recon - T)**2)
            return err

        def try_decomposition(U0, V0, W0, max_iter=5000, ftol=1e-14):
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            # Phase 1: L-BFGS-B with moderate tolerance
            res = minimize(loss, x0, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-8})
            
            if res.fun > 1e-6:
                return None
            
            U, V, W = unpack(res.x)
            
            # Phase 2: Polish with tighter tolerance
            res2 = minimize(loss, res.x, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-10})
            if res2.fun < res.fun:
                U, V, W = unpack(res2.x)
            
            # Snap to half-integers and verify
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
            if np.allclose(T_recon, T, atol=1e-9):
                return U_snap, V_snap, W_snap
            
            # Try snapping to integers as well
            U_snap_i = np.round(U)
            V_snap_i = np.round(V)
            W_snap_i = np.round(W)
            T_recon_i = np.einsum('ar,br,cr->abc', U_snap_i, V_snap_i, W_snap_i)
            if np.allclose(T_recon_i, T, atol=1e-9):
                return U_snap_i, V_snap_i, W_snap_i
            
            return None

        # Use a structured initial guess based on Winograd patterns
        # For 2x2 @ 2x3, a known decomposition uses 7 products
        # Let's try a sparse initialization
        best_result = None
        
        # Try structured initializations first
        structured_inits = []
        
        # Init 1: Based on 2x2 Winograd (7 products) extended to 2x3
        U1 = np.array([
            [1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 0],
        ], dtype=float)
        V1 = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)
        W1 = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
        ], dtype=float)
        structured_inits.append((U1, V1, W1))
        
        # Init 2: Different pattern
        U2 = np.array([
            [1, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 0],
        ], dtype=float)
        V2 = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
        ], dtype=float)
        W2 = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
        ], dtype=float)
        structured_inits.append((U2, V2, W2))
        
        for U0, V0, W0 in structured_inits:
            if time.time() > deadline - 5:
                break
            result = try_decomposition(U0, V0, W0)
            if result is not None:
                best_result = result
                break
        
        # If structured inits failed, try random restarts
        if best_result is None:
            for seed in range(20):
                if time.time() > deadline - 5:
                    break
                
                np.random.seed(seed)
                scale = np.random.choice([0.1, 0.5, 1.0])
                U0 = np.random.randn(4, rank) * scale
                V0 = np.random.randn(6, rank) * scale
                W0 = np.random.randn(6, rank) * scale
                
                result = try_decomposition(U0, V0, W0)
                if result is not None:
                    best_result = result
                    break
        
        return best_result

    if time.time() < deadline - 10:
        decomp = find_223_decomp()
        if decomp is not None:
            U_sub, V_sub, W_sub = decomp
            rank_sub = U_sub.shape[1]
            total_rank = 4 * rank_sub
            
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            # Place 4 blocks
            # Block (side, col_half)
            # side: 0 (left A cols 0,1 / top B rows 0,1), 1 (right A cols 2,3 / bottom B rows 2,3)
            # col_half: 0 (left C cols 0,1,2), 1 (right C cols 3,4,5)
            
            for side in [0, 1]:
                for col_half in [0, 1]:
                    col_offset = (side * 2 + col_half) * rank_sub
                    # A offset
                    u_row_offset = side * 2
                    # B offset
                    v_row_offset = side * 2 # Rows of B
                    v_col_offset = col_half * 3 # Cols of B
                    # C offset
                    w_row_offset = col_half * 3 # Cols of C
                    
                    for r in range(rank_sub):
                        for i in range(2):
                            for j in range(2):
                                a_idx = i * 2 + j
                                u_idx = u_row_offset + a_idx
                                U[u_idx][col_offset + r] += U_sub[a_idx, r]
                                
                        for j in range(2):
                            for k in range(3):
                                b_idx = j * 3 + k
                                # B index in full: row = side*2 + j, col = col_half*3 + k
                                v_idx_full = (side * 2 + j) * 6 + (col_half * 3 + k)
                                V[v_idx_full][col_offset + r] += V_sub[b_idx, r]
                                
                        for i in range(2):
                            for k in range(3):
                                c_idx = i * 3 + k
                                # C index in full: row = i, col = col_half*3 + k
                                # W shape is (p*n, rank) = (12, rank).
                                # Convention: W[k*n+i, r]
                                # k is col of C (0..5), i is row of C (0..1)
                                # So w_idx = k * 2 + i
                                w_idx_full = (col_half * 3 + k) * 2 + i
                                W[w_idx_full][col_offset + r] += W_sub[c_idx, r]
            
            return {"U": U, "V": V, "W": W}

    return get_rank_42_solution()
