"""Matrix multiplication 4,4,9 tensor decomposition - Laderman-based with targeted search."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Build the target tensor for verification
    n_dim = n * m  # 16
    m_dim = m * p  # 36
    p_dim = p * n  # 36
    
    target = np.zeros(n_dim * m_dim * p_dim, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                flat_idx = a_idx * (m_dim * p_dim) + b_idx * p_dim + c_idx
                target[flat_idx] = 1.0
    
    def verify_decomposition(U, V, W, rank):
        """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == target tensor."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T.flatten(), target, atol=1e-10)
    
    # Laderman's rank-19 4x4x4 decomposition
    # Each entry is (u_dict, v_dict, w_dict) with (i,j) or (j,k) or (i,k) keys
    laderman_4x4 = [
        # 19 terms for 4x4x4
        ({(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}),
        ({(0,2):1,(0,3):1,(1,2):1,(1,3):1}, {(0,2):1,(0,3):1,(1,2):1,(1,3):1}, {(0,2):1,(0,3):1,(1,2):1,(1,3):1}),
        ({(2,0):1,(2,1):1,(3,0):1,(3,1):1}, {(2,0):1,(2,1):1,(3,0):1,(3,1):1}, {(2,0):1,(2,1):1,(3,0):1,(3,1):1}),
        ({(2,2):1,(2,3):1,(3,2):1,(3,3):1}, {(2,2):1,(2,3):1,(3,2):1,(3,3):1}, {(2,2):1,(2,3):1,(3,2):1,(3,3):1}),
        ({(0,2):-1,(0,3):-1,(1,2):-1,(1,3):-1}, {(0,2):1,(0,3):1,(1,2):1,(1,3):1}, {(0,2):-1,(0,3):-1,(1,2):-1,(1,3):-1}),
        ({(2,0):-1,(2,1):-1,(3,0):-1,(3,1):-1}, {(2,0):1,(2,1):1,(3,0):1,(3,1):1}, {(2,0):-1,(2,1):-1,(3,0):-1,(3,1):-1}),
        ({(0,0):1,(0,1):-1,(1,0):-1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):-1,(1,1):-1}, {(0,0):1,(0,1):-1,(1,0):-1,(1,1):1}),
        ({(0,2):1,(0,3):-1,(1,2):-1,(1,3):1}, {(0,2):1,(0,3):1,(1,2):-1,(1,3):-1}, {(0,2):1,(0,3):-1,(1,2):-1,(1,3):1}),
        ({(2,0):1,(2,1):-1,(3,0):-1,(3,1):1}, {(2,0):1,(2,1):1,(3,0):-1,(3,1):-1}, {(2,0):1,(2,1):-1,(3,0):-1,(3,1):1}),
        ({(2,2):1,(2,3):-1,(3,2):-1,(3,3):1}, {(2,2):1,(2,3):1,(3,2):-1,(3,3):-1}, {(2,2):1,(2,3):-1,(3,2):-1,(3,3):1}),
        ({(0,0):-1,(0,1):1,(1,0):1,(1,1):-1}, {(0,0):1,(0,1):-1,(1,0):1,(1,1):-1}, {(0,0):-1,(0,1):1,(1,0):1,(1,1):-1}),
        ({(0,2):-1,(0,3):1,(1,2):1,(1,3):-1}, {(0,2):1,(0,3):-1,(1,2):1,(1,3):-1}, {(0,2):-1,(0,3):1,(1,2):1,(1,3):-1}),
        ({(2,0):-1,(2,1):1,(3,0):1,(3,1):-1}, {(2,0):1,(2,1):-1,(3,0):1,(3,1):-1}, {(2,0):-1,(2,1):1,(3,0):1,(3,1):-1}),
        ({(2,2):-1,(2,3):1,(3,2):1,(3,3):-1}, {(2,2):1,(2,3):-1,(3,2):1,(3,3):-1}, {(2,2):-1,(2,3):1,(3,2):1,(3,3):-1}),
        ({(0,0):1,(1,1):-1}, {(0,0):1,(1,1):-1}, {(0,0):1,(1,1):-1}),
        ({(0,2):1,(1,3):-1}, {(0,2):1,(1,3):-1}, {(0,2):1,(1,3):-1}),
        ({(2,0):1,(3,1):-1}, {(2,0):1,(3,1):-1}, {(2,0):1,(3,1):-1}),
        ({(2,2):1,(3,3):-1}, {(2,2):1,(3,3):-1}, {(2,2):1,(3,3):-1}),
        ({(0,0):-1,(0,1):1,(1,0):-1,(1,1):1}, {(0,0):1,(0,1):-1,(1,0):1,(1,1):-1}, {(0,0):1,(0,1):-1,(1,0):-1,(1,1):1}),
    ]
    
    # Verify Laderman works for 4x4x4
    def build_laderman_4x4(A_i_base, A_j_base, B_j_base, B_k_base, C_i_base, C_k_base):
        """Build a 4x4x4 Laderman decomposition. Returns list of (u_vec, v_vec, w_vec)."""
        terms = []
        for u_c, v_c, w_c in laderman_4x4:
            u_vec = [0] * (n*m)
            v_vec = [0] * (m*p)
            w_vec = [0] * (p*n)
            
            for (loc_i, loc_j), coeff in u_c.items():
                gi = A_i_base + loc_i
                gj = A_j_base + loc_j
                u_vec[gi * m + gj] += coeff
            
            for (loc_j, loc_k), coeff in v_c.items():
                gj = B_j_base + loc_j
                gk = B_k_base + loc_k
                v_vec[gj * p + gk] += coeff
            
            for (loc_i, loc_k), coeff in w_c.items():
                gi = C_i_base + loc_i
                gk = C_k_base + loc_k
                w_vec[gk * n + gi] += coeff
            
            terms.append((u_vec, v_vec, w_vec))
        
        return terms
    
    # Build rank-54 solution: 19 + 19 + 16 = 54
    # Two 4x4x4 Laderman blocks + schoolbook for last column
    def construct_laderman_54():
        all_terms = []
        all_terms.extend(build_laderman_4x4(0, 0, 0, 0, 0, 0))
        all_terms.extend(build_laderman_4x4(0, 0, 0, 4, 0, 4))
        for i in range(n):
            for j in range(m):
                u_vec = [0] * (n*m)
                v_vec = [0] * (m*p)
                w_vec = [0] * (p*n)
                u_vec[i * m + j] = 1
                v_vec[j * p + 8] = 1
                w_vec[8 * n + i] = 1
                all_terms.append((u_vec, v_vec, w_vec))
        return all_terms
    
    # Build rank-114 Strassen solution as fallback
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    def build_strassen_4x4(A_i_base, A_j_base, B_j_base, B_k_base, C_i_base, C_k_base):
        terms = []
        def add_inner(A_blocks, B_blocks, C_blocks):
            for u_c, v_c, w_c in strassen_2x2:
                u_vec = [0] * (n*m)
                v_vec = [0] * (m*p)
                w_vec = [0] * (p*n)
                for (loc_i, loc_j), coeff in u_c.items():
                    for (br, bc, bc_coeff) in A_blocks:
                        gi = A_i_base + br * 2 + loc_i
                        gj = A_j_base + bc * 2 + loc_j
                        u_vec[gi * m + gj] += coeff * bc_coeff
                for (loc_j, loc_k), coeff in v_c.items():
                    for (br, bc, bc_coeff) in B_blocks:
                        gj = B_j_base + br * 2 + loc_j
                        gk = B_k_base + bc * 2 + loc_k
                        v_vec[gj * p + gk] += coeff * bc_coeff
                for (loc_i, loc_k), coeff in w_c.items():
                    for (br, bc, bc_coeff) in C_blocks:
                        gi = C_i_base + br * 2 + loc_i
                        gk = C_k_base + bc * 2 + loc_k
                        w_vec[gk * n + gi] += coeff * bc_coeff
                terms.append((u_vec, v_vec, w_vec))
        
        add_inner([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
        add_inner([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
        add_inner([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])
        add_inner([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
        add_inner([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
        add_inner([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
        add_inner([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
        return terms
    
    def construct_strassen_114():
        all_terms = []
        all_terms.extend(build_strassen_4x4(0, 0, 0, 0, 0, 0))
        all_terms.extend(build_strassen_4x4(0, 0, 0, 4, 0, 4))
        for i in range(n):
            for j in range(m):
                u_vec = [0] * (n*m)
                v_vec = [0] * (m*p)
                w_vec = [0] * (p*n)
                u_vec[i * m + j] = 1
                v_vec[j * p + 8] = 1
                w_vec[8 * n + i] = 1
                all_terms.append((u_vec, v_vec, w_vec))
        return all_terms
    
    def terms_to_matrices(all_terms):
        rank = len(all_terms)
        U = np.zeros((n_dim, rank))
        V = np.zeros((m_dim, rank))
        W = np.zeros((p_dim, rank))
        for r, (u_vec, v_vec, w_vec) in enumerate(all_terms):
            for i in range(n_dim):
                if u_vec[i] != 0:
                    U[i, r] = u_vec[i]
            for i in range(m_dim):
                if v_vec[i] != 0:
                    V[i, r] = v_vec[i]
            for i in range(p_dim):
                if w_vec[i] != 0:
                    W[i, r] = w_vec[i]
        return U, V, W, rank
    
    # Try Laderman first
    all_terms_laderman = construct_laderman_54()
    U_l, V_l, W_l, rank_l = terms_to_matrices(all_terms_laderman)
    
    if verify_decomposition(U_l, V_l, W_l, rank_l):
        U, V, W, rank = U_l, V_l, W_l, rank_l
    else:
        # Fallback to Strassen
        all_terms_s = construct_strassen_114()
        U, V, W, rank = terms_to_matrices(all_terms_s)
        if not verify_decomposition(U, V, W, rank):
            # Ultimate fallback: schoolbook
            rank_sb = n * m * p
            U_sb = np.zeros((n_dim, rank_sb))
            V_sb = np.zeros((m_dim, rank_sb))
            W_sb = np.zeros((p_dim, rank_sb))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j, col] = 1
                        V_sb[j * p + k, col] = 1
                        W_sb[k * n + i, col] = 1
                        col += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Targeted search for lower rank
    def try_targeted_search(target_rank, max_iters=30, seed=42):
        import random
        rng = random.Random(seed)
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            U = np.zeros((n_dim, target_rank))
            V = np.zeros((m_dim, target_rank))
            
            for r in range(target_rank):
                u_count = rng.randint(2, 4)
                u_support = rng.sample(range(n_dim), u_count)
                for pos in u_support:
                    U[pos, r] = rng.choice([-1, 1])
                
                v_count = rng.randint(2, 4)
                v_support = rng.sample(range(m_dim), v_count)
                for pos in v_support:
                    V[pos, r] = rng.choice([-1, 1])
            
            # Build design matrix: D[a*m_dim+b, r] = U[a,r]*V[b,r]
            D = np.zeros((n_dim * m_dim, target_rank))
            for a in range(n_dim):
                for b in range(m_dim):
                    D[a * m_dim + b, :] = U[a, :] * V[b, :]
            
            T_reshaped = target.reshape(n_dim * m_dim, p_dim)
            
            try:
                X, residuals, rank_sys, sv = np.linalg.lstsq(D, T_reshaped.T, rcond=None)
                W_sol = X.T
            except Exception:
                continue
            
            # Check exactness
            T_check = np.einsum('ar,br,cr->abc', U, V, W_sol)
            if np.allclose(T_check.flatten(), target, atol=1e-9):
                # Check half-integer
                all_vals = np.concatenate([U.flatten(), V.flatten(), W_sol.flatten()])
                scaled = all_vals * 2
                if np.allclose(scaled, np.round(scaled), atol=1e-6):
                    U_exact = np.round(U * 2) / 2.0
                    V_exact = np.round(V * 2) / 2.0
                    W_exact = np.round(W_sol * 2) / 2.0
                    T_final = np.einsum('ar,br,cr->abc', U_exact, V_exact, W_exact)
                    if np.allclose(T_final.flatten(), target, atol=1e-10):
                        return U_exact, V_exact, W_exact, target_rank
        
        return None
    
    # Also try structured search: fix U,V from Laderman pattern, solve for W
    def try_structured_search(target_rank, max_iters=20):
        """Use Laderman-like sparse patterns for U and V, solve for W."""
        import random
        rng = random.Random(123)
        
        # Generate Laderman-inspired support patterns
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            U = np.zeros((n_dim, target_rank))
            V = np.zeros((m_dim, target_rank))
            
            for r in range(target_rank):
                # U pattern: select a 2x2 or 4x4 block
                if rng.random() < 0.5:
                    # 2x2 block
                    bi = rng.randint(0, 1) * 2
                    bj = rng.randint(0, 1) * 2
                    for di in range(2):
                        for dj in range(2):
                            U[(bi+di)*m + (bj+dj), r] = rng.choice([-1, 1])
                else:
                    # 4x4 full
                    for di in range(4):
                        for dj in range(4):
                            U[di*m + dj, r] = rng.choice([-1, 1])
                
                # V pattern: select columns
                if rng.random() < 0.5:
                    bk = rng.randint(0, 2) * 3  # 0, 3, or 6
                    for dj in range(4):
                        for dk in range(min(3, 9-bk)):
                            V[dj*p + (bk+dk), r] = rng.choice([-1, 1])
                else:
                    bk = rng.randint(0, 5)
                    for dj in range(4):
                        for dk in range(min(4, 9-bk)):
                            V[dj*p + (bk+dk), r] = rng.choice([-1, 1])
            
            D = np.zeros((n_dim * m_dim, target_rank))
            for a in range(n_dim):
                for b in range(m_dim):
                    D[a * m_dim + b, :] = U[a, :] * V[b, :]
            
            T_reshaped = target.reshape(n_dim * m_dim, p_dim)
            
            try:
                X, residuals, rank_sys, sv = np.linalg.lstsq(D, T_reshaped.T, rcond=None)
                W_sol = X.T
            except Exception:
                continue
            
            T_check = np.einsum('ar,br,cr->abc', U, V, W_sol)
            if np.allclose(T_check.flatten(), target, atol=1e-9):
                all_vals = np.concatenate([U.flatten(), V.flatten(), W_sol.flatten()])
                scaled = all_vals * 2
                if np.allclose(scaled, np.round(scaled), atol=1e-6):
                    U_exact = np.round(U * 2) / 2.0
                    V_exact = np.round(V * 2) / 2.0
                    W_exact = np.round(W_sol * 2) / 2.0
                    T_final = np.einsum('ar,br,cr->abc', U_exact, V_exact, W_exact)
                    if np.allclose(T_final.flatten(), target, atol=1e-10):
                        return U_exact, V_exact, W_exact, target_rank
        
        return None
    
    best_result = None
    for target_rank in range(rank - 1, max(rank - 10, 30), -1):
        if time.time() > deadline - 20:
            break
        result = try_targeted_search(target_rank, max_iters=15, seed=target_rank)
        if result is not None:
            best_result = result
            break
        result = try_structured_search(target_rank, max_iters=15)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U, V, W, rank_out = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
