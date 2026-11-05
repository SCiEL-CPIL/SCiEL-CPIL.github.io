import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0


def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5

    U_cols = []
    V_cols = []
    W_cols = []

    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def add_term(i, k, j, coeff_u=1.0, coeff_v=1.0, coeff_w=1.0):
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        u_vec[i * m + k] = coeff_u
        v_vec[k * p + j] = coeff_v
        w_vec[j * n + i] = coeff_w
        add_rank1(u_vec, v_vec, w_vec)

    # ---- Part 1: 4x4x4 via Strassen (7 outer x 7 inner = 49 terms) ----
    strassen_products = [
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}},
        {'A': {2: 1, 3: 1}, 'B': {0: 1}},
        {'A': {0: 1}, 'B': {1: 1, 3: -1}},
        {'A': {3: 1}, 'B': {2: 1, 0: -1}},
        {'A': {0: 1, 1: 1}, 'B': {3: 1}},
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}},
        {'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1}},
    ]
    c_contributions = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1},
    ]

    def a_global(block, li, lk):
        if block == 0: return li, lk
        elif block == 1: return li, lk + 2
        elif block == 2: return li + 2, lk
        elif block == 3: return li + 2, lk + 2

    def b_global(block, lk, lj):
        if block == 0: return lk, lj
        elif block == 1: return lk, lj + 2
        elif block == 2: return lk + 2, lj
        elif block == 3: return lk + 2, lj + 2

    def c_global(block, li, lj):
        if block == 0: return li, lj
        elif block == 1: return li, lj + 2
        elif block == 2: return li + 2, lj
        elif block == 3: return li + 2, lj + 2

    inner_strassen = [
        {'A': [(0, 0, 1), (1, 1, 1)], 'B': [(0, 0, 1), (1, 1, 1)], 'C': [(0, 0, 1), (1, 1, 1)]},
        {'A': [(1, 0, 1), (1, 1, 1)], 'B': [(0, 0, 1)], 'C': [(1, 0, 1), (1, 1, -1)]},
        {'A': [(0, 0, 1)], 'B': [(0, 1, 1), (1, 1, -1)], 'C': [(0, 1, 1), (1, 1, 1)]},
        {'A': [(1, 1, 1)], 'B': [(1, 0, 1), (0, 0, -1)], 'C': [(0, 0, 1), (1, 0, 1)]},
        {'A': [(0, 0, 1), (0, 1, 1)], 'B': [(1, 1, 1)], 'C': [(0, 0, -1), (0, 1, 1)]},
        {'A': [(1, 0, 1), (0, 0, -1)], 'B': [(0, 0, 1), (0, 1, 1)], 'C': [(1, 1, 1)]},
        {'A': [(0, 1, 1), (1, 1, -1)], 'B': [(1, 0, 1), (1, 1, 1)], 'C': [(0, 0, 1)]},
    ]

    for prod_idx, prod in enumerate(strassen_products):
        a_form = prod['A']
        b_form = prod['B']
        c_contrib = c_contributions[prod_idx]
        for inner in inner_strassen:
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            for a_block, a_sign in a_form.items():
                for li, lk, s in inner['A']:
                    gi, gk = a_global(a_block, li, lk)
                    u_vec[gi * m + gk] += a_sign * s
            for b_block, b_sign in b_form.items():
                for lk, lj, s in inner['B']:
                    gk, gj = b_global(b_block, lk, lj)
                    v_vec[gk * p + gj] += b_sign * s
            for c_block, c_sign in c_contrib.items():
                for li, lj, s in inner['C']:
                    gi, gj = c_global(c_block, li, lj)
                    w_vec[gj * n + gi] += c_sign * s
            add_rank1(u_vec, v_vec, w_vec)

    # ---- Parts 2 & 3: the two 5th rows/cols (j=4 and k=4), schoolbook ----
    for i in range(4):
        for k in range(4):
            add_term(i, k, 4, 1.0, 1.0, 1.0)
    for i in range(4):
        for j in range(5):
            add_term(i, 4, j, 1.0, 1.0, 1.0)

    if len(U_cols) == 0:
        add_term(0, 0, 0, 1.0, 1.0, 1.0)

    # ---- Verification machinery ----
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    def compute_C(Uc, Vc, Wc):
        C = np.zeros((n, p))
        for r in range(len(Uc)):
            u = Uc[r]
            v = Vc[r]
            w = Wc[r]
            for i in range(n):
                for j in range(p):
                    s = 0.0
                    for k in range(m):
                        s += u[i * m + k] * v[k * p + j]
                    C[i, j] += s * w[j * n + i]
        return C

    def verify(Uc, Vc, Wc, atol=1e-6):
        return np.allclose(compute_C(Uc, Vc, Wc), C_expected, atol=atol)

    # ---- Aggressive compression: find true tensor rank via iterative dependency elimination ----
    def compress_aggressive(Uc, Vc, Wc, tol=1e-9, max_iters=30):
        """
        More aggressive compression:
        1. First pass: remove terms where U, V, or W are in the null space of the other two
        2. Second pass: find linear dependencies among terms by checking if removing
           one term and adjusting others preserves the decomposition
        3. Use SVD on the combined structure to find true rank
        """
        for _ in range(max_iters):
            if time.time() - _START > _DEADLINE - 20:
                break
            if len(Uc) <= 1:
                break
            
            Ua = np.array(Uc, dtype=float).T
            Va = np.array(Vc, dtype=float).T
            Wa = np.array(Wc, dtype=float).T
            R = len(Uc)
            
            U_s, _, U_Vt = np.linalg.svd(Ua, full_matrices=True)
            V_s, _, V_Vt = np.linalg.svd(Va, full_matrices=True)
            W_s, _, W_Vt = np.linalg.svd(Wa, full_matrices=True)
            rU = int(np.sum(U_s > tol))
            rV = int(np.sum(V_s > tol))
            rW = int(np.sum(W_s > tol))
            
            # Find terms that are redundant: a term r is redundant if
            # U[r] is in span of other U's AND V[r] is in span of other V's AND W[r] is in span of other W's
            # More precisely: if we can express term r as a linear combination of other terms
            
            # Strategy: for each term, check if it's in the null space of the "joint" constraint
            # A simpler approach: check if U[r] is linearly dependent on other U's, etc.
            
            # Find the term with the highest "redundancy score"
            best_drop = -1
            best_score = 0.0
            
            for r in range(R):
                # Check if U[r] is in the column space of U without column r
                if R <= 1:
                    break
                U_without = np.delete(Ua, r, axis=1)
                V_without = np.delete(Va, r, axis=1)
                W_without = np.delete(Wa, r, axis=1)
                
                # Distance of U[r] from col(U_without)
                if U_without.shape[1] == 0:
                    u_dist = np.linalg.norm(Ua[:, r])
                else:
                    # Project U[r] onto col(U_without)
                    Q, _ = np.linalg.qr(U_without)
                    proj = Q @ (Q.T @ Ua[:, r])
                    u_dist = np.linalg.norm(Ua[:, r] - proj)
                
                if V_without.shape[1] == 0:
                    v_dist = np.linalg.norm(Va[:, r])
                else:
                    Q, _ = np.linalg.qr(V_without)
                    proj = Q @ (Q.T @ Va[:, r])
                    v_dist = np.linalg.norm(Va[:, r] - proj)
                
                if W_without.shape[1] == 0:
                    w_dist = np.linalg.norm(Wa[:, r])
                else:
                    Q, _ = np.linalg.qr(W_without)
                    proj = Q @ (Q.T @ Wa[:, r])
                    w_dist = np.linalg.norm(Wa[:, r] - proj)
                
                # A term is "redundant" if all three distances are small
                # Use the max distance as the redundancy measure
                redundancy = max(u_dist, v_dist, w_dist)
                
                if redundancy < best_score or (redundancy == 0 and best_score > 0):
                    # Actually, we want to drop the term with the SMALLEST max distance
                    pass
            
            # Better approach: find the term whose removal causes the least damage
            # i.e., the term that is most "explainable" by the others
            best_drop = -1
            min_max_dist = np.inf
            
            for r in range(R):
                U_without = np.delete(Ua, r, axis=1)
                V_without = np.delete(Va, r, axis=1)
                W_without = np.delete(Wa, r, axis=1)
                
                if U_without.shape[1] == 0:
                    u_dist = np.linalg.norm(Ua[:, r])
                else:
                    Q, _ = np.linalg.qr(U_without)
                    proj = Q @ (Q.T @ Ua[:, r])
                    u_dist = np.linalg.norm(Ua[:, r] - proj)
                
                if V_without.shape[1] == 0:
                    v_dist = np.linalg.norm(Va[:, r])
                else:
                    Q, _ = np.linalg.qr(V_without)
                    proj = Q @ (Q.T @ Va[:, r])
                    v_dist = np.linalg.norm(Va[:, r] - proj)
                
                if W_without.shape[1] == 0:
                    w_dist = np.linalg.norm(Wa[:, r])
                else:
                    Q, _ = np.linalg.qr(W_without)
                    proj = Q @ (Q.T @ Wa[:, r])
                    w_dist = np.linalg.norm(Wa[:, r] - proj)
                
                max_dist = max(u_dist, v_dist, w_dist)
                if max_dist < min_max_dist:
                    min_max_dist = max_dist
                    best_drop = r
            
            if best_drop == -1 or min_max_dist > tol:
                # Try the original joint null space approach
                N_U = U_Vt[rU:]
                N_V = V_Vt[rV:]
                N_W = W_Vt[rW:]
                joint = []
                for a in N_U:
                    if np.allclose(a, 0.0, atol=1e-12):
                        continue
                    if (len(N_V) == 0 or np.allclose(N_V @ a, 0.0, atol=1e-10)) and \
                       (len(N_W) == 0 or np.allclose(N_W @ a, 0.0, atol=1e-10)):
                        joint.append(a)
                if not joint:
                    break
                J = np.array(joint)
                J_s, _, J_Vt = np.linalg.svd(J, full_matrices=False)
                rJ = int(np.sum(J_s > tol))
                keep = J_Vt[:rJ]
                drop_idx = int(np.argmax(np.abs(keep[0])))
                if abs(keep[0, drop_idx]) < 1e-12:
                    break
                best_drop = drop_idx
            
            if best_drop == -1:
                break
            
            new_U = [Uc[i] for i in range(len(Uc)) if i != best_drop]
            new_V = [Vc[i] for i in range(len(Vc)) if i != best_drop]
            new_W = [Wc[i] for i in range(len(Wc)) if i != best_drop]
            
            if verify(new_U, new_V, new_W):
                Uc, Vc, Wc = new_U, new_V, new_W
            else:
                break
        
        return Uc, Vc, Wc

    U_cols, V_cols, W_cols = compress_aggressive(U_cols, V_cols, W_cols)

    # Final safety check
    if not verify(U_cols, V_cols, W_cols):
        # Regenerate from scratch
        U_cols = []
        V_cols = []
        W_cols = []
        
        for prod_idx, prod in enumerate(strassen_products):
            a_form = prod['A']
            b_form = prod['B']
            c_contrib = c_contributions[prod_idx]
            for inner in inner_strassen:
                u_vec = np.zeros(n * m)
                v_vec = np.zeros(m * p)
                w_vec = np.zeros(p * n)
                for a_block, a_sign in a_form.items():
                    for li, lk, s in inner['A']:
                        gi, gk = a_global(a_block, li, lk)
                        u_vec[gi * m + gk] += a_sign * s
                for b_block, b_sign in b_form.items():
                    for lk, lj, s in inner['B']:
                        gk, gj = b_global(b_block, lk, lj)
                        v_vec[gk * p + gj] += b_sign * s
                for c_block, c_sign in c_contrib.items():
                    for li, lj, s in inner['C']:
                        gi, gj = c_global(c_block, li, lj)
                        w_vec[gj * n + gi] += c_sign * s
                add_rank1(u_vec, v_vec, w_vec)
        
        for i in range(4):
            for k in range(4):
                add_term(i, k, 4, 1.0, 1.0, 1.0)
        for i in range(4):
            for j in range(5):
                add_term(i, 4, j, 1.0, 1.0, 1.0)
        
        if not verify(U_cols, V_cols, W_cols):
            # Absolute fallback: schoolbook
            U_cols = []
            V_cols = []
            W_cols = []
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        add_term(i, k, j, 1.0, 1.0, 1.0)

    U = np.stack(U_cols, axis=1)
    V = np.stack(V_cols, axis=1)
    W = np.stack(W_cols, axis=1)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
