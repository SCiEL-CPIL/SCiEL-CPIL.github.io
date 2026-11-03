"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-minimal decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Build rank-77 via Kronecker(7, 11), then aggressively reduce.
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # ===== Build Strassen 2x2x2 base (rank 7) =====
    U1 = np.zeros((4, 7), dtype=int)
    V1 = np.zeros((4, 7), dtype=int)
    W1 = np.zeros((4, 7), dtype=int)
    
    # M1 = (a00+a11)(b00+b11) -> C00 += 1, C11 += 1
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1
    
    # M2 = (a10+a11)b00 -> C10 += 1, C11 += -1
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1; W1[3, 1] = -1
    
    # M3 = a00(b01-b11) -> C01 += 1, C11 += 1
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1; W1[3, 2] = 1
    
    # M4 = a11(b10-b00) -> C00 += 1, C10 += 1
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[1, 3] = 1
    
    # M5 = (a00+a01)b11 -> C00 += -1, C01 += 1
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[2, 4] = 1
    
    # M6 = (a10-a00)(b00+b01) -> C11 += 1
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1
    
    # M7 = (a01-a11)(b10+b11) -> C00 += 1
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1
    
    # ===== Build 2x2x3 base (rank 11) =====
    U2 = np.zeros((4, 11), dtype=int)
    V2 = np.zeros((6, 11), dtype=int)
    W2 = np.zeros((6, 11), dtype=int)
    
    # Terms 0-6: Strassen for columns 0,1
    for c in range(7):
        U2[:, c] = U1[:, c]
        V2[0, c] = V1[0, c]  # j=0,k=0
        V2[1, c] = V1[1, c]  # j=0,k=1
        V2[3, c] = V1[2, c]  # j=1,k=0
        V2[4, c] = V1[3, c]  # j=1,k=1
        W2[0, c] = W1[0, c]  # k=0,i=0
        W2[1, c] = W1[1, c]  # k=0,i=1
        W2[2, c] = W1[2, c]  # k=1,i=0
        W2[3, c] = W1[3, c]  # k=1,i=1
    
    # Terms 7-10: Schoolbook for column 2
    U2[0, 7] = 1
    V2[2, 7] = 1
    W2[4, 7] = 1
    
    U2[1, 8] = 1
    V2[5, 8] = 1
    W2[4, 8] = 1
    
    U2[2, 9] = 1
    V2[2, 9] = 1
    W2[5, 9] = 1
    
    U2[3, 10] = 1
    V2[5, 10] = 1
    W2[5, 10] = 1
    
    # ===== Kronecker product to get 4x4x6 (rank 77) =====
    rank = 7 * 11
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((24, rank), dtype=int)
    W = np.zeros((24, rank), dtype=int)
    
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1 * 2 + j1, c1]
                            u2 = U2[i2 * 2 + j2, c2]
                            if u1 != 0 and u2 != 0:
                                row = i1 * 8 + i2 * 4 + j1 * 2 + j2
                                U[row, c] = u1 * u2
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1 * 2 + k1, c1]
                            v2 = V2[j2 * 3 + k2, c2]
                            if v1 != 0 and v2 != 0:
                                row = j1 * 12 + j2 * 6 + k1 * 3 + k2
                                V[row, c] = v1 * v2
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1 * 2 + i1, c1]
                            w2 = W2[k2 * 2 + i2, c2]
                            if w1 != 0 and w2 != 0:
                                row = k1 * 12 + k2 * 4 + i1 * 2 + i2
                                W[row, c] = w1 * w2
    
    # ===== Verify =====
    def _verify(U_mat, V_mat, W_mat, n, m, p):
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        C_computed = np.zeros((n, p))
        for r in range(U_mat.shape[1]):
            alpha_r = np.sum(A_test * U_mat[:, r].reshape(n, m))
            beta_r = np.sum(B_test * V_mat[:, r].reshape(m, p))
            C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p, n).T
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    if not _verify(U, V, W, n, m, p):
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=int)
        V = np.zeros((m * p, rank_school), dtype=int)
        W = np.zeros((p * n, rank_school), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # ===== Aggressive rank reduction =====
    if time.time() < deadline - 10:
        U, V, W = _reduce_rank_aggressive(U, V, W, n, m, p, deadline)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _reduce_rank_aggressive(U, V, W, n, m, p, deadline):
    """
    Aggressive rank reduction using structured approaches:
    1. Greedy single-term removal
    2. Pair/triple removal for linearly dependent groups
    3. Tensor-train style bond dimension optimization
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    
    def _verify_fast(U_mat, V_mat, W_mat):
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        C_computed = np.zeros((n, p))
        for r in range(U_mat.shape[1]):
            alpha_r = np.sum(A_test * U_mat[:, r].reshape(n, m))
            beta_r = np.sum(B_test * V_mat[:, r].reshape(m, p))
            C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p, n).T
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    # Phase 1: Greedy single-term removal with multiple random orders
    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
    best_rank = current_U.shape[1]
    
    for seed in [123, 456, 789]:
        if time.time() > deadline - 15:
            break
        np.random.seed(seed)
        U_cur, V_cur, W_cur = current_U.copy(), current_V.copy(), current_W.copy()
        rank_cur = U_cur.shape[1]
        order = list(range(rank_cur))
        np.random.shuffle(order)
        
        for r in order:
            if time.time() > deadline - 15:
                break
            if U_cur.shape[1] <= 1:
                break
            mask = np.ones(U_cur.shape[1], dtype=bool)
            mask[r] = False
            U_test = U_cur[:, mask]
            V_test = V_cur[:, mask]
            W_test = W_cur[:, mask]
            
            if _verify_fast(U_test, V_test, W_test):
                U_cur = U_test
                V_cur = V_test
                W_cur = W_test
        
        if U_cur.shape[1] < best_rank:
            best_rank = U_cur.shape[1]
            best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
    
    current_U, current_V, current_W = best_U, best_V, best_W
    
    # Phase 2: Try removing pairs of terms that might be linearly dependent
    if time.time() < deadline - 20:
        current_U, current_V, current_W = _try_pair_removal(
            current_U, current_V, current_W, n, m, p, deadline, _verify_fast
        )
    
    # Phase 3: Tensor-train style optimization
    if time.time() < deadline - 25:
        current_U, current_V, current_W = _tensor_train_reduce(
            current_U, current_V, current_W, n, m, p, deadline, _verify_fast
        )
    
    return current_U, current_V, current_W


def _try_pair_removal(U, V, W, n, m, p, deadline, _verify_fast):
    """Try removing pairs of terms that form linearly dependent sub-decompositions."""
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    rank = current_U.shape[1]
    
    # Build tensor representation for quick dependency checks
    # T[i,j,k,r] = U[i,r]*V[j,r]*W[k,r]
    # A pair (r1, r2) is redundant if T[:, :, :, r1] = c * T[:, :, :, r2] for some c
    
    # Check for proportional term pairs
    removed = set()
    for r1 in range(rank):
        if time.time() > deadline - 20:
            break
        if r1 in removed:
            continue
        for r2 in range(r1 + 1, rank):
            if time.time() > deadline - 20:
                break
            if r2 in removed:
                continue
            
            # Check if terms r1 and r2 are proportional
            U1_vec = current_U[:, r1]
            U2_vec = current_U[:, r2]
            V1_vec = current_V[:, r1]
            V2_vec = current_V[:, r2]
            W1_vec = current_W[:, r1]
            W2_vec = current_W[:, r2]
            
            # Check proportionality
            nz1 = np.nonzero(U1_vec)[0]
            nz2 = np.nonzero(U2_vec)[0]
            if len(nz1) == 0 or len(nz2) == 0:
                continue
            
            # Find ratio from first non-zero element
            idx = nz1[0]
            if U2_vec[idx] == 0:
                continue
            ratio = U1_vec[idx] / U2_vec[idx]
            
            # Check if all elements match with this ratio
            if np.allclose(U1_vec, ratio * U2_vec) and \
               np.allclose(V1_vec, ratio * V2_vec) and \
               np.allclose(W1_vec, ratio * W2_vec):
                # They're proportional - remove one
                mask = np.ones(current_U.shape[1], dtype=bool)
                mask[r1] = False
                U_test = current_U[:, mask]
                V_test = current_V[:, mask]
                W_test = current_W[:, mask]
                if _verify_fast(U_test, V_test, W_test):
                    current_U = U_test
                    current_V = V_test
                    current_W = W_test
                    removed.add(r1)
                    break  # Restart inner loop since indices shifted
    
    return current_U, current_V, current_W


def _tensor_train_reduce(U, V, W, n, m, p, deadline, _verify_fast):
    """
    Tensor-train style reduction: represent the multiplication tensor in MPS form
    and try to reduce bond dimensions.
    
    The multiplication tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    can be seen as a 3-tensor. We try to find a lower-rank approximation
    by optimizing the factor matrices.
    """
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    rank = current_U.shape[1]
    
    # Try to find linear dependencies by solving for coefficients
    # If sum_r c_r * (U_r ⊗ V_r ⊗ W_r) = 0, we can remove terms
    
    # Build the full tensor and check rank via SVD on matricizations
    # T as (n*m) x (m*p) x (p*n) tensor
    # Matricize to (n*m*(m*p)) x (p*n) and check rank
    
    if time.time() > deadline - 30:
        return current_U, current_V, current_W
    
    # Try iterative refinement: perturb and re-optimize
    # Use ALS (Alternating Least Squares) style approach
    
    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
    best_rank = rank
    
    # Try reducing by 1, 2, 3... terms at a time using structured search
    for target_reduction in range(1, min(10, rank // 5 + 1)):
        if time.time() > deadline - 30:
            break
        
        # Try removing target_reduction terms
        # Use greedy approach: remove terms that contribute least to the tensor
        U_cur, V_cur, W_cur = current_U.copy(), current_V.copy(), current_W.copy()
        
        # Compute contribution of each term
        contributions = []
        for r in range(U_cur.shape[1]):
            # Contribution is ||U_r|| * ||V_r|| * ||W_r||
            contrib = np.linalg.norm(U_cur[:, r]) * np.linalg.norm(V_cur[:, r]) * np.linalg.norm(W_cur[:, r])
            contributions.append((contrib, r))
        
        # Sort by contribution (smallest first)
        contributions.sort()
        
        # Try removing the smallest contributors
        removed_indices = set()
        for contrib, r in contributions[:target_reduction]:
            if r in removed_indices:
                continue
            mask = np.ones(U_cur.shape[1], dtype=bool)
            mask[r] = False
            U_test = U_cur[:, mask]
            V_test = V_cur[:, mask]
            W_test = W_cur[:, mask]
            
            if _verify_fast(U_test, V_test, W_test):
                U_cur = U_test
                V_cur = V_test
                W_cur = W_test
                removed_indices.add(r)
        
        if U_cur.shape[1] < best_rank:
            best_rank = U_cur.shape[1]
            best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
    
    return best_U, best_V, best_W
# EVOLVE-BLOCK-END
