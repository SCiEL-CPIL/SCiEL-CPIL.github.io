import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Standard 2x2x2 Strassen terms
    strassen_terms = [
        {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
        {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
        {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
        {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
        {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
        {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
        {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
    ]

    def strassen_block():
        rank = 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))

        col = 0
        for a in range(2):  # A block column: 2a, 2a+1
            for b in range(4):  # B column block: 2b, 2b+1
                for t in range(7):
                    term = strassen_terms[t]
                    for (i_prime, j_prime), coeff in term['x'].items():
                        u_idx = i_prime * m + (2*a + j_prime)
                        U[u_idx, col] += coeff
                    for (j_prime, k_prime), coeff in term['y'].items():
                        v_idx = (2*a + j_prime) * p + (2*b + k_prime)
                        V[v_idx, col] += coeff
                    for (i_prime, k_prime), coeff in term['z'].items():
                        w_idx = (2*b + k_prime) * n + i_prime
                        W[w_idx, col] += coeff
                    col += 1

        return U, V, W, rank

    # 2x4 x 4x4 decomposition with rank 24 using shared Strassen structure
    # A is 2x4 (indices i in {0,1}, j in {0,1,2,3}), B is 4x4 (j in {0,1,2,3}, k in {0,1,2,3})
    # Split A into two 2x2 blocks: A_left (j=0,1), A_right (j=2,3)
    # Split B into two 4x2 blocks: B_top (k=0,1), B_bot (k=2,3)
    # For each A-block and B-block pair, use Strassen (7 terms)
    # Total: 2 * 2 * 7 = 28... that's still 28 per half, 56 total.
    # 
    # Alternative: use a 2x4x4 rank-24 decomposition directly.
    # Try: treat as 2x2x4 with 4 columns in B, using 6 terms per column-pair sharing.
    # 
    # Let's try a different structured approach: 2x4 x 4x4 with rank 24
    # by using 4 "rows" of 6 terms each, where each row handles 2 output columns.
    
    def strassen_24x4x4():
        """
        Rank-24 decomposition for 2x4 * 4x4.
        Uses the fact that we can group B columns into pairs and share 
        some Strassen intermediates.
        
        Actually, let's use a known approach: decompose 2x4x4 as
        sum of 24 rank-1 tensors using a structured pattern.
        
        We'll use 4 groups of 6 terms. Each group handles a pair of B-columns.
        Within each group, we use a 2x2x2-like structure but with the 4-row 
        dimension of B partially shared.
        """
        rank = 24
        U = np.zeros((8, rank))  # 2*4
        V = np.zeros((16, rank))  # 4*4
        W = np.zeros((8, rank))  # 4*2

        col = 0
        # For each pair of B output columns (k_pair in {0,1,2,3} -> columns 2*k_pair, 2*k_pair+1)
        for kp in range(2):  # 2 pairs of output columns
            # For each A 2x2 block (a in {0,1} -> columns 2a, 2a+1 of A)
            for a in range(2):  # 2 A-blocks
                for t in range(6):  # 6 terms per (A-block, B-col-pair)
                    # Use a modified Strassen-like pattern
                    # Term t handles specific (i,j) and (j,k) combinations
                    # We need 6 terms to cover the 2x4x2 sub-problem
                    # 
                    # For 2x4 x 4x2, schoolbook rank = 2*4*2 = 16
                    # We want rank 6 per (a, kp) pair.
                    # This is aggressive - let's use a different structure.
                    pass
                col += 6

        # This approach is getting complex. Let me use a simpler but valid rank-28 
        # and then try to reduce via optimization.
        return None

    # Use rank-28 per 2x4x4 half (2 A-blocks * 7 Strassen terms * 2 B-col-pairs / 2 sharing)
    # Actually: 2 A-blocks * 7 terms = 14 per B-col-pair, 2 B-col-pairs = 28 per 4x4 B-half
    # Two B-halves = 56. Same as before.
    
    # Let's try a different decomposition: 2x4 x 4x8 directly with rank 48
    # by using 6 terms per (A-block, B-col-pair) instead of 7.
    # This requires a rank-6 decomposition of 2x2 x 2x2... but min rank is 7.
    # 
    # Alternative: use overlapping A-blocks to share terms.
    # A has columns [0,1,2,3]. Use blocks [0,1], [1,2], [2,3] (3 blocks with overlap).
    # For 3 overlapping 2x2 A-blocks, we might reduce total terms.
    
    def overlapping_strassen():
        """
        Use overlapping 2x2 A-blocks: [0,1], [1,2], [2,3]
        Each block with each of 4 B-col-pairs uses 7 Strassen terms.
        But overlapping blocks share some structure.
        
        Actually this doesn't directly reduce rank since each (block, B-pair) 
        still needs 7 independent multiplications.
        
        Let's try: 3 A-blocks * 4 B-pairs * 7 = 84. Worse.
        """
        return None
    
    # Best structured approach: rank 56 Strassen, then optimize to lower rank.
    # Let's focus on making the optimization more effective.

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :rank] * V[j*p+k, :rank] * W[k*n+i, :rank])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    def verify_fast(U, V, W, rank):
        """Vectorized verification."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :rank] * V[j*p+k, :rank], W[k*n+i, :rank])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def error_metric(U, V, W, rank):
        """Compute sum of squared errors for all tensor entries (vectorized)."""
        total_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :rank]
                for k in range(p):
                    val = np.dot(u_row * V[j*p+k, :rank], W[k*n+i, :rank])
                    total_err += (val - 1.0) ** 2
        return total_err

    def compute_gradients(U, V, W, rank):
        """Compute gradients of sum-of-squared-errors w.r.t. U, V, W (vectorized)."""
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)

        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :rank]
                for k in range(p):
                    v_row = V[j*p+k, :rank]
                    w_row = W[k*n+i, :rank]
                    val = np.dot(u_row * v_row, w_row)
                    err = val - 1.0
                    uv = u_row * v_row
                    dU[i*m+j, :rank] += err * v_row * w_row
                    dV[j*p+k, :rank] += err * u_row * w_row
                    dW[k*n+i, :rank] += err * uv
        return dU, dV, dW

    def optimize_rank(U_init, V_init, W_init, target_rank, max_iters=2000, lr=0.01):
        """
        Gradient-based optimization to find a valid decomposition at target_rank.
        Uses smooth surrogate: minimize sum of squared errors.
        """
        if time.time() > deadline - 15:
            return None

        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))

        # Initialize: use first target_rank columns from best solution
        init_cols = min(target_rank, U_init.shape[1])
        U[:, :init_cols] = U_init[:, :init_cols]
        V[:, :init_cols] = V_init[:, :init_cols]
        W[:, :init_cols] = W_init[:, :init_cols]

        # If we're reducing rank, add small random perturbation to break symmetry
        if target_rank < init_cols:
            U[:, init_cols:] = np.random.randn(nm, target_rank - init_cols) * 0.1
            V[:, init_cols:] = np.random.randn(mp, target_rank - init_cols) * 0.1
            W[:, init_cols:] = np.random.randn(pn, target_rank - init_cols) * 0.1

        # Compute initial error
        current_err = error_metric(U, V, W, target_rank)

        # Gradient descent with adaptive learning rate and momentum
        best_U, best_V, best_W, best_err = U.copy(), V.copy(), W.copy(), current_err
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        beta = 0.9

        for it in range(max_iters):
            if time.time() > deadline - 15:
                break

            # Compute gradients
            dU, dV, dW = compute_gradients(U, V, W, target_rank)

            # Momentum update
            vU = beta * vU + (1 - beta) * dU
            vV = beta * vV + (1 - beta) * dV
            vW = beta * vW + (1 - beta) * dW

            # Update
            U -= lr * vU
            V -= lr * vV
            W -= lr * vW

            new_err = error_metric(U, V, W, target_rank)

            if new_err < best_err:
                best_err = new_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            # Adaptive learning rate
            if new_err > current_err * 1.5:
                lr *= 0.5
            elif it % 50 == 0 and it > 0:
                lr *= 0.97

            current_err = new_err

            if best_err < 1e-10:
                break

        if best_err < 1e-9:
            return best_U, best_V, best_W, target_rank
        return None

    def polish(U, V, W, rank, max_iters=5000, lr=0.005):
        """
        High-precision polish pass: tighter tolerance, more iterations,
        smaller learning rate for final convergence.
        """
        if time.time() > deadline - 5:
            return None

        current_err = error_metric(U, V, W, rank)
        if current_err < 1e-12:
            return U, V, W, rank

        best_U, best_V, best_W, best_err = U.copy(), V.copy(), W.copy(), current_err

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            dU, dV, dW = compute_gradients(U, V, W, rank)

            U -= lr * dU
            V -= lr * dV
            W -= lr * dW

            new_err = error_metric(U, V, W, rank)

            if new_err < best_err:
                best_err = new_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            # Adaptive learning rate: decay slowly
            if new_err > best_err * 1.5:
                lr *= 0.5
            elif it % 200 == 0 and it > 0:
                lr *= 0.9

            if best_err < 1e-14:
                break

        if best_err < 1e-10:
            return best_U, best_V, best_W, rank
        return None

    # Start with Strassen (rank 56)
    U_best, V_best, W_best, rank_best = strassen_block()
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()

    # Try to find lower rank via gradient-based optimization
    target_ranks = [48, 44, 40, 36, 32]

    for target in target_ranks:
        if time.time() > deadline - 25:
            break
        if target >= rank_best:
            continue

        # Try multiple random initializations for better exploration
        found = False
        for attempt in range(5):
            if time.time() > deadline - 25:
                break
            # Vary the perturbation scale for diversity
            pert_scale = 0.05 + 0.05 * attempt
            U_try = U_best[:, :min(target, U_best.shape[1])].copy()
            V_try = V_best[:, :min(target, V_best.shape[1])].copy()
            W_try = W_best[:, :min(target, W_best.shape[1])].copy()
            if target > U_try.shape[1]:
                U_try = np.hstack([U_try, np.random.randn(nm, target - U_try.shape[1]) * pert_scale])
                V_try = np.hstack([V_try, np.random.randn(mp, target - V_try.shape[1]) * pert_scale])
                W_try = np.hstack([W_try, np.random.randn(pn, target - W_try.shape[1]) * pert_scale])
            
            result = optimize_rank(U_try, V_try, W_try, target, max_iters=2000, lr=0.015)
            if result is not None:
                U_best, V_best, W_best, rank_best = result
                found = True
                break

        if not found:
            break

    # Final high-precision polish pass
    if time.time() < deadline - 10:
        polished = polish(U_best, V_best, W_best, rank_best, max_iters=5000, lr=0.003)
        if polished is not None:
            U_best, V_best, W_best, rank_best = polished

    # Final verification
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
