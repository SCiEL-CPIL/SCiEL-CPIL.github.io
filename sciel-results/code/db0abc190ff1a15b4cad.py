import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 145

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :rank] * V[j*p+k, :rank] * W[k*n+i, :rank])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    def verify_fast(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :rank] * V[j*p+k, :rank], W[k*n+i, :rank])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def winograd_rank48():
        """
        Rank-48 decomposition for 2x4 * 4x8 using Winograd F(4,2) convolution.
        
        For each row i of A (n=2 rows) and each column k of B (p=8 columns),
        we use 3 Winograd multiplications to compute the dot product of A[i,:] 
        (length 4) with B[:,k] (length 2 kernel applied to get 2 outputs... 
        actually B is 4x8, so B[:,k] is length 4).
        
        Wait - let me reconsider. A is 2x4, B is 4x8, C is 2x8.
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        For fixed i and k, this is a dot product of two length-4 vectors.
        Winograd F(4,2) reduces a length-4 dot product to 3 multiplications.
        
        Winograd F(4,2):
          Input a = [a0, a1, a2, a3] (from A[i,:])
          Input b = [b0, b1, b2, b3] (from B[:,k])
          
          Forward transform of a (length 4 -> 3):
            d0 = a0 + a1
            d1 = a1
            d2 = a2 + a3
          
          Forward transform of b (length 4 -> 3):
            e0 = b0 + b1
            e1 = b0
            e2 = b1
          
          Wait, that's not right. Let me use the standard Winograd F(4,2):
          
          For computing y = [a0*b0 + a1*b1, a2*b2 + a3*b3]... no.
          
          Actually, the dot product of two length-4 vectors is just one scalar.
          Winograd is for convolution, not dot product.
          
          Let me reconsider. The dot product sum_j A[i,j]*B[j,k] for j=0..3
          is just a single bilinear form. The minimum rank of a 4x4 identity-like
          tensor (the dot product) is 4 (you can't do better than schoolbook for
          a single dot product of two independent vectors).
          
          So Winograd doesn't directly help for a single dot product.
          
          BUT - if we consider the structure differently:
          C[i,k] = sum_j A[i,j] * B[j,k]
          
          We can group the j dimension. If we split j into pairs:
          C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
          
          Each pair is a 2-term dot product, which needs rank 2 (can't reduce).
          So total rank per (i,k) is still 4.
          
          Hmm, Winograd doesn't reduce a simple dot product. It reduces convolution
          where the kernel is shared across multiple output positions.
          
          Let me think about this differently. The full matrix multiply is:
          C[i,k] = sum_j A[i,j] * B[j,k]
          
          This is a 3-tensor T[i,j,k] = delta (the identity in j).
          The rank of this tensor is n*m*p = 2*4*8 = 64 (schoolbook).
          
          Can we do better? The bilinear complexity of 2x4 times 4x8...
          
          Actually, let me use a different approach. Consider the problem as:
          For each pair (i,k), we need to compute sum_j A[i,j]*B[j,k].
          
          We can use the following trick with 3 multiplications per (i,k) pair
          if we allow linear combinations:
          
          Let me try:
          u0 = A[i,0] + A[i,1]
          u1 = A[i,1]
          u2 = A[i,2] + A[i,3]
          
          v0 = B[0,k] + B[1,k]
          v1 = B[0,k]
          v2 = B[1,k]
          
          p0 = u0 * v0 = (A[i,0]+A[i,1])(B[0,k]+B[1,k])
          p1 = u1 * v1 = A[i,1] * B[0,k]
          p2 = u2 * v2 = (A[i,2]+A[i,3]) * B[1,k]
          
          c0 = p0 - p1 = A[i,0]*B[0,k] + A[i,0]*B[1,k] + A[i,1]*B[1,k]
          c1 = p0 - p2 = A[i,0]*B[0,k] + A[i,0]*B[1,k] + A[i,1]*B[0,k] - A[i,2]*B[1,k] - A[i,3]*B[1,k]
          
          That doesn't give us the full dot product. This approach doesn't work
          for a general dot product.
          
          Let me go back to Strassen-based approach but try to be smarter.
        """
        # Let me just use the Strassen block approach with rank 56 as base
        # and try to optimize down.
        return None

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

    # Strassen terms for 2x2x2
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
        """Rank-56 Strassen decomposition: 2 A-blocks x 4 B-col-pairs x 7 terms"""
        rank = 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))

        col = 0
        for a in range(2):  # A block: columns 2a, 2a+1
            for kp in range(4):  # B column pair: columns 2*kp, 2*kp+1
                for t in range(7):
                    term = strassen_terms[t]
                    for (i_prime, j_prime), coeff in term['x'].items():
                        u_idx = i_prime * m + (2*a + j_prime)
                        U[u_idx, col] += coeff
                    for (j_prime, k_prime), coeff in term['y'].items():
                        v_idx = (2*a + j_prime) * p + (2*kp + k_prime)
                        V[v_idx, col] += coeff
                    for (i_prime, k_prime), coeff in term['z'].items():
                        w_idx = (2*kp + k_prime) * n + i_prime
                        W[w_idx, col] += coeff
                    col += 1

        return U, V, W, rank

    def error_metric(U, V, W, rank):
        total_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :rank]
                for k in range(p):
                    val = np.dot(u_row * V[j*p+k, :rank], W[k*n+i, :rank])
                    total_err += (val - 1.0) ** 2
        return total_err

    def compute_gradients(U, V, W, rank):
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

    def optimize_rank(U_init, V_init, W_init, target_rank, max_iters=5000, lr=0.02):
        if time.time() > deadline - 10:
            return None

        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))

        init_cols = min(target_rank, U_init.shape[1])
        U[:, :init_cols] = U_init[:, :init_cols]
        V[:, :init_cols] = V_init[:, :init_cols]
        W[:, :init_cols] = W_init[:, :init_cols]

        if target_rank > init_cols:
            U[:, init_cols:] = np.random.randn(nm, target_rank - init_cols) * 0.1
            V[:, init_cols:] = np.random.randn(mp, target_rank - init_cols) * 0.1
            W[:, init_cols:] = np.random.randn(pn, target_rank - init_cols) * 0.1

        current_err = error_metric(U, V, W, target_rank)
        best_U, best_V, best_W, best_err = U.copy(), V.copy(), W.copy(), current_err
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        beta = 0.9
        lr_min = 1e-6
        lr_max = 0.05

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            dU, dV, dW = compute_gradients(U, V, W, target_rank)

            vU = beta * vU + (1 - beta) * dU
            vV = beta * vV + (1 - beta) * dV
            vW = beta * vW + (1 - beta) * dW

            U -= lr * vU
            V -= lr * vV
            W -= lr * vW

            new_err = error_metric(U, V, W, target_rank)

            if new_err < best_err:
                best_err = new_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if new_err > current_err * 1.5:
                lr = max(lr * 0.5, lr_min)
            elif new_err < current_err * 0.8:
                lr = min(lr * 1.2, lr_max)
            elif it % 200 == 0 and it > 0:
                lr = max(lr * 0.95, lr_min)

            current_err = new_err

            if best_err < 1e-12:
                break

        if best_err < 1e-10:
            return best_U, best_V, best_W, target_rank
        return None

    def polish(U, V, W, rank, max_iters=10000, lr=0.001):
        if time.time() > deadline - 5:
            return None

        current_err = error_metric(U, V, W, rank)
        if current_err < 1e-14:
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

            if new_err > best_err * 1.5:
                lr *= 0.5
            elif it % 500 == 0 and it > 0:
                lr *= 0.9

            if best_err < 1e-16:
                break

        if best_err < 1e-12:
            return best_U, best_V, best_W, rank
        return None

    # Start with Strassen rank-56
    U_best, V_best, W_best, rank_best = strassen_block()
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()

    # Try to reduce rank via optimization
    target_ranks = [48, 44, 40, 36, 32]

    for target in target_ranks:
        if time.time() > deadline - 30:
            break
        if target >= rank_best:
            continue

        found = False
        for attempt in range(10):
            if time.time() > deadline - 25:
                break
            pert_scale = 0.02 + 0.02 * attempt
            U_try = U_best[:, :min(target, U_best.shape[1])].copy()
            V_try = V_best[:, :min(target, V_best.shape[1])].copy()
            W_try = W_best[:, :min(target, W_best.shape[1])].copy()
            if target > U_try.shape[1]:
                U_try = np.hstack([U_try, np.random.randn(nm, target - U_try.shape[1]) * pert_scale])
                V_try = np.hstack([V_try, np.random.randn(mp, target - V_try.shape[1]) * pert_scale])
                W_try = np.hstack([W_try, np.random.randn(pn, target - W_try.shape[1]) * pert_scale])
            elif target < U_try.shape[1] and attempt > 0:
                cols = np.random.choice(U_try.shape[1], target, replace=False)
                U_try = U_try[:, cols].copy()
                V_try = V_try[:, cols].copy()
                W_try = W_try[:, cols].copy()

            result = optimize_rank(U_try, V_try, W_try, target, max_iters=4000, lr=0.015)
            if result is not None:
                U_best, V_best, W_best, rank_best = result
                found = True
                break

        if not found:
            break

    # Final polish
    if time.time() < deadline - 8:
        polished = polish(U_best, V_best, W_best, rank_best, max_iters=10000, lr=0.001)
        if polished is not None:
            U_best, V_best, W_best, rank_best = polished

    # Final verification
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
