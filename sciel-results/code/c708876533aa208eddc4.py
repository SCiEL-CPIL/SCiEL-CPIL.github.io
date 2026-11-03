import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 118

    dim_U = n * m  # 10
    dim_V = m * p  # 25
    dim_W = p * n  # 10

    T_target = np.zeros((dim_U, dim_V, dim_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    def check_and_snap(U, V, W):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
        if np.allclose(T_recon, T_target, atol=1e-10):
            return U_snap, V_snap, W_snap
        return None

    def local_refine(U, V, W, max_iters=500):
        U = np.copy(U)
        V = np.copy(V)
        W = np.copy(W)
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2

        def calc_error():
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            return np.sum((T_recon - T_target) ** 2)

        current_err = calc_error()
        if current_err < 1e-10:
            return U, V, W

        for _ in range(5):
            if current_err < 1e-10:
                break
            for a in range(dim_U):
                for r_idx in range(U.shape[1]):
                    orig_val = U[a, r_idx]
                    U[a, r_idx] = orig_val + 0.5
                    new_err = calc_error()
                    if new_err < current_err:
                        current_err = new_err
                    else:
                        U[a, r_idx] = orig_val
                    if current_err < 1e-10:
                        return U, V, W

            for b in range(dim_V):
                for r_idx in range(V.shape[1]):
                    orig_val = V[b, r_idx]
                    V[b, r_idx] = orig_val + 0.5
                    new_err = calc_error()
                    if new_err < current_err:
                        current_err = new_err
                    else:
                        V[b, r_idx] = orig_val
                    if current_err < 1e-10:
                        return U, V, W

            for c in range(dim_W):
                for r_idx in range(W.shape[1]):
                    orig_val = W[c, r_idx]
                    W[c, r_idx] = orig_val + 0.5
                    new_err = calc_error()
                    if new_err < current_err:
                        current_err = new_err
                    else:
                        W[c, r_idx] = orig_val
                    if current_err < 1e-10:
                        return U, V, W

        return None

    def try_rank(r, seed=42):
        if time.time() > deadline:
            return None

        np.random.seed(seed)

        U = np.random.randn(dim_U, r) * 0.5
        V = np.random.randn(dim_V, r) * 0.5
        W = np.random.randn(dim_W, r) * 0.5

        size_U = dim_U * r
        size_V = dim_V * r
        size_W = dim_W * r

        def loss(params):
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U + size_V]
            W_flat = params[size_U + size_V:]

            U_p = U_flat.reshape((dim_U), r)
            V_p = V_flat.reshape((dim_V), r)
            W_p = W_flat.reshape((dim_W), r)

            T_recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
            err = np.sum((T_recon - T_target) ** 2)
            return err

        def grad(params):
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U + size_V]
            W_flat = params[size_U + size_V:]

            U_p = U_flat.reshape((dim_U), r)
            V_p = V_flat.reshape((dim_V), r)
            W_p = W_flat.reshape((dim_W), r)

            T_recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
            diff = T_recon - T_target

            grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V_p, W_p)
            grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U_p, W_p)
            grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U_p, V_p)

            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

        try:
            res = minimize(loss, x0, jac=grad, method='L-BFGS-B',
                          options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        except Exception:
            return None

        if res.fun < 1e-3:
            U_sol = res.x[:size_U].reshape((dim_U), r)
            V_sol = res.x[size_U:size_U + size_V].reshape((dim_V), r)
            W_sol = res.x[size_U + size_V:].reshape((dim_W), r)

            snapped = check_and_snap(U_sol, V_sol, W_sol)
            if snapped:
                return snapped[0].tolist(), snapped[1].tolist(), snapped[2].tolist()

            refined = local_refine(U_sol, V_sol, W_sol)
            if refined:
                return refined[0].tolist(), refined[1].tolist(), refined[2].tolist()

        return None

    # New approach: exploit the small output dimension
    # The output has only 10 entries, so we can try to find a rank-10 decomposition
    # by identifying the 10 independent bilinear forms that span the output

    def try_structured_rank10():
        """
        Try to find a rank-10 decomposition by exploiting the structure of the 2x5x5 multiplication.
        
        Key insight: The output C = A @ B has 10 entries C[i,k] where i in {0,1}, k in {0,1,2,3,4}.
        Each C[i,k] = sum_j A[i,j] * B[j,k].
        
        We can try to use the 10 output entries themselves as the 10 "intermediate products",
        but we need to verify if this gives a valid tensor decomposition.
        
        Alternative: Use a structured initialization based on the matrix multiplication structure.
        """
        if time.time() > deadline:
            return None

        r = 10
        # Initialize with structured values based on the multiplication pattern
        U = np.zeros((dim_U, r))
        V = np.zeros((dim_V, r))
        W = np.zeros((dim_W, r))

        # For each output entry (i,k), create one rank-1 term
        # The bilinear form for C[i,k] is sum_j A[i,j] * B[j,k]
        # We can represent this as a sum of 5 rank-1 terms (one per j), but we want only 10 total
        # So we need to be more clever.
        
        # Try a different approach: use the fact that n=2 is small
        # For each k (0..4) and i (0..1), we need C[i,k] = sum_j A[i,j] B[j,k]
        # That's 10 equations, each involving 5 products
        # Total 50 products, but many can be shared
        
        # Let's try to initialize with a pattern that might work
        np.random.seed(12345)
        U = np.random.randn(dim_U, r) * 0.3
        V = np.random.randn(dim_V, r) * 0.3
        W = np.random.randn(dim_W, r) * 0.3

        # Add structured components
        for i in range(n):
            for k in range(p):
                col = i * p + k
                # This term should contribute to C[i,k]
                for j in range(m):
                    U[i * m + j, col] += 1.0
                    V[j * p + k, col] += 1.0
                W[k * n + i, col] += 1.0

        size_U = dim_U * r
        size_V = dim_V * r
        size_W = dim_W * r

        def loss(params):
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U + size_V]
            W_flat = params[size_U + size_V:]

            U_p = U_flat.reshape((dim_U), r)
            V_p = V_flat.reshape((dim_V), r)
            W_p = W_flat.reshape((dim_W), r)

            T_recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
            err = np.sum((T_recon - T_target) ** 2)
            return err

        def grad(params):
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U + size_V]
            W_flat = params[size_U + size_V:]

            U_p = U_flat.reshape((dim_U), r)
            V_p = V_flat.reshape((dim_V), r)
            W_p = W_flat.reshape((dim_W), r)

            T_recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
            diff = T_recon - T_target

            grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V_p, W_p)
            grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U_p, W_p)
            grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U_p, V_p)

            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

        try:
            res = minimize(loss, x0, jac=grad, method='L-BFGS-B',
                          options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
        except Exception:
            return None

        if res.fun < 1e-6:
            U_sol = res.x[:size_U].reshape((dim_U), r)
            V_sol = res.x[size_U:size_U + size_V].reshape((dim_V), r)
            W_sol = res.x[size_U + size_V:].reshape((dim_W), r)

            snapped = check_and_snap(U_sol, V_sol, W_sol)
            if snapped:
                return snapped[0].tolist(), snapped[1].tolist(), snapped[2].tolist()

            refined = local_refine(U_sol, V_sol, W_sol)
            if refined:
                return refined[0].tolist(), refined[1].tolist(), refined[2].tolist()

        return None

    # First, try the structured rank-10 approach
    result10 = try_structured_rank10()
    if result10:
        U_list, V_list, W_list = result10
        best_result = {"U": U_list, "V": V_list, "W": W_list}
        best_rank = 10
        return best_result

    # If that fails, try the original approach with lower ranks
    ranks_to_try = list(range(11, 50))
    seeds = [42, 123, 999, 111, 222, 333, 444, 555, 666, 777]

    for r in ranks_to_try:
        if time.time() > deadline:
            break

        found = False
        for seed in seeds:
            if time.time() > deadline:
                break
            result = try_rank(r, seed)
            if result:
                U_list, V_list, W_list = result
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r
                found = True
                break

        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
