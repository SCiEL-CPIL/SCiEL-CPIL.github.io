"""
Rank-minimal exact tensor decomposition for 2x4 x 4x7 matrix multiplication.
Implements a fast randomized rank-reduction search (Bun-style heuristic)
with a hard deadline guard, returning the best rank found or the schoolbook
rank 56 as a safe fallback.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]
WALL_LIMIT = 150.0          # hard wall-clock ceiling in seconds
SAFETY_MARGIN = 10.0        # seconds to leave for cleanup

# EVOLVE-BLOCK-START
def construct():
    t0 = time.perf_counter()
    deadline = t0 + WALL_LIMIT - SAFETY_MARGIN

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 56

    def make_schoolbook():
        r = schoolbook_rank
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return _to_output(U, V, W)

    def _to_output(U, V, W):
        # Convert to lists of ints/half-ints, rounding to nearest 0.5 if close
        def conv(arr):
            out = np.empty_like(arr, dtype=np.float64)
            # round to nearest 0.5
            rounded = np.round(arr * 2.0) / 2.0
            # if within tolerance of rounded, use rounded; else use arr
            tol = 1e-8
            mask = np.abs(arr - rounded) < tol
            out = np.where(mask, rounded, arr)
            return out
        Uc = conv(U).tolist()
        Vc = conv(V).tolist()
        Wc = conv(W).tolist()
        return {"U": Uc, "V": Vc, "W": Wc}

    # Try rank-reduction search
    best_rank = schoolbook_rank
    best_factors = None

    for r in range(36, schoolbook_rank + 1):
        if time.perf_counter() > deadline:
            break

        # Multiple random restarts per rank
        max_restarts = 20 if r >= 50 else 50
        found = False
        for restart in range(max_restarts):
            if time.perf_counter() > deadline:
                break

            # Initialize U, V, W randomly with small entries
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5

            # Perform gradient descent-like optimization
            # The matrix multiplication tensor T has entries T[i*m+j, j*p+k, k*n+i]
            # We want: T = sum_c U[:,c] V[:,c] W[:,c]
            # Equivalently: for each (i,j,k), U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] summed over c = 1

            # Use a simple iterative approach: fix two factors, solve for the third
            # This is a rank-1 update / alternating minimization

            lr = 0.01
            for step in range(300):
                if time.perf_counter() > deadline:
                    break

                # Compute residual: for each (i,j,k), compute
                # residual[i*m+j, j*p+k, k*n+i] = 1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                # This is expensive to form explicitly; instead do gradient steps.

                # Update W (p*n x r):
                # For each (i,j,k), the contribution to W[k*n+i,c] is U[i*m+j,c]*V[j*p+k,c]
                # Gradient for W[k*n+i,c] w.r.t. loss 0.5*sum(T - T_hat)^2:
                # dW[k*n+i,c] = sum over all (i',j',k') where (i'=i,k'=k) and j' free:
                #   Actually the indices are: T is indexed by (i,j,k) -> position (i*m+j, j*p+k, k*n+i)
                #   For a fixed W row (k*n+i), the (i,j,k) indices with this i,k are: j ranges 0..m-1
                #   So: for j in range(m):
                #     T_idx = (i*m+j, j*p+k, k*n+i)
                #     residual at T_idx contributes to gradient

                # This is getting complex. Let's use a simpler approach:
                # Directly form the full residual tensor and compute gradients.

                # Form T_hat as a (n*m, m*p, p*n) tensor
                # T_hat[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # This is rank-r product of 3 matrices -> use einsum
                T_hat = np.einsum('ar,br,cr->abc', U, V, W)

                # Target tensor T: T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k; else 0
                # Actually T is sparse: only entries corresponding to valid (i,j,k) triplets are 1
                # But wait: the matrix multiplication tensor is:
                # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
                # So the tensor has one non-zero (value 1) for each (i,j,k) tuple, at position
                #   U-index: i*m+j
                #   V-index: j*p+k
                #   W-index: k*n+i
                # All other entries are 0.

                # So we need:
                # For each valid triplet (i,j,k): T_hat[i*m+j, j*p+k, k*n+i] should equal 1
                # For all other (a,b,c): T_hat[a,b,c] should equal 0

                # Build the target as a sparse set of indices
                # Number of non-zero targets: n*m*p = 56

                # Compute loss and gradients
                # For efficiency, compute residual only at the 56 non-zero positions
                # and also need to ensure off-target positions are ~0.

                # Let's compute the full residual but only sum at target positions for the main loss
                # and add a small penalty on the rest.

                # Indices of target positions
                target_idx = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            target_idx.append((i * m + j, j * p + k, k * n + i))

                # Residual at targets
                residuals = []
                for (a, b, c) in target_idx:
                    residuals.append(T_hat[a, b, c] - 1.0)
                residuals = np.array(residuals)
                loss_targets = 0.5 * np.sum(residuals ** 2)

                # Also penalize off-target: compute sum of squares of all T_hat entries
                # that are not at target positions. This is expensive but n*m=8, m*p=28, p*n=14
                # so T_hat is 8*28*14 = 3136 entries. Manageable.
                off_loss = 0.5 * np.sum(T_hat ** 2)  # simple L2 on all entries

                total_loss = loss_targets + off_loss

                # Compute gradients
                # d(T_hat[a,b,c])/dU[x,r] = delta_{x,a} * V[b,r] * W[c,r]
                # Gradient of loss w.r.t. U[x,r]:
                #   from target residual at (a,b,c) where x=a:
                #     sum over (b,c) in targets with a=x: residuals[idx] * V[b,r] * W[c,r]
                #   from off_loss:
                #     T_hat[x,b,c] * V[b,r] * W[c,r] for all b,c

                # This is still complex. Let's use numpy einsum for gradients.
                # grad_U[x,r] = sum_{b,c} V[b,r] W[c,r] T_hat[x,b,c]
                #              + sum_{(a,b,c) in targets, a=x} residuals_at(a,b,c) * V[b,r] * W[c,r]

                # Compute the first term:
                # tmp1 = np.einsum('br,cr,bc->xr', V, W, T_hat)  # shape (n*m, r)
                # Second term: need to add residuals at target positions
                # For each target (a,b,c), add residual * V[b,r] * W[c,r] to grad_U[a,r]

                # Let's build the residual tensor with zeros everywhere except targets
                R = np.zeros((n * m, m * p, p * n))
                for idx, (a, b, c) in enumerate(target_idx):
                    R[a, b, c] = residuals[idx]

                # grad_U = np.einsum('br,cr,bc->xr', V, W, R) + np.einsum('br,cr,xbc->xr', V, W, T_hat)
                # Simplify:
                # grad_U[x,r] = sum_{b,c} V[b,r] W[c,r] (R[x,b,c] + T_hat[x,b,c])
                # But R and T_hat have different roles. Let me just do:
                grad_U = np.einsum('br,cr,bc->xr', V, W, R + T_hat)
                # Wait, that's not quite right either. Let me think again.
                #
                # The loss is: 0.5*sum_{targets}(T_hat-1)^2 + 0.5*sum_all T_hat^2
                # dL/dU[x,r] = sum_{targets: a=x} (T_hat[a,b,c]-1) V[b,r] W[c,r] + sum_{b,c} T_hat[x,b,c] V[b,r] W[c,r]
                # The first sum is over the specific (b,c) for each target with a=x.
                # R[x,b,c] is non-zero only at target positions, and R[a,b,c] = T_hat[a,b,c] - 1.
                # So: first term = sum_{b,c} R[x,b,c] V[b,r] W[c,r]  (since R is zero off-target)
                # Second term = sum_{b,c} T_hat[x,b,c] V[b,r] W[c,r]
                # So grad_U = sum_{b,c} (R + T_hat) V W = einsum('br,cr,bc->xr', V, W, R + T_hat) ✓

                grad_V = np.einsum('ar,cr,ac->br', U, W, R + T_hat)
                grad_W = np.einsum('ar,br,ab->cr', U, V, R + T_hat)

                # Gradient descent step
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W

                # Check convergence
                if loss_targets < 1e-6 and off_loss < 1e-4:
                    found = True
                    break

            if found:
                best_rank = r
                best_factors = (U, V, W)
                break

    if best_factors is not None:
        return _to_output(*best_factors)
    else:
        return make_schoolbook()
# EVOLVE-BLOCK-END
