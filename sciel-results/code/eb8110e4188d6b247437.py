import time
import sys
import numpy as np

DIMENSIONS = [3, 4, 7]


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check whether the rank-R decomposition exactly reproduces T[i,j,k]=1.
    Uses the separability along j: for each r, accumulate U_r ⊗ V_r ⊗ W_r
    into a (n, p, m) tensor, then check that the sum over r equals 1 everywhere.
    """
    R = U.shape[1]
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    # Reshape: U -> (R, n, m), V -> (R, m, p), W -> (R, p, n)
    U3 = U.reshape(R, n, m)
    V3 = V.reshape(R, m, p)
    W3 = W.reshape(R, p, n)
    # Accumulate outer product along r: result[i, k, j] = sum_r U3[r,i,j]*V3[r,j,k]*W3[r,k,i]
    # Efficient: for each r, U3[r] (n,m) * V3[r] (m,p) gives (n,p) but we need the j structure.
    # Actually the full tensor is T[i,j,k] = sum_r U3[r,i,j] * V3[r,j,k] * W3[r,k,i]
    # We can compute this as: for each r, form M_r = U3[r] * V3[r] elementwise -> (n,m) and (m,p) don't align.
    # Better: T[i,j,k] = sum_r (U3[r,i,j] * W3[r,k,i]) * V3[r,j,k]
    # Let A_r[i,j] = U3[r,i,j], B_r[j,k] = V3[r,j,k], C_r[k,i] = W3[r,k,i]
    # T[i,j,k] = sum_r A_r[i,j] * B_r[j,k] * C_r[k,i]
    # We can loop over r and accumulate, or use einsum.
    # einsum: 'rij,rjk,rki->ijk'
    T = np.einsum('rij,rjk,rki->ijk', U3, V3, W3)
    return bool(np.allclose(T, 1.0, atol=tol))


def _schoolbook(n, m, p):
    R = n * m * p
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, max_steps=200, tol=1e-9):
    """Attempt to reduce rank by zeroing one direction at a time and repairing.
    Returns (U, V, W, new_rank) where new_rank <= original rank.
    """
    R0 = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    current_R = R0

    for step in range(max_steps):
        # Try to zero out each column r in [0, current_R)
        best_r = -1
        best_residual = None
        for r in range(current_R):
            # Zero out column r
            U[:, r] = 0.0
            V[:, r] = 0.0
            W[:, r] = 0.0
            # Check if remaining columns still span the tensor
            # Compute residual: T_reconstructed - 1
            U3 = U[:, :current_R].reshape(current_R, n, m)
            V3 = V[:, :current_R].reshape(current_R, m, p)
            W3 = W[:, :current_R].reshape(current_R, p, n)
            T = np.einsum('rij,rjk,rki->ijk', U3, V3, W3)
            residual = 1.0 - T  # (n, m, p)
            if np.allclose(residual, 0.0, atol=tol):
                # This direction was redundant! Remove it.
                best_r = r
                break
            else:
                # Restore
                pass  # We'll restore below if this r doesn't work

        if best_r >= 0:
            # Remove column best_r by shifting
            U[:, best_r] = 0.0
            V[:, best_r] = 0.0
            W[:, best_r] = 0.0
            current_R -= 1
            continue

        # No single direction is redundant. Try a more aggressive approach:
        # Pick a random direction, zero it, and try to repair via least squares on others.
        # This is more complex; for now, try random zeroing with repair.
        r = np.random.randint(0, current_R)
        saved_U = U[:, r].copy()
        saved_V = V[:, r].copy()
        saved_W = W[:, r].copy()
        U[:, r] = 0.0
        V[:, r] = 0.0
        W[:, r] = 0.0

        # Compute residual with column r zeroed
        U3 = U[:, :current_R].reshape(current_R, n, m)
        V3 = V[:, :current_R].reshape(current_R, m, p)
        W3 = W[:, :current_R].reshape(current_R, p, n)
        T = np.einsum('rij,rjk,rki->ijk', U3, V3, W3)
        residual = 1.0 - T

        # Try to repair: find delta_U, delta_V, delta_W for column r that minimizes ||residual||
        # The contribution of column r to T is: U_r[i,j] * V_r[j,k] * W_r[k,i]
        # We want sum_{i,j,k} (residual[i,j,k] - U_r[i,j]*V_r[j,k]*W_r[k,i])^2 = 0
        # This is a non-convex problem. Use a simple iterative approach.
        # Initialize U_r, V_r, W_r with small random values
        np.random.seed(step + 42)
        Ur = np.random.randn(n, m) * 0.1
        Vr = np.random.randn(m, p) * 0.1
        Wr = np.random.randn(p, n) * 0.1

        # Iterative refinement: alternate solving for each factor
        for it in range(50):
            # Fix V_r and W_r, solve for U_r:
            # residual[i,j,k] ≈ Ur[i,j] * Vr[j,k] * Wr[k,i]
            # => Ur[i,j] = sum_k residual[i,j,k] * Vr[j,k] * Wr[k,i] / sum_k (Vr[j,k]*Wr[k,i])^2
            denom_U = (Vr ** 2) * (Wr.T ** 2)  # (m, p) * (n, p).T -> need (j,k) * (i,k)
            # Actually: sum_k (Vr[j,k]*Wr[k,i])^2 = sum_k Vr[j,k]^2 * Wr[k,i]^2
            # This is (Vr^2 @ Wr^2.T)[j,i]
            Wr2 = Wr ** 2  # (p, n)
            Vr2 = Vr ** 2  # (m, p)
            denom_U = Vr2 @ Wr2.T  # (m, n) -> transpose to (n, m)
            denom_U = denom_U.T  # (n, m)
            # Numerator: sum_k residual[i,j,k] * Vr[j,k] * Wr[k,i]
            # = residual @ (Vr.T * Wr) ... let's compute directly
            num_U = np.zeros((n, m))
            for j in range(m):
                for k in range(p):
                    num_U[:, j] += residual[:, j, k] * Vr[j, k] * Wr[k, :]
            mask = denom_U > 1e-12
            Ur_new = np.zeros_like(Ur)
            Ur_new[mask] = num_U[mask] / denom_U[mask]
            Ur = Ur_new

            # Fix U_r and W_r, solve for V_r:
            # residual[i,j,k] ≈ Ur[i,j] * Vr[j,k] * Wr[k,i]
            # => Vr[j,k] = sum_i residual[i,j,k] * Ur[i,j] * Wr[k,i] / sum_i (Ur[i,j]*Wr[k,i])^2
            denom_V = np.zeros((m, p))
            for j in range(m):
                for k in range(p):
                    denom_V[j, k] = np.sum((Ur[:, j] * Wr[k, :]) ** 2)
            num_V = np.zeros((m, p))
            for j in range(m):
                for k in range(p):
                    num_V[j, k] = np.sum(residual[:, j, k] * Ur[:, j] * Wr[k, :])
            mask_V = denom_V > 1e-12
            Vr_new = np.zeros_like(Vr)
            Vr_new[mask_V] = num_V[mask_V] / denom_V[mask_V]
            Vr = Vr_new

            # Fix U_r and V_r, solve for W_r:
            # residual[i,j,k] ≈ Ur[i,j] * Vr[j,k] * Wr[k,i]
            # => Wr[k,i] = sum_j residual[i,j,k] * Ur[i,j] * Vr[j,k] / sum_j (Ur[i,j]*Vr[j,k])^2
            denom_W = np.zeros((p, n))
            for k in range(p):
                for i in range(n):
                    denom_W[k, i] = np.sum((Ur[i, :] * Vr[:, k]) ** 2)
            num_W = np.zeros((p, n))
            for k in range(p):
                for i in range(n):
                    num_W[k, i] = np.sum(residual[i, :, k] * Ur[i, :] * Vr[:, k])
            mask_W = denom_W > 1e-12
            Wr_new = np.zeros_like(Wr)
            Wr_new[mask_W] = num_W[mask_W] / denom_W[mask_W]
            Wr = Wr_new

        # Check if repair worked
        U3 = U[:, :current_R].reshape(current_R, n, m)
        V3 = V[:, :current_R].reshape(current_R, m, p)
        W3 = W[:, :current_R].reshape(current_R, p, n)
        T = np.einsum('rij,rjk,rki->ijk', U3, V3, W3)
        T_new = T + Ur.reshape(n, m, 1) * Vr.reshape(1, m, p) * Wr.reshape(p, n, 1)
        # Wait, the contribution is Ur[i,j]*Vr[j,k]*Wr[k,i]
        T_r = np.einsum('ij,jk,ki->ijk', Ur, Vr, Wr)
        T_new = T + T_r
        if np.allclose(T_new, 1.0, atol=tol):
            # Repair successful! Set column r to the repaired values.
            U[:, r] = Ur.flatten()
            V[:, r] = Vr.flatten()
            W[:, r] = Wr.flatten()
            current_R -= 1
            continue

        # Repair failed, restore
        U[:, r] = saved_U
        V[:, r] = saved_V
        W[:, r] = saved_W

    # Final verification
    U3 = U[:, :current_R].reshape(current_R, n, m)
    V3 = V[:, :current_R].reshape(current_R, m, p)
    W3 = W[:, :current_R].reshape(current_R, p, n)
    T = np.einsum('rij,rjk,rki->ijk', U3, V3, W3)
    if not np.allclose(T, 1.0, atol=tol):
        return None

    # Trim to current_R
    U = U[:, :current_R]
    V = V[:, :current_R]
    W = W[:, :current_R]
    return U, V, W, current_R


def _random_search(n, m, p, start_rank, time_budget, tol=1e-9):
    """Randomized search for lower-rank decompositions."""
    best_rank = start_rank
    best_result = None
    start_time = time.time()

    # Start from schoolbook
    U, V, W = _schoolbook(n, m, p)
    R = U.shape[1]

    # Try greedy reduction
    result = _try_reduce_rank(U, V, W, n, m, p, max_steps=100, tol=tol)
    if result is not None:
        U_r, V_r, W_r, R_r = result
        if R_r < best_rank:
            best_rank = R_r
            best_result = (U_r, V_r, W_r)

    elapsed = time.time() - start_time
    if elapsed > time_budget * 0.5:
        if best_result is not None:
            return best_result
        return (U, V, W)

    # Random restarts with perturbations
    for restart in range(50):
        if time.time() - start_time > time_budget:
            break

        # Perturb the schoolbook solution
        U_p = U.copy()
        V_p = V.copy()
        W_p = W.copy()
        # Add small random noise to encourage different local minima
        noise_scale = 0.01 * (restart + 1)
        U_p += np.random.randn(*U_p.shape) * noise_scale
        V_p += np.random.randn(*V_p.shape) * noise_scale
        W_p += np.random.randn(*W_p.shape) * noise_scale

        result = _try_reduce_rank(U_p, V_p, W_p, n, m, p, max_steps=50, tol=tol)
        if result is not None:
            U_r, V_r, W_r, R_r = result
            if R_r < best_rank:
                best_rank = R_r
                best_result = (U_r, V_r, W_r)

    if best_result is not None:
        return best_result
    return (U, V, W)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p

    # Get schoolbook solution
    U, V, W = _schoolbook(n, m, p)

    # Try to reduce rank
    U_r, V_r, W_r, R_r = _random_search(n, m, p, schoolbook_rank, 130.0)

    # Final check
    if time.time() > deadline - 5.0:
        # Return schoolbook if we're running out of time
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify the result
    if _verify(U_r, V_r, W_r, n, m, p):
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
