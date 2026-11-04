"""Matrix multiplication 2x3 x 3x4 - rank-minimal tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit

    # Tensor dimensions
    U_dims = n * m  # 6
    V_dims = m * p  # 12
    W_dims = p * n  # 8

    # Build the matrix multiplication tensor
    # T[i*m+j, j*p+k, k*n+i] = 1
    nz = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz.append((a, b, c, 1.0))

    # Dense tensor for verification
    T_dense = np.zeros((U_dims, V_dims, W_dims))
    for (a, b, c, val) in nz:
        T_dense[a, b, c] = val

    def verify_decomposition(U, V, W, tol=1e-8):
        """Verify that the decomposition exactly reconstructs T."""
        T_rec = np.einsum('at,bt,ct->abc', U, V, W)
        err = np.linalg.norm(T_rec - T_dense)
        return err < tol, err

    def snap_to_half_integers(mat, tol=1e-9):
        """Snap values to nearest half-integer and verify."""
        snapped = np.round(mat * 2) / 2.0
        # Check if snapping is accurate
        if np.linalg.norm(snapped - mat) > 1e-6:
            return None
        return snapped

    def to_list(mat):
        """Convert numpy array to list of lists with exact half-integers."""
        result = []
        for row in mat:
            rrow = []
            for val in row:
                v2 = round(val * 2)
                if abs(val * 2 - v2) < 1e-9:
                    rrow.append(v2 / 2.0)
                else:
                    rrow.append(val)
            result.append(rrow)
        return result

    def als_cp(rank, n_iters=300, n_restarts=20, seed=0):
        """Run ALS for CP decomposition with multiple restarts."""
        best_err = float('inf')
        best_UVW = None
        rng = random.Random(seed)

        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            # Random initialization with scaling
            scale = 1.0 / (rank ** 0.25)
            U = rng.gauss(0, scale, (U_dims, rank))
            V = rng.gauss(0, scale, (V_dims, rank))
            W = rng.gauss(0, scale, (W_dims, rank))

            # Normalize columns to prevent divergence
            for t in range(rank):
                nu = np.linalg.norm(U[:, t])
                nv = np.linalg.norm(V[:, t])
                nw = np.linalg.norm(W[:, t])
                if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                    U[:, t] /= nu
                    V[:, t] /= nv
                    W[:, t] /= nw

            for it in range(n_iters):
                if time.time() > deadline:
                    break

                # Update U
                M = np.einsum('bt,ct->btc', V, W)
                M_flat = M.reshape(rank, V_dims * W_dims)
                new_U = np.zeros((U_dims, rank))
                for a in range(U_dims):
                    T_a = T_dense[a].reshape(-1)
                    try:
                        new_U[a] = np.linalg.lstsq(M_flat, T_a, rcond=None)[0]
                    except:
                        new_U[a] = U[a]
                U = new_U

                # Update V
                M2 = np.einsum('at,ct->act', U, W)
                M2_flat = M2.reshape(rank, U_dims * W_dims)
                new_V = np.zeros((V_dims, rank))
                for b in range(V_dims):
                    T_b = T_dense[:, b, :].reshape(-1)
                    try:
                        new_V[b] = np.linalg.lstsq(M2_flat, T_b, rcond=None)[0]
                    except:
                        new_V[b] = V[b]
                V = new_V

                # Update W
                M3 = np.einsum('at,bt->abt', U, V)
                M3_flat = M3.reshape(rank, U_dims * V_dims)
                new_W = np.zeros((W_dims, rank))
                for c in range(W_dims):
                    T_c = T_dense[:, :, c].reshape(-1)
                    try:
                        new_W[c] = np.linalg.lstsq(M3_flat, T_c, rcond=None)[0]
                    except:
                        new_W[c] = W[c]
                W = new_W

                # Compute error
                T_rec = np.einsum('at,bt,ct->abc', U, V, W)
                err = np.linalg.norm(T_rec - T_dense)
                if err < 1e-10:
                    break

                # Periodic normalization
                if it % 10 == 0:
                    for t in range(rank):
                        nu = np.linalg.norm(U[:, t])
                        nv = np.linalg.norm(V[:, t])
                        nw = np.linalg.norm(W[:, t])
                        if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                            U[:, t] /= nu
                            V[:, t] /= nv
                            W[:, t] /= nw

            # Final error
            T_rec = np.einsum('at,bt,ct->abc', U, V, W)
            err = np.linalg.norm(T_rec - T_dense)

            if err < best_err:
                best_err = err
                best_UVW = (U.copy(), V.copy(), W.copy())

            if best_err < 1e-10:
                break

        return best_UVW, best_err

    def try_structured_decomposition(rank):
        """Try a structured decomposition using the symmetry of the problem."""
        # For 2x3 * 3x4, we can exploit the small dimensions
        # Try to find a decomposition where entries are from {-1, -0.5, 0, 0.5, 1}
        
        if time.time() > deadline:
            return None

        # Use ALS with structured initialization
        rng = random.Random(rank * 7919)
        
        for restart in range(10):
            if time.time() > deadline:
                break
            
            # Initialize with sparse structure
            U = np.zeros((U_dims, rank))
            V = np.zeros((V_dims, rank))
            W = np.zeros((W_dims, rank))
            
            for t in range(rank):
                # Each rank-1 component touches a subset of indices
                # Randomly assign non-zero entries
                for a in range(U_dims):
                    if rng.random() < 0.5:
                        U[a, t] = rng.choice([-1, -0.5, 0.5, 1])
                for b in range(V_dims):
                    if rng.random() < 0.5:
                        V[b, t] = rng.choice([-1, -0.5, 0.5, 1])
                for c in range(W_dims):
                    if rng.random() < 0.5:
                        W[c, t] = rng.choice([-1, -0.5, 0.5, 1])
            
            # Run ALS from this initialization
            for it in range(200):
                if time.time() > deadline:
                    break
                
                M = np.einsum('bt,ct->btc', V, W)
                M_flat = M.reshape(rank, V_dims * W_dims)
                new_U = np.zeros((U_dims, rank))
                for a in range(U_dims):
                    T_a = T_dense[a].reshape(-1)
                    try:
                        new_U[a] = np.linalg.lstsq(M_flat, T_a, rcond=None)[0]
                    except:
                        new_U[a] = U[a]
                U = new_U

                M2 = np.einsum('at,ct->act', U, W)
                M2_flat = M2.reshape(rank, U_dims * W_dims)
                new_V = np.zeros((V_dims, rank))
                for b in range(V_dims):
                    T_b = T_dense[:, b, :].reshape(-1)
                    try:
                        new_V[b] = np.linalg.lstsq(M2_flat, T_b, rcond=None)[0]
                    except:
                        new_V[b] = V[b]
                V = new_V

                M3 = np.einsum('at,bt->abt', U, V)
                M3_flat = M3.reshape(rank, U_dims * V_dims)
                new_W = np.zeros((W_dims, rank))
                for c in range(W_dims):
                    T_c = T_dense[:, :, c].reshape(-1)
                    try:
                        new_W[c] = np.linalg.lstsq(M3_flat, T_c, rcond=None)[0]
                    except:
                        new_W[c] = W[c]
                W = new_W

                T_rec = np.einsum('at,bt,ct->abc', U, V, W)
                err = np.linalg.norm(T_rec - T_dense)
                if err < 1e-10:
                    break
            
            T_rec = np.einsum('at,bt,ct->abc', U, V, W)
            err = np.linalg.norm(T_rec - T_dense)
            
            if err < 1e-8:
                # Try to snap to half-integers
                U_snap = snap_to_half_integers(U)
                V_snap = snap_to_half_integers(V)
                W_snap = snap_to_half_integers(W)
                
                if U_snap is not None and V_snap is not None and W_snap is not None:
                    ok, _ = verify_decomposition(U_snap, V_snap, W_snap)
                    if ok:
                        return {'U': to_list(U_snap), 'V': to_list(V_snap), 'W': to_list(W_snap)}
                
                # If snapping fails, return as-is if error is tiny
                if err < 1e-10:
                    return {'U': to_list(U), 'V': to_list(V), 'W': to_list(W)}
        
        return None

    # Try ranks from 19 to 24
    best_result = None
    
    for r in range(19, n * m * p + 1):
        if time.time() > deadline:
            break
        
        # Method 1: Standard ALS
        if time.time() < deadline - 30:
            try:
                UVW, err = als_cp(r, n_iters=200, n_restarts=15, seed=r * 1000)
                if UVW is not None and err < 1e-8:
                    U, V, W = UVW
                    U_snap = snap_to_half_integers(U)
                    V_snap = snap_to_half_integers(V)
                    W_snap = snap_to_half_integers(W)
                    
                    if U_snap is not None and V_snap is not None and W_snap is not None:
                        ok, _ = verify_decomposition(U_snap, V_snap, W_snap)
                        if ok:
                            best_result = {'U': to_list(U_snap), 'V': to_list(V_snap), 'W': to_list(W_snap)}
                            break
                    elif err < 1e-10:
                        best_result = {'U': to_list(U), 'V': to_list(V), 'W': to_list(W)}
                        break
            except Exception:
                pass
        
        # Method 2: Structured search
        if time.time() < deadline - 20:
            result = try_structured_decomposition(r)
            if result is not None:
                best_result = result
                break
    
    # Fallback: schoolbook
    if best_result is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(U_dims)]
        V = [[0] * rank for _ in range(V_dims)]
        W = [[0] * rank for _ in range(W_dims)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}

    return best_result
# EVOLVE-BLOCK-END
