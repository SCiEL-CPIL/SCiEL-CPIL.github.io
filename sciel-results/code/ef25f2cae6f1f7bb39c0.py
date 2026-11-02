import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
                
    def compute_residual(params, r):
        U = params[:r * nm].reshape(nm, r)
        V = params[r * nm:r * (nm + mp)].reshape(mp, r)
        W = params[r * (nm + mp):].reshape(pn, r)
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return (T_approx - target).flatten()

    def polish_half_integer(U, V, W, r, max_iters=100):
        """
        Local search on the half-integer grid to minimize residual.
        """
        U_curr = np.round(U * 2) / 2
        V_curr = np.round(V * 2) / 2
        W_curr = np.round(W * 2) / 2
        
        # Flatten for easier indexing
        U_flat = U_curr.flatten()
        V_flat = V_curr.flatten()
        W_flat = W_curr.flatten()
        
        def get_residual_from_flat(uf, vf, wf):
            u = uf.reshape(nm, r)
            v = vf.reshape(mp, r)
            w = wf.reshape(pn, r)
            t = np.einsum('ak,bk,ck->abc', u, v, w)
            return np.sum((t - target)**2)

        current_cost = get_residual_from_flat(U_flat, V_flat, W_flat)
        
        if current_cost < 1e-10:
            return U_curr, V_curr, W_curr, True

        # Simple hill climbing / local search
        # Try changing one element at a time
        # To speed up, we only try a few random moves per iteration
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            # Try random few moves
            for _ in range(10):
                # Pick a random matrix and index
                mat_choice = np.random.randint(3)
                if mat_choice == 0:
                    idx = np.random.randint(len(U_flat))
                    old_val = U_flat[idx]
                    # Try neighbors: +/- 0.5
                    for delta in [0.5, -0.5]:
                        new_val = old_val + delta
                        # Optional: constrain to small range to keep search focused
                        if abs(new_val) > 5: continue
                        U_flat[idx] = new_val
                        new_cost = get_residual_from_flat(U_flat, V_flat, W_flat)
                        if new_cost < current_cost:
                            current_cost = new_cost
                            improved = True
                        else:
                            U_flat[idx] = old_val
                elif mat_choice == 1:
                    idx = np.random.randint(len(V_flat))
                    old_val = V_flat[idx]
                    for delta in [0.5, -0.5]:
                        new_val = old_val + delta
                        if abs(new_val) > 5: continue
                        V_flat[idx] = new_val
                        new_cost = get_residual_from_flat(U_flat, V_flat, W_flat)
                        if new_cost < current_cost:
                            current_cost = new_cost
                            improved = True
                        else:
                            V_flat[idx] = old_val
                else:
                    idx = np.random.randint(len(W_flat))
                    old_val = W_flat[idx]
                    for delta in [0.5, -0.5]:
                        new_val = old_val + delta
                        if abs(new_val) > 5: continue
                        W_flat[idx] = new_val
                        new_cost = get_residual_from_flat(U_flat, V_flat, W_flat)
                        if new_cost < current_cost:
                            current_cost = new_cost
                            improved = True
                        else:
                            W_flat[idx] = old_val
                
            if not improved:
                # Stuck in local minimum, try a bigger jump (restart-like)
                # Perturb a few random values
                for _ in range(3):
                    mat_choice = np.random.randint(3)
                    if mat_choice == 0:
                        idx = np.random.randint(len(U_flat))
                        U_flat[idx] += np.random.choice([-1.0, 1.0, -0.5, 0.5])
                    elif mat_choice == 1:
                        idx = np.random.randint(len(V_flat))
                        V_flat[idx] += np.random.choice([-1.0, 1.0, -0.5, 0.5])
                    else:
                        idx = np.random.randint(len(W_flat))
                        W_flat[idx] += np.random.choice([-1.0, 1.0, -0.5, 0.5])
                
                new_cost = get_residual_from_flat(U_flat, V_flat, W_flat)
                if new_cost < current_cost * 1.1: # Accept worse to escape
                    current_cost = new_cost
                else:
                    # Revert? We don't have easy revert. Just continue.
                    pass

        U_final = U_flat.reshape(nm, r)
        V_final = V_flat.reshape(mp, r)
        W_final = W_flat.reshape(pn, r)
        
        if current_cost < 1e-9:
            return U_final, V_final, W_final, True
        return None, None, None, False

    def try_rank(r, num_restarts=50):
        if time.time() > deadline:
            return None
        
        size = r * (nm + mp + pn)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialization
            if restart % 2 == 0:
                # Random
                params = np.random.randn(size) * 0.5
            else:
                # Structured: Schoolbook perturbed
                U_init = np.zeros((nm, r))
                V_init = np.zeros((mp, r))
                W_init = np.zeros((pn, r))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            col = i * m * p + j * p + k
                            if col < r:
                                U_init[i * m + j, col] = 1.0
                                V_init[j * p + k, col] = 1.0
                                W_init[k * n + i, col] = 1.0
                params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                params += np.random.randn(size) * 0.1

            try:
                # Use least_squares to get close
                res = least_squares(compute_residual, params, args=(r,), method='lm', max_nfev=200)
                
                U = res.x[:r * nm].reshape(nm, r)
                V = res.x[r * nm:r * (nm + mp)].reshape(mp, r)
                W = res.x[r * (nm + mp):].reshape(pn, r)
                
                # Polish on half-integer grid
                U_p, V_p, W_p, success = polish_half_integer(U, V, W, r, max_iters=20)
                
                if success:
                    return U_p, V_p, W_p

            except Exception:
                continue

        return None

    # Try rank 23
    res = try_rank(23, num_restarts=20)
    if res is not None:
        U, V, W = res
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 24
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
