# EVOLVE-BLOCK-START
import numpy as np
import time


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Exploits separability: base_x depends only on i, base_y only on j.
    The 3D union volume = (1D x-union integral) * (1D y-union integral).
    We optimize the 1D problem independently for x and y using a vectorized
    Nelder-Mead optimizer, starting from the Keich binary construction.
    """
    N = 8
    start_time = time.time()
    deadline = 630.0
    safety_margin = 10.0

    # Keich binary construction as initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    def union_length_1d_vec(z, c_samples):
        """
        Vectorized: for each c in c_samples, compute the length of the union
        of intervals [z[m] + c*m/N, z[m] + 1/N + c*m/N] for m=0..N-1.
        z: shape (N,), c_samples: shape (C,)
        Returns: shape (C,)
        """
        m = np.arange(N, dtype=np.float64)
        # starts: shape (C, N)
        starts = z[None, :] + c_samples[:, None] * m[None, :] / N
        ends = starts + 1.0 / N
        
        C = len(c_samples)
        lengths = np.zeros(C)
        
        for ci in range(C):
            s = starts[ci]
            e = ends[ci]
            order = np.argsort(s)
            s_s = s[order]
            e_s = e[order]
            
            total = 0.0
            cur_s = s_s[0]
            cur_e = e_s[0]
            for k in range(1, N):
                if s_s[k] <= cur_e:
                    if e_s[k] > cur_e:
                        cur_e = e_s[k]
                else:
                    total += cur_e - cur_s
                    cur_s = s_s[k]
                    cur_e = e_s[k]
            total += cur_e - cur_s
            lengths[ci] = total
        
        return lengths

    def cost_1d(z_free):
        """
        z_free: shape (N-1,), representing z_1..z_{N-1}. z_0 = 0.
        Returns the integral of union length over c in [0,1].
        """
        full_z = np.zeros(N)
        full_z[1:] = z_free
        c_samples = np.linspace(0, 1, 201)
        lengths = union_length_1d_vec(full_z, c_samples)
        # Trapezoidal rule
        return np.trapz(lengths, c_samples)

    def nelder_mead(f, x0, max_iter=500, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5,
                    xatol=1e-7, fatol=1e-8, time_limit=None):
        """
        Simple Nelder-Mead implementation with time guard.
        f: cost function, x0: initial point (1D array)
        """
        n = len(x0)
        # Create simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0
        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += 0.05 if x0[i] == 0 else 0.05 * abs(x0[i])
        
        fvals = np.array([f(simplex[i]) for i in range(n + 1)])
        
        for it in range(max_iter):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Sort by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # Check convergence
            if abs(fvals[0] - fvals[-1]) < fatol:
                break
            
            centroid = simplex[:n].mean(axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = f(xr)
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                else:
                    # Inside contraction
                    xc = centroid + rho * (xr - centroid)
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
        
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]

    # Optimize 1D problem
    # Initial guess from Keich (relative to z_0)
    z0_keich = keich[1:] - keich[0]
    
    best_z = z0_keich.copy()
    best_cost = cost_1d(z0_keich)
    
    # Try a few random starts
    rng = np.random.default_rng(777)
    time_limit = deadline - safety_margin
    
    for attempt in range(8):
        if time.time() > time_limit:
            break
        
        if attempt == 0:
            x_start = z0_keich.copy()
        else:
            x_start = z0_keich + rng.uniform(-0.1, 0.1, N - 1)
        
        try:
            z_opt, cost_opt = nelder_mead(cost_1d, x_start, max_iter=300,
                                           time_limit=time_limit)
            if cost_opt < best_cost:
                best_cost = cost_opt
                best_z = z_opt
        except Exception:
            pass
    
    # Construct full z array
    z_full = np.zeros(N)
    z_full[1:] = best_z
    
    # Build 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = z_full[i_flat]
    base_y = z_full[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
