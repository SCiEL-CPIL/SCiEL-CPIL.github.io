# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _union_length_1d(positions, c, N):
    """Compute the 1D union length of intervals at height c.
    positions: array of shape (N,), base positions x_i
    c: scalar height
    N: grid parameter
    Returns: total length of union of [x_i + c*i/N, x_i + c*i/N + 1/N]
    """
    starts = positions + c * np.arange(N) / N
    ends = starts + 1.0 / N
    # Sort by start
    order = np.argsort(starts)
    s = starts[order]
    e = ends[order]
    # Merge intervals
    total = 0.0
    cur_s = s[0]
    cur_e = e[0]
    for k in range(1, N):
        if s[k] <= cur_e:
            cur_e = max(cur_e, e[k])
        else:
            total += cur_e - cur_s
            cur_s = s[k]
            cur_e = e[k]
    total += cur_e - cur_s
    return total


def _volume_exact(x_pos, y_pos, N, num_c_samples=200):
    """Compute the 3D volume exactly by integrating over c.
    Uses adaptive sampling: find critical c values where union structure changes.
    """
    # The union structure changes when interval endpoints cross.
    # For x: endpoints are x_i + c*i/N and x_i + 1/N + c*i/N
    # Two x-endpoints are equal when:
    # x_i + c*i/N = x_j + c*j/N  =>  c = (x_j - x_i) / ((i-j)/N) = N*(x_j-x_i)/(i-j)
    # x_i + c*i/N = x_j + 1/N + c*j/N => c = (x_j - x_i + 1/N) / ((i-j)/N)
    # etc.
    
    # Collect all critical c values in [0,1]
    critical_c = set([0.0, 1.0])
    
    # For x-direction
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # x_i + c*i/N = x_j + c*j/N
            denom = (i - j) / N
            if abs(denom) > 1e-15:
                c_val = (x_pos[j] - x_pos[i]) / denom
                if 0.0 <= c_val <= 1.0:
                    critical_c.add(c_val)
            # x_i + c*i/N = x_j + 1/N + c*j/N
            denom = (i - j) / N
            if abs(denom) > 1e-15:
                c_val = (x_pos[j] + 1.0/N - x_pos[i]) / denom
                if 0.0 <= c_val <= 1.0:
                    critical_c.add(c_val)
            # x_i + 1/N + c*i/N = x_j + c*j/N
            denom = (i - j) / N
            if abs(denom) > 1e-15:
                c_val = (x_pos[j] - x_pos[i] - 1.0/N) / denom
                if 0.0 <= c_val <= 1.0:
                    critical_c.add(c_val)
            # x_i + 1/N + c*i/N = x_j + 1/N + c*j/N
            denom = (i - j) / N
            if abs(denom) > 1e-15:
                c_val = (x_pos[j] - x_pos[i]) / denom
                if 0.0 <= c_val <= 1.0:
                    critical_c.add(c_val)
    
    # For y-direction
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            denom = (i - j) / N
            if abs(denom) > 1e-15:
                c_val = (y_pos[j] - y_pos[i]) / denom
                if 0.0 <= c_val <= 1.0:
                    critical_c.add(c_val)
            denom = (i - j) / N
            if abs(denom) > 1e-15:
                c_val = (y_pos[j] + 1.0/N - y_pos[i]) / denom
                if 0.0 <= c_val <= 1.0:
                    critical_c.add(c_val)
            denom = (i - j) / N
            if abs(denom) > 1e-15:
                c_val = (y_pos[j] - y_pos[i] - 1.0/N) / denom
                if 0.0 <= c_val <= 1.0:
                    critical_c.add(c_val)
            denom = (i - j) / N
            if abs(denom) > 1e-15:
                c_val = (y_pos[j] - y_pos[i]) / denom
                if 0.0 <= c_val <= 1.0:
                    critical_c.add(c_val)
    
    critical_c = sorted(critical_c)
    
    # Integrate using trapezoidal rule on these intervals
    # Add midpoints for better accuracy
    c_points = [critical_c[0]]
    for k in range(len(critical_c) - 1):
        c_points.append((critical_c[k] + critical_c[k+1]) / 2)
    c_points.append(critical_c[-1])
    c_points = np.array(c_points)
    
    # Compute union lengths at each c point
    x_lengths = np.array([_union_length_1d(x_pos, c, N) for c in c_points])
    y_lengths = np.array([_union_length_1d(y_pos, c, N) for c in c_points])
    
    # Integrate product
    integrand = x_lengths * y_lengths
    volume = np.trapz(integrand, c_points)
    
    return volume


def _volume_fast(x_pos, y_pos, N, num_samples=500):
    """Fast volume estimate using uniform sampling in c."""
    c_samples = np.linspace(0, 1, num_samples)
    x_lengths = np.array([_union_length_1d(x_pos, c, N) for c in c_samples])
    y_lengths = np.array([_union_length_1d(y_pos, c, N) for c in c_samples])
    integrand = x_lengths * y_lengths
    volume = np.trapz(integrand, c_samples)
    return volume


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses numerical optimization to find base positions that minimize the union volume.
    Exploits the separable structure: x depends only on i, y depends only on j.
    """
    N = 8
    start_time = time.time()
    deadline = 620.0  # Safety margin
    
    # Initial guess: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    x_init = keich.copy()
    y_init = keich.copy()
    
    # Objective function for optimization
    def objective(params):
        x_pos = params[:N]
        y_pos = params[N:]
        elapsed = time.time() - start_time
        if elapsed > deadline:
            # Return a large value to stop optimization
            return 1e10
        return _volume_fast(x_pos, y_pos, N, num_samples=300)
    
    # Run L-BFGS-B optimization
    try:
        result = minimize(
            objective,
            np.concatenate([x_init, y_init]),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.success or result.fun < 1e10:
            best_params = result.x
        else:
            best_params = np.concatenate([x_init, y_init])
    except Exception:
        best_params = np.concatenate([x_init, y_init])
    
    x_opt = best_params[:N]
    y_opt = best_params[N:]
    
    # Try a second round with more accurate volume if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        def objective_accurate(params):
            x_pos = params[:N]
            y_pos = params[N:]
            elapsed = time.time() - start_time
            if elapsed > deadline:
                return 1e10
            return _volume_fast(x_pos, y_pos, N, num_samples=500)
        
        try:
            result2 = minimize(
                objective_accurate,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result2.success or result2.fun < 1e10:
                best_params = result2.x
        except Exception:
            pass
    
    x_final = best_params[:N]
    y_final = best_params[N:]
    
    # Build the 64x2 position array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_final[i_flat]
    base_y = y_final[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    # Validate
    if not np.all(np.isfinite(positions)):
        # Fall back to Keich
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
