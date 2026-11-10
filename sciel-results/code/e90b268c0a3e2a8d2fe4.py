# EVOLVE-BLOCK-START
import numpy as np
import time


def _keich_positions(N: int) -> np.ndarray:
    """Keich binary construction for n = 2^k base positions."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _compute_volume(positions: np.ndarray, N: int, n_samples: int = 500000) -> float:
    """Compute the Monte Carlo estimate of the union volume."""
    i_flat = np.arange(N * N) // N
    j_flat = np.arange(N * N) % N
    
    x_k = positions[:, 0]
    y_k = positions[:, 1]
    
    # Compute the bounding box
    X0 = np.min(x_k)
    X1 = np.max(x_k + (i_flat + 1) / N)
    Y0 = np.min(y_k)
    Y1 = np.max(y_k + (j_flat + 1) / N)
    
    area = (X1 - X0) * (Y1 - Y0)
    if area <= 1e-9:
        return float('inf')
    
    rng = np.random.default_rng(12345)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # For each tube, check if point is inside
    # Tube (i, j) at height c: [x_k + c*i/N, x_k + c*i/N + 1/N] x [y_k + c*j/N, y_k + c*j/N + 1/N]
    # Vectorized check: for each sample point, is it in at least one tube?
    
    # We can do this efficiently by checking all tubes at once using broadcasting
    # But 500000 * 64 might be memory heavy. Let's do it in chunks or use a different approach.
    
    # Actually, let's compute the union indicator directly.
    # For each point (a, b, c), it's in tube k if:
    # x_k + c*i_k/N <= a <= x_k + c*i_k/N + 1/N
    # AND y_k + c*j_k/N <= b <= y_k + c*j_k/N + 1/N
    
    # Let's compute this for all tubes and take the OR.
    # To save memory, we can process in chunks of tubes.
    
    chunk_size = 16  # Process 16 tubes at a time
    inside_any = np.zeros(n_samples, dtype=bool)
    
    for start in range(0, N * N, chunk_size):
        end = min(start + chunk_size, N * N)
        x_chunk = x_k[start:end]  # shape (chunk,)
        y_chunk = y_k[start:end]  # shape (chunk,)
        i_chunk = i_flat[start:end]  # shape (chunk,)
        j_chunk = j_flat[start:end]  # shape (chunk,)
        
        # For each sample point and each tube in chunk, check if inside
        # a: (n_samples,), x_chunk: (chunk,) -> a[:, None] - (x_chunk + c[:, None] * i_chunk[None, :] / N)
        # This gives (n_samples, chunk)
        
        x_lo = x_chunk[None, :] + c[:, None] * i_chunk[None, :] / N
        x_hi = x_lo + 1.0 / N
        y_lo = y_chunk[None, :] + c[:, None] * j_chunk[None, :] / N
        y_hi = y_lo + 1.0 / N
        
        # Check if a is in [x_lo, x_hi] and b is in [y_lo, y_hi]
        in_x = (a[:, None] >= x_lo) & (a[:, None] <= x_hi)
        in_y = (b[:, None] >= y_lo) & (b[:, None] <= y_hi)
        in_tube = in_x & in_y  # (n_samples, chunk)
        
        inside_any |= in_tube.any(axis=1)
    
    fraction = np.mean(inside_any)
    volume = fraction * area
    return volume


def _local_search(positions: np.ndarray, N: int, max_iter: int = 200, step_size: float = 0.01, deadline: float = None) -> np.ndarray:
    """Perform local search to minimize the union volume."""
    current_pos = positions.copy()
    current_vol = _compute_volume(current_pos, N)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        best_vol = current_vol
        best_pos = current_pos.copy()
        
        # Try perturbing each position
        for k in range(N * N):
            for dx in [-step_size, step_size]:
                for dy in [-step_size, step_size]:
                    trial = current_pos.copy()
                    trial[k, 0] += dx
                    trial[k, 1] += dy
                    vol = _compute_volume(trial, N)
                    if vol < best_vol:
                        best_vol = vol
                        best_pos = trial
        
        if best_vol < current_vol:
            current_pos = best_pos
            current_vol = best_vol
        else:
            break  # No improvement, stop
    
    return current_pos


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Returns:
        positions: np.ndarray of shape (64, 2). Row k = tube k in (i, j) raster order.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    N = 8
    
    # Start with Keich construction
    keich = _keich_positions(N)
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = keich[i_flat]
    base_y = keich[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    # Compute initial volume
    initial_vol = _compute_volume(positions, N)
    best_pos = positions.copy()
    best_vol = initial_vol
    
    # Perform local search
    improved_pos = _local_search(positions, N, max_iter=50, step_size=0.005, deadline=deadline)
    improved_vol = _compute_volume(improved_pos, N)
    
    if improved_vol < best_vol:
        best_pos = improved_pos
        best_vol = improved_vol
    
    # Check if we have time for more iterations with smaller steps
    if time.time() < deadline - 60:
        further_improved = _local_search(best_pos, N, max_iter=50, step_size=0.002, deadline=deadline)
        further_vol = _compute_volume(further_improved, N)
        if further_vol < best_vol:
            best_pos = further_improved
            best_vol = further_vol
    
    return best_pos


# EVOLVE-BLOCK-END
