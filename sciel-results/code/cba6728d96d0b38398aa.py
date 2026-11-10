# EVOLVE-BLOCK-START
import numpy as np


def _deterministic_volume(positions, n_samples=20):
    """
    Compute union volume using deterministic 3D grid sampling.
    positions: (64, 2) array of (x_k, y_k).
    Returns estimated volume.
    """
    N = 8
    n = n_samples
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N
    x_top = positions[:, 0] + (i_idx + 1.0) / N
    y_top = positions[:, 1] + (j_idx + 1.0) / N
    X0 = positions[:, 0].min()
    X1 = x_top.max()
    Y0 = positions[:, 1].min()
    Y1 = y_top.max()
    
    if X1 - X0 < 1e-9 or Y1 - Y0 < 1e-9:
        return 0.0
    
    xs = np.linspace(X0 + (X1 - X0) / (2 * n), X1 - (X1 - X0) / (2 * n), n)
    ys = np.linspace(Y0 + (Y1 - Y0) / (2 * n), Y1 - (Y1 - Y0) / (2 * n), n)
    cs = np.linspace(0.5 / n, 1.0 - 0.5 / n, n)
    
    cell_vol = (X1 - X0) * (Y1 - Y0) / (n * n * n)
    
    cx, cy, cz = np.meshgrid(xs, ys, cs, indexing='ij')
    cx = cx.ravel()
    cy = cy.ravel()
    cz = cz.ravel()
    
    inside_any = np.zeros(len(cx), dtype=bool)
    for k in range(64):
        i_k = k // N
        j_k = k % N
        x_low = positions[k, 0] + cz * i_k / N
        x_high = x_low + 1.0 / N
        y_low = positions[k, 1] + cz * j_k / N
        y_high = y_low + 1.0 / N
        
        mask = (cx >= x_low) & (cx <= x_high) & (cy >= y_low) & (cy <= y_high)
        inside_any |= mask
    
    return np.sum(inside_any) * cell_vol


def _local_search(positions, n_iterations=800, step_size=0.03, deterministic_n=15):
    """
    Local search to minimize union volume.
    """
    current = positions.copy()
    current_vol = _deterministic_volume(current, deterministic_n)
    
    best = current.copy()
    best_vol = current_vol
    
    rng = np.random.default_rng(12345)
    
    for it in range(n_iterations):
        k = rng.integers(64)
        coord = rng.integers(2)
        
        best_step = 0.0
        best_new_vol = current_vol
        
        for _ in range(3):
            delta = rng.uniform(-step_size, step_size)
            new_pos = current.copy()
            new_pos[k, coord] += delta
            
            new_vol = _deterministic_volume(new_pos, deterministic_n)
            
            if new_vol < best_new_vol:
                best_new_vol = new_vol
                best_step = delta
        
        if best_step != 0.0:
            current[k, coord] += best_step
            current_vol = best_new_vol
            
            if current_vol < best_vol:
                best_vol = current_vol
                best = current.copy()
    
    return best


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses randomized local search to minimize the union volume of 64 sheared tubes.
    
    Returns:
        positions: np.ndarray of shape (64, 2).
    """
    best_positions = None
    best_vol = np.inf
    
    for restart in range(3):
        if restart == 0:
            positions = np.zeros((64, 2))
        else:
            positions = np.random.default_rng(restart + 100).uniform(-0.2, 0.2, (64, 2))
        
        optimized = _local_search(positions, n_iterations=600, step_size=0.03, deterministic_n=12)
        
        vol = _deterministic_volume(optimized, 20)
        
        if vol < best_vol:
            best_vol = vol
            best_positions = optimized
    
    return best_positions


# EVOLVE-BLOCK-END
