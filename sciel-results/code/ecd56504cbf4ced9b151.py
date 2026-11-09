# EVOLVE-BLOCK-START
import numpy as np
import time


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    This implementation uses a 'Star' or 'Convergent' construction where all tubes
    are positioned to pass through a common central region, maximizing overlap and
    minimizing the bounding box volume.
    
    Returns:
        positions: np.ndarray of shape (64, 2). Row k = tube k in (i, j) raster order.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep margin
    
    N = 8
    # Generate indices i and j for the 64 tubes
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    # 'Star' construction:
    # We want the center of the tube cross-section at c=0.5 to be at (Cx, Cy).
    # x_center(c) = x_k + c*i/N + 1/(2N)
    # At c=0.5: x_k + 0.5*i/N + 1/(2N) = Cx
    # x_k = Cx - 0.5*i/N - 1/(2N) = Cx - (2i + 1)/(2N)
    
    # Let's choose Cx and Cy to center the resulting bounding box.
    # The x-extent of tube i is [x_k, x_k + i/N + 1/N]
    # x_k = Cx - (2i+1)/(2N)
    # L_i = Cx - (2i+1)/(2N)
    # R_i = Cx - (2i+1)/(2N) + (i+1)/N = Cx + (2i+2 - 2i - 1)/(2N) = Cx + 1/(2N)
    # Wait, R_i = x_k + (i+1)/N.
    # x_k + (i+1)/N = Cx - (2i+1)/(2N) + 2(i+1)/(2N) = Cx + (2i + 2 - 2i - 1)/(2N) = Cx + 1/(2N).
    # So R_i is constant for all i?
    # Let's re-calculate carefully.
    # Tube i x-interval at c is [x_k + c*i/N, x_k + c*i/N + 1/N].
    # Union over c in [0,1] is [x_k, x_k + i/N + 1/N].
    # x_k = Cx - (2i+1)/(2N).
    # L_i = Cx - (2i+1)/(2N).
    # R_i = x_k + (i+1)/N = Cx - (2i+1)/(2N) + 2(i+1)/(2N) = Cx + 1/(2N).
    # Yes, R_i is constant!
    # Min L_i is at max i (i=N-1): Cx - (2N-1)/(2N) = Cx - 1 + 1/(2N).
    # Max R_i is Cx + 1/(2N).
    # Width = (Cx + 1/(2N)) - (Cx - 1 + 1/(2N)) = 1.
    # So the width is always 1, regardless of Cx.
    # We can just set Cx = 0.5 to keep coordinates in a reasonable range like [0, 1].
    
    Cx = 0.5
    Cy = 0.5
    
    base_x = Cx - (2 * i_flat + 1) / (2 * N)
    base_y = Cy - (2 * j_flat + 1) / (2 * N)
    
    positions = np.stack([base_x, base_y], axis=1)
    
    # Check for non-finite values
    if not np.all(np.isfinite(positions)):
        # Fallback to seed-like behavior if something went wrong, though unlikely
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)

    return positions


# EVOLVE-BLOCK-END
