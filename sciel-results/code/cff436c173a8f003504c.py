# EVOLVE-BLOCK-START
import numpy as np


def _differentiable_union_volume(positions, nx=32, ny=32, nz=16, sigma=0.02):
    """
    Compute differentiable approximation of union volume using soft indicators.
    Returns (volume, gradients) where gradients have shape (64, 2).
    """
    N = 8
    num_tubes = 64
    
    i_flat = np.repeat(np.arange(N), N)
    j_flat = np.tile(np.arange(N), N)
    
    X0 = np.min(positions[:, 0])
    X1 = np.max(positions[:, 0] + (i_flat + 1) / N)
    Y0 = np.min(positions[:, 1])
    Y1 = np.max(positions[:, 1] + (j_flat + 1) / N)
    
    box_area = (X1 - X0) * (Y1 - Y0)
    if box_area <= 1e-9:
        return 0.0, np.zeros((num_tubes, 2))
    
    # Voxel grid
    xs = np.linspace(X0, X1, nx + 1)
    ys = np.linspace(Y0, Y1, ny + 1)
    zs = np.linspace(0, 1, nz + 1)
    
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    dz = zs[1] - zs[0]
    voxel_vol = dx * dy * dz
    
    # Voxel centers
    xc = 0.5 * (xs[:-1] + xs[1:])
    yc = 0.5 * (ys[:-1] + ys[1:])
    zc = 0.5 * (zs[:-1] + zs[1:])
    
    # Create 3D grid of voxel centers
    XX, YY, ZZ = np.meshgrid(xc, yc, zc, indexing='ij')
    
    # Soft indicator for each tube using sigmoid-like function
    # For a tube k with base (xk, yk) and shear (i, j):
    # At height c, the tube occupies [xk + c*i/N, xk + c*i/N + 1/N] x [yk + c*j/N, yk + c*j/N + 1/N]
    # We use a soft indicator that is 1 inside, 0 outside, with smooth transition
    
    # Process in batches to save memory
    batch_size = 8
    soft_union = np.zeros((nx, ny, nz), dtype=np.float64)
    gradients = np.zeros((num_tubes, 2), dtype=np.float64)
    
    for start in range(0, num_tubes, batch_size):
        end = min(start + batch_size, num_tubes)
        batch_pos = positions[start:end]
        batch_i = i_flat[start:end]
        batch_j = j_flat[start:end]
        
        xk = batch_pos[:, 0].reshape(1, 1, 1, -1)
        yk = batch_pos[:, 1].reshape(1, 1, 1, -1)
        ik = batch_i.reshape(1, 1, 1, -1).astype(np.float64)
        jk = batch_j.reshape(1, 1, 1, -1).astype(np.float64)
        
        ZZ_exp = ZZ[:, :, :, np.newaxis]
        XX_exp = XX[:, :, :, np.newaxis]
        YY_exp = YY[:, :, :, np.newaxis]
        
        # Tube center at height c
        x_center = xk + ZZ_exp * ik / N
        y_center = yk + ZZ_exp * jk / N
        
        # Distance from voxel to tube boundary
        # Inside if: x_center <= XX <= x_center + 1/N AND y_center <= YY <= y_center + 1/N
        # Use soft indicator: sigmoid((distance_to_boundary)/sigma)
        
        # For x-dimension:
        # left_dist = XX - x_center (positive if to the right of left edge)
        # right_dist = (x_center + 1/N) - XX (positive if to the left of right edge)
        # Inside when both are positive. Use min of the two distances.
        
        left_x = XX_exp - x_center
        right_x = (x_center + 1.0 / N) - XX_exp
        min_x = np.minimum(left_x, right_x)
        
        left_y = YY_exp - y_center
        right_y = (y_center + 1.0 / N) - YY_exp
        min_y = np.minimum(left_y, right_y)
        
        # Combined distance: min of min_x and min_y
        min_dist = np.minimum(min_x, min_y)
        
        # Soft indicator: sigmoid(min_dist / sigma)
        # This is 1 when well inside, 0 when well outside, smooth in between
        soft_ind = 1.0 / (1.0 + np.exp(-min_dist / sigma))
        
        # For union: we want 1 if ANY tube covers the voxel
        # Use: 1 - prod(1 - soft_ind_k) for each batch, then combine across batches
        # But this is not differentiable in a simple way for gradients.
        
        # Alternative: track which voxels are covered by which tubes
        # For gradient computation, we need d(volume)/d(positions)
        
        # Let's compute the contribution of each tube to the union
        # A voxel is in the union if covered by at least one tube
        # The gradient of the union volume w.r.t. position k only matters
        # for voxels where tube k is the "sole" coverer or contributes significantly
        
        # Simplified approach: use the max soft indicator
        # union_soft = max_k soft_ind_k
        # This is differentiable almost everywhere
        
        if start == 0:
            batch_union = soft_ind
        else:
            batch_union = np.maximum(batch_union, soft_ind)
        
        # Gradient of max: d(max)/d(x_k) = 1 if x_k is the max, else 0
        # But we need to accumulate contributions
        
        # For now, store the soft indicators for later gradient computation
        if start == 0:
            all_soft = np.zeros((nx, ny, nz, num_tubes), dtype=np.float64)
        all_soft[:, :, :, start:end] = soft_ind
    
    # Compute union as max over all tubes
    union_soft = np.max(all_soft, axis=-1)
    volume = np.sum(union_soft) * voxel_vol
    
    # Compute gradients using the "winner-take-all" approach
    # For each voxel, find which tube(s) achieved the max
    max_indices = np.argmax(all_soft, axis=-1)
    
    # Gradient of soft_ind w.r.t. xk:
    # soft_ind = sigmoid(min_dist / sigma)
    # d(soft_ind)/d(xk) = soft_ind * (1 - soft_ind) * (1/sigma) * d(min_dist)/d(xk)
    # 
    # min_dist = min(min_x, min_y)
    # min_x = min(XX - xk - c*i/N, xk + c*i/N + 1/N - XX)
    # d(min_x)/d(xk): 
    #   If XX - xk - c*i/N < xk + c*i/N + 1/N - XX (i.e., XX < xk + c*i/N + 1/(2N)):
    #     min_x = XX - xk - c*i/N, d(min_x)/d(xk) = -1
    #   Else:
    #     min_x = xk + c*i/N + 1/N - XX, d(min_x)/d(xk) = +1
    # Similarly for y
    
    # This is complex. Let's use a simpler finite difference approach for gradients
    # or use a different formulation.
    
    # Alternative: use the fact that for the max, the gradient flows only to the max tube
    # For each voxel, compute gradient contribution to the winning tube
    
    # Let's compute gradients analytically for the winning tube
    for start in range(0, num_tubes, batch_size):
        end = min(start + batch_size, num_tubes)
        batch_pos = positions[start:end]
        batch_i = i_flat[start:end]
        batch_j = j_flat[start:end]
        
        xk = batch_pos[:, 0].reshape(1, 1, 1, -1)
        yk = batch_pos[:, 1].reshape(1, 1, 1, -1)
        ik = batch_i.reshape(1, 1, 1, -1).astype(np.float64)
        jk = batch_j.reshape(1, 1, 1, -1).astype(np.float64)
        
        ZZ_exp = ZZ[:, :, :, np.newaxis]
        XX_exp = XX[:, :, :, np.newaxis]
        YY_exp = YY[:, :, :, np.newaxis]
        
        x_center = xk + ZZ_exp * ik / N
        y_center = yk + ZZ_exp * jk / N
        
        left_x = XX_exp - x_center
        right_x = (x_center + 1.0 / N) - XX_exp
        min_x = np.minimum(left_x, right_x)
        
        left_y = YY_exp - y_center
        right_y = (y_center + 1.0 / N) - YY_exp
        min_y = np.minimum(left_y, right_y)
        
        min_dist = np.minimum(min_x, min_y)
        soft_ind = 1.0 / (1.0 + np.exp(-min_dist / sigma))
        
        # d(soft_ind)/d(min_dist) = soft_ind * (1 - soft_ind) / sigma
        dsigmoid = soft_ind * (1.0 - soft_ind) / sigma
        
        # d(min_dist)/d(xk):
        # min_dist = min(min_x, min_y)
        # If min_x <= min_y: d(min_dist)/d(xk) = d(min_x)/d(xk)
        # Else: d(min_dist)/d(xk) = 0
        
        # d(min_x)/d(xk):
        # If left_x <= right_x: min_x = left_x, d(min_x)/d(xk) = -1
        # Else: min_x = right_x, d(min_x)/d(xk) = +1
        
        dxk_min_x = np.where(left_x <= right_x, -1.0, 1.0)
        dyk_min_y = np.where(left_y <= right_y, -1.0, 1.0)
        
        # d(min_dist)/d(xk) = dxk_min_x if min_x <= min_y else 0
        dxk_min_dist = np.where(min_x <= min_y, dxk_min_x, 0.0)
        dyk_min_dist = np.where(min_y <= min_x, dyk_min_y, 0.0)
        
        # d(soft_ind)/d(xk) = dsigmoid * dxk_min_dist
        dxk_grad = dsigmoid * dxk_min_dist
        dyk_grad = dsigmoid * dyk_min_dist
        
        # Only contribute to gradient if this tube is the winner for the voxel
        # Create mask: is this tube the argmax for each voxel?
        # all_soft[:, :, :, start:end] contains the soft indicators for this batch
        batch_soft = all_soft[:, :, :, start:end]
        
        # For each voxel, check if any tube in this batch is the winner
        # max_value = union_soft (scalar per voxel)
        # is_winner = (batch_soft >= max_value - epsilon)
        
        max_val = union_soft[:, :, :, np.newaxis]
        is_winner = (batch_soft >= max_val - 1e-8)
        
        # Sum gradient contributions for winning voxels
        grad_x = np.sum(is_winner * dxk_grad, axis=(0, 1, 2)) * voxel_vol
        grad_y = np.sum(is_winner * dyk_grad, axis=(0, 1, 2)) * voxel_vol
        
        gradients[start:end, 0] = grad_x
        gradients[start:end, 1] = grad_y
    
    return volume, gradients


def _compute_union_volume_fast(positions, nx=48, ny=48, nz=24):
    """
    Compute the union volume using a voxel grid with optimized computation.
    positions: (64, 2) array of base positions (x_k, y_k)
    Returns: estimated union volume
    """
    N = 8
    num_tubes = 64
    
    i_flat = np.repeat(np.arange(N), N)
    j_flat = np.tile(np.arange(N), N)
    
    X0 = np.min(positions[:, 0])
    X1 = np.max(positions[:, 0] + (i_flat + 1) / N)
    Y0 = np.min(positions[:, 1])
    Y1 = np.max(positions[:, 1] + (j_flat + 1) / N)
    
    box_area = (X1 - X0) * (Y1 - Y0)
    if box_area <= 1e-9:
        return 0.0
    
    # Voxel grid
    xs = np.linspace(X0, X1, nx + 1)
    ys = np.linspace(Y0, Y1, ny + 1)
    zs = np.linspace(0, 1, nz + 1)
    
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    dz = zs[1] - zs[0]
    voxel_vol = dx * dy * dz
    
    # Voxel centers
    xc = 0.5 * (xs[:-1] + xs[1:])
    yc = 0.5 * (ys[:-1] + ys[1:])
    zc = 0.5 * (zs[:-1] + zs[1:])
    
    # Create 3D grid of voxel centers
    XX, YY, ZZ = np.meshgrid(xc, yc, zc, indexing='ij')
    
    # Process tubes in batches
    batch_size = 16
    occupied = np.zeros((nx, ny, nz), dtype=bool)
    
    for start in range(0, num_tubes, batch_size):
        end = min(start + batch_size, num_tubes)
        batch_pos = positions[start:end]
        batch_i = i_flat[start:end]
        batch_j = j_flat[start:end]
        
        xk = batch_pos[:, 0].reshape(1, 1, 1, -1)
        yk = batch_pos[:, 1].reshape(1, 1, 1, -1)
        ik = batch_i.reshape(1, 1, 1, -1)
        jk = batch_j.reshape(1, 1, 1, -1)
        
        ZZ_exp = ZZ[:, :, :, np.newaxis]
        XX_exp = XX[:, :, :, np.newaxis]
        YY_exp = YY[:, :, :, np.newaxis]
        
        x_center = xk + ZZ_exp * ik / N
        y_center = yk + ZZ_exp * jk / N
        
        cond_x = (XX_exp >= x_center) & (XX_exp <= x_center + 1.0 / N)
        cond_y = (YY_exp >= y_center) & (YY_exp <= y_center + 1.0 / N)
        
        occupied |= np.any(cond_x & cond_y, axis=-1)
    
    num_occupied = np.sum(occupied)
    return num_occupied * voxel_vol


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Uses grid-based morphological representation with gradient-based optimization.
    """
    N = 8
    num_tubes = 64
    
    i_flat = np.repeat(np.arange(N), N)
    j_flat = np.tile(np.arange(N), N)
    
    np.random.seed(42)
    
    # Initialize: place tubes so they all roughly pass through center at c=0.5
    positions = np.zeros((num_tubes, 2))
    for i in range(N):
        for j in range(N):
            k = i * N + j
            positions[k, 0] = 0.5 - i / (2.0 * N) + 0.01 * np.random.randn()
            positions[k, 1] = 0.5 - j / (2.0 * N) + 0.01 * np.random.randn()
    
    # Gradient-based optimization
    nx, ny, nz = 32, 32, 16
    sigma = 0.02
    
    best_positions = positions.copy()
    best_vol, _ = _differentiable_union_volume(best_positions, nx, ny, nz, sigma)
    
    # Learning rate schedule
    lr = 0.01
    momentum = 0.9
    velocity = np.zeros_like(best_positions)
    
    for iteration in range(300):
        vol, grad = _differentiable_union_volume(best_positions, nx, ny, nz, sigma)
        
        # Update velocity with momentum
        velocity = momentum * velocity - lr * grad
        
        # Update positions
        new_positions = best_positions + velocity
        
        # Check if volume improved
        new_vol, _ = _differentiable_union_volume(new_positions, nx, ny, nz, sigma)
        
        if new_vol < best_vol:
            best_positions = new_positions
            best_vol = new_vol
        else:
            # Revert and reduce learning rate
            velocity *= 0.5
            lr *= 0.95
        
        # Decay learning rate over time
        lr = max(0.001, 0.01 * (0.99 ** iteration))
    
    # Final verification with finer grid
    final_vol = _compute_union_volume_fast(best_positions, 48, 48, 24)
    if final_vol <= 0:
        best_positions = np.random.uniform(0, 1, (num_tubes, 2))
    
    return best_positions


# EVOLVE-BLOCK-END
