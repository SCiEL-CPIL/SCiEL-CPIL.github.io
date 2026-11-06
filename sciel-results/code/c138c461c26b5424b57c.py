"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _apply_symmetry(cell):
    """Enforce 4-fold symmetry about center: flip rows, flip cols, transpose."""
    cell = np.minimum(cell, cell[::-1, :].copy())
    cell = np.minimum(cell, cell[:, ::-1].copy())
    cell = np.minimum(cell, cell.T.copy())
    return cell

def _mlp_forward(x, y, W1, b1, W2, b2):
    """Forward pass of 2-layer MLP with tanh activation."""
    h = np.tanh(x @ W1 + b1)
    return h @ W2 + b2

def _generate_nn_cell(size, W1, b1, W2, b2, threshold=0.0):
    """Generate cell from MLP weights."""
    coords = np.arange(size) / size - 0.5
    X, Y = np.meshgrid(coords, coords, indexing='ij')
    x = X.flatten().reshape(-1, 1)
    y = Y.flatten().reshape(-1, 1)
    inputs = np.hstack([x, y])
    field = _mlp_forward(inputs, None, W1, b1, W2, b2).flatten()
    cell = (field > threshold).astype(int).reshape(size, size)
    return _apply_symmetry(cell)

def _quick_gap_estimate(cell, size=32, n_pw=13, n_kpts=6):
    """Fast surrogate evaluation using reduced plane-wave basis."""
    eps = np.array(cell, dtype=float)
    eps_high = 12.0
    eps_low = 1.0
    eps = eps_low + (eps_high - eps_low) * eps
    
    # Compute Fourier coefficients
    if n_pw % 2 == 1:
        half = n_pw // 2
        pw_idx = np.arange(-half, half+1)
    else:
        half = n_pw // 2
        pw_idx = np.arange(-half, half)
    
    eps_ft = np.zeros((len(pw_idx), len(pw_idx)), dtype=complex)
    coords = np.arange(size) / size
    for im, m in enumerate(pw_idx):
        for in_, n in enumerate(pw_idx):
            phase = np.exp(-1j * 2.0 * np.pi * (m * coords + n * coords))
            phase2d = phase[:, None] * phase[None, :]
            eps_ft[im, in_] = np.sum(eps * phase2d) / (size * size)
    
    # Evaluate at key k-points
    k_points = [
        (0.0, 0.0),      # Gamma
        (0.5, 0.0),      # X
        (0.5, 0.5),      # M
        (0.25, 0.25),    # interior
        (0.3, 0.1),      # interior
        (0.1, 0.3),      # interior
    ][:n_kpts]
    
    band4_max = 0.0
    band5_min = np.inf
    
    for kx, ky in k_points:
        # Build TM eigenvalue matrix
        n_pw_actual = len(pw_idx)
        H = np.zeros((n_pw_actual * n_pw_actual, n_pw_actual * n_pw_actual), dtype=complex)
        
        for im, m in enumerate(pw_idx):
            for in_, n in enumerate(pw_idx):
                idx = im * n_pw_actual + in_
                kx_g = kx + m
                ky_g = ky + n
                k_g_sq = kx_g**2 + ky_g**2
                
                for im2, m2 in enumerate(pw_idx):
                    for in2, n2 in enumerate(pw_idx):
                        idx2 = im2 * n_pw_actual + in2
                        dm = m - m2
                        dn = n - n2
                        
                        # Find index of (dm, dn) in pw_idx
                        dm_idx = None
                        dn_idx = None
                        for p, pm in enumerate(pw_idx):
                            if pm == dm:
                                dm_idx = p
                                break
                        for p, pn in enumerate(pw_idx):
                            if pn == dn:
                                dn_idx = p
                                break
                        
                        if dm_idx is not None and dn_idx is not None:
                            H[idx, idx2] += k_g_sq * eps_ft[dm_idx, dn_idx]
                        else:
                            if dm == 0 and dn == 0:
                                H[idx, idx2] += k_g_sq * eps_ft[0, 0]
        
        # Solve eigenvalue problem
        eigvals = np.linalg.eigvalsh(H)
        eigvals = np.real(eigvals)
        
        # Get bands 4 and 5 (1-indexed, so indices 3 and 4)
        if len(eigvals) >= 5:
            band4_max = max(band4_max, eigvals[3])
            band5_min = min(band5_min, eigvals[4])
    
    if band5_min <= band4_max:
        return -1.0
    
    gap = band5_min - band4_max
    midgap = (band5_min + band4_max) / 2.0
    if midgap == 0:
        return 0.0
    
    return 2.0 * gap / midgap

def _optimize_nn_weights(size, W1, b1, W2, b2, threshold, n_iter=50, lr=0.1):
    """Simple gradient-free optimization of NN weights."""
    best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
    best_score = _quick_gap_estimate(_generate_nn_cell(size, *best_weights, threshold), size)
    
    for _ in range(n_iter):
        # Random perturbation
        scale = lr * (1.0 + random.gauss(0, 0.5))
        W1_p = W1 + scale * random.gauss(0, 1, W1.shape)
        b1_p = b1 + scale * random.gauss(0, 1, b1.shape)
        W2_p = W2 + scale * random.gauss(0, 1, W2.shape)
        b2_p = b2 + scale * random.gauss(0, 1, b2.shape)
        
        cell = _generate_nn_cell(size, W1_p, b1_p, W2_p, b2_p, threshold)
        score = _quick_gap_estimate(cell, size)
        
        if score > best_score:
            best_score = score
            best_weights = (W1_p, b1_p, W2_p, b2_p)
            W1, b1, W2, b2 = W1_p, b1_p, W2_p, b2_p
            # Reduce learning rate on improvement
            lr *= 0.95
    
    return best_weights, best_score

def design(problem):
    """Design a 32x32 photonic crystal unit cell to maximize the 4-5 TM band gap."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s kill
    
    best_cell = None
    best_score = -np.inf
    
    # Strategy 1: Neural network-based design
    # 2-layer MLP: 2 inputs -> 8 hidden -> 1 output
    n_hidden = 8
    n_restarts = 15
    iters_per_restart = 30
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Initialize weights
        W1 = random.gauss(0, 1.0, (2, n_hidden))
        b1 = random.gauss(0, 0.5, n_hidden)
        W2 = random.gauss(0, 1.0, (n_hidden, 1))
        b2 = random.gauss(0, 0.5, 1)
        threshold = random.uniform(-0.5, 0.5)
        
        # Optimize
        W1, b1, W2, b2 = np.array(W1), np.array(b1), np.array(W2), np.array(b2)
        weights, score = _optimize_nn_weights(size, W1, b1, W2, b2, threshold, 
                                               n_iter=iters_per_restart, lr=0.1)
        
        if score > best_score:
            best_score = score
            best_cell = _generate_nn_cell(size, *weights, threshold)
    
    # Strategy 2: Fine-tune boundary pixels of best NN cell
    if best_cell is not None and time.time() < deadline - 15.0:
        best_arr = np.array(best_cell)
        improved = True
        max_passes = 3
        
        for _ in range(max_passes):
            if time.time() > deadline - 10.0:
                break
            improved = False
            for i in range(size):
                if time.time() > deadline - 5.0:
                    break
                for j in range(size):
                    if time.time() > deadline - 3.0:
                        break
                    # Check if it's a boundary pixel
                    is_boundary = False
                    if best_arr[i, j] == 1:
                        if (i > 0 and best_arr[i-1, j] == 0) or \
                           (i < size-1 and best_arr[i+1, j] == 0) or \
                           (j > 0 and best_arr[i, j-1] == 0) or \
                           (j < size-1 and best_arr[i, j+1] == 0):
                            is_boundary = True
                    elif best_arr[i, j] == 0:
                        if (i > 0 and best_arr[i-1, j] == 1) or \
                           (i < size-1 and best_arr[i+1, j] == 1) or \
                           (j > 0 and best_arr[i, j-1] == 1) or \
                           (j < size-1 and best_arr[i, j+1] == 1):
                            is_boundary = True
                    
                    if is_boundary:
                        trial = best_arr.copy()
                        trial[i, j] = 1 - trial[i, j]
                        trial = _apply_symmetry(trial)
                        score = _quick_gap_estimate(trial.tolist(), size)
                        if score > best_score:
                            best_score = score
                            best_cell = trial.tolist()
                            best_arr = trial
                            improved = True
            
            if not improved:
                break
    
    # Strategy 3: If NN didn't find good result, try parametric fallback
    if best_score < 2.0:
        # Four-rod lattice
        for offset in [0.22, 0.25, 0.28, 0.30]:
            if time.time() > deadline - 10.0:
                break
            for radius in [0.10, 0.12, 0.14, 0.16]:
                if time.time() > deadline - 5.0:
                    break
                cell = np.zeros((size, size), dtype=int)
                centers = [(-offset, -offset), (offset, -offset), (-offset, offset), (offset, offset)]
                for i in range(size):
                    for j in range(size):
                        x = (i + 0.5) / size - 0.5
                        y = (j + 0.5) / size - 0.5
                        for cx, cy in centers:
                            dx = x - cx
                            dy = y - cy
                            if np.sqrt(dx*dx + dy*dy) < radius:
                                cell[i, j] = 1
                cell = _apply_symmetry(cell).tolist()
                score = _quick_gap_estimate(cell, size)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    if best_cell is None:
        # Fallback: simple four-rod
        best_cell = np.zeros((size, size), dtype=int).tolist()
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                for cx, cy in [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]:
                    if np.sqrt((x-cx)**2 + (y-cy)**2) < 0.12:
                        best_cell[i][j] = 1
        best_cell = _apply_symmetry(np.array(best_cell)).tolist()
    
    return best_cell
# EVOLVE-BLOCK-END
