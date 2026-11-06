"""
Design a 32x32 unit cell with D4 symmetry to maximize the TM band gap
between bands 4 and 5 using a wavelet-inspired multi-scale representation.
"""
import numpy as np
import time
import math

def design(problem):
    """
    Design a 32x32 unit cell with D4 symmetry to maximize the TM band gap
    between bands 4 and 5.
    
    Strategy: Use a superposition of dyadic-scale features to create a
    multi-scale dielectric structure. Start with a square pillar (which
    showed improvement over circular rods) and add wavelet-like modulations
    at multiple scales to capture features that couple specific reciprocal
    lattice vectors.
    """
    size = problem["pixels"]
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit
    
    # Start with a square pillar design (known to be better than circular)
    # Square pillar with side length ~0.7 of the cell
    half_size = size // 2
    
    # Base design: square pillar
    base_cell = np.zeros((size, size), dtype=int)
    pillar_half = 0.35 * size  # pillar extends from center by this amount
    for i in range(size):
        for j in range(size):
            # Distance from center in Chebyshev metric (square)
            dx = abs(i - (size - 1) / 2.0)
            dy = abs(j - (size - 1) / 2.0)
            if dx < pillar_half and dy < pillar_half:
                base_cell[i, j] = 1
    
    best_cell = base_cell.copy()
    best_score = -np.inf
    
    # Compute a proxy score for the base design
    best_score = _compute_proxy_score(base_cell)
    
    # Try different pillar sizes and positions
    pillar_sizes = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    for ps in pillar_sizes:
        if time.time() > deadline:
            break
        cell = np.zeros((size, size), dtype=int)
        half = ps * size
        for i in range(size):
            for j in range(size):
                dx = abs(i - (size - 1) / 2.0)
                dy = abs(j - (size - 1) / 2.0)
                if dx < half and dy < half:
                    cell[i, j] = 1
        score = _compute_proxy_score(cell)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # Add multi-scale wavelet-like modulations
    # Scale 1: fine detail (period ~4 pixels)
    # Scale 2: medium detail (period ~8 pixels)
    # Scale 3: coarse detail (period ~16 pixels)
    
    scales = [4, 8, 16]
    best_cell_modified = best_cell.copy()
    best_modified_score = best_score
    
    for scale in scales:
        if time.time() > deadline:
            break
        
        # Create a wavelet-like modulation pattern
        # Use a sign pattern based on a dyadic wavelet
        modulation = np.zeros((size, size), dtype=float)
        period = scale
        
        for i in range(size):
            for j in range(size):
                # Daubechies-like sign pattern: alternating blocks
                # At each scale, the sign changes every period/2 pixels
                block_i = (i // (period // 2)) % 2
                block_j = (j // (period // 2)) % 2
                # XOR-like combination for 2D wavelet
                if block_i == block_j:
                    modulation[i, j] = 1.0
                else:
                    modulation[i, j] = -1.0
        
        # Add modulation to the best cell with varying amplitudes
        for amp in [0.1, 0.2, 0.3, 0.4, 0.5]:
            if time.time() > deadline:
                break
            
            # Apply modulation: where modulation is positive, add to dielectric
            # where negative, subtract
            modified = best_cell_modified.astype(float)
            modified += amp * modulation
            
            # Threshold to binary
            modified_binary = (modified > 0.5).astype(int)
            
            # Enforce D4 symmetry
            modified_binary = _enforce_d4(modified_binary)
            
            score = _compute_proxy_score(modified_binary)
            if score > best_modified_score:
                best_modified_score = score
                best_cell_modified = modified_binary.copy()
    
    # Local search: try flipping small patches to improve the score
    if time.time() < deadline - 10:
        best_cell_modified = _local_search(best_cell_modified, deadline)
    
    # Convert to nested list
    cell_list = best_cell_modified.tolist()
    
    return cell_list


def _enforce_d4(cell):
    """Enforce D4 symmetry: 90-degree rotational symmetry and reflection."""
    size = cell.shape[0]
    # Average over all 8 symmetry operations
    sym_sum = np.zeros_like(cell, dtype=float)
    
    # Identity
    sym_sum += cell
    
    # 90-degree rotation
    sym_sum += np.rot90(cell, 1)
    
    # 180-degree rotation
    sym_sum += np.rot90(cell, 2)
    
    # 270-degree rotation
    sym_sum += np.rot90(cell, 3)
    
    # Reflections
    sym_sum += np.fliplr(cell)
    sym_sum += np.flipud(cell)
    sym_sum += np.fliplr(np.flipud(cell))
    
    # Threshold at 0.5
    return (sym_sum > 4).astype(int)


def _compute_proxy_score(cell):
    """
    Compute a proxy score based on the dielectric contrast and structure.
    Higher contrast and more complex multi-scale structure should correlate
    with larger band gaps.
    """
    size = cell.shape[0]
    cell_float = cell.astype(float)
    
    # Compute Fourier coefficients
    fft_cell = np.fft.fft2(cell_float)
    fft_mag = np.abs(fft_cell)
    
    # Exclude DC component
    fft_mag[0, 0] = 0
    
    # Compute the ratio of high-frequency to low-frequency content
    # Higher ratio indicates more fine structure
    total_energy = np.sum(fft_mag ** 2)
    if total_energy < 1e-10:
        return -np.inf
    
    # Split into low and high frequency components
    center = size // 2
    low_freq_mask = np.zeros((size, size), dtype=bool)
    high_freq_mask = np.zeros((size, size), dtype=bool)
    
    for i in range(size):
        for j in range(size):
            dist = math.sqrt((i - center) ** 2 + (j - center) ** 2)
            if dist < center * 0.3:
                low_freq_mask[i, j] = True
            elif dist > center * 0.6:
                high_freq_mask[i, j] = True
    
    low_energy = np.sum(fft_mag[low_freq_mask] ** 2)
    high_energy = np.sum(fft_mag[high_freq_mask] ** 2)
    
    # Score: reward high-frequency content and overall contrast
    contrast = np.std(cell_float)
    freq_ratio = high_energy / (low_energy + 1e-10)
    
    return contrast * (1 + 0.5 * freq_ratio)


def _local_search(cell, deadline):
    """Local search: try flipping small patches to improve the score."""
    size = cell.shape[0]
    best_cell = cell.copy()
    best_score = _compute_proxy_score(best_cell)
    
    # Try flipping 2x2 and 4x4 patches at various positions
    patch_sizes = [2, 4]
    
    for _ in range(50):  # Limit iterations
        if time.time() > deadline:
            break
        
        improved = False
        for ps in patch_sizes:
            if time.time() > deadline:
                break
            
            for i in range(0, size - ps + 1, ps // 2):
                for j in range(0, size - ps + 1, ps // 2):
                    if time.time() > deadline:
                        break
                    
                    # Create a copy and flip the patch
                    trial = best_cell.copy()
                    trial[i:i+ps, j:j+ps] = 1 - trial[i:i+ps, j:j+ps]
                    
                    # Enforce D4 symmetry
                    trial = _enforce_d4(trial)
                    
                    score = _compute_proxy_score(trial)
                    if score > best_score:
                        best_score = score
                        best_cell = trial.copy()
                        improved = True
        
        if not improved:
            break
    
    return best_cell
