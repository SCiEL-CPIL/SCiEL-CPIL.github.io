<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """
    Design a C4v-symmetric 32x32 unit cell using Perlin noise thresholding.
    The approach uses a multi-scale Perlin noise field with C4v symmetry
    to create complex dielectric textures that maximize TM band gap between
    bands 4 and 5.
    """
    size = problem["pixels"]
    
    # Generate Perlin noise with C4v symmetry
    # We'll use a combination of sinusoidal and noise-based patterns
    # that are inherently C4v symmetric
    
    def perlin_noise_2d(x, y, seed=42):
        """Simple 2D value noise with smooth interpolation."""
        # Hash function for grid values
        def hash_2d(ix, iy):
            h = (ix * 374761393 + iy * 668265263 + seed * 1442695041) & 0x7FFFFFFF
            return h / 0x7FFFFFFF
        
        # Bilinear interpolation
        ix, iy = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - ix, y - iy
        
        # Smoothstep
        sx = fx * fx * (3 - 2 * fx)
        sy = fy * fy * (3 - 2 * fy)
        
        # Corner values
        v00 = hash_2d(ix, iy)
        v10 = hash_2d(ix + 1, iy)
        v01 = hash_2d(ix, iy + 1)
        v11 = hash_2d(ix + 1, iy + 1)
        
        # Bilinear interpolation
        v0 = v00 + sx * (v10 - v00)
        v1 = v01 + sx * (v11 - v01)
        return v0 + sy * (v1 - v0)
    
    # Create a C4v-symmetric noise field
    # Use multiple octaves for multi-scale texture
    cell = np.zeros((size, size), dtype=np.float64)
    
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    
    # Parameters for noise generation
    base_freq = 4.0
    octaves = 4
    persistence = 0.5
    lacunarity = 2.0
    
    # Generate multi-scale noise
    for octave in range(octaves):
        freq = base_freq * (lacunarity ** octave)
        amp = persistence ** octave
        
        for i in range(size):
            for j in range(size):
                # Normalize coordinates to [-0.5, 0.5]
                dx = (i - cx) / size
                dy = (j - cy) / size
                
                # Apply C4v symmetry by using absolute values and
                # creating a pattern that's symmetric under 90-degree rotation
                # and reflection
                ax = abs(dx)
                ay = abs(dy)
                
                # Create C4v symmetric coordinates
                # Use min/max to enforce symmetry
                sx = min(ax, ay)
                sy = max(ax, ay)
                
                # Sample noise at symmetric points
                noise_val = 0.0
                for ox in [sx, sy]:
                    for oy in [sx, sy]:
                        noise_val += perlin_noise_2d(ox * freq, oy * freq, seed=42 + octave)
                noise_val /= 4.0
                
                # Add angular modulation for C4v symmetry
                theta = np.arctan2(dy, dx)
                angular = np.cos(4 * theta)
                
                # Combine noise and angular components
                cell[i, j] += amp * (0.7 * noise_val + 0.3 * angular)
    
    # Add a radial component to create ring-like structures
    for i in range(size):
        for j in range(size):
            dx = (i - cx) / size
            dy = (j - cy) / size
            r = np.sqrt(dx*dx + dy*dy)
            radial = np.cos(2 * np.pi * 3 * r)
            cell[i, j] += 0.2 * radial
    
    # Normalize and threshold
    cell_min = cell.min()
    cell_max = cell.max()
    cell_norm = (cell - cell_min) / (cell_max - cell_min)
    
    # Threshold to create binary pattern
    threshold = 0.45
    cell_binary = (cell_norm > threshold).astype(int)
    
    # Enforce C4v symmetry explicitly
    # 1. Horizontal flip symmetry
    cell_binary = (cell_binary + cell_binary[:, ::-1]) // 2
    # 2. Vertical flip symmetry
    cell_binary = (cell_binary + cell_binary[::-1, :]) // 2
    # 3. Transpose symmetry
    cell_binary = (cell_binary + cell_binary.T) // 2
    # 4. Diagonal flip symmetry (anti-diagonal)
    cell_binary = (cell_binary + cell_binary[::-1, ::-1]) // 2
    
    # Convert to nested list
    cell_list = cell_binary.tolist()
    
    return cell_list
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """
    Design a C4v-symmetric 32x32 unit cell using multi-scale Perlin noise
    with optimized parameters for maximizing TM band gap between bands 4 and 5.
    """
    size = problem["pixels"]
    
    def hash_2d(ix, iy, seed):
        """Deterministic hash for 2D grid."""
        h = (ix * 374761393 + iy * 668265263 + seed * 1442695041) & 0x7FFFFFFF
        return h / 0x7FFFFFFF
    
    def value_noise_2d(x, y, seed):
        """2D value noise with smooth interpolation."""
        ix, iy = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - ix, y - iy
        
        # Smoothstep interpolation
        sx = fx * fx * (3 - 2 * fx)
        sy = fy * fy * (3 - 2 * fy)
        
        # Corner values
        v00 = hash_2d(ix, iy, seed)
        v10 = hash_2d(ix + 1, iy, seed)
        v01 = hash_2d(ix, iy + 1, seed)
        v11 = hash_2d(ix + 1, iy + 1, seed)
        
        # Bilinear interpolation
        v0 = v00 + sx * (v10 - v00)
        v1 = v01 + sx * (v11 - v01)
        return v0 + sy * (v1 - v0)
    
    # Optimized parameters for better band gap
    base_freq = 6.0
    octaves = 5
    persistence = 0.45
    lacunarity = 1.8
    
    # Seed for noise generation - tuned for better gap
    noise_seed = 137
    
    # Create C4v-symmetric noise field
    cell = np.zeros((size, size), dtype=np.float64)
    
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    
    # Generate multi-scale noise with C4v symmetry
    for octave in range(octaves):
        freq = base_freq * (lacunarity ** octave)
        amp = persistence ** octave
        seed = noise_seed + octave * 7919
        
        for i in range(size):
            for j in range(size):
                # Normalize coordinates
                dx = (i - cx) / size
                dy = (j - cy) / size
                
                # C4v symmetric coordinates
                ax = abs(dx)
                ay = abs(dy)
                
                # Use min/max for C4v symmetry
                sx = min(ax, ay)
                sy = max(ax, ay)
                
                # Sample noise at symmetric points
                noise_val = 0.0
                for ox in [sx, sy]:
                    for oy in [sx, sy]:
                        noise_val += value_noise_2d(ox * freq, oy * freq, seed)
                noise_val /= 4.0
                
                # Angular modulation for C4v symmetry
                theta = np.arctan2(dy, dx)
                angular = np.cos(4 * theta)
                
                # Combine noise and angular components
                cell[i, j] += amp * (0.6 * noise_val + 0.4 * angular)
    
    # Add radial component for ring-like structures
    for i in range(size):
        for j in range(size):
            dx = (i - cx) / size
            dy = (j - cy) / size
            r = np.sqrt(dx*dx + dy*dy)
            radial = np.cos(2 * np.pi * 4 * r)
            cell[i, j] += 0.15 * radial
    
    # Normalize and threshold
    cell_min = cell.min()
    cell_max = cell.max()
    if cell_max > cell_min:
        cell_norm = (cell - cell_min) / (cell_max - cell_min)
    else:
        cell_norm = cell
    
    # Threshold - tuned for optimal dielectric fraction
    threshold = 0.52
    cell_binary = (cell_norm > threshold).astype(np.int32)
    
    # Enforce C4v symmetry explicitly
    # 1. Horizontal flip symmetry
    cell_binary = (cell_binary + cell_binary[:, ::-1]) // 2
    # 2. Vertical flip symmetry
    cell_binary = (cell_binary + cell_binary[::-1, :]) // 2
    # 3. Transpose symmetry
    cell_binary = (cell_binary + cell_binary.T) // 2
    # 4. Diagonal flip symmetry (anti-diagonal)
    cell_binary = (cell_binary + cell_binary[::-1, ::-1]) // 2
    
    # Convert to nested list
    cell_list = cell_binary.tolist()
    
    return cell_list
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
