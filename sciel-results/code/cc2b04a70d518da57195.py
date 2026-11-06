"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
import random

def design(problem):
    """
    Design a 32x32 unit cell by parameterizing the dielectric distribution
    as a superposition of low-order Fourier harmonics that couple to the
    4th and 5th TM bands, optimizing amplitudes and phases.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    
    # For C4v symmetry on 32x32, we parameterize 1/epsilon as a sum of
    # Fourier modes that respect the symmetry. The key modes for the 4-5 gap
    # are (2,0), (0,2), (2,2), (1,1), (1,-1).
    
    # We'll work in the symmetric fundamental domain: i in [0,15], j in [i,15]
    # and expand to full cell.
    
    def make_cell_from_params(params):
        """
        params: list of (amplitude, phase) pairs for Fourier modes.
        We construct 1/epsilon as:
          1/eps(x,y) = c0 + sum_k [A_k * cos(2*pi*(kx*x + ky*y) + phi_k)]
        
        Then threshold to get 0/1 cell (dielectric where 1/eps < threshold).
        """
        # c0 is the mean of 1/eps
        # For a binary cell with fraction f of dielectric:
        #   c0 = f/eps_high + (1-f)/eps_low
        
        # We parameterize:
        # - f: dielectric fraction (0 to 1)
        # - A20, phi20: (2,0) mode
        # - A02, phi02: (0,2) mode  
        # - A22, phi22: (2,2) mode
        # - A11, phi11: (1,1) mode
        # - A1m1, phi1m1: (1,-1) mode
        
        f = params[0]
        A20, phi20 = params[1], params[2]
        A02, phi02 = params[3], params[4]
        A22, phi22 = params[5], params[6]
        A11, phi11 = params[7], params[8]
        A1m1, phi1m1 = params[9], params[10]
        
        # Mean value of 1/eps
        c0 = f / eps_high + (1.0 - f) / eps_low
        
        # Build 1/eps on the 32x32 grid
        # Coordinates: (i+0.5)/32 - 0.5 for i=0..31, so range [-0.46875, 0.46875]
        x = np.arange(size, dtype=float) / size - 0.5  # [-0.46875, ..., 0.46875]
        X, Y = np.meshgrid(x, x, indexing='ij')  # X varies along axis 0
        
        # Fourier modes: exp(i*2*pi*(kx*X + ky*Y))
        # For real symmetric patterns, we use cos terms with C4v-compatible phases
        
        inv_eps = np.full((size, size), c0, dtype=float)
        
        # (2,0) mode: cos(2*pi*2*X + phi20)
        inv_eps += A20 * np.cos(2 * np.pi * 2 * X + phi20)
        
        # (0,2) mode: cos(2*pi*2*Y + phi02)
        inv_eps += A02 * np.cos(2 * np.pi * 2 * Y + phi02)
        
        # (2,2) mode: cos(2*pi*2*(X+Y) + phi22)
        inv_eps += A22 * np.cos(2 * np.pi * 2 * (X + Y) + phi22)
        
        # (1,1) mode: cos(2*pi*(X+Y) + phi11)
        inv_eps += A11 * np.cos(2 * np.pi * (X + Y) + phi11)
        
        # (1,-1) mode: cos(2*pi*(X-Y) + phi1m1)
        inv_eps += A1m1 * np.cos(2 * np.pi * (X - Y) + phi1m1)
        
        # Threshold: dielectric (1) where inv_eps < median value
        # This ensures approximately f fraction is dielectric
        threshold = np.median(inv_eps)
        
        # 1 = dielectric (high eps, low 1/eps)
        cell = ((inv_eps < threshold).astype(int)).tolist()
        
        # Enforce C4v symmetry exactly
        # Check and fix: cell should be symmetric under:
        # 1. Transpose: cell[i][j] == cell[j][i]
        # 2. 180 rotation: cell[i][j] == cell[31-i][31-j]
        # 3. Reflection: cell[i][j] == cell[i][31-j]
        
        # Apply symmetry by averaging (rounding to nearest)
        sym_cell = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                # All 8 symmetry-equivalent positions
                equiv = [
                    (i, j),
                    (j, i),
                    (size-1-i, j),
                    (i, size-1-j),
                    (size-1-i, size-1-j),
                    (size-1-j, i),
                    (size-1-j, size-1-i),
                    (j, size-1-i)
                ]
                val = sum(cell[ei][ej] for ei, ej in equiv) / 8.0
                sym_cell[i][j] = 1 if val >= 0.5 else 0
        
        return sym_cell
    
    def fitness(cell):
        """
        Surrogate: maximize the (2,0) and (0,2) Fourier components of 1/eps,
        which drive the 4-5 TM band gap.
        """
        eps_arr = np.where(np.array(cell) == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_arr
        
        fft_inv_eps = np.fft.fft2(inv_eps)
        
        # (2,0), (0,2) and their negatives
        m_20 = np.abs(fft_inv_eps[2, 0])
        m_02 = np.abs(fft_inv_eps[0, 2])
        m_m20 = np.abs(fft_inv_eps[size-2, 0])
        m_0m2 = np.abs(fft_inv_eps[0, size-2])
        
        # Also (2,2) which couples to higher bands
        m_22 = np.abs(fft_inv_eps[2, 2])
        m_m2m2 = np.abs(fft_inv_eps[size-2, size-2])
        
        # (1,1) for lower band structure
        m_11 = np.abs(fft_inv_eps[1, 1])
        
        # Weighted score
        score = 3.0 * (m_20 + m_02 + m_m20 + m_0m2) + 1.5 * (m_22 + m_m2m2) + 0.5 * m_11
        return score
    
    # Optimize parameters using Nelder-Mead or grid + local search
    # Parameters: [f, A20, phi20, A02, phi02, A22, phi22, A11, phi11, A1m1, phi1m1]
    # Ranges:
    #   f: [0.3, 0.7]
    #   A: [-0.1, 0.1] (small perturbations around c0)
    #   phi: [0, 2*pi)
    
    best_params = None
    best_score = -1.0
    
    # Multi-start local search
    num_starts = 20
    steps_per_start = 5000
    
    for s in range(num_starts):
        if time.time() > deadline - 15.0:
            break
        
        # Initialize parameters
        if s == 0:
            # Start with a reasonable guess: moderate dielectric fraction,
            # strong (2,0) and (0,2) modes with zero phase
            params = [
                0.5,    # f
                0.05, 0.0,   # A20, phi20
                0.05, 0.0,   # A02, phi02
                0.02, 0.0,   # A22, phi22
                0.01, 0.0,   # A11, phi11
                0.01, 0.0    # A1m1, phi1m1
            ]
        else:
            params = [
                random.uniform(0.3, 0.7),
                random.uniform(-0.08, 0.08), random.uniform(0, 2*math.pi),
                random.uniform(-0.08, 0.08), random.uniform(0, 2*math.pi),
                random.uniform(-0.04, 0.04), random.uniform(0, 2*math.pi),
                random.uniform(-0.02, 0.02), random.uniform(0, 2*math.pi),
                random.uniform(-0.02, 0.02), random.uniform(0, 2*math.pi)
            ]
        
        current_score = fitness(make_cell_from_params(params))
        if current_score > best_score:
            best_score = current_score
            best_params = params[:]
        
        # Local search: perturb random parameter
        step_size = 0.01
        for step in range(steps_per_start):
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random parameter to perturb
            pidx = random.randint(0, 10)
            old_val = params[pidx]
            
            # Perturb
            if pidx == 0:  # f
                params[pidx] = max(0.2, min(0.8, old_val + random.gauss(0, step_size)))
            elif pidx % 2 == 1:  # amplitude
                params[pidx] = old_val + random.gauss(0, step_size)
            else:  # phase
                params[pidx] = (old_val + random.gauss(0, step_size)) % (2 * math.pi)
            
            new_score = fitness(make_cell_from_params(params))
            
            if new_score >= current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_params = params[:]
            else:
                params[pidx] = old_val
        
        # Decay step size for next start
        step_size *= 0.8
    
    if best_params is None:
        # Fallback
        return [[0]*size for _ in range(size)]
    
    return make_cell_from_params(best_params)

# EVOLVE-BLOCK-END
