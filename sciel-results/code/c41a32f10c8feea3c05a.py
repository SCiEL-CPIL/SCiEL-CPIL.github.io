import numpy as np
import time
import math
import random

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 190.0  # Leave 10s margin for safety against 200s limit

    # 1. Setup Fast Band Solver
    # We use a slightly reduced set of plane waves for the internal search loop to be faster,
    # but the evaluator uses 289. The physics is similar enough for ranking.
    # However, to be safe and accurate, we stick to a decent resolution.
    # Let's use 15x15 = 225 plane waves for the internal loop, which is fast.
    half_pw = 7 
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs) * (2.0 * np.pi)
    nG = len(G_vecs)
    
    # Precompute k-points.
    # The evaluator uses 30 points. We use a representative subset for speed.
    # Irreducible BZ for square lattice: 0 <= kx <= ky <= pi
    k_points = []
    n_k = 8
    for i in range(n_k + 1):
        for j in range(i, n_k + 1): # Ensure kx <= ky
            s = i / n_k
            t = j / n_k
            k_points.append((s * np.pi, t * np.pi))
    # Add high symmetry points specifically
    k_points.append((np.pi, 0))
    k_points.append((np.pi/2, np.pi/2))
    k_points.append((np.pi, np.pi))
    k_points = np.array(k_points)
    
    def compute_score(cell):
        """
        Compute the 4-5 TM band gap score for a given cell (32x32 array of 0/1).
        """
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        
        # FFT of epsilon
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        inv_eps_F = 1.0 / eps_F
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
            kG2 = kG[:, 0]**2 + kG[:, 1]**2
            
            def matvec(v):
                w = kG2 * v
                u = np.fft.fft2(w.reshape(15, 15))
                z = inv_eps_F * u
                y = np.fft.ifft2(z)
                return y.flatten().real
            
            try:
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import LinearOperator
                A_op = LinearOperator((nG, nG), matvec=matvec, dtype=complex)
                vals, vecs = eigsh(A_op, k=6, which='SA', tol=1e-4)
                vals = np.sort(vals.real)
                
                if len(vals) < 5:
                    return -1.0
                
                omega4 = np.sqrt(max(0, vals[3]))
                omega5 = np.sqrt(max(0, vals[4]))
                
                if omega5 < min_omega5:
                    min_omega5 = omega5
                if omega4 > max_omega4:
                    max_omega4 = omega4
                    
            except Exception:
                return -1.0
        
        if min_omega5 == np.inf or max_omega4 == 0.0:
            return -1.0
            
        if min_omega5 <= max_omega4:
            return -1.0
            
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

    def get_fourier_targets(cell):
        """
        Calculate the magnitude of key Fourier coefficients for the dielectric constant.
        We want to maximize |eps_G| for G=(2,0) and G=(2,2) and minimize for G=(0,0) (mean).
        Actually, the gap is driven by the coupling between bands.
        For TM modes in square lattice, the 4-5 gap is strongly influenced by 
        the (2,0) and (2,2) Fourier components of the permittivity inverse or permittivity itself.
        Let's target the Fourier coefficients of the permittivity grid.
        """
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        eps_F = np.fft.fft2(eps_grid)
        
        # Get coefficients for (2,0) and (2,2)
        # Note: FFT indices are shifted. 
        # For a 32x32 grid, frequency index 2 corresponds to G=2.
        # We need to handle the periodicity.
        
        # Extract specific coefficients
        # G=(2,0) -> index [2, 0] or [30, 0]
        # G=(2,2) -> index [2, 2] or [30, 30]
        
        c20 = np.abs(eps_F[2, 0]) + np.abs(eps_F[-2, 0])
        c22 = np.abs(eps_F[2, 2]) + np.abs(eps_F[-2, -2])
        
        # We want to maximize these
        return c20, c22

    # 2. Initialize Cell
    # Start with a configuration known to be decent: 4 squares or circles.
    # Let's start with 4 squares at (8,8), (23,8), (8,23), (23,23) with size 5x5
    cell = np.zeros((size, size), dtype=int)
    centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
    for cx, cy in centers:
        cell[cx-2:cx+3, cy-2:cy+3] = 1
        
    # Ensure symmetry is maintained by construction in the search?
    # The problem requires the cell to be symmetric.
    # Strategy: Operate on a fundamental domain (1/8th of the cell) and mirror.
    # Fundamental domain: 0 <= x < y < size/2 ? No, symmetry is about center.
    # Symmetry group: C4v.
    # We can define the cell by its top-left 16x16 quadrant? No, symmetry is about center (16,16).
    # Let's define the cell by the region 0 <= i < 16, 0 <= j < 16 and mirror.
    # Actually, simpler: generate random flips in the full cell, but enforce symmetry immediately.
    
    def enforce_symmetry(c):
        """
        Enforce 4-fold rotational and reflection symmetry about the center.
        Center is at (15.5, 15.5) for 32x32? No, indices 0..31. Center is between 15 and 16.
        Symmetry:
        c[i, j] == c[31-i, 31-j] (180 rot)
        c[i, j] == c[j, i] (transpose)
        c[i, j] == c[i, 31-j] (vertical flip)
        """
        # We can just average or take the majority? 
        # Better: define the cell by the fundamental domain 0 <= i <= 15, 0 <= j <= 15.
        # And map any pixel (i,j) to the fundamental domain.
        pass

    # Let's use a simpler approach: 
    # Generate a 16x16 top-left block, and mirror it to form the 32x32 cell.
    # Mirroring:
    # Top-Left (0..15, 0..15)
    # Top-Right (0..15, 16..31) is flip of Top-Left horizontally? 
    # If symmetric about center (16,16), then:
    # c[i, j] = c[31-i, 31-j]
    # c[i, j] = c[i, 31-j]
    # c[i, j] = c[31-i, j]
    # c[i, j] = c[j, i]
    
    # Let's just use the full 32x32 but only modify pixels in the fundamental region:
    # i in [0, 15], j in [0, 15] AND i <= j (triangle).
    # Actually, let's just use i in [0, 15], j in [0, 15].
    # And apply symmetry to the rest.
    
    def generate_symmetric_cell_from_block(block):
        """
        block is 16x16.
        Construct 32x32 cell.
        Symmetry about center (16,16).
        c[i, j] for i<16, j<16 comes from block.
        c[i, 31-j] = c[i, j]
        c[31-i, j] = c[i, j]
        c[31-i, 31-j] = c[i, j]
        Also c[i, j] = c[j, i].
        This implies block must be symmetric? 
        If we define block[0:16, 0:16], we must ensure block[i,j] == block[j,i].
        Let's just enforce symmetry on the full 32x32 array directly.
        """
        c = np.zeros((size, size), dtype=int)
        # Fill from block
        # We assume block defines the top-left 16x16.
        # To satisfy c[i,j] == c[j,i], block must be symmetric.
        # We will enforce this in the search step.
        
        # For now, let's just work with the full 32x32 and enforce symmetry after every move.
        # It's slower but safer.
        pass

    # Revised Search Strategy:
    # 1. Start with a good seed (4 squares).
    # 2. In each step, pick a random pixel (i,j).
    # 3. Flip it.
    # 4. Enforce full symmetry (C4v) on the resulting grid.
    # 5. Calculate score.
    # 6. Accept if better or with probability exp(-delta/T).
    
    def enforce_full_symmetry(c):
        """
        Enforce C4v symmetry on 32x32 array.
        We will overwrite the array to be symmetric.
        We define the value of a pixel by the "canonical" pixel in the fundamental domain.
        Fundamental domain: 0 <= i < 16, 0 <= j < 16, i <= j.
        Actually, simpler: 
        For every pixel (i,j), map it to a representative in the top-left 16x16 quadrant.
        Then ensure the top-left 16x16 is symmetric about its diagonal?
        No, the symmetry is about the global center.
        
        Let's just iterate through all pixels and set them to the majority value of their symmetry group.
        Or better: 
        c[i,j] = c[31-i, 31-j]
        c[i,j] = c[i, 31-j]
        c[i,j] = c[31-i, j]
        c[i,j] = c[j, i]
        
        We can just take the average of the 8 symmetric pixels and threshold?
        Since it's binary, let's just pick one representative and copy.
        Representative: i in [0,15], j in [0,15], i <= j.
        
        Mapping:
        Given (i,j):
        1. If i > 15: i = 31 - i
        2. If j > 15: j = 31 - j
        3. If i > j: i, j = j, i
        
        Now (i,j) is in the fundamental triangle.
        We assign the value from the original c at this representative location?
        No, we need to ensure consistency.
        
        Let's do this:
        Create a new array c_sym.
        For every (i,j) in the fundamental domain (0<=i<=15, 0<=j<=15, i<=j):
            val = c[i,j]
            Set all 8 symmetric positions to val.
        """
        c_sym = np.zeros_like(c)
        for i in range(16):
            for j in range(i, 16):
                val = c[i, j]
                # 1. (i, j)
                c_sym[i, j] = val
                # 2. (j, i)
                c_sym[j, i] = val
                # 3. (31-i, 31-j)
                c_sym[31-i, 31-j] = val
                # 4. (31-j, 31-i)
                c_sym[31-j, 31-i] = val
                # 5. (i, 31-j)
                c_sym[i, 31-j] = val
                # 6. (31-j, i)
                c_sym[31-j, i] = val
                # 7. (31-i, j)
                c_sym[31-i, j] = val
                # 8. (j, 31-i)
                c_sym[j, 31-i] = val
        return c_sym

    best_score = compute_score(cell)
    best_cell = cell.copy()
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    steps = 0
    max_steps = 100000 # Safety cap
    
    while time.time() < deadline - 5.0 and steps < max_steps:
        steps += 1
        
        # Pick a random pixel
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)
        
        # Flip it
        new_cell = cell.copy()
        new_cell[i, j] = 1 - new_cell[i, j]
        
        # Enforce symmetry
        new_cell = enforce_full_symmetry(new_cell)
        
        # If no change, skip
        if np.array_equal(new_cell, cell):
            continue
            
        # Calculate score
        new_score = compute_score(new_cell)
        
        # Acceptance criterion
        if new_score > best_score:
            best_score = new_score
            best_cell = new_cell.copy()
            cell = new_cell
            # Reset temperature? No, just keep cooling
        else:
            delta = new_score - best_score
            if delta < 0:
                p = math.exp(delta / T)
                if random.random() < p:
                    cell = new_cell
            
        T = max(T_min, T * cooling)
        
        # Check time frequently
        if steps % 100 == 0 and time.time() > deadline - 10.0:
            break

    # Final check: if best_cell is invalid or score is bad, try a few random restarts?
    # No, we have time. Let's just return the best found.
    
    # Ensure the returned cell is a list of lists of ints
    return best_cell.tolist()
