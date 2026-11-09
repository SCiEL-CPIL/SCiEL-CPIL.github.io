import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell using local search on a symmetry-constrained binary matrix.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s budget
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Precompute for fast evaluation
    N_pw = 8
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    Gx_flat = G_vectors[:, 0]
    Gy_flat = G_vectors[:, 1]
    Diff_x = Gx_flat[:, None] - Gx_flat[None, :]
    Diff_y = Gy_flat[:, None] - Gy_flat[None, :]
    
    idx_x = (Diff_x + size) % size
    idx_y = (Diff_y + size) % size
    
    # K-points: 30 points in irreducible BZ
    k_points = []
    for i in range(10):
        kx = (i + 0.5) * (np.pi / 10)
        k_points.append((kx, 0))
    for i in range(10):
        ky = (i + 0.5) * (np.pi / 10)
        k_points.append((np.pi, ky))
    for i in range(10):
        t = (i + 0.5) * (np.pi / 10)
        k_points.append((t, t))
        
    k_points = np.array(k_points)
    
    KG_SQ = np.zeros((len(k_points), num_pw))
    for i, k in enumerate(k_points):
        kx, ky = k
        kg_x = kx + 2 * np.pi * Gx_flat
        kg_y = ky + 2 * np.pi * Gy_flat
        KG_SQ[i] = kg_x**2 + kg_y**2
        
    def evaluate_cell(cell):
        """
        cell: 32x32 array of 0s and 1s.
        Returns gap-midgap ratio.
        """
        eps = np.where(cell == 1, eps_high, eps_low)
        mu = 1.0 / eps
        
        mu_fft = np.fft.fft2(mu)
        mu_fft /= (size * size)
        
        mu_diff = mu_fft[idx_x, idx_y]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for i_k in range(len(k_points)):
            kg_sq = KG_SQ[i_k]
            H = mu_diff * kg_sq[None, :]
            
            evals = np.linalg.eigvalsh(H)
            evals = np.real(evals)
            
            if len(evals) >= 5:
                w4 = np.sqrt(evals[3])
                w5 = np.sqrt(evals[4])
                
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)
                
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        return 2 * gap / midgap

    def expand_symmetry(core):
        """
        core: 16x16 symmetric array.
        Returns 32x32 array.
        """
        cell = np.zeros((32, 32), dtype=int)
        cell[0:16, 0:16] = core
        cell[0:16, 16:32] = core[:, ::-1]
        cell[16:32, 0:16] = core[::-1, :]
        cell[16:32, 16:32] = core[::-1, ::-1]
        return cell

    def generate_initial_core():
        """
        Generates a 16x16 symmetric core with 4 air holes in a dielectric background.
        Holes centered at (8,8), (8,23), (23,8), (23,23) in 32x32 coords.
        In 16x16 core (top-left), the hole is at (8,8) with radius ~3-4 pixels.
        """
        core = np.ones((16, 16), dtype=int)
        # Hole center in core coords: (7.5, 7.5) approx?
        # 32x32 center is 16. Hole at 8 is 8 pixels from left.
        # Core is 0..15. So index 7 or 8?
        # Pixel 8 in 32x32 corresponds to index 8 in 0..31.
        # In core 0..15, index 8 is present.
        cx, cy = 7.5, 7.5 # Center of pixel 7 and 8?
        # Let's use distance to (7.5, 7.5)
        I, J = np.meshgrid(np.arange(16), np.arange(16), indexing='ij')
        dist = np.sqrt((I - 7.5)**2 + (J - 7.5)**2)
        radius = 3.5
        core[dist < radius] = 0
        # Enforce symmetry (should be symmetric already due to circle, but ensure)
        core = (core + core.T) // 2
        return core

    best_score = -np.inf
    best_core = generate_initial_core()
    best_cell = expand_symmetry(best_core)
    best_score = evaluate_cell(best_cell)
    
    # Local Search
    # Independent pixels: i in 0..15, j in 0..i
    indices = [(i, j) for i in range(16) for j in range(i + 1)]
    
    stagnation_count = 0
    max_stagnation = 50
    
    while time.time() < deadline - 5:
        # Pick a random index
        idx = np.random.choice(len(indices))
        i, j = indices[idx]
        
        # Flip
        old_val = best_core[i, j]
        new_val = 1 - old_val
        best_core[i, j] = new_val
        best_core[j, i] = new_val # Maintain symmetry
        
        cell = expand_symmetry(best_core)
        score = evaluate_cell(cell)
        
        if score > best_score:
            best_score = score
            stagnation_count = 0
        else:
            # Revert
            best_core[i, j] = old_val
            best_core[j, i] = old_val
            stagnation_count += 1
            
            # If stagnated, try a random restart or larger move?
            # For simplicity, just keep flipping random spots.
            # If stuck for too long, maybe perturb a few pixels at once?
            if stagnation_count > max_stagnation:
                # Random walk: flip 3-5 random pixels
                num_flips = np.random.randint(3, 6)
                for _ in range(num_flips):
                    idx_r = np.random.choice(len(indices))
                    ir, jr = indices[idx_r]
                    best_core[ir, jr] = 1 - best_core[ir, jr]
                    best_core[jr, ir] = best_core[ir, jr]
                cell = expand_symmetry(best_core)
                score = evaluate_cell(cell)
                if score > best_score:
                    best_score = score
                    stagnation_count = 0
                else:
                    # Revert the multi-flip? Hard to track easily.
                    # Just accept the new state as the new "current" but not "best".
                    # Actually, for hill climbing, we should revert.
                    # But we lost the old state.
                    # Let's store the old core.
                    pass
                # To keep it simple and robust, let's just reset stagnation counter
                # and continue. The "revert" logic above only handles single flips.
                # Let's refine: Store old_core.
                
    # The loop above has a flaw: if we do multi-flip and it fails, we don't revert properly.
    # Let's rewrite the loop to be safer.
    
    # Reset for a cleaner loop
    best_score = -np.inf
    best_core = generate_initial_core()
    best_cell = expand_symmetry(best_core)
    best_score = evaluate_cell(best_cell)
    current_core = best_core.copy()
    
    stagnation_count = 0
    
    while time.time() < deadline - 5:
        # Single flip move
        idx = np.random.choice(len(indices))
        i, j = indices[idx]
        
        old_val = current_core[i, j]
        new_val = 1 - old_val
        
        current_core[i, j] = new_val
        current_core[j, i] = new_val
        
        cell = expand_symmetry(current_core)
        score = evaluate_cell(cell)
        
        if score > best_score:
            best_score = score
            best_core = current_core.copy()
            stagnation_count = 0
        else:
            # Revert
            current_core[i, j] = old_val
            current_core[j, i] = old_val
            stagnation_count += 1
            
            if stagnation_count > 100:
                # Random restart from best
                current_core = best_core.copy()
                stagnation_count = 0
                
    final_cell = expand_symmetry(best_core)
    
    result = []
    for i in range(32):
        row = []
        for j in range(32):
            row.append(int(final_cell[i, j]))
        result.append(row)
        
    return result
