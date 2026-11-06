import numpy as np
import time

def _get_d4_orbits(n=32):
    """Get all D4 orbits of pixels in an n x n grid."""
    orbits = []
    seen = set()
    for i in range(n):
        for j in range(n):
            if (i, j) in seen:
                continue
            orbit = set()
            orbit.add((i, j))
            orbit.add((j, n-1-i))
            orbit.add((n-1-i, n-1-j))
            orbit.add((n-1-j, i))
            orbit.add((i, n-1-j))
            orbit.add((n-1-i, j))
            orbit.add((j, i))
            orbit.add((n-1-j, n-1-i))
            orbit = list(orbit)
            for p in orbit:
                seen.add(p)
            orbits.append(orbit)
    return orbits

def _build_eps_fourier(cell_grid, size=32, eps_low=1.0, eps_high=12.0):
    """Compute Fourier coefficients of epsilon from the pixel grid."""
    eps = eps_low + (eps_high - eps_low) * cell_grid.astype(float)
    F = np.fft.fft2(eps)
    F /= (size * size)
    return F

def _get_G_vecs(half=6):
    """Get diamond-shaped G vectors."""
    G_vecs = []
    for m in range(-half, half+1):
        for n in range(-half, half+1):
            if abs(m) + abs(n) <= half:
                G_vecs.append((2*np.pi*m, 2*np.pi*n))
    return np.array(G_vecs, dtype=float)

def _get_k_points(n_per_seg=10):
    """Get k-points along the irreducible BZ boundary for square lattice."""
    k_points = []
    for i in range(n_per_seg):
        t = i / (n_per_seg - 1)
        k_points.append((t * np.pi, 0.0))
    for i in range(1, n_per_seg):
        t = i / (n_per_seg - 1)
        k_points.append((np.pi, t * np.pi))
    for i in range(1, n_per_seg):
        t = i / (n_per_seg - 1)
        k_points.append(((1 - t) * np.pi, (1 - t) * np.pi))
    return k_points

def _solve_bands_batched(eps_F_batch, G_vecs, k_points, size=32, n_bands=6):
    """
    Solve TM bands for a batch of epsilon Fourier coefficients.
    eps_F_batch: shape (N, size, size) - Fourier coefficients for N candidates
    Returns: array of shape (N, n_kpoints, n_bands) with omega values
    """
    N = eps_F_batch.shape[0]
    nG = len(G_vecs)
    nk = len(k_points)
    
    # Precompute G vector indices
    G_int = np.round(G_vecs / (2*np.pi)).astype(int)  # shape (nG, 2)
    
    # For each k-point, precompute the A matrix structure
    # A[n,m] = eps_F[G_n - G_m]
    # G_n - G_m has integer components
    
    # Precompute all (dm, dn) pairs for the A matrix
    dm_all = G_int[:, 0][:, None] - G_int[:, 0][None, :]  # (nG, nG)
    dn_all = G_int[:, 1][:, None] - G_int[:, 1][None, :]  # (nG, nG)
    
    # For each k-point, compute kG2
    results = np.zeros((N, nk, n_bands))
    
    for ki, (kx, ky) in enumerate(k_points):
        # kG = k + G
        kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
        kG2 = kG[:, 0]**2 + kG[:, 1]**2
        kG2_reg = np.where(kG2 < 1e-12, 1e-12, kG2)
        
        # For each candidate in batch, build and solve
        for ni in range(N):
            # Build A matrix: A[n,m] = eps_F[ni, (dn_all[n,m]) % size, (dm_all[n,m]) % size]
            idx_n = dn_all % size
            idx_m = dm_all % size
            A = eps_F_batch[ni, idx_n, idx_m]
            
            # B = D^{-1} A
            B = A / kG2_reg[:, None]
            B = (B + B.conj().T) / 2
            
            try:
                eigvals = np.linalg.eigvalsh(B)
                eigvals = eigvals.real
                eigvals = np.sort(eigvals)
                omega2 = np.zeros_like(eigvals)
                valid = np.abs(eigvals) > 1e-12
                omega2[valid] = 1.0 / eigvals[valid]
                omega2[~valid] = np.inf
                omega2 = np.sort(omega2)
                # Take first n_bands
                results[ni, ki, :min(n_bands, len(omega2))] = omega2[:n_bands]
            except:
                results[ni, ki, :] = np.inf
    
    return results

def _compute_gap_score_fast(cell_grid, k_points, G_vecs, size=32, eps_low=1.0, eps_high=12.0):
    """Compute the 4-5 TM band gap score for a single cell."""
    eps_F = _build_eps_fourier(cell_grid, size, eps_low, eps_high)
    bands = _solve_bands_batched(eps_F[None, :, :], G_vecs, k_points, size)[0]
    
    # bands shape: (nk, n_bands)
    # Band 4 is index 3, Band 5 is index 4
    omega4 = np.sqrt(np.maximum(0, bands[:, 3]))
    omega5 = np.sqrt(np.maximum(0, bands[:, 4]))
    
    max_omega4 = np.max(omega4)
    min_omega5 = np.min(omega5)
    
    denom = min_omega5 + max_omega4
    if denom < 1e-12:
        return -1.0
    return 2.0 * (min_omega5 - max_omega4) / denom

def _compute_gap_score_batch(cell_grids, k_points, G_vecs, size=32, eps_low=1.0, eps_high=12.0):
    """Compute gap scores for a batch of cells. cell_grids: list of (size,size) arrays."""
    N = len(cell_grids)
    eps_F_batch = np.zeros((N, size, size), dtype=complex)
    for i, cell in enumerate(cell_grids):
        eps_F_batch[i] = _build_eps_fourier(cell, size, eps_low, eps_high)
    
    bands = _solve_bands_batched(eps_F_batch, G_vecs, k_points, size)
    # bands shape: (N, nk, n_bands)
    
    scores = np.zeros(N)
    for ni in range(N):
        omega4 = np.sqrt(np.maximum(0, bands[ni, :, 3]))
        omega5 = np.sqrt(np.maximum(0, bands[ni, :, 4]))
        max_omega4 = np.max(omega4)
        min_omega5 = np.min(omega5)
        denom = min_omega5 + max_omega4
        if denom < 1e-12:
            scores[ni] = -1.0
        else:
            scores[ni] = 2.0 * (min_omega5 - max_omega4) / denom
    return scores

def _is_symmetric(cell):
    """Check if cell has D4 symmetry."""
    n = len(cell)
    for i in range(n):
        for j in range(n):
            if cell[i][j] != cell[j][i]:
                return False
            if cell[i][j] != cell[n-1-i][n-1-j]:
                return False
            if cell[i][j] != cell[n-1-i][j]:
                return False
            if cell[i][j] != cell[i][n-1-j]:
                return False
    return True

def _make_superlattice_cell(block_4x4, size=32):
    """
    Create a 32x32 cell from a 4x4 block by tiling.
    The 4x4 block must have D4 symmetry about its center (1.5, 1.5).
    This creates a cell with 8x8 periodicity that also has D4 symmetry about (15.5, 15.5).
    """
    cell = np.zeros((size, size), dtype=int)
    # Tile the 4x4 block 8x8 times
    for i in range(size):
        for j in range(size):
            cell[i, j] = block_4x4[i % 4, j % 4]
    return cell

def _get_4x4_d4_orbits():
    """Get D4 orbits for a 4x4 grid about center (1.5, 1.5)."""
    n = 4
    orbits = []
    seen = set()
    for i in range(n):
        for j in range(n):
            if (i, j) in seen:
                continue
            orbit = set()
            # D4 symmetries about center (1.5, 1.5)
            # Identity
            orbit.add((i, j))
            # 90 deg rotation about (1.5, 1.5): (i,j) -> (1.5 + (j-1.5), 1.5 - (i-1.5)) = (j, 3-i)
            orbit.add((j, 3-i))
            # 180 deg: (i,j) -> (3-i, 3-j)
            orbit.add((3-i, 3-j))
            # 270 deg: (i,j) -> (3-j, i)
            orbit.add((3-j, i))
            # Mirror x: (i,j) -> (i, 3-j)
            orbit.add((i, 3-j))
            # Mirror y: (i,j) -> (3-i, j)
            orbit.add((3-i, j))
            # Diagonal: (i,j) -> (j, i)
            orbit.add((j, i))
            # Anti-diagonal: (i,j) -> (3-j, 3-i)
            orbit.add((3-j, 3-i))
            orbit = list(orbit)
            for p in orbit:
                seen.add(p)
            orbits.append(orbit)
    return orbits

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s kill
    
    # Setup
    G_vecs = _get_G_vecs(half=6)
    k_points = _get_k_points(n_per_seg=10)
    
    # Phase 1: Optimize 4x4 superlattice block
    # The 4x4 block has very few D4 orbits
    orbits_4x4 = _get_4x4_d4_orbits()
    n_orbits_4x4 = len(orbits_4x4)
    
    # Initialize with a few candidate 4x4 blocks
    rng = np.random.RandomState(42)
    
    best_score = -np.inf
    best_cell = np.zeros((size, size), dtype=int)
    
    # Generate diverse initial 4x4 blocks
    candidates_4x4 = []
    
    # All zeros
    candidates_4x4.append(np.zeros((4, 4), dtype=int))
    # All ones
    candidates_4x4.append(np.ones((4, 4), dtype=int))
    
    # Random blocks
    for _ in range(20):
        block = rng.randint(0, 2, (4, 4))
        candidates_4x4.append(block)
    
    # Some structured blocks
    # Center dot
    block = np.zeros((4, 4), dtype=int)
    block[1, 1] = 1
    block[1, 2] = 1
    block[2, 1] = 1
    block[2, 2] = 1
    candidates_4x4.append(block)
    
    # Cross pattern
    block = np.zeros((4, 4), dtype=int)
    block[1, :] = 1
    block[2, :] = 1
    block[:, 1] = 1
    block[:, 2] = 1
    candidates_4x4.append(block)
    
    # Checkerboard
    block = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                block[i, j] = 1
    candidates_4x4.append(block)
    
    # Evaluate all initial candidates
    cells_4x4 = [_make_superlattice_cell(b, size) for b in candidates_4x4]
    scores_4x4 = _compute_gap_score_batch(cells_4x4, k_points, G_vecs, size, eps_low, eps_high)
    
    for i, s in enumerate(scores_4x4):
        if s > best_score:
            best_score = s
            best_cell = cells_4x4[i].copy()
    
    # Local search on 4x4 blocks
    # There are only a few orbits, so we can do exhaustive or near-exhaustive search
    # For each orbit, try flipping it
    
    current_block = candidates_4x4[np.argmax(scores_4x4)].copy()
    current_score = best_score
    
    # Convert to orbit representation
    orbit_vals = np.zeros(n_orbits_4x4, dtype=int)
    for oi, orbit in enumerate(orbits_4x4):
        i, j = orbit[0]
        orbit_vals[oi] = current_block[i, j]
    
    # Simulated annealing on 4x4 orbit values
    T_start = 1.0
    T_end = 0.05
    n_iter_4x4 = 2000
    
    for it in range(n_iter_4x4):
        if time.time() > deadline - 80:
            break
        
        t_frac = it / max(1, n_iter_4x4 - 1)
        T = T_start + (T_end - T_start) * t_frac
        
        # Pick random orbit to flip
        oi = rng.randint(n_orbits_4x4)
        old_val = orbit_vals[oi]
        orbit_vals[oi] = 1 - old_val
        
        # Reconstruct block
        new_block = np.zeros((4, 4), dtype=int)
        for oi2, orbit in enumerate(orbits_4x4):
            for (i, j) in orbit:
                new_block[i, j] = orbit_vals[oi2]
        
        new_cell = _make_superlattice_cell(new_block, size)
        new_score = _compute_gap_score_fast(new_cell, k_points, G_vecs, size, eps_low, eps_high)
        
        if new_score > current_score:
            current_score = new_score
            current_block = new_block.copy()
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell.copy()
        else:
            delta = new_score - current_score
            if T > 1e-12 and rng.rand() < np.exp(delta / T):
                current_score = new_score
                current_block = new_block.copy()
            else:
                orbit_vals[oi] = old_val
    
    # Phase 2: Refine at full 32x32 resolution using batched local search
    # Get full D4 orbits
    orbits_full = _get_d4_orbits(size)
    n_orbits_full = len(orbits_full)
    
    # Convert best_cell to orbit values
    orbit_vals_full = np.zeros(n_orbits_full, dtype=int)
    for oi, orbit in enumerate(orbits_full):
        i, j = orbit[0]
        orbit_vals_full[oi] = best_cell[i, j]
    
    current_cell = best_cell.copy()
    current_score = best_score
    
    # Batched local search: evaluate multiple flips at once
    batch_size = 50
    n_iter_full = 3000
    
    for it in range(n_iter_full):
        if time.time() > deadline - 30:
            break
        
        # Generate batch of candidate flips
        flip_indices = rng.randint(0, n_orbits_full, batch_size)
        
        # Create batch of cells
        batch_cells = []
        for oi in flip_indices:
            old_val = orbit_vals_full[oi]
            orbit_vals_full[oi] = 1 - old_val
            cell = np.zeros((size, size), dtype=int)
            for oi2, orbit in enumerate(orbits_full):
                for (i, j) in orbit:
                    cell[i, j] = orbit_vals_full[oi2]
            batch_cells.append(cell)
            orbit_vals_full[oi] = old_val  # Revert
        
        # Evaluate batch
        batch_scores = _compute_gap_score_batch(batch_cells, k_points, G_vecs, size, eps_low, eps_high)
        
        # Find best in batch
        best_idx = np.argmax(batch_scores)
        best_batch_score = batch_scores[best_idx]
        
        if best_batch_score > current_score:
            # Accept the best flip
            oi = flip_indices[best_idx]
            orbit_vals_full[oi] = 1 - orbit_vals_full[oi]
            current_cell = batch_cells[best_idx].copy()
            current_score = best_batch_score
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell.copy()
        else:
            # Try a few random single flips with Metropolis
            for _ in range(3):
                oi = rng.randint(n_orbits_full)
                old_val = orbit_vals_full[oi]
                orbit_vals_full[oi] = 1 - old_val
                cell = np.zeros((size, size), dtype=int)
                for oi2, orbit in enumerate(orbits_full):
                    for (i, j) in orbit:
                        cell[i, j] = orbit_vals_full[oi2]
                score = _compute_gap_score_fast(cell, k_points, G_vecs, size, eps_low, eps_high)
                if score > current_score:
                    current_cell = cell.copy()
                    current_score = score
                    if score > best_score:
                        best_score = score
                        best_cell = current_cell.copy()
                else:
                    orbit_vals_full[oi] = old_val
    
    # Phase 3: Try some geometric templates as additional candidates
    templates = []
    
    # Four circles at corners of inner square
    for radius in [3, 4, 5, 6]:
        cell = np.zeros((size, size), dtype=int)
        centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
        for i in range(size):
            for j in range(size):
                for (cx, cy) in centers:
                    dx = i - cx
                    dy = j - cy
                    if dx*dx + dy*dy < radius*radius:
                        cell[i, j] = 1
                        break
        templates.append(cell)
    
    # Cross pattern
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if abs(i - 15.5) < 3 or abs(j - 15.5) < 3:
                cell[i, j] = 1
    templates.append(cell)
    
    # Evaluate templates
    template_scores = _compute_gap_score_batch(templates, k_points, G_vecs, size, eps_low, eps_high)
    for i, s in enumerate(template_scores):
        if s > best_score:
            best_score = s
            best_cell = templates[i].copy()
    
    # Final: ensure symmetry and return
    best_list = [[int(best_cell[i, j]) for j in range(size)] for i in range(size)]
    
    if not _is_symmetric(best_list):
        # Force symmetry by averaging
        sym_cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                # Take majority vote of all D4 symmetric equivalents
                vals = [
                    best_cell[i, j],
                    best_cell[j, i],
                    best_cell[size-1-i, size-1-j],
                    best_cell[size-1-j, i],
                    best_cell[i, size-1-j],
                    best_cell[size-1-i, j],
                    best_cell[size-1-i, size-1-j],
                    best_cell[j, i]
                ]
                sym_cell[i, j] = 1 if sum(vals) >= 4 else 0
        best_list = [[int(sym_cell[i, j]) for j in range(size)] for i in range(size)]
    
    return best_list
