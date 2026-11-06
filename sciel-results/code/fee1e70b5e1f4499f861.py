"""Optimized square-hole photonic crystal design with local search."""
import time
import numpy as np

def design(problem):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # The seed is a dielectric rod in air.
    # We switch to an air hole in dielectric (square hole topology).
    # This is known to open larger TM band gaps in square lattices.
    
    # We will search for the optimal size of the central air square.
    # The cell is symmetric. The hole is centered.
    
    best_score = -1.0
    best_cell = None
    
    # Fast surrogate evaluation function
    def evaluate_cell(cell):
        # Convert cell to numpy array for speed
        eps = np.array(cell, dtype=float)
        eps[eps == 1] = eps_high
        eps[eps == 0] = eps_low
        
        # Use a small plane-wave expansion for speed.
        # N=3 (3x3=9 waves) or N=5 (5x5=25 waves) is fast.
        # Let's use N=5 for better accuracy than N=3.
        N = 5
        # G vectors
        gx = np.arange(-N//2, N//2 + 1)
        gy = np.arange(-N//2, N//2 + 1)
        Gx, Gy = np.meshgrid(gx, gy, indexing='ij')
        G = np.stack([Gx.ravel(), Gy.ravel()], axis=1)
        num_G = len(G)
        
        # Fourier coefficients of 1/epsilon (for TM) or epsilon (for TE)?
        # TM: H field. Equation involves 1/epsilon.
        # We need the Fourier coefficients of 1/epsilon.
        inv_eps = 1.0 / eps
        
        # Calculate Fourier coefficients
        # C_G = 1/S * sum_{r} inv_eps(r) exp(-i G . r)
        # For a 32x32 pixel grid, we can use FFT if we pad, but direct sum is small enough?
        # 32x32 = 1024 points. 25 G vectors. 25 * 1024 ops = 25k ops. Very fast.
        
        x = np.linspace(0, 1, size, endpoint=False)
        y = np.linspace(0, 1, size, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Compute coefficients
        C = np.zeros(num_G, dtype=complex)
        for idx, g in enumerate(G):
            # exp(-i * 2pi * (gx*X + gy*Y))
            phase = np.exp(-1j * 2 * np.pi * (g[0]*X + g[1]*Y))
            C[idx] = np.sum(inv_eps * phase) / (size * size)
        
        # Define k-points in irreducible BZ
        # Gamma (0,0), M (0.5, 0), K (0.5, 0.5)
        # Also some interior points might be needed for accurate min/max, 
        # but for a surrogate, these 3 are good.
        k_points = [
            (0.0, 0.0),
            (0.5, 0.0),
            (0.5, 0.5)
        ]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            # Build eigenvalue problem
            # For TM: (G + k) x (G' + k) / (2pi)^2 * delta_{GG'} - 1/(2pi)^2 * C_{G-G'} * ...
            # Standard form: A H = omega^2 H
            # A_{GG'} = ( (G+k)_x (G'+k)_x + (G+k)_y (G'+k)_y ) * delta_{GG'} - C_{G-G'}
            # Wait, the equation is:
            # sum_{G'} [ (G+k) x (G'+k) delta_{GG'} - C_{G-G'} ] H_{G'} = omega^2 H_G
            # Note: C is Fourier coeff of 1/eps.
            
            A = np.zeros((num_G, num_G), dtype=complex)
            for i in range(num_G):
                for j in range(num_G):
                    Gi = G[i]
                    Gj = G[j]
                    ki = Gi + np.array([kx, ky])
                    kj = Gj + np.array([kx, ky])
                    dot = ki[0]*kj[0] + ki[1]*kj[1]
                    A[i, j] = dot * (np.pi**2) # Scaling factor depends on convention, but ratio is invariant
                    if i == j:
                        pass # delta term is included in dot when i=j? No, delta is explicit in sum.
                    # Actually, the term is (G+k) dot (G'+k) * delta_{GG'}
                    # So if i != j, the delta term is 0.
                    # The C term is always present.
                    # C_{G_i - G_j}
                    diff = Gi - Gj
                    # Find index of diff in G list?
                    # Since G is symmetric and small, we can just search or precompute map.
                    # For surrogate, let's just compute C on the fly if not found?
                    # Better: precompute C for all differences?
                    # Or simpler: since N is small, just loop.
                    
            # This double loop is slow in Python. Let's vectorize or simplify.
            # Given the surrogate nature, maybe just use Gamma point?
            # No, gap is often at M or K.
            
            # Let's stick to a simpler heuristic if the solver is too slow.
            # But 25x25 matrix eig is fast. The bottleneck is filling A.
            
            # Re-construct A efficiently
            # We need C_{G_i - G_j}.
            # Let's create a dictionary for C.
            C_dict = {}
            for idx, g in enumerate(G):
                C_dict[tuple(g)] = C[idx]
            
            for i in range(num_G):
                ki = G[i] + np.array([kx, ky])
                for j in range(num_G):
                    kj = G[j] + np.array([kx, ky])
                    dot = ki[0]*kj[0] + ki[1]*kj[1]
                    val = dot
                    if i == j:
                        pass # delta term is just dot
                    else:
                        val = 0.0 # delta term is 0
                    # Subtract C term
                    diff = tuple(G[i] - G[j])
                    c_val = C_dict.get(diff, 0.0)
                    A[i, j] = val - c_val
            
            # Solve eigenvalues
            # A is Hermitian?
            # (G+k) dot (G'+k) is real and symmetric.
            # C_{G-G'} is Fourier coeff of real function, so C_{-h} = conj(C_h).
            # Thus A is Hermitian.
            try:
                eigvals = np.linalg.eigvalsh(A)
                # Filter out negative/zero due to numerical noise?
                eigvals = np.sort(eigvals.real)
                if len(eigvals) >= 5:
                    # We want bands 4 and 5 (1-indexed). So indices 3 and 4.
                    # But wait, the first few bands might be spurious or low.
                    # In a 32x32 cell, we expect many bands.
                    # The "4th and 5th TM bands" refers to the global band index.
                    # With N=5, we only have 25 bands.
                    # If the 4th and 5th bands are within the first 25, we can find them.
                    # Usually, the gap of interest is in the lower spectrum.
                    w4 = eigvals[3]
                    w5 = eigvals[4]
                    
                    # Update global min/max
                    if w5 < min_omega_5: min_omega_5 = w5
                    if w4 > max_omega_4: max_omega_4 = w4
                    
            except Exception:
                pass
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    # Search for best square hole size
    # The hole is a square of air in the center.
    # Size is determined by 'half_width' in pixels.
    # Center is at 15.5, 15.5.
    # Pixels are 0..31.
    # A square from i_start to i_end.
    # To be symmetric about center, if we pick a size S (odd or even), we center it.
    
    # Let's try half-widths w = 1 to 10.
    # If w=1, we turn off pixels (15,15) to (16,16)?
    # Let's define the mask based on distance from center.
    # Center is (15.5, 15.5).
    # Pixel (i,j) is in hole if |i - 15.5| <= w and |j - 15.5| <= w.
    
    best_w = 5 # Default fallback
    best_surrogate_score = -1.0
    
    for w in range(1, 12):
        if time.time() > deadline - 5.0:
            break
            
        cell = []
        for i in range(size):
            row = []
            for j in range(size):
                # Check if in central square
                if abs(i - 15.5) <= w and abs(j - 15.5) <= w:
                    row.append(0) # Air
                else:
                    row.append(1) # Dielectric
            cell.append(row)
            
        score = evaluate_cell(cell)
        if score > best_surrogate_score:
            best_surrogate_score = score
            best_w = w
            best_cell = [row[:] for row in cell] # Deep copy
            
    if best_cell is None:
        # Fallback to a reasonable square hole
        w = 5
        cell = []
        for i in range(size):
            row = []
            for j in range(size):
                if abs(i - 15.5) <= w and abs(j - 15.5) <= w:
                    row.append(0)
                else:
                    row.append(1)
            cell.append(row)
        best_cell = cell

    return best_cell
