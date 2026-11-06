"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]  # 32
    eps_high = problem.get("eps_high", 12)
    eps_low = problem.get("eps_low", 1)
    
    start_time = time.time()
    deadline = 220.0  # seconds, with safety margin before 230s hard kill
    margin = 15.0  # seconds before hard kill
    
    def elapsed():
        return time.time() - start_time
    
    def time_ok():
        return elapsed() < deadline - margin

    def enforce_symmetry(cell):
        """Enforce full d4 (square) symmetry by averaging over the symmetry group."""
        n = len(cell)
        sym_cell = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                vals = [
                    cell[i][j],
                    cell[n - 1 - i][n - 1 - j],
                    cell[j][i],
                    cell[n - 1 - j][n - 1 - i],
                ]
                sym_cell[i][j] = 1 if sum(vals) >= 2 else 0
        return sym_cell

    def check_symmetry(cell):
        """Check if cell has the required square symmetry."""
        n = len(cell)
        for i in range(n):
            for j in range(n):
                if cell[i][j] != cell[n - 1 - i][n - 1 - j]:
                    return False
                if cell[i][j] != cell[j][i]:
                    return False
        return True

    # Build reciprocal lattice vectors and Fourier coefficients using FFT
    def build_fourier_data(cell, n_gmax=8):
        """
        Compute the inverse-epsilon matrix Fourier coefficients needed for PWE.
        Returns inv_eps_coeffs: a dict mapping (mx, my) -> complex coefficient
        """
        n = size
        inv_eps = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if cell[i][j] == 1:
                    inv_eps[i, j] = 1.0 / eps_high
                else:
                    inv_eps[i, j] = 1.0 / eps_low
        
        # FFT to get Fourier coefficients
        fft_inv = np.fft.fft2(inv_eps)
        fft_inv = fft_inv / (n * n)
        
        inv_eps_coeffs = {}
        for mx in range(-n_gmax, n_gmax + 1):
            for my in range(-n_gmax, n_gmax + 1):
                # Map negative indices to FFT indices
                idx_x = mx % n
                idx_y = my % n
                inv_eps_coeffs[(mx, my)] = fft_inv[idx_x, idx_y]
        
        return inv_eps_coeffs

    # Precompute k-points in the irreducible Brillouin zone
    # The irreducible BZ of a square lattice is the triangle: kx >= ky, kx >= 0, kx + ky <= 1 (in units of pi/a)
    # Actually, the reciprocal lattice vectors are 2*pi, so the first BZ is [-pi, pi] x [-pi, pi].
    # The irreducible BZ (1/8 of the full BZ) is: 0 <= ky <= kx, kx + ky <= 1.0 (in units of pi/a? No, let's be precise)
    # Let's use units where the reciprocal lattice vector is 2*pi. The first BZ is [-pi, pi] in each direction.
    # The symmetry group D4 maps the BZ to 8 equivalent triangles. One such triangle has vertices at:
    # (0,0), (pi, 0), and (pi/2, pi/2). Wait, the full BZ corner is (pi, pi).
    # The 1/8 triangle is bounded by kx=0, ky=kx, and kx+ky=pi? No.
    # The standard irreducible BZ for square lattice D4 symmetry:
    # Vertices: Gamma(0,0), X(pi, 0), M(pi, pi).
    # The fundamental domain is the triangle Gamma-X-M? No, that's 1/4.
    # D4 has 8 elements. The BZ area is (2*pi)^2 = 4*pi^2.
    # The irreducible BZ area should be (4*pi^2)/8 = pi^2/2.
    # The triangle with vertices (0,0), (pi,0), (pi,pi) has area 1/2 * pi * pi = pi^2/2.
    # But this triangle is NOT symmetric under all D4 operations.
    # Let's use the standard one: 0 <= ky <= kx, kx <= pi, kx + ky <= pi.
    # Wait, the point (pi, 0) is on the boundary. The point (pi/2, pi/2) is on the boundary kx+ky=pi.
    # The vertices are (0,0), (pi,0), and (pi/2, pi/2).
    # Area = 1/2 * base * height. Base from (0,0) to (pi,0) is pi. Height of (pi/2,pi/2) above x-axis is pi/2.
    # Area = 1/2 * pi * pi/2 = pi^2/4. That's 1/4 of the BZ, not 1/8.
    # I need to be more careful.
    # The BZ is [-pi, pi] x [-pi, pi].
    # Symmetry operations:
    # 1. Identity
    # 2. Rotation 90 deg: (kx, ky) -> (-ky, kx)
    # 3. Rotation 180 deg: (kx, ky) -> (-kx, -ky)
    # 4. Rotation 270 deg: (kx, ky) -> (ky, -kx)
    # 5. Reflection across x-axis: (kx, ky) -> (kx, -ky)
    # 6. Reflection across y-axis: (kx, ky) -> (-kx, ky)
    # 7. Reflection across kx=ky: (kx, ky) -> (ky, kx)
    # 8. Reflection across kx=-ky: (kx, ky) -> (-ky, -kx)
    #
    # A fundamental domain can be defined by:
    # kx >= 0, ky >= 0, kx >= ky, kx + ky <= pi.
    # Let's check the vertices of this region:
    # 1. Intersection of kx=0, ky=0: (0,0)
    # 2. Intersection of kx=0, kx+ky=pi: (0, pi) -> but kx>=ky is violated if ky=pi, kx=0.
    #    Actually, kx>=ky and kx>=0, ky>=0 implies we're in the first quadrant with kx>=ky.
    #    The line kx+ky=pi intersects kx=ky at (pi/2, pi/2).
    #    It intersects ky=0 at (pi, 0).
    #    So the vertices are: (0,0), (pi, 0), (pi/2, pi/2).
    #    Wait, is (0,0) a vertex? The region is bounded by kx=0, ky=kx, and kx+ky=pi.
    #    At kx=0, ky must be 0 (since ky>=0 and kx>=ky). So (0,0) is a corner.
    #    At ky=0, kx can go from 0 to pi. So (pi,0) is a corner.
    #    At kx+ky=pi and kx=ky, we get (pi/2, pi/2).
    #    So the triangle is (0,0) - (pi,0) - (pi/2, pi/2).
    #    Area = 1/2 * |det([pi, 0], [pi/2, pi/2])| = 1/2 * |pi*pi/2 - 0| = pi^2/4.
    #    This is still 1/4.
    #    
    #    I think I'm making a mistake. Let's reconsider.
    #    The full BZ has 8 symmetry-equivalent regions.
    #    The triangle (0,0)-(pi,0)-(pi/2,pi/2) is actually 1/8 of the BZ if we also account for the fact that the BZ is a square.
    #    Let's just generate points on a grid in the full BZ and pick unique ones.
    #    For simplicity, I'll generate points in the triangle (0,0)-(pi,0)-(pi/2,pi/2) and trust that this is the standard irreducible BZ.
    #    Many references use this triangle for square lattice photonic crystals.
    
    # Generate k-points
    n_k = 6  # Number of points along each edge of the triangle
    k_points = []
    # Triangle vertices: A=(0,0), B=(pi,0), C=(pi/2, pi/2)
    for i in range(n_k):
        for j in range(n_k - i):
            # Point = A + (i/(n_k-1))*(B-A) + (j/(n_k-1-i))*(C-B)
            # Actually, let's use barycentric or a simple grid.
            # Let u = i/(n_k-1), v = j/(n_k-1-i) if i < n_k-1 else 0
            # This is getting complicated. Let's just do a simple grid in the triangle.
            pass
    
    # Simpler approach: generate a grid in the full BZ and keep unique points
    k_grid_size = 8
    k_points_set = set()
    k_points = []
    for ix in range(k_grid_size):
        for iy in range(k_grid_size):
            kx = -math.pi + 2*math.pi*ix/(k_grid_size-1)
            ky = -math.pi + 2*math.pi*iy/(k_grid_size-1)
            # Normalize to irreducible BZ
            # Apply symmetries to map to fundamental domain
            candidates = [
                (kx, ky),
                (-kx, -ky),
                (ky, kx),
                (-ky, -kx),
                (kx, -ky),
                (-kx, ky),
                (-ky, kx),
                (ky, -kx),
            ]
            # Pick the one in the fundamental domain: kx>=0, ky>=0, kx>=ky
            best = None
            for ckx, cky in candidates:
                if ckx >= -1e-9 and cky >= -1e-9 and ckx >= cky - 1e-9 and ckx <= math.pi + 1e-9 and cky <= math.pi + 1e-9:
                    if best is None or (ckx, cky) < (best[0], best[1]):
                        best = (ckx, cky)
            if best is not None:
                key = (round(best[0], 6), round(best[1], 6))
                if key not in k_points_set:
                    k_points_set.add(key)
                    k_points.append(best)
    
    # Ensure we have at least a few key points
    if len(k_points) < 10:
        # Add specific high-symmetry points
        high_sym = [
            (0, 0),
            (math.pi, 0),
            (math.pi/2, math.pi/2),
            (math.pi/4, 0),
            (math.pi/2, 0),
            (math.pi/3, math.pi/6),
        ]
        for pt in high_sym:
            key = (round(pt[0], 6), round(pt[1], 6))
            if key not in k_points_set:
                k_points_set.add(key)
                k_points.append(pt)
    
    print(f"Using {len(k_points)} k-points")
    
    def solve_bands(cell, n_pw=5, k_points_list=None):
        """
        Solve for the first 5 TM bands at given k-points.
        Returns (min_omega_5, max_omega_4) in units where a=1.
        """
        if k_points_list is None:
            k_points_list = k_points
        
        inv_eps_coeffs = build_fourier_data(cell, n_gmax=n_pw)
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points_list:
            # Build G-vector list
            G_list = []
            for mx in range(-n_pw, n_pw + 1):
                for my in range(-n_pw, n_pw + 1):
                    Gx = 2*math.pi*mx + kx
                    Gy = 2*math.pi*my + ky
                    G_list.append((Gx, Gy))
            
            n_G = len(G_list)
            if n_G == 0:
                continue
            
            # Build the matrix A where A_{ij} = delta_{ij} * |k+G_i|^2 / eps_{G_i - G_j}
            # For TM: (k+G)^2 / eps_{G-G'} c_{G'} = omega^2 c_G
            A = np.zeros((n_G, n_G), dtype=np.complex128)
            
            for i, (Gxi, Gyi) in enumerate(G_list):
                kGi2 = Gxi*Gxi + Gyi*Gyi
                for j, (Gxj, Gyj) in enumerate(G_list):
                    dGx = mx - (j - n_pw)  # This is wrong, let's compute properly
                    # Actually, we need G_i - G_j
                    # G_i = (2*pi*m_i + kx, 2*pi*n_i + ky)
                    # G_j = (2*pi*m_j + kx, 2*pi*n_j + ky)
                    # G_i - G_j = (2*pi*(m_i-m_j), 2*pi*(n_i-n_j))
                    mi = i // (2*n_pw+1)
                    ni = i % (2*n_pw+1)
                    mj = j // (2*n_pw+1)
                    nj = j % (2*n_pw+1)
                    dGx = 2*math.pi*(mi - mj)
                    dGy = 2*math.pi*(ni - nj)
                    A[i, j] = kGi2 * inv_eps_coeffs.get((mi-mj, ni-nj), 0.0)
            
            # Solve eigenvalue problem
            try:
                eigvals, eigvecs = np.linalg.eig(A)
                # Take real parts (should be real)
                eigvals_real = np.real(eigvals)
                eigvals_real = np.sort(eigvals_real)
                if len(eigvals_real) >= 5:
                    omega_4 = math.sqrt(max(0, eigvals_real[3]))
                    omega_5 = math.sqrt(max(0, eigvals_real[4]))
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)
            except:
                continue
        
        if min_omega_5 == float('inf') or max_omega_4 == 0.0:
            return 0.0, 0.0
        
        return min_omega_5, max_omega_4
    
    def compute_gap(cell):
        """Compute the gap-midgap ratio."""
        min_omega_5, max_omega_4 = solve_bands(cell, n_pw=4)
        if min_omega_5 + max_omega_4 == 0:
            return 0.0
        gap = 2 * (min_omega_5 - max_omega_4)
        midgap = min_omega_5 + max_omega_4
        return gap / midgap
    
    # Generate diverse candidate patterns
    def make_cell_from_pattern(pattern_func):
        """Create a cell from a pattern function."""
        n = size
        cell = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                x = (i + 0.5) / n - 0.5
                y = (j + 0.5) / n - 0.5
                cell[i][j] = 1 if pattern_func(x, y) else 0
        return enforce_symmetry(cell)
    
    def four_rod_pattern(radius, offset):
        def f(x, y):
            for sx in [1, -1]:
                for sy in [1, -1]:
                    dx = x - sx * offset
                    dy = y - sy * offset
                    if dx * dx + dy * dy < radius * radius:
                        return True
            return False
        return f
    
    def ring_pattern(inner_r, outer_r):
        def f(x, y):
            r2 = x * x + y * y
            return inner_r * inner_r < r2 < outer_r * outer_r
        return f
    
    def four_rings_pattern(inner_r, outer_r, offset):
        def f(x, y):
            for sx in [1, -1]:
                for sy in [1, -1]:
                    dx = x - sx * offset
                    dy = y - sy * offset
                    r2 = dx * dx + dy * dy
                    if inner_r * inner_r < r2 < outer_r * outer_r:
                        return True
            return False
        return f
    
    candidates = []
    
    # Four-rod patterns
    for r in [0.06, 0.08, 0.10, 0.12]:
        for off in [0.20, 0.25, 0.30]:
            candidates.append(four_rod_pattern(r, off))
    
    # Ring patterns
    for ir, orr in [(0.10, 0.20), (0.15, 0.25), (0.12, 0.22)]:
        candidates.append(ring_pattern(ir, orr))
    
    # Four small rings
    for ir, orr in [(0.04, 0.10), (0.06, 0.14)]:
        for off in [0.22, 0.28]:
            candidates.append(four_rings_pattern(ir, orr, off))
    
    # Evaluate all candidates using the actual gap computation
    best_cell = None
    best_gap = -float('inf')
    
    for idx, pattern_func in enumerate(candidates):
        if not time_ok():
            break
        
        cell = make_cell_from_pattern(pattern_func)
        
        if not check_symmetry(cell):
            continue
        
        gap = compute_gap(cell)
        
        if gap > best_gap:
            best_gap = gap
            best_cell = cell
    
    # Local search: try to improve the best candidate by flipping pixels
    if best_cell is not None and time_ok():
        current_best = best_cell
        current_gap = best_gap
        n = size
        
        for iteration in range(100):
            if not time_ok():
                break
            
            improved = False
            for _ in range(30):
                if not time_ok():
                    break
                
                ri = random.randint(0, n - 1)
                rj = random.randint(0, n - 1)
                
                new_cell = [row[:] for row in current_best]
                orbit = set()
                i, j = ri, rj
                orbit.add((i, j))
                orbit.add((n - 1 - i, n - 1 - j))
                orbit.add((j, i))
                orbit.add((n - 1 - j, n - 1 - i))
                
                for (oi, oj) in orbit:
                    new_cell[oi][oj] = 1 - new_cell[oi][oj]
                
                new_gap = compute_gap(new_cell)
                if new_gap > current_gap:
                    current_best = new_cell
                    current_gap = new_gap
                    best_cell = new_cell
                    best_gap = new_gap
                    improved = True
                    break
            
            if not improved and time_ok():
                # Try multi-pixel flip
                new_cell = [row[:] for row in current_best]
                for _ in range(3):
                    ri = random.randint(0, n - 1)
                    rj = random.randint(0, n - 1)
                    orbit = set()
                    i, j = ri, rj
                    orbit.add((i, j))
                    orbit.add((n - 1 - i, n - 1 - j))
                    orbit.add((j, i))
                    orbit.add((n - 1 - j, n - 1 - i))
                    for (oi, oj) in orbit:
                        new_cell[oi][oj] = 1 - new_cell[oi][oj]
                
                new_gap = compute_gap(new_cell)
                if new_gap > current_gap:
                    current_best = new_cell
                    current_gap = new_gap
                    best_cell = new_cell
                    best_gap = new_gap
    
    # If no valid cell was found, use a default
    if best_cell is None:
        best_cell = make_cell_from_pattern(four_rod_pattern(0.10, 0.25))
    
    return best_cell
# EVOLVE-BLOCK-END
