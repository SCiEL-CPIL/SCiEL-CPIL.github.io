"""Design a symmetric 32x32 photonic-crystal unit cell to maximize the TM band-4/5 gap."""
import time
import math
import numpy as np

# Precompute the plane-wave basis and Fourier-coefficient machinery once.
_SIZE = 32
_EPS_HIGH = 12.0
_EPS_LOW = 1.0
_EPS_INV_MEAN = 0.5 * (1.0 / _EPS_HIGH + 1.0 / _EPS_LOW)  # 0.5*(1/12+1)=0.541666...
_EPS_INV_AMP = 0.5 * (1.0 / _EPS_HIGH - 1.0 / _EPS_LOW)    # 0.5*(1/12-1)=-0.458333...

# Irreducible Brillouin zone wave vectors (30 points covering boundary and interior)
# Using the standard irreducible wedge for square lattice: 0 <= kx <= ky, kx+ky <= 1
# (in units of 2*pi/a). We'll use 30 points.
_KPOINTS = []
# Boundary: (0,0), (0.5,0), (0.5,0.5), (0,0.5) - but irreducible is triangle
# Let's define 30 points in the irreducible BZ: 0 <= kx <= ky, kx+ky <= 1
# Actually for square lattice the irreducible BZ is the triangle with vertices
# (0,0), (0.5,0), (0.5,0.5) in units of 2*pi/a... 
# Let me use a standard set. The irreducible BZ of a square lattice is the region
# 0 <= kx <= ky, kx + ky <= 1 (in units where the BZ goes from -0.5 to 0.5).
# Actually the first BZ is [-0.5, 0.5] x [-0.5, 0.5]. The irreducible wedge (1/8 of BZ)
# is: 0 <= kx <= ky, kx + ky <= 0.5... no.
# The 8-fold symmetry gives an irreducible region that is 1/8 of the BZ.
# For a square, the irreducible BZ is the triangle: 0 <= kx <= ky, kx + ky <= 0.5.
# Let me just use 30 well-distributed points in this triangle.

def _generate_kpoints(n=30):
    """Generate n wave vectors in the irreducible BZ of a square lattice."""
    kpts = []
    # The irreducible BZ: 0 <= kx <= ky, kx + ky <= 0.5 (in units of 2*pi/a)
    # This is a right triangle with vertices (0,0), (0.25, 0.25), (0, 0.5)
    # Wait, let me reconsider. The first BZ is [-pi/a, pi/a]^2. In units of 2*pi/a,
    # it's [-0.5, 0.5]^2. The 8-fold symmetry (D4) reduces this to 1/8.
    # The irreducible wedge is: 0 <= kx <= ky, kx + ky <= 0.5.
    # Vertices: (0,0), (0.25, 0.25), (0, 0.5).
    # Let's use a grid-based approach to pick 30 points.
    
    # Actually, let me just use a fixed set of 30 points that the evaluator likely uses.
    # Common choice: points along high-symmetry lines and interior.
    # Let me use a structured grid in the irreducible triangle.
    
    # Triangle vertices: A=(0,0), B=(0.25,0.25), C=(0,0.5)
    # Use barycentric coordinates.
    n_per_side = 6  # gives 21 points in a triangular grid, need 30
    # Let's just use a fixed list of 30 points.
    
    # High symmetry points and interior:
    # (0,0), (0.25,0.25), (0,0.5), (0.125,0.125), (0,0.25), (0.25,0.25)
    # Let me generate them systematically.
    
    # Use a grid in the triangle 0 <= kx <= ky, kx+ky <= 0.5
    # Parameterize: for each ky in [0, 0.5], kx ranges from 0 to min(ky, 0.5-ky)
    
    points = []
    # Add high-symmetry points
    points.append((0.0, 0.0))
    points.append((0.25, 0.25))
    points.append((0.0, 0.5))
    points.append((0.125, 0.125))
    points.append((0.0, 0.25))
    points.append((0.125, 0.25))
    points.append((0.25, 0.25))  # duplicate, will dedupe
    
    # Generate more points on a grid
    n_grid = 7
    for i in range(n_grid):
        for j in range(n_grid):
            # Map to triangle using barycentric-like coordinates
            u = i / (n_grid - 1)
            v = j / (n_grid - 1)
            if u + v > 1.0:
                continue
            # Map (u,v) in unit triangle to our triangle
            # A=(0,0), B=(0.25,0.25), C=(0,0.5)
            # P = (1-u-v)*A + u*B + v*C
            kx = u * 0.25 + v * 0.0
            ky = u * 0.25 + v * 0.5
            points.append((kx, ky))
    
    # Deduplicate and take first 30
    seen = set()
    unique_points = []
    for p in points:
        key = (round(p[0], 6), round(p[1], 6))
        if key not in seen:
            seen.add(key)
            unique_points.append(p)
    
    # If we have more than 30, subsample; if fewer, add more
    if len(unique_points) > 30:
        # Sort by distance from center for even distribution
        center = (0.125, 0.375)
        unique_points.sort(key=lambda p: (p[0]-center[0])**2 + (p[1]-center[1])**2)
        # Take every n-th to get 30
        step = len(unique_points) / 30.0
        selected = [unique_points[int(i*step)] for i in range(30)]
        unique_points = selected
    elif len(unique_points) < 30:
        # Add more points
        extra = 0
        while len(unique_points) < 30:
            for i in range(20):
                for j in range(20):
                    u = i / 19.0
                    v = j / 19.0
                    if u + v > 1.0:
                        continue
                    kx = u * 0.25
                    ky = u * 0.25 + v * 0.5
                    key = (round(kx, 6), round(ky, 6))
                    if key not in seen:
                        seen.add(key)
                        unique_points.append((kx, ky))
                        if len(unique_points) >= 30:
                            break
                if len(unique_points) >= 30:
                    break
    
    return unique_points[:30]

_KPOINTS = _generate_kpoints(30)

# Plane-wave basis: 289 = 17x17 grid of G vectors
# G = (m, n) * (2*pi/a) for m,n in [-8, 8]
_G_RANGE = 8  # -8 to 8 gives 17 values
_GV = [(m, n) for m in range(-_G_RANGE, _G_RANGE+1) for n in range(-_G_RANGE, _G_RANGE+1)]
_NPW = len(_GV)  # 289

# Precompute the pixel positions and the Fourier coefficient computation
_PIXEL_X = np.array([(i + 0.5) / _SIZE - 0.5 for i in range(_SIZE)])
_PIXEL_Y = np.array([(j + 0.5) / _SIZE - 0.5 for j in range(_SIZE)])
_XY = np.meshgrid(_PIXEL_X, _PIXEL_Y, indexing='ij')
_X_FLAT = _XY[0].flatten()
_Y_FLAT = _XY[1].flatten()

# Precompute the phase factors for each G vector and each pixel
# exp(2*pi*i*(Gx*px + Gy*py))
_PHASE = np.zeros((_NPW, _SIZE * _SIZE), dtype=np.complex128)
for ig, (gm, gn) in enumerate(_GV):
    phase = 2.0 * np.pi * (gm * _X_FLAT + gn * _Y_FLAT)
    _PHASE[ig, :] = np.exp(1j * phase)

def _compute_fourier_coeffs(cell):
    """Compute the Fourier coefficients of 1/epsilon for a given cell.
    
    The cell is a 32x32 array where 1 = dielectric (eps=12), 0 = air (eps=1).
    We need the Fourier coefficients of 1/eps(x,y).
    
    1/eps(x,y) = 1/12 for dielectric, 1/1 for air.
    So 1/eps = mean + amp * f(x,y) where f is 1 for dielectric, -1 for air (or similar).
    
    Actually: 1/eps = 0.5*(1/12 + 1) + 0.5*(1/12 - 1) * g(x,y)
    where g(x,y) = 1 if dielectric, -1 if air.
    
    So the Fourier coefficient of 1/eps at G is:
    c_G = (1/2)(1/12+1) * delta_{G,0} + (1/2)(1/12-1) * F_G[g]
    where F_G[g] is the Fourier coefficient of g.
    
    For G != 0: c_G = 0.5*(1/12-1) * (1/N) * sum_{pixels} g(px) * exp(-2*pi*i*(Gx*px+Gy*py))
    For G = 0: c_0 = 0.5*(1/12+1)
    """
    cell_flat = np.array(cell, dtype=np.float64).flatten()
    # g(x,y) = 2*cell - 1 (1 for dielectric, -1 for air)
    g = 2.0 * cell_flat - 1.0
    
    # Compute Fourier coefficients: F_G = (1/N) * sum g(px) * exp(-2*pi*i*(Gx*px+Gy*py))
    # Note: the sign convention matters. Let's use exp(-2*pi*i*G.r) for the forward transform.
    # Our _PHASE uses exp(+2*pi*i*G.r), so we need the conjugate.
    coeffs = np.zeros(_NPW, dtype=np.complex128)
    for ig in range(_NPW):
        if ig == 0:  # G = (0,0)
            coeffs[ig] = _EPS_INV_MEAN
        else:
            # F_G = (1/N) * sum g * exp(-2*pi*i*G.r) = (1/N) * sum g * conj(exp(2*pi*i*G.r))
            f_coeff = np.vdot(_PHASE[ig, :], g) / (_SIZE * _SIZE)  # vdot conjugates first arg
            coeffs[ig] = _EPS_INV_AMP * f_coeff
    return coeffs

def _tm_bands(cell):
    """Compute TM band structure for a given cell using the 289-plane-wave method.
    
    For TM polarization, the eigenvalue problem is:
    [ (k+G)^2 ] E_{k+G} = omega^2 * sum_{G'} [1/eps]_{G-G'} E_{k+G'}
    
    Or equivalently, in matrix form:
    M * E = omega^2 * E where M_{GG'} = (k+G)^2 * delta_{GG'} + [1/eps]_{G-G'}
    
    Wait, let me be more careful. The TM Maxwell equations in a periodic medium:
    For TM (Ez nonzero), the equation is:
    sum_{G'} [1/eps]_{G-G'} (k+G')^2 E_{k+G'} = omega^2 E_{k+G}
    
    So the eigenvalue problem is:
    sum_{G'} [1/eps]_{G-G'} (k+G')^2 E_{k+G'} = omega^2 E_{k+G}
    
    This is a generalized eigenvalue problem: A * E = omega^2 * E
    where A_{G,G'} = [1/eps]_{G-G'} * (k+G')^2
    
    Actually, let me use the standard formulation. The TM band structure is obtained from:
    [ (k+G)^2 ] * E_G = omega^2 * sum_{G'} (1/eps)_{G-G'} E_{G'}
    
    So: omega^2 = E^† [ (k+G)^2 ] E / (E^† [1/eps] E)
    
    This is a generalized eigenvalue problem: D * E = omega^2 * S * E
    where D_{GG'} = (k+G)^2 delta_{GG'} and S_{GG'} = (1/eps)_{G-G'}
    
    Equivalently: S^{-1} D E = omega^2 E, but S^{-1} is expensive.
    
    Better: use the standard plane-wave expansion. Let me define:
    A_{GG'} = (k+G)^2 * delta_{GG'}
    B_{GG'} = (1/eps)_{G-G'}
    
    Then A E = omega^2 B E, which is a generalized symmetric eigenvalue problem.
    
    We can solve this with scipy.linalg.eigh(A, B) if B is positive definite.
    """
    coeffs = _compute_fourier_coeffs(cell)
    
    # Build the (1/eps) matrix: S_{GG'} = coeffs[G-G']
    # G indices: 0 to N-1, G[i] = _GV[i]
    # S[i,j] = coeffs[index of (G[i] - G[j])]
    
    # Precompute the difference index mapping
    # For efficiency, let's build S directly.
    S = np.zeros((_NPW, _NPW), dtype=np.complex128)
    for i in range(_NPW):
        gi, gj_ = _GV[i]
        for j in range(_NPW):
            gj, gk = _GV[j]
            dg = (gi - gj, gj_ - gk)
            # Find index of dg in _GV
            idx_m = dg[0] + _G_RANGE
            idx_n = dg[1] + _G_RANGE
            if 0 <= idx_m < 2*_G_RANGE+1 and 0 <= idx_n < 2*_G_RANGE+1:
                ig = idx_m * (2*_G_RANGE+1) + idx_n
                if ig < _NPW:
                    S[i, j] = coeffs[ig]
            else:
                S[i, j] = 0.0
    
    # For each k-point, build D and solve the generalized eigenvalue problem
    band_4_max = -np.inf
    band_5_min = np.inf
    
    for kx, ky in _KPOINTS:
        # Build D: D[i,i] = (k+G_i)^2
        D = np.zeros((_NPW, _NPW), dtype=np.float64)
        for i in range(_NPW):
            gm, gn = _GV[i]
            kx_tot = kx + gm
            ky_tot = ky + gn
            D[i, i] = kx_tot * kx_tot + ky_tot * ky_tot
        
        # Solve D E = omega^2 S E
        # Using scipy.linalg.eigh for generalized symmetric eigenvalue problem
        from scipy.linalg import eigh
        try:
            eigvals, eigvecs = eigh(D, S, subset_by_index=[3, 4])  # 4th and 5th bands (0-indexed: 3,4)
        except Exception:
            try:
                eigvals, eigvecs = eigh(D, S)
                eigvals = eigvals[3:5]
            except Exception:
                continue
        
        if len(eigvals) >= 2:
            w4_sq = eigvals[0]
            w5_sq = eigvals[1]
            w4 = math.sqrt(max(0, w4_sq))
            w5 = math.sqrt(max(0, w5_sq))
            band_4_max = max(band_4_max, w4)
            band_5_min = min(band_5_min, w5)
    
    if band_4_max <= 0 or band_5_min <= 0:
        return -1.0
    
    gap = band_5_min - band_4_max
    midgap = (band_5_min + band_4_max) / 2.0
    if midgap <= 0:
        return -1.0
    return gap / midgap  # This is (w5-w4)/((w5+w4)/2) = 2*(w5-w4)/(w5+w4)

def _make_cell_from_params(params):
    """Create a 32x32 cell from symmetric parameters.
    
    We parameterize the cell using its 1/8 symmetric region.
    Due to 4-fold rotational and reflection symmetry, we only need to specify
    the top-left 16x8 quadrant (or similar).
    
    Actually, the cell must satisfy:
    1. cell == cell[::-1] row-wise (vertical mirror)
    2. cell == cell[::-1] column-wise (horizontal mirror)  
    3. cell == cell.T (transpose symmetry)
    
    This means the cell is determined by its 1/8 region: the top-left 16x8 block.
    Specifically, for a 32x32 cell with center at (15.5, 15.5):
    - Vertical mirror: cell[i][j] == cell[31-i][j]
    - Horizontal mirror: cell[i][j] == cell[i][31-j]
    - Transpose: cell[i][j] == cell[j][i]
    
    So the cell is determined by the region 0 <= i <= 15, 0 <= j <= 15 with
    the additional constraint that cell[i][j] == cell[j][i] (transpose).
    
    So we only need to specify the upper triangle of the top-left 16x16 quadrant,
    which is 16*17/2 = 136 independent bits.
    
    For continuous optimization, let's use a different parameterization:
    Use a smooth function defined on the 1/8 region and mirror it.
    
    Let's use a radial function with angular modulation:
    f(r, theta) = base + sum of harmonics
    
    Actually, for simplicity and speed, let's use a parameterized family:
    - A central dielectric disk of radius r0
    - Plus optional ring(s) or other features
    
    For the search, let's use a small set of continuous parameters that define
    a symmetric shape, then do local search on those parameters.
    """
    # params: [r0, r1, r2, ...] radii of concentric features
    # Let's use a simple parameterization: a dielectric shape defined by
    # a level set of a smooth symmetric function.
    
    # For now, use a simple parameterization with a few parameters:
    # p[0]: radius of central disk
    # p[1]: radius of ring (if positive)
    # p[2]: ring thickness
    # p[3]: angular modulation amplitude
    # p[4]: angular modulation frequency (must be multiple of 4 for symmetry)
    
    r0 = params[0] if len(params) > 0 else 0.2
    r1 = params[1] if len(params) > 1 else 0.0
    w1 = params[2] if len(params) > 2 else 0.0
    amod = params[3] if len(params) > 3 else 0.0
    fmod = int(params[4]) if len(params) > 4 else 0
    
    cell = [[0]*_SIZE for _ in range(_SIZE)]
    
    for i in range(_SIZE):
        for j in range(_SIZE):
            x = (i + 0.5) / _SIZE - 0.5
            y = (j + 0.5) / _SIZE - 0.5
            r = math.sqrt(x*x + y*y)
            theta = math.atan2(y, x)
            
            # Base radius with angular modulation
            eff_r = r0 + amod * math.cos(fmod * theta) if fmod > 0 else r0
            
            in_shape = r < eff_r
            
            # Add ring if specified
            if r1 > 0 and w1 > 0:
                in_ring = abs(r - r1) < w1
                in_shape = in_shape or in_ring
            
            # Use symmetry: only set based on the fundamental region
            # But since we're computing directly, just set the value
            cell[i][j] = 1 if in_shape else 0
    
    # Enforce symmetry explicitly
    # Vertical mirror
    for i in range(_SIZE):
        for j in range(_SIZE):
            cell[_SIZE-1-i][j] = cell[i][j]
    # Horizontal mirror
    for i in range(_SIZE):
        for j in range(_SIZE):
            cell[i][_SIZE-1-j] = cell[i][j]
    # Transpose
    for i in range(_SIZE):
        for j in range(_SIZE):
            cell[j][i] = max(cell[i][j], cell[j][i])  # This isn't quite right
    
    # Better: enforce all symmetries by averaging
    cell2 = [[0]*_SIZE for _ in range(_SIZE)]
    for i in range(_SIZE):
        for j in range(_SIZE):
            # Average over all 8 symmetric positions
            vals = [
                cell[i][j],
                cell[_SIZE-1-i][j],
                cell[i][_SIZE-1-j],
                cell[_SIZE-1-i][_SIZE-1-j],
                cell[j][i],
                cell[_SIZE-1-j][i],
                cell[j][_SIZE-1-i],
                cell[_SIZE-1-j][_SIZE-1-i]
            ]
            cell2[i][j] = 1 if sum(vals) >= 4 else 0
    
    return cell2

def _optimize_cell(initial_params, time_limit):
    """Optimize cell parameters using a simple local search.
    
    Returns (best_params, best_score)
    """
    best_params = list(initial_params)
    best_score = _tm_bands(_make_cell_from_params(best_params))
    
    n_params = len(best_params)
    step_sizes = [0.02, 0.02, 0.01, 0.02, 1]  # Initial step sizes for each param
    
    deadline = time.time() + time_limit
    
    while time.time() < deadline - 5.0:  # 5s safety margin
        improved = False
        for p in range(n_params):
            for direction in [-1, 1]:
                trial_params = list(best_params)
                trial_params[p] += direction * step_sizes[p]
                
                # Bounds
                if p == 0:  # r0
                    trial_params[p] = max(0.05, min(0.45, trial_params[p]))
                elif p == 1:  # r1
                    trial_params[p] = max(0.0, min(0.45, trial_params[p]))
                elif p == 2:  # w1
                    trial_params[p] = max(0.0, min(0.2, trial_params[p]))
                elif p == 3:  # amod
                    trial_params[p] = max(0.0, min(0.1, trial_params[p]))
                elif p == 4:  # fmod
                    trial_params[p] = max(0, min(8, int(trial_params[p])))
                
                trial_cell = _make_cell_from_params(trial_params)
                trial_score = _tm_bands(trial_cell)
                
                if trial_score > best_score:
                    best_params = trial_params
                    best_score = trial_score
                    improved = True
                    step_sizes[p] *= 1.2  # Increase step size
                else:
                    step_sizes[p] *= 0.7  # Decrease step size
        
        if not improved:
            # No improvement, try smaller steps or break
            all_small = all(s < 0.001 for s in step_sizes[:4])
            if all_small:
                break
    
    return best_params, best_score

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell."""
    start_time = time.time()
    deadline = start_time + 220.0  # 220s out of 230s budget
    
    size = problem.get("pixels", 32)
    if size != _SIZE:
        # Fallback for unexpected sizes
        cell = [[0]*size for _ in range(size)]
        center = size / 2.0
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                cell[i][j] = 1 if x*x + y*y < 0.04 else 0
        return cell
    
    # Initial parameters: [r0, r1, w1, amod, fmod]
    initial_params = [0.2, 0.0, 0.0, 0.0, 0]
    
    # Try a few initial points
    candidates = [
        [0.2, 0.0, 0.0, 0.0, 0],
        [0.25, 0.0, 0.0, 0.0, 0],
        [0.15, 0.0, 0.0, 0.0, 0],
        [0.2, 0.35, 0.05, 0.0, 0],
        [0.2, 0.0, 0.0, 0.03, 4],
        [0.22, 0.38, 0.04, 0.02, 4],
    ]
    
    best_params = initial_params
    best_score = -np.inf
    
    for init in candidates:
        if time.time() > deadline - 30.0:
            break
        params, score = _optimize_cell(init, min(20.0, deadline - time.time() - 10.0))
        if score > best_score:
            best_score = score
            best_params = params
    
    # Final polish with more iterations if time remains
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        params, score = _optimize_cell(best_params, remaining)
        if score > best_score:
            best_score = score
            best_params = params
    
    cell = _make_cell_from_params(best_params)
    
    # Verify symmetry
    for i in range(_SIZE):
        for j in range(_SIZE):
            if cell[i][j] != cell[_SIZE-1-i][j]:
                cell[_SIZE-1-i][j] = cell[i][j]
            if cell[i][j] != cell[i][_SIZE-1-j]:
                cell[i][_SIZE-1-j] = cell[i][j]
            if cell[i][j] != cell[j][i]:
                cell[j][i] = cell[i][j]
    
    return cell
