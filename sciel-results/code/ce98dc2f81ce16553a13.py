import numpy as np
from scipy.special import ellipk, ellipe

N = 26

def construct():
    # Schwarz-Christoffel map from unit disk to unit square
    # z = C * integral_0^zeta 1/sqrt(1-t^4) dt
    # The constant C is chosen so that the square is [0,1]x[0,1]
    # The half-diagonal of the square is 1/sqrt(2).
    # The integral from 0 to 1 is (1/4) * B(1/4, 1/2) = (1/4) * Gamma(1/4)*Gamma(1/2)/Gamma(3/4)
    
    # We can use the elliptic integral of the first kind K(k)
    # Integral_0^1 dt/sqrt(1-t^4) = (1/2) * K(1/sqrt(2)) * 2^(-1/4) ? No.
    # Let t^2 = u. dt = du/(2sqrt(u)).
    # Int 0->1 1/sqrt(1-u^2) * 1/(2sqrt(u)) du. This is not standard K.
    
    # Standard result:
    # Side length of square for unit disk is 2 * K(1/sqrt(2)) / (Gamma(1/4)^2 / (2*pi)) ... 
    # Actually, let's just compute the integral numerically for the map and normalize.
    
    # Pre-compute the map and its derivative on a grid or use analytic approximations?
    # For speed, we will compute the map and derivative on the fly using numerical integration? 
    # No, that's too slow for SA.
    
    # Alternative: Use the fact that the map is conformal. 
    # We can define the map using the elliptic integral.
    # f(zeta) = C * F(arcsin(zeta), 1/sqrt(2)) ? 
    # Actually, the map from disk to square is related to the lemniscate.
    
    # Let's use a simpler, robust numerical approach for the initialization only.
    # We will generate a hexagonal packing in the disk, map it to the square, 
    # and then refine in the square.
    
    # To map efficiently, we can pre-compute a lookup table for the map and derivative?
    # Or, we can just use a high-quality random initialization in the square, 
    # but the directive asks for the conformal map approach.
    
    # Let's implement the map using scipy.integrate.quad for initialization only (once).
    # Then run SA in the square.
    
    import scipy.integrate as integrate
    import scipy.special as spec

    # The conformal map from the unit disk to the square [-1,1]x[-1,1] (centered)
    # is z = C * ∫_0^ζ (1-t^4)^(-1/2) dt.
    # The constant C is such that f(1) = 1 + i.
    # f(1) = C * I, where I = ∫_0^1 (1-t^4)^(-1/2) dt.
    # I = (1/4) * B(1/4, 1/2) = (1/4) * Γ(1/4)Γ(1/2)/Γ(3/4).
    
    def sc_map(zeta):
        # zeta is complex
        # We integrate from 0 to zeta
        # For speed in initialization, we can use a fixed number of points or analytic approx.
        # Since we only need this for initialization, we can afford a bit of cost.
        
        # Use the fact that ∫ (1-t^4)^(-1/2) dt can be computed via elliptic integrals.
        # Let t^2 = sin(theta). 
        # A simpler way: use scipy.integrate.quad on real and imag parts? 
        # No, complex integration is tricky.
        
        # Let's use a series expansion or a known approximation for the map.
        # Or, just use a dense grid initialization in the square that mimics the disk packing density.
        # The disk packing has uniform density in the disk. 
        # The map stretches the center and compresses the corners (or vice versa?).
        # The derivative |f'(zeta)| is largest at the center and smallest at the corners?
        # f'(zeta) = 1/sqrt(1-zeta^4). At zeta=0, f'=1. At zeta=1, f'->inf.
        # So the map stretches the boundary. The corners of the square correspond to the boundary of the disk.
        # The center of the square corresponds to the center of the disk.
        # So a uniform grid in the disk maps to a grid in the square that is denser near the center?
        # No, if f' is large near boundary, the image of a small disk area is large.
        # So a uniform grid in disk -> non-uniform in square, with larger spacing near boundary?
        
        # Let's just generate a hexagonal grid in the disk, map it using a simple numerical integration for each point (only 26 points), 
        # and then refine in the square.
        
        # We need to map complex zeta to complex z.
        # z = C * ∫_0^ζ (1-t^4)^(-1/2) dt
        
        # We can compute this integral for each of the 26 points.
        
        def integral_sc(zeta_val):
            # Integrate 1/sqrt(1-t^4) from 0 to zeta_val
            # Split into real and imaginary? No, complex path.
            # Since the integrand is analytic in the unit disk, the integral is path independent.
            # We can integrate along the straight line from 0 to zeta_val.
            
            # Use scipy.integrate.quad for complex? No.
            # We can integrate real and imaginary parts separately if we parameterize the path.
            # t(s) = s * zeta_val, s from 0 to 1.
            # dt = zeta_val ds.
            # ∫_0^1 1/sqrt(1 - (s*zeta_val)^4) * zeta_val ds
            
            def integrand(s):
                t = s * zeta_val
                # Handle singularity if t is near 4th roots of unity (boundary)
                # Our points are inside, so it's safe.
                return zeta_val / np.sqrt(1 - t**4)
            
            # quad doesn't handle complex directly well in all versions, but we can do real/imag
            # Actually, numpy complex works in quad if we split?
            # Let's use a simple trapezoidal rule with high N for initialization.
            n_steps = 100
            s_vals = np.linspace(0, 1, n_steps)
            t_vals = s_vals * zeta_val
            # Avoid division by zero or negative sqrt issues if close to boundary
            # Add small epsilon
            vals = zeta_val / np.sqrt(1 - t_vals**4 + 1e-12)
            # Trapezoidal rule
            result = np.trapz(vals, s_vals)
            return result

        I_01 = integral_sc(1.0) # This is real
        C = (1 + 1j) / I_01 # Map 1 to 1+i (corner of square centered at 0)
        
        z = C * integral_sc(zeta)
        return z

    # 1. Generate Hexagonal Grid in Disk
    points_disk = []
    s = 0.15 # spacing
    h = s * np.sqrt(3) / 2
    for i in range(-10, 10):
        for j in range(-10, 10):
            x = j * s + (i % 2) * s / 2
            y = i * h
            if x*x + y*y < 0.9**2: # Keep inside disk with margin
                points_disk.append(complex(x, y))
                
    np.random.shuffle(points_disk)
    if len(points_disk) < N:
        # Fallback to random
        points_disk = [complex(np.random.rand(), np.random.rand()) * 0.9 for _ in range(N)]
        
    selected_disk = points_disk[:N]
    
    # 2. Map to Square
    # The map above maps to [-1,1]x[-1,1]. We need [0,1]x[0,1].
    # z_square = (z_map + 1) / 2
    
    init_xs = np.zeros(N)
    init_ys = np.zeros(N)
    
    for i, zeta in enumerate(selected_disk):
        z = sc_map(zeta)
        # z is complex, maps to centered square [-1,1]x[-1,1]
        # Shift and scale to [0,1]x[0,1]
        x = (z.real + 1) / 2
        y = (z.imag + 1) / 2
        
        # Clip to [0,1] to be safe
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        
        init_xs[i] = x
        init_ys[i] = y

    positions = np.zeros(2*N)
    positions[:N] = init_xs
    positions[N:] = init_ys

    # 3. Refine in Square using Simulated Annealing
    def compute_radii(positions):
        xs = positions[:N]
        ys = positions[N:]
        rs = np.zeros(N)
        
        # Wall constraints
        for i in range(N):
            rs[i] = min(xs[i], 1 - xs[i], ys[i], 1 - ys[i])
            
        # Neighbor constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                val = dist / 2.0
                if val < rs[i]:
                    rs[i] = val
                if val < rs[j]:
                    rs[j] = val
        return rs

    def objective(positions):
        rs = compute_radii(positions)
        return -np.sum(rs)

    best_pos = positions.copy()
    best_score = np.sum(compute_radii(positions))
    
    current_pos = positions.copy()
    current_obj = objective(current_pos)
    
    T = 0.05
    T_min = 0.001
    alpha = 0.995
    steps = 5000
    
    for step in range(steps):
        new_pos = current_pos.copy()
        idx = np.random.randint(2*N)
        delta = np.random.normal(0, T)
        new_pos[idx] += delta
        new_pos[idx] = np.clip(new_pos[idx], 0.0, 1.0)
        
        new_obj = objective(new_pos)
        delta_obj = new_obj - current_obj
        
        if delta_obj < 0 or np.random.rand() < np.exp(-delta_obj / T):
            current_pos = new_pos
            current_obj = new_obj
            
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Update best
        rs = compute_radii(current_pos)
        score = np.sum(rs)
        if score > best_score:
            best_score = score
            best_pos = current_pos.copy()

    xs = best_pos[:N]
    ys = best_pos[N:]
    rs = compute_radii(best_pos)
    
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = rs
    
    return result
