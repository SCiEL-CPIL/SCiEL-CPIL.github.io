import numpy as np
import time
import math
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def compute_radius(centers):
        """
        Computes the max radius for the given centers.
        Returns 0 if invalid.
        """
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        # Check bounds
        if np.any(np.abs(centers) > 0.5 + 1e-9):
            return 0.0
        
        # Wall distance
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        # Pair distance
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        # Set diagonal to inf
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
        min_pair = np.sqrt(min_dist_sq)
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def hexagonal_init(rows, cols, jitter=0.0, seed=0):
        """
        Generates a hexagonal lattice packing.
        """
        rng = np.random.RandomState(seed)
        # Approximate d
        d = 1.0 / cols * 0.99
        
        # Check if it fits vertically
        row_spacing = d * math.sqrt(3) / 2
        total_height = (rows - 1) * row_spacing + d
        if total_height > 1.0:
            d = 1.0 / ((rows - 1) * math.sqrt(3) / 2 + 1) * 0.99
            
        pts = []
        for row in range(rows):
            y = -0.5 + d/2 + row * (d * math.sqrt(3) / 2)
            if row % 2 == 0:
                x_start = -0.5 + d/2
                for col in range(cols):
                    x = x_start + col * d
                    if jitter > 0:
                        x += rng.uniform(-jitter*d, jitter*d)
                        y += rng.uniform(-jitter*d, jitter*d)
                    pts.append([x, y])
            else:
                x_start = -0.5 + d
                for col in range(cols - 1):
                    x = x_start + col * d
                    if jitter > 0:
                        x += rng.uniform(-jitter*d, jitter*d)
                        y += rng.uniform(-jitter*d, jitter*d)
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            return pts[:N]
        else:
            return None

    def lj_potential_and_grad(x_flat):
        """
        Computes the Lennard-Jones potential energy and gradient.
        Potential: E = sum_{i<j} [ 4 * (1/d^12 - 1/d^6) ] + sum_i 0.5 * k * (d_wall)^2
        We use a softening parameter to avoid singularities, though BFGS should handle it if init is good.
        Actually, standard LJ has a minimum at d=1 (in reduced units). 
        Here we want to maximize separation, so we want the repulsive part to dominate or tune the potential.
        
        A common approach for packing is to minimize a potential that is purely repulsive for d < d_target and attractive otherwise, 
        or simply use the LJ potential with a specific sigma.
        
        Let's use the standard LJ form: V(r) = 4 * epsilon * (sigma/r)^12 - 4 * epsilon * (sigma/r)^6.
        The minimum is at r = 2^(1/6) * sigma.
        
        We want the circles to pack. The "ideal" distance is related to the final radius.
        However, LJ is a fixed potential. 
        Alternative: Use a purely repulsive potential like 1/r^2 or 1/r^4 to push them apart, 
        but that doesn't have a minimum, it just pushes to infinity (or walls).
        With walls, it will find a configuration where forces balance.
        
        Let's use a potential that penalizes overlap heavily and encourages separation.
        V_pair = 1 / (d^2) is simple. Gradient is -2 / d^4 * diff.
        But this doesn't have a "preferred" distance, just "as far as possible".
        With walls, they will spread out. This is effectively maximizing the minimum distance.
        
        Let's try a slightly softer repulsion to allow BFGS to converge smoothly:
        V_pair = 1 / (d^4)
        dV/dd = -4 / d^5
        dV/dx = dV/dd * dx/dd = (-4/d^5) * (dx/d) = -4 * dx / d^6
        
        Wall potential: V_wall = 0.5 * k * (0.5 - |x|)^2 for |x| < 0.5
        Gradient: -k * (0.5 - x) for x > 0, k * (0.5 + x) for x < 0.
        Simply: -k * (0.5 - x) * sign(x)? No.
        Let w = 0.5 - |x|.
        If x > 0, w = 0.5 - x. dw/dx = -1. dV/dx = k * w * (-1) = -k(0.5-x).
        If x < 0, w = 0.5 + x. dw/dx = 1. dV/dx = k * w * (1) = k(0.5+x).
        This can be written as -k * (0.5 - |x|) * sign(x) ??
        If x>0: -k(0.5-x)(1) = k(x-0.5). Correct (pushes left).
        If x<0: -k(0.5+x)(-1) = k(0.5+x). Correct (pushes right).
        
        Let's use k=10.0 to keep them well inside.
        """
        centers = x_flat.reshape(N, 2)
        grad_flat = np.zeros_like(x_flat)
        energy = 0.0
        
        # Pairwise interactions
        # To be efficient, we can loop or vectorize. With N=200, N^2=40000, vectorization is fast.
        # But computing full matrix every gradient step might be okay.
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        # Avoid division by zero
        dist_sq = np.maximum(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)
        
        # Potential V = 1/d^4 = d^-4
        # We only want i < j to avoid double counting, or just sum all and divide by 2.
        # Summing all i!=j is easier to vectorize.
        # V_total = 0.5 * sum_{i!=j} (1/d^4)
        
        inv_d4 = 1.0 / (dist_sq * dist_sq)
        # Set diagonal to 0
        np.fill_diagonal(inv_d4, 0.0)
        
        energy += 0.5 * np.sum(inv_d4)
        
        # Gradient of 0.5 * sum_{i!=j} d^-4
        # d/dx_i (d^-4) = -4 * d^-5 * (x_i - x_j)/d = -4 * (x_i - x_j) / d^6
        # So grad_i = sum_{j!=i} -4 * (x_i - x_j) / d^6
        # Note: d^6 = dist_sq^3
        inv_d6 = 1.0 / (dist_sq * dist_sq * dist_sq)
        np.fill_diagonal(inv_d6, 0.0)
        
        # grad_pair[i] = sum_j -4 * diff[i,j] * inv_d6[i,j]
        # diff[i,j] = c_i - c_j
        # grad_pair = -4 * sum_j (c_i - c_j) * inv_d6[i,j]
        # This is equivalent to:
        # grad_pair = -4 * (c_i * sum_j inv_d6[i,j] - sum_j c_j * inv_d6[i,j])
        
        # Let's compute sum_j c_j * inv_d6[i,j]
        # weighted_centers = inv_d6 @ centers
        weighted_centers = inv_d6 @ centers
        sum_inv_d6 = np.sum(inv_d6, axis=1)
        
        grad_pair = -4.0 * (centers * sum_inv_d6[:, np.newaxis] - weighted_centers)
        
        # Wall potential
        # V_wall = sum 0.5 * k * (0.5 - |c|)^2
        # k = 10.0
        k_wall = 10.0
        abs_c = np.abs(centers)
        w = 0.5 - abs_c
        # If w < 0, it's outside. Penalty should be high.
        # For simplicity, assume we stay inside or use a quadratic penalty that grows.
        # V = 0.5 * k * w^2
        # dV/dc = k * w * dw/dc
        # dw/dc = -sign(c)
        # dV/dc = -k * w * sign(c)
        
        signs = np.sign(centers)
        # Handle 0 carefully, but sign(0)=0 so grad=0 which is fine for symmetry
        grad_wall = -k_wall * w * signs
        
        energy += np.sum(0.5 * k_wall * w * w)
        
        grad_flat = (grad_pair + grad_wall).flatten()
        
        return energy, grad_flat

    # Main Logic
    best_centers = None
    best_radius = 0.0
    
    # Generate diverse initializations
    inits = []
    
    # 1. Hexagonal grids with different row counts and jitters
    for rows in range(12, 18):
        cols = math.ceil(N / rows)
        pts = hexagonal_init(rows, cols, jitter=0.0, seed=rows)
        if pts is not None:
            inits.append(pts)
        for jitter in [0.05, 0.1, 0.2]:
            pts = hexagonal_init(rows, cols, jitter=jitter, seed=rows*100 + int(jitter*100))
            if pts is not None:
                inits.append(pts)
                
    # 2. Square grids
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    # 3. Random
    for seed in range(5):
        rng = np.random.RandomState(seed)
        pts = rng.uniform(-0.49, 0.49, (N, 2))
        inits.append(pts)

    # Evaluate all inits
    for i, init in enumerate(inits):
        if time.time() > deadline - 5:
            break
        r = compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()
            
    # Sort inits by radius to prioritize better starts
    init_scores = [(compute_radius(init), i) for i, init in enumerate(inits)]
    init_scores.sort(reverse=True)
    
    # Run BFGS on the best few initializations
    tried_indices = set()
    for score, idx in init_scores:
        if len(tried_indices) >= 5:
            break
        if idx in tried_indices:
            continue
        tried_indices.add(idx)
        
        if time.time() > deadline - 20:
            break
            
        x0 = inits[idx].flatten()
        
        # BFGS options
        # maxiter: limit iterations to prevent hanging
        # gtol: gradient tolerance
        # disp: display (False for speed)
        # callback: for time checking? BFGS doesn't easily support external time checks in callback without global state.
        # We'll rely on maxiter and the fact that BFGS is fast.
        
        try:
            res = minimize(
                lj_potential_and_grad, 
                x0, 
                jac=True, 
                method='BFGS', 
                options={'gtol': 1e-8, 'maxiter': 1000, 'disp': False}
            )
            
            optimized_centers = res.x.reshape(N, 2)
            
            # Ensure validity (BFGS might step outside if wall potential is weak)
            # Clip to safe bounds? No, the wall potential should keep them in.
            # But just in case, check.
            r_opt = compute_radius(optimized_centers)
            
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized_centers.copy()
                
        except Exception:
            pass

    # Final polish: Run BFGS again on the best result found, perhaps with tighter tolerance
    if best_centers is not None and time.time() < deadline - 5:
        try:
            x0 = best_centers.flatten()
            res = minimize(
                lj_potential_and_grad, 
                x0, 
                jac=True, 
                method='BFGS', 
                options={'gtol': 1e-10, 'maxiter': 2000, 'disp': False}
            )
            final_centers = res.x.reshape(N, 2)
            r_final = compute_radius(final_centers)
            if r_final > best_radius:
                best_radius = r_final
                best_centers = final_centers.copy()
        except Exception:
            pass

    if best_centers is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-END
