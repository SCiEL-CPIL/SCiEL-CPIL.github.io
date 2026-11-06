"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    def compute_energy(c):
        """
        Computes the energy to be minimized.
        Minimizing sum(1/d_ij^2) encourages maximizing the minimum distance.
        Boundary penalty keeps spheres inside [-0.5, 0.5].
        """
        # c is (N, 3)
        # Pairwise squared distances
        # Using GEMM for speed: c @ c.T - 0.5 * (||c_i||^2 + ||c_j||^2)
        # ||c_i||^2 is sum of squares of each row
        c_sq = np.sum(c * c, axis=1)
        d2 = c_sq[:, np.newaxis] + c_sq[np.newaxis, :] - 2.0 * (c @ c.T)
        
        # Clip small negative values due to numerical precision
        np.maximum(d2, 1e-12, out=d2)
        
        # Mask diagonal (distance to self)
        np.fill_diagonal(d2, 1.0) # Set to 1 so 1/d2 is 1, we subtract N later
        
        e_pair = np.sum(1.0 / d2) - N  # Subtract the N diagonal terms
        
        # Boundary penalty
        # Distance to wall: 0.5 - |c|
        # If |c| > 0.5, it's outside.
        abs_c = np.abs(c)
        violations = np.maximum(0.0, abs_c - 0.5)
        e_boundary = np.sum(violations * violations) * 100.0 # High penalty for boundary
        
        return e_pair + e_boundary

    def get_radius(c):
        """
        Computes the actual radius score for the given centers.
        r = min(min_pair_dist/2, min_dist_to_wall)
        """
        c_sq = np.sum(c * c, axis=1)
        d2 = c_sq[:, np.newaxis] + c_sq[np.newaxis, :] - 2.0 * (c @ c.T)
        np.maximum(d2, 0.0, out=d2)
        np.fill_diagonal(d2, 1.0)
        min_d2 = np.min(d2)
        min_dist = np.sqrt(min_d2)
        
        # Distance to wall
        abs_c = np.abs(c)
        min_dist_to_wall = np.min(0.5 - abs_c)
        
        r = min(min_dist / 2.0, min_dist_to_wall)
        return r

    def sa_run(init, time_budget):
        """
        Runs Simulated Annealing starting from init.
        Returns the best center found.
        """
        current_c = init.copy()
        current_e = compute_energy(current_c)
        best_c = current_c.copy()
        best_e = current_e
        best_r = get_radius(best_c)
        
        T = 1.0
        cooling_rate = 0.9999
        min_T = 1e-6
        step_size = 0.05
        
        while time.time() < deadline - 5.0 and T > min_T:
            # Propose move
            i = np.random.randint(N)
            delta = np.random.normal(0, step_size, size=3)
            new_c = current_c.copy()
            new_c[i] += delta
            
            # Ensure new_c is not crazy, but boundary penalty handles it
            new_e = compute_energy(new_c)
            
            delta_e = new_e - current_e
            
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
                current_c = new_c
                current_e = new_e
                
                # Check if improved
                if current_e < best_e:
                    best_c = current_c.copy()
                    best_e = current_e
                    # Only compute radius occasionally to save time, or every time?
                    # Radius calculation is cheap (O(N^2)), same as energy.
                    r = get_radius(best_c)
                    if r > best_r:
                        best_r = r
            
            T *= cooling_rate
            # Occasionally adjust step size? Keep it simple for now.
            
        return best_c, best_r

    # Initialize multiple configurations
    inits = []
    
    # 1. Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    # 2. Random
    inits.append(np.random.uniform(-0.4, 0.4, size=(N, 3)))
    
    # 3. BCC-like
    # Create a 3x3x3 grid of unit cells, each with 2 points (corners + center)
    # This is a bit complex to generate exactly N=50.
    # Let's just use a dense grid and perturb.
    m_bcc = 4
    ticks_bcc = (np.arange(m_bcc) + 0.5) / m_bcc - 0.5
    grid_bcc = np.array([(x, y, z) for z in ticks_bcc for y in ticks_bcc for x in ticks_bcc])
    # Add body centers
    offsets = [0.0, 0.5/m_bcc]
    # This is getting messy. Let's just use Random and Grid.
    # Maybe add a "Clustered" init?
    
    # 4. FCC-like (just another random with different seed)
    inits.append(np.random.uniform(-0.4, 0.4, size=(N, 3)))
    inits.append(np.random.uniform(-0.4, 0.4, size=(N, 3)))
    
    best_overall_c = inits[0].copy()
    best_overall_r = get_radius(best_overall_c)
    
    # Run SA on each init
    # Allocate time based on number of inits
    n_inits = len(inits)
    # We have ~110s total. 5 inits -> 20s each?
    # But we have a global deadline check in sa_run.
    # Let's just run them sequentially, sa_run will stop when deadline is near.
    
    for init in inits:
        if time.time() > deadline - 10.0:
            break
        c, r = sa_run(init, 0) # time_budget not used, uses global deadline
        if r > best_overall_r:
            best_overall_r = r
            best_overall_c = c.copy()
            
    # Final Polish
    if time.time() < deadline - 2.0:
        # Use Nelder-Mead to polish the best result
        # We need to minimize the energy function
        # But we want to maximize radius.
        # Minimizing energy (sum 1/d^2) is a good proxy.
        
        # Define a wrapper for minimize
        def obj(x):
            c = x.reshape(N, 3)
            return compute_energy(c)
        
        # Flatten the best center
        x0 = best_overall_c.flatten()
        
        try:
            res = minimize(obj, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
            if res.success or time.time() < deadline - 1.0:
                c_polished = res.x.reshape(N, 3)
                r_polished = get_radius(c_polished)
                if r_polished > best_overall_r:
                    best_overall_c = c_polished
                    best_overall_r = r_polished
        except Exception:
            pass

    return best_overall_c

# EVOLVE-BLOCK-END
