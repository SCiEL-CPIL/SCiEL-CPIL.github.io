# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import morse_energy_gradient

def construct(instance):
    """
    Find global minima of Morse clusters using stochastic collision dynamics
    (velocity Verlet with thermostatting) initialized from high-temperature gas.
    """
    n = instance["n"]
    rho = instance["rho"]
    
    # Time budget: ~45 seconds per instance (6 instances in 300s)
    # We'll use a multi-replica approach with annealing
    
    def morse_energy(pos, rho=rho):
        """Compute Morse energy."""
        n = pos.shape[0]
        E = 0.0
        for i in range(n):
            for j in range(i+1, n):
                r = np.linalg.norm(pos[i] - pos[j])
                if r < 1e-8:
                    return 1e10
                x = rho * (1.0 - r)
                E += np.exp(x) * (np.exp(x) - 2.0)
        return E
    
    def morse_force(pos, rho=rho):
        """Compute Morse forces."""
        n = pos.shape[0]
        F = np.zeros_like(pos)
        for i in range(n):
            for j in range(i+1, n):
                rij = pos[i] - pos[j]
                r = np.linalg.norm(rij)
                if r < 1e-8:
                    continue
                x = rho * (1.0 - r)
                exp_x = np.exp(x)
                # dE/dr = 2 * exp(x) * (exp(x) - 1) * (-rho)
                # F = -dE/dr * rij/r
                dE_dr = 2.0 * exp_x * (exp_x - 1.0) * (-rho)
                F[i] += dE_dr * rij / r
                F[j] -= dE_dr * rij / r
        return F
    
    def velocity_verlet_step(pos, vel, dt, T, gamma=0.1):
        """Velocity Verlet with Langevin thermostatting."""
        n = pos.shape[0]
        F0 = morse_force(pos)
        
        # Update velocity with force and thermostat
        vel_new = vel + F0 * dt
        # Langevin thermostat: add random kick and damping
        vel_new *= (1.0 - gamma * dt)
        vel_new += np.sqrt(2.0 * gamma * T * dt) * np.random.randn(*vel_new.shape)
        
        # Update position
        pos_new = pos + vel_new * dt
        
        # Compute new force
        F1 = morse_force(pos_new)
        
        # Final velocity update
        vel_final = vel_new + 0.5 * (F0 + F1) * dt
        vel_final *= (1.0 - 0.5 * gamma * dt)
        
        return pos_new, vel_final
    
    def anneal_cluster(pos_init, T_start=50.0, T_end=0.01, n_steps=5000, dt=0.005, gamma=0.5):
        """Simulate cluster annealing with velocity Verlet and thermostat."""
        pos = pos_init.copy()
        vel = np.random.randn(n, 3) * np.sqrt(T_start / n)
        
        best_pos = pos.copy()
        best_E = morse_energy(pos)
        
        for step in range(n_steps):
            # Linear annealing schedule
            frac = step / n_steps
            T = T_start * (T_end / T_start) ** frac
            
            pos, vel = velocity_verlet_step(pos, vel, dt, T, gamma)
            
            # Check minimum distance constraint
            min_dist = np.inf
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < min_dist:
                        min_dist = d
                    if d < 0.5:
                        # Push atoms apart
                        push = (pos[i] - pos[j]) / (d + 1e-10) * (0.5 - d) * 0.5
                        pos[i] += push
                        pos[j] -= push
                        break
                else:
                    continue
                break
            
            # Center of mass
            pos -= np.mean(pos, axis=0)
            
            E = morse_energy(pos)
            if E < best_E:
                best_E = E
                best_pos = pos.copy()
        
        return best_pos, best_E
    
    def local_refine(pos):
        """Local refinement using gradient-based optimization."""
        try:
            def energy_func(x):
                pos_flat = x.reshape(-1, 3)
                # Enforce minimum distance
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.linalg.norm(pos_flat[i] - pos_flat[j])
                        if d < 0.5:
                            return 1e10
                return morse_energy(pos_flat)
            
            result = minimize(energy_func, pos.flatten(), method='BFGS', 
                            options={'maxiter': 200, 'gtol': 1e-6})
            if result.success or result.fun < morse_energy(pos):
                pos_ref = result.x.reshape(-1, 3)
                # Ensure minimum distance
                valid = True
                for i in range(n):
                    for j in range(i+1, n):
                        if np.linalg.norm(pos_ref[i] - pos_ref[j]) < 0.5:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return pos_ref
        except:
            pass
        return pos
    
    # Multi-replica annealing
    rng = np.random.default_rng(42 + n)
    
    # Initial gas state: random positions in a sphere
    radius = n ** (1.0/3.0) * 1.5
    best_pos = None
    best_E = np.inf
    
    n_replicas = 8
    steps_per_replica = 4000
    
    for rep in range(n_replicas):
        # Random initial positions
        pos_init = rng.uniform(-radius, radius, (n, 3))
        # Ensure no overlaps initially
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos_init[i] - pos_init[j])
                if d < 0.5:
                    pos_init[i] += rng.uniform(0.5 - d, 0.5 - d + 0.1, 3)
        
        # Anneal
        pos_annealed, E_annealed = anneal_cluster(pos_init, T_start=30.0, T_end=0.005, 
                                                   n_steps=steps_per_replica, dt=0.005, gamma=0.5)
        
        # Local refinement
        pos_refined = local_refine(pos_annealed)
        E_refined = morse_energy(pos_refined)
        
        if E_refined < best_E:
            best_E = E_refined
            best_pos = pos_refined.copy()
    
    # Additional rounds with lower temperatures if time permits
    for round2 in range(3):
        # Use best so far as starting point with some perturbation
        pos_perturbed = best_pos + rng.normal(0, 0.1, (n, 3))
        pos_annealed, E_annealed = anneal_cluster(pos_perturbed, T_start=5.0, T_end=0.001,
                                                   n_steps=3000, dt=0.005, gamma=0.8)
        pos_refined = local_refine(pos_annealed)
        E_refined = morse_energy(pos_refined)
        
        if E_refined < best_E:
            best_E = E_refined
            best_pos = pos_refined.copy()
    
    return best_pos
# EVOLVE-BLOCK-END
