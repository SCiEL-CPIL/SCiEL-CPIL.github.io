"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Initialize with a hexagonal-like grid
    def hex_init():
        pts = []
        # Try different spacings to get at least N points in unit disk
        for spacing in np.linspace(0.25, 0.1, 20):
            k = int(1.0 / spacing) + 2
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if x**2 + y**2 <= 0.9:
                        pts.append([x, y])
            if len(pts) >= N:
                pts = np.array(pts[:N])
                # Shuffle to break symmetry
                idx = np.random.permutation(N)
                return pts[idx]
        # Fallback: random points in disk
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N)) * 0.9
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    def compute_score(pts):
        """Compute the packing radius (minimum of half-min-pair-dist and min-container-margin)."""
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Container margins
        radii = np.sqrt(np.sum(pts**2, axis=1))
        min_margin = 1.0 - np.min(radii)
        
        return min(min_pair_dist / 2.0, min_margin)

    def energy(pts, target_r):
        """Energy = sum of overlap penalties + container violation penalties."""
        # Pairwise overlaps
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        overlaps = np.maximum(0, 2*target_r - dists)
        pair_energy = np.sum(overlaps**2)
        
        # Container violations
        radii = np.sqrt(np.sum(pts**2, axis=1))
        container_viol = np.maximum(0, radii + target_r - 1.0)
        container_energy = np.sum(container_viol**2)
        
        return pair_energy + container_energy

    # Simulated annealing
    def simulated_annealing(initial_pts, max_iter=50000, T_start=0.1, T_end=0.001, target_r=None):
        if target_r is None:
            target_r = compute_score(initial_pts)
        
        pts = initial_pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        best_energy = energy(pts, target_r)
        
        for it in range(max_iter):
            T = T_start * (T_end / T_start) ** (it / max_iter)
            
            # Random perturbation
            delta = np.random.normal(0, 0.05, (N, 2))
            new_pts = pts + delta
            
            # Keep in container (hard constraint for stability)
            new_radii = np.sqrt(np.sum(new_pts**2, axis=1))
            mask = new_radii > 0.95
            if np.any(mask):
                new_pts[mask] = 0.95 * new_pts[mask] / new_radii[mask]
            
            new_energy = energy(new_pts, target_r)
            new_score = compute_score(new_pts)
            
            # Acceptance
            if new_energy < best_energy or np.random.random() < np.exp(-(new_energy - best_energy) / max(T, 1e-10)):
                pts = new_pts
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    best_energy = new_energy
            
            # Update target_r periodically
            if it % 1000 == 0:
                target_r = best_score * 1.01  # Slightly optimistic target
            
            if time.time() > deadline:
                break
        
        return best_pts, best_score

    # Differential evolution
    def differential_evolution(pts, target_r, max_iter=2000, F=0.8, CR=0.9, pop_size=50):
        pop = [pts.copy() for _ in range(pop_size)]
        # Diversify population
        for i in range(1, pop_size):
            pop[i] = pts + np.random.normal(0, 0.1, (N, 2))
            # Keep in container
            radii = np.sqrt(np.sum(pop[i]**2, axis=1))
            mask = radii > 0.95
            if np.any(mask):
                pop[i][mask] = 0.95 * pop[i][mask] / radii[mask]
        
        best_idx = np.argmin([energy(p, target_r) for p in pop])
        best_pts = pop[best_idx].copy()
        best_score = compute_score(best_pts)
        
        for it in range(max_iter):
            new_pop = pop[:]
            for i in range(pop_size):
                # Mutation
                a, b, c = np.random.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover
                trial = mutant.copy()
                cross_mask = np.random.random((N, 2)) < CR
                trial[cross_mask] = pop[i][cross_mask]
                
                # Keep in container
                radii = np.sqrt(np.sum(trial**2, axis=1))
                mask = radii > 0.95
                if np.any(mask):
                    trial[mask] = 0.95 * trial[mask] / radii[mask]
                
                # Selection
                if energy(trial, target_r) <= energy(pop[i], target_r):
                    new_pop[i] = trial
                    score_i = compute_score(trial)
                    if score_i > best_score:
                        best_score = score_i
                        best_pts = trial.copy()
                        best_idx = i
                else:
                    new_pop[i] = pop[i]
            
            pop = new_pop
            
            if time.time() > deadline:
                break
        
        return best_pts, best_score

    # Local refinement with gradient descent on score
    def local_refine(pts, max_iter=1000):
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        for it in range(max_iter):
            # Compute gradients numerically
            score = best_score
            # Try small moves in random directions
            improved = False
            for _ in range(50):
                direction = np.random.normal(0, 1, (N, 2))
                direction /= np.sqrt(np.sum(direction**2))
                step = 0.001
                new_pts = best_pts + step * direction
                new_radii = np.sqrt(np.sum(new_pts**2, axis=1))
                mask = new_radii > 0.99
                if np.any(mask):
                    new_pts[mask] = 0.99 * new_pts[mask] / new_radii[mask]
                new_score = compute_score(new_pts)
                if new_score > best_score:
                    best_pts = new_pts
                    best_score = new_score
                    improved = True
            
            if not improved:
                break
            
            if time.time() > deadline:
                break
        
        return best_pts, best_score

    # Main optimization loop
    best_pts = hex_init()
    best_score = compute_score(best_pts)
    
    # Phase 1: Simulated annealing with increasing target radius
    for phase in range(5):
        if time.time() > deadline:
            break
        target_r = best_score * (1.0 + 0.02 * (phase + 1))
        pts, score = simulated_annealing(best_pts, max_iter=20000, T_start=0.05, T_end=0.001, target_r=target_r)
        if score > best_score:
            best_pts = pts
            best_score = score
    
    # Phase 2: Differential evolution
    for phase in range(3):
        if time.time() > deadline:
            break
        target_r = best_score
        pts, score = differential_evolution(best_pts, target_r, max_iter=1000, F=0.7, CR=0.8, pop_size=30)
        if score > best_score:
            best_pts = pts
            best_score = score
    
    # Phase 3: Local refinement
    if time.time() < deadline - 10:
        best_pts, best_score = local_refine(best_pts, max_iter=2000)
    
    # Final check: ensure validity
    final_score = compute_score(best_pts)
    if final_score < best_score:
        best_score = final_score
    
    return best_pts
# EVOLVE-BLOCK-END
