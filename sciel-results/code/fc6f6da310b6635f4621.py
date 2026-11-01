"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random

def _compute_score(centers):
    """Compute the radius for a set of centers."""
    # Boundary constraint
    boundary = 0.5 - np.max(np.abs(centers), axis=1)
    r_boundary = np.min(boundary)
    
    # Pairwise distances
    # For N=200, we can compute all pairwise distances
    # But to be efficient, let's use a smarter approach
    # Actually for scoring, we need the minimum distance
    dists = np.zeros((N, N))
    for i in range(N):
        diff = centers[i] - centers[i+1:]
        d = np.sqrt(np.sum(diff**2, axis=1))
        dists[i, i+1:] = d
        dists[i+1:, i] = d
    
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r_pairs = min_dist / 2.0
    
    return min(r_boundary, r_pairs)

def _score_for_optimization(centers):
    """Score function for optimization (we want to maximize, so minimize negative)."""
    return -_compute_score(centers)

def _hexagonal_init():
    """Create a hexagonal packing initialization."""
    centers = []
    # Hexagonal packing: rows offset
    # For a unit square, we can fit about 14x14 in a grid
    # Hexagonal is denser
    
    # Let's try a hexagonal lattice approach
    # In hexagonal packing, rows are spaced by sqrt(3)/2 * d
    # and alternate rows are offset by d/2
    
    # Start with a grid and perturb
    m = int(np.ceil(np.sqrt(N)))
    
    # Create hexagonal pattern
    rows = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    cols = int(np.ceil(np.sqrt(N * np.sqrt(3) / 2)))
    
    # Actually, let's use a simpler approach: start with square grid
    # and apply a hexagonal distortion
    
    # For now, let's use a square grid as initialization
    # and the optimization will improve it
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    centers += np.random.normal(0, 0.001, size=centers.shape)
    
    return centers

def _force_relaxation(centers, max_iter=1000, dt=0.01, cooling=0.999):
    """Simple force-directed relaxation to increase minimum distance."""
    centers = centers.copy()
    current_r = _compute_score(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    for iteration in range(max_iter):
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i+1, N):
                diff = centers[i] - centers[j]
                dist = np.sqrt(np.sum(diff**2))
                if dist < 1e-10:
                    # Random nudge
                    forces[i] += np.random.normal(0, 0.001, size=2)
                    forces[j] -= np.random.normal(0, 0.001, size=2)
                else:
                    # Repulsive force, stronger when closer
                    # We want to push apart circles that are too close
                    if dist < 2 * current_r * 1.1:
                        force_mag = (2 * current_r * 1.1 - dist) / dist
                        forces[i] += force_mag * diff
                        forces[j] -= force_mag * diff
        
        # Boundary attraction (pull back if outside)
        for i in range(N):
            for k in range(2):
                if centers[i][k] > 0.5 - current_r:
                    forces[i][k] -= (centers[i][k] - (0.5 - current_r)) * 10
                elif centers[i][k] < -0.5 + current_r:
                    forces[i][k] += ((-0.5 + current_r) - centers[i][k]) * 10
        
        # Update positions
        centers += dt * forces
        dt *= cooling
        
        # Compute new radius
        new_r = _compute_score(centers)
        if new_r > best_r:
            best_r = new_r
            best_centers = centers.copy()
        
        # If we've been stuck, add some random perturbation
        if iteration % 100 == 99 and new_r <= current_r:
            centers += np.random.normal(0, 0.005, size=centers.shape)
            current_r = _compute_score(centers)
    
    return best_centers, best_r

def _simulated_annealing(centers, max_iter=5000, T_start=0.01, T_end=0.0001):
    """Simulated annealing to improve the packing."""
    centers = centers.copy()
    current_r = _compute_score(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / max_iter)
    
    for iteration in range(max_iter):
        # Generate a random move
        i = np.random.randint(N)
        move = np.random.normal(0, 0.005, size=2)
        new_centers = centers.copy()
        new_centers[i] += move
        
        # Check if still in bounds (with some margin)
        if np.any(np.abs(new_centers[i]) > 0.5):
            new_r = 0
        else:
            new_r = _compute_score(new_centers)
        
        # Accept or reject
        if new_r > current_r:
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new_r - current_r) / T)
            delta = new_r - current_r
            if np.random.random() < np.exp(delta / T):
                centers = new_centers
                current_r = new_r
        
        T *= cooling
    
    return best_centers, best_r

def construct():
    """Construct the centers of N equal circles inside the unit square."""
    start_time = time.time()
    deadline = 130.0  # Leave 20s margin before 150s limit
    
    # Initial configuration
    centers = _hexagonal_init()
    best_centers = centers.copy()
    best_r = _compute_score(centers)
    
    # Phase 1: Force relaxation
    if time.time() - start_time < deadline - 60:
        centers, r = _force_relaxation(centers, max_iter=500)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Phase 2: Simulated annealing
    if time.time() - start_time < deadline - 20:
        centers, r = _simulated_annealing(centers, max_iter=3000)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Phase 3: Final force relaxation
    if time.time() - start_time < deadline - 5:
        centers, r = _force_relaxation(centers, max_iter=300)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Ensure all centers are within the square
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
