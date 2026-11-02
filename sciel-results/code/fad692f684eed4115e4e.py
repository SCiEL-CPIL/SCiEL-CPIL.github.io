"""Improved packing search for 50 equal circles in a unit square."""
import numpy as np
from scipy.spatial import cKDTree
import time
import random

N = 50
DEADLINE = 145.0  # safety margin before 150s kill

def compute_score(centers):
    """Compute the radius score for a given set of centers."""
    # Boundary constraint
    dist_to_boundary = np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    )
    min_boundary = np.min(dist_to_boundary)
    
    # Pairwise distances
    tree = cKDTree(centers)
    pairs = tree.query_pairs(1.0)  # query all pairs within distance 1
    if len(pairs) == 0:
        min_pair_dist = float('inf')
    else:
        dists = []
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            dists.append(d)
        min_pair_dist = min(dists) if dists else float('inf')
    
    return min(min_pair_dist / 2.0, min_boundary)


def hexagonal_init():
    """Generate a hexagonal packing initialization."""
    centers = []
    # Try different hexagonal arrangements
    best_centers = None
    best_score = -1
    
    for offset_x in np.linspace(-0.4, 0.4, 7):
        for offset_y in np.linspace(-0.4, 0.4, 7):
            # Generate hexagonal grid points
            rows = []
            row = 0
            x = -0.5 + offset_x
            while len(rows) < N:
                y = -0.5 + offset_y + row * 0.14
                if y > 0.5:
                    break
                col = 0
                while len(rows) < N:
                    xx = -0.5 + offset_x + col * 0.16
                    if xx > 0.5:
                        break
                    rows.append([xx, y])
                    col += 1
                row += 1
            
            if len(rows) >= N:
                cand = np.array(rows[:N])
                # Check if all points are within bounds
                if np.all(np.abs(cand) <= 0.5):
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_centers = cand.copy()
    
    if best_centers is not None:
        return best_centers
    return None


def grid_init():
    """Generate a grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def random_init():
    """Generate a random initialization."""
    return np.random.uniform(-0.45, 0.45, size=(N, 2))


def local_refine(centers, max_iters=5000, step_scale=1.0):
    """Local refinement using gradient-free optimization."""
    centers = centers.copy()
    current_score = compute_score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    step_size = 0.01 * step_scale
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Pick a random point to move
        idx = np.random.randint(N)
        
        # Try several random perturbations
        improved = False
        for _ in range(3):
            new_centers = centers.copy()
            new_centers[idx] += np.random.normal(0, step_size, size=2)
            
            # Clamp to valid region
            new_centers[idx] = np.clip(new_centers[idx], -0.499, 0.499)
            
            new_score = compute_score(new_centers)
            
            if new_score > current_score + 1e-12:
                centers = new_centers
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_centers = centers.copy()
                no_improve = 0
                break
        
        if not improved:
            no_improve += 1
            if no_improve > 50:
                step_size *= 0.95
                no_improve = 0
                if step_size < 1e-6:
                    break
        else:
            step_size = min(step_size * 1.05, 0.05)
    
    return best_centers, best_score


def simulated_annealing(centers, max_iters=20000, t_start=0.01, t_end=0.0001):
    """Simulated annealing for global search."""
    centers = centers.copy()
    current_score = compute_score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Temperature schedule
        progress = it / max_iters
        temperature = t_start * (t_end / t_start) ** progress
        
        # Pick a random point to move
        idx = np.random.randint(N)
        
        new_centers = centers.copy()
        new_centers[idx] += np.random.normal(0, temperature * 0.5, size=2)
        new_centers[idx] = np.clip(new_centers[idx], -0.499, 0.499)
        
        new_score = compute_score(new_centers)
        delta = new_score - current_score
        
        if delta > 0:
            centers = new_centers
            current_score = new_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
        else:
            # Accept with probability exp(delta / temperature)
            if np.random.random() < np.exp(delta / max(temperature, 1e-10)):
                centers = new_centers
                current_score = new_score
                no_improve += 1
            else:
                no_improve += 1
        
        if no_improve > 1000:
            # Restart from best
            centers = best_centers.copy()
            current_score = best_score
            no_improve = 0
    
    return best_centers, best_score


def construct():
    """Main entry point: find the best packing of 50 circles."""
    start_time = time.time()
    
    # Multiple initializations
    inits = []
    
    # Grid initialization
    inits.append(grid_init())
    
    # Hexagonal initialization
    hex_init = hexagonal_init()
    if hex_init is not None:
        inits.append(hex_init)
    
    # Random initializations
    for _ in range(3):
        inits.append(random_init())
    
    best_centers = None
    best_score = -1
    
    # Phase 1: Local refinement from each initialization
    for i, init in enumerate(inits):
        if time.time() > DEADLINE - 30:
            break
        
        refined, score = local_refine(init, max_iters=3000, step_scale=1.0)
        if score > best_score:
            best_score = score
            best_centers = refined.copy()
    
    # Phase 2: Simulated annealing from the best result
    if time.time() < DEADLINE - 20:
        sa_centers, sa_score = simulated_annealing(
            best_centers, max_iters=15000, t_start=0.008, t_end=0.00005
        )
        if sa_score > best_score:
            best_score = sa_score
            best_centers = sa_centers.copy()
    
    # Phase 3: Final high-precision polish
    if time.time() < DEADLINE - 10:
        polished, polish_score = local_refine(
            best_centers, max_iters=5000, step_scale=0.5
        )
        if polish_score > best_score:
            best_score = polish_score
            best_centers = polished.copy()
    
    # Ensure output is valid
    if best_centers is None:
        best_centers = grid_init()
    
    # Final clamp to ensure validity
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
