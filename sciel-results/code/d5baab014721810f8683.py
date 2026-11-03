"""Pack 200 equal circles in a unit square using grid-based genetic algorithm."""
import numpy as np
import time
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 145  # seconds, leave margin before 150s kill


def _get_radius(pts):
    """Calculate the maximum possible radius for given centers."""
    if len(pts) == 0:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5 - 1e-12) or np.any(pts > 0.5 + 1e-12):
        return 0.0

    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    wall_dist = min(min_x + 0.5, 0.5 - max_x, min_y + 0.5, 0.5 - max_y)
    if wall_dist < 0:
        return 0.0

    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair_dist = np.min(D)

    return min(wall_dist, min_pair_dist / 2.0)


def _grid_to_points(occupancy, grid_res=100):
    """Convert binary occupancy mask to point coordinates."""
    idx = np.argwhere(occupancy)
    if len(idx) == 0:
        return np.array([])
    # Convert grid indices to coordinates in [-0.5, 0.5]
    coords = (idx.astype(float) + 0.5) / grid_res - 0.5
    return coords


def _points_to_grid(pts, grid_res=100):
    """Convert point coordinates to binary occupancy mask."""
    occupancy = np.zeros((grid_res, grid_res), dtype=bool)
    if len(pts) == 0:
        return occupancy
    idx_x = np.clip(((pts[:, 0] + 0.5) * grid_res).astype(int), 0, grid_res - 1)
    idx_y = np.clip(((pts[:, 1] + 0.5) * grid_res).astype(int), 0, grid_res - 1)
    occupancy[idx_y, idx_x] = True
    return occupancy


def _check_non_overlap(occupancy, grid_res=100, min_separation=1):
    """Check that no two occupied cells are within min_separation grid units."""
    if not np.any(occupancy):
        return True
    
    idx = np.argwhere(occupancy)
    if len(idx) <= 1:
        return True
    
    # Use KDTree for efficient distance checking
    tree = KDTree(idx)
    pairs = tree.query_pairs(r=min_separation)
    return len(pairs) == 0


def _evaluate_occupancy(occupancy, grid_res=100):
    """Evaluate a binary occupancy mask: return (num_points, min_separation_in_grid_units)."""
    if not np.any(occupancy):
        return 0, 0
    
    idx = np.argwhere(occupancy)
    n_pts = len(idx)
    if n_pts <= 1:
        return n_pts, grid_res  # infinite separation
    
    tree = KDTree(idx)
    # Find the minimum distance between any pair
    # Use query_pairs with increasing radius to find min separation
    min_sep = grid_res  # start with max possible
    
    # Binary search for minimum separation
    lo, hi = 1, grid_res
    while lo < hi:
        mid = (lo + hi) // 2
        pairs = tree.query_pairs(r=mid)
        if len(pairs) > 0:
            hi = mid
        else:
            lo = mid + 1
    
    min_sep = lo
    return n_pts, min_sep


def _init_occupancy_random(grid_res=100, n_points=N):
    """Generate a random valid occupancy mask with n_points non-overlapping cells."""
    occupancy = np.zeros((grid_res, grid_res), dtype=bool)
    min_sep = 1  # minimum grid units between centers
    
    attempts = 0
    max_attempts = n_points * 50
    while np.sum(occupancy) < n_points and attempts < max_attempts:
        attempts += 1
        y = np.random.randint(0, grid_res)
        x = np.random.randint(0, grid_res)
        
        # Check if this cell can be added
        if occupancy[y, x]:
            continue
        
        # Check against all existing points
        idx = np.argwhere(occupancy)
        if len(idx) > 0:
            dists = np.sqrt((idx[:, 0] - y)**2 + (idx[:, 1] - x)**2)
            if np.min(dists) < min_sep:
                continue
        
        occupancy[y, x] = True
    
    return occupancy


def _init_occupancy_grid(grid_res=100, n_points=N):
    """Generate a grid-based initial occupancy."""
    # Create a regular grid pattern
    cells_per_row = int(np.sqrt(n_points))
    while cells_per_row * cells_per_row < n_points:
        cells_per_row += 1
    
    spacing = (grid_res - 1) / (cells_per_row - 1) if cells_per_row > 1 else grid_res
    
    occupancy = np.zeros((grid_res, grid_res), dtype=bool)
    count = 0
    for i in range(cells_per_row):
        for j in range(cells_per_row):
            if count >= n_points:
                break
            y = int(i * spacing)
            x = int(j * spacing)
            if 0 <= y < grid_res and 0 <= x < grid_res:
                occupancy[y, x] = True
                count += 1
    
    return occupancy


def _crossover(parent1, parent2, grid_res=100, n_points=N):
    """Uniform crossover: randomly select cells from each parent."""
    child = np.zeros((grid_res, grid_res), dtype=bool)
    
    # Get all occupied cells from both parents
    cells1 = np.argwhere(parent1)
    cells2 = np.argwhere(parent2)
    
    # Randomly select from each parent
    n1 = np.random.randint(max(1, len(cells1) // 2), max(2, len(cells1) - 1))
    n2 = n_points - n1
    
    selected1 = cells1[np.random.choice(len(cells1), size=min(n1, len(cells1)), replace=False)]
    selected2 = cells2[np.random.choice(len(cells2), size=min(n2, len(cells2)), replace=False)]
    
    # Combine and check for overlaps
    all_cells = np.vstack([selected1, selected2]) if len(selected2) > 0 else selected1
    
    # Remove duplicates
    unique_cells = np.unique(all_cells, axis=0)
    
    # If we have too many or too few, adjust
    if len(unique_cells) > n_points:
        unique_cells = unique_cells[:n_points]
    elif len(unique_cells) < n_points:
        # Fill remaining from either parent
        remaining = n_points - len(unique_cells)
        all_available = np.vstack([cells1, cells2])
        mask = np.zeros((grid_res, grid_res), dtype=bool)
        for cell in unique_cells:
            mask[cell[0], cell[1]] = True
        available = np.argwhere(~mask)
        if len(available) > 0:
            fill_cells = available[np.random.choice(len(available), size=min(remaining, len(available)), replace=False)]
            unique_cells = np.vstack([unique_cells, fill_cells])
    
    # Set occupancy
    for cell in unique_cells:
        child[cell[0], cell[1]] = True
    
    return child


def _mutate(occupancy, grid_res=100, n_points=N, mutation_rate=0.1):
    """Apply mutation: randomly move some occupied cells."""
    child = occupancy.copy()
    
    idx = np.argwhere(child)
    if len(idx) == 0:
        return child
    
    n_mutate = int(len(idx) * mutation_rate)
    
    for _ in range(n_mutate):
        # Remove a random occupied cell
        remove_idx = np.random.randint(0, len(idx))
        y, x = idx[remove_idx]
        child[y, x] = False
        idx = np.argwhere(child)
        
        # Add a new random cell
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            attempts += 1
            new_y = np.random.randint(0, grid_res)
            new_x = np.random.randint(0, grid_res)
            
            if child[new_y, new_x]:
                continue
            
            # Check non-overlap
            current_idx = np.argwhere(child)
            if len(current_idx) > 0:
                dists = np.sqrt((current_idx[:, 0] - new_y)**2 + (current_idx[:, 1] - new_x)**2)
                if np.min(dists) < 1:
                    continue
            
            child[new_y, new_x] = True
            break
    
    return child


def _tournament_selection(population, fitnesses, k=3):
    """Tournament selection: pick k random individuals, return the best."""
    indices = np.random.choice(len(population), size=k, replace=False)
    best_idx = indices[np.argmax(fitnesses[indices])]
    return population[best_idx].copy()


def _genetic_algorithm(grid_res=100, n_points=N, pop_size=100, generations=50, 
                       deadline=140.0, tournament_k=3, mutation_rate=0.1):
    """Run genetic algorithm to evolve optimal occupancy mask."""
    t_start = time.time()
    
    # Initialize population
    population = []
    fitnesses = []
    
    # Mix of random and grid initializations
    for i in range(pop_size):
        if i < pop_size // 2:
            occ = _init_occupancy_random(grid_res, n_points)
        else:
            occ = _init_occupancy_grid(grid_res, n_points)
            # Add some randomness
            occ = _mutate(occ, grid_res, n_points, mutation_rate=0.2)
        population.append(occ)
        n_pts, min_sep = _evaluate_occupancy(occ, grid_res)
        # Fitness: want many points with large separation
        # Convert grid separation to approximate radius
        approx_radius = min_sep / (2 * grid_res)
        fitness = min_sep * n_pts  # maximize product
        fitnesses.append(fitness)
    
    best_fitness = max(fitnesses) if fitnesses else 0
    best_occ = population[np.argmax(fitnesses)].copy() if population else np.zeros((grid_res, grid_res), dtype=bool)
    
    for gen in range(generations):
        if time.time() > deadline:
            break
        
        # Evaluate fitness
        for i, occ in enumerate(population):
            n_pts, min_sep = _evaluate_occupancy(occ, grid_res)
            fitnesses[i] = min_sep * n_pts
        
        # Update best
        if max(fitnesses) > best_fitness:
            best_fitness = max(fitnesses)
            best_occ = population[np.argmax(fitnesses)].copy()
        
        # Create next generation
        new_population = []
        new_fitnesses = []
        
        # Elitism: keep top 10%
        elite_size = max(1, pop_size // 10)
        elite_indices = np.argsort(fitnesses)[-elite_size:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
            new_fitnesses.append(fitnesses[idx])
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            
            parent1 = _tournament_selection(population, fitnesses, tournament_k)
            parent2 = _tournament_selection(population, fitnesses, tournament_k)
            
            child = _crossover(parent1, parent2, grid_res, n_points)
            child = _mutate(child, grid_res, n_points, mutation_rate)
            
            new_population.append(child)
            n_pts, min_sep = _evaluate_occupancy(child, grid_res)
            new_fitnesses.append(min_sep * n_pts)
        
        population = new_population
        fitnesses = new_fitnesses
    
    return best_occ


def _grid_polish(occupancy, grid_res=100, n_points=N, iterations=500, deadline=140.0):
    """Local search: try moving individual cells to improve minimum separation."""
    occupancy = occupancy.copy()
    
    for _ in range(iterations):
        if time.time() > deadline:
            break
        
        idx = np.argwhere(occupancy)
        if len(idx) != n_points:
            break
        
        # Find the pair with minimum separation
        tree = KDTree(idx)
        pairs = tree.query_pairs(r=grid_res)
        if not pairs:
            break
        
        # Find minimum distance pair
        min_dist = float('inf')
        min_pair = None
        for i, j in pairs:
            d = np.sqrt((idx[i][0] - idx[j][0])**2 + (idx[i][1] - idx[j][1])**2)
            if d < min_dist:
                min_dist = d
                min_pair = (i, j)
        
        if min_pair is None:
            break
        
        i, j = min_pair
        y1, x1 = idx[i]
        y2, x2 = idx[j]
        
        # Try moving one of them to improve separation
        best_move = None
        best_new_min = min_dist
        
        # Try moving point i
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                new_y1 = y1 + dy
                new_x1 = x1 + dx
                if not (0 <= new_y1 < grid_res and 0 <= new_x1 < grid_res):
                    continue
                if occupancy[new_y1, new_x1]:
                    continue
                
                # Check if this improves things
                test_occ = occupancy.copy()
                test_occ[y1, x1] = False
                test_occ[new_y1, new_x1] = True
                
                test_idx = np.argwhere(test_occ)
                test_tree = KDTree(test_idx)
                test_pairs = test_tree.query_pairs(r=grid_res)
                
                test_min = float('inf')
                for pi, pj in test_pairs:
                    d = np.sqrt((test_idx[pi][0] - test_idx[pj][0])**2 + 
                               (test_idx[pi][1] - test_idx[pj][1])**2)
                    if d < test_min:
                        test_min = d
                
                if test_min > best_new_min:
                    best_new_min = test_min
                    best_move = (y1, x1, new_y1, new_x1)
        
        # Try moving point j
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                new_y2 = y2 + dy
                new_x2 = x2 + dx
                if not (0 <= new_y2 < grid_res and 0 <= new_x2 < grid_res):
                    continue
                if occupancy[new_y2, new_x2]:
                    continue
                
                test_occ = occupancy.copy()
                test_occ[y2, x2] = False
                test_occ[new_y2, new_x2] = True
                
                test_idx = np.argwhere(test_occ)
                test_tree = KDTree(test_idx)
                test_pairs = test_tree.query_pairs(r=grid_res)
                
                test_min = float('inf')
                for pi, pj in test_pairs:
                    d = np.sqrt((test_idx[pi][0] - test_idx[pj][0])**2 + 
                               (test_idx[pi][1] - test_idx[pj][1])**2)
                    if d < test_min:
                        test_min = d
                
                if test_min > best_new_min:
                    best_new_min = test_min
                    best_move = (y2, x2, new_y2, new_x2)
        
        if best_move is not None:
            old_y, old_x, new_y, new_x = best_move
            occupancy[old_y, old_x] = False
            occupancy[new_y, new_x] = True
        else:
            break
    
    return occupancy


def _refine_with_continuous(occupancy, grid_res=100, n_points=N, deadline=140.0):
    """Take grid solution and refine with continuous optimization."""
    pts = _grid_to_points(occupancy, grid_res)
    if len(pts) != n_points:
        return pts
    
    # Simple local optimization: push points apart
    for _ in range(200):
        if time.time() > deadline:
            break
        
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        
        # Find minimum distance pair
        min_idx = np.unravel_index(np.argmin(D), D.shape)
        i, j = min_idx[0], min_idx[1]
        
        diff = pts[i] - pts[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-10:
            break
        
        # Push apart
        force = diff / dist * 0.001
        pts[i] += force
        pts[j] -= force
        
        # Keep in bounds
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
    
    return pts


def construct():
    t0 = time.time()
    deadline = t0 + _TIME_LIMIT - 5
    
    np.random.seed(42)
    
    best_r = 0.0
    best_pts = None
    
    # Run genetic algorithm with different grid resolutions
    for grid_res in [50, 80, 100, 120]:
        if time.time() > deadline - 30:
            break
        
        # Run GA
        occ = _genetic_algorithm(grid_res, N, pop_size=50, generations=30, 
                                deadline=deadline - 20, tournament_k=3, mutation_rate=0.1)
        
        # Polish
        occ = _grid_polish(occ, grid_res, N, iterations=200, deadline=deadline - 10)
        
        # Convert to points
        pts = _grid_to_points(occ, grid_res)
        
        if len(pts) != N:
            continue
        
        # Refine with continuous optimization
        pts = _refine_with_continuous(pts, grid_res, N, deadline=deadline - 5)
        
        r = _get_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
        
        # Try different GA parameters
        if time.time() < deadline - 25:
            occ2 = _genetic_algorithm(grid_res, N, pop_size=30, generations=40, 
                                     deadline=deadline - 20, tournament_k=5, mutation_rate=0.15)
            occ2 = _grid_polish(occ2, grid_res, N, iterations=100, deadline=deadline - 10)
            pts2 = _grid_to_points(occ2, grid_res)
            if len(pts2) == N:
                pts2 = _refine_with_continuous(pts2, grid_res, N, deadline=deadline - 5)
                r2 = _get_radius(pts2)
                if r2 > best_r:
                    best_r = r2
                    best_pts = pts2.copy()
    
    # Fallback: simple grid
    if best_pts is None:
        rows, cols = 14, 15
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]
    
    return best_pts


if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
