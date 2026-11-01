"""Improved program using grid-based genetic algorithm with local refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def hexagon_vertices(center, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon."""
    cx, cy = center
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.radians(60 * i)
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def sat_min_separation(c1, c2):
    """Compute minimum separation between two hexagons using SAT."""
    v1 = hexagon_vertices((c1[0], c1[1]), 1.0, c1[2])
    v2 = hexagon_vertices((c2[0], c2[1]), 1.0, c2[2])
    
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i + 1) % 6] - v[i]
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            n = np.array([-edge[1], edge[0]]) / edge_len
            axes.append(n)
    
    min_sep = float('inf')
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
        if sep < min_sep:
            min_sep = sep
    
    return min_sep

def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    max_extent = 0.0
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    
    for (px, py, pa) in centers:
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        for v in inner_verts:
            rel = np.array([v[0] - outer_center[0], v[1] - outer_center[1]])
            rx = rel[0] * cos_a - rel[1] * sin_a
            ry = rel[0] * sin_a + rel[1] * cos_a
            
            for k in range(6):
                theta = np.radians(60 * k)
                nx = np.cos(theta)
                ny = np.sin(theta)
                d = abs(rx * nx + ry * ny)
                max_extent = max(max_extent, d)
    
    return max_extent / (np.sqrt(3) / 2)

def check_validity(centers, outer_center, outer_side, outer_angle):
    """Check if packing is valid: all inner hexagons inside outer, no overlaps."""
    actual_side = required_outer_side(centers, outer_center, outer_angle)
    if actual_side > outer_side + 1e-9:
        return False
    
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            s = sat_min_separation(centers[i], centers[j])
            if s < -1e-9:
                return False
    return True

def compute_all_clearances(centers, outer_center, outer_angle, outer_side):
    """Compute all pairwise and outer clearances."""
    n = len(centers)
    pair_clearances = []
    outer_clearances = []
    
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    apothem = outer_side * np.sqrt(3) / 2
    
    # Precompute vertices
    inner_verts = []
    for c in centers:
        v = hexagon_vertices((c[0], c[1]), 1.0, c[2])
        inner_verts.append(v)
    
    # Pairwise separations
    for i in range(n):
        for j in range(i + 1, n):
            v1 = inner_verts[i]
            v2 = inner_verts[j]
            
            axes = []
            for v in [v1, v2]:
                for k in range(6):
                    edge = v[(k + 1) % 6] - v[k]
                    elen = np.linalg.norm(edge)
                    if elen > 1e-12:
                        nrm = np.array([-edge[1], edge[0]]) / elen
                        axes.append(nrm)
            
            min_sep = float('inf')
            for axis in axes:
                proj1 = np.dot(v1, axis)
                proj2 = np.dot(v2, axis)
                sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
                if sep < min_sep:
                    min_sep = sep
            
            pair_clearances.append(min_sep)
    
    # Outer clearance
    for i in range(n):
        v = inner_verts[i]
        for vert in v:
            rel = np.array([vert[0] - outer_center[0], vert[1] - outer_center[1]])
            rx = rel[0] * cos_a - rel[1] * sin_a
            ry = rel[0] * sin_a + rel[1] * cos_a
            
            max_proj = 0.0
            for k in range(6):
                theta = np.radians(60 * k)
                nx = np.cos(theta)
                ny = np.sin(theta)
                d = abs(rx * nx + ry * ny)
                if d > max_proj:
                    max_proj = d
            
            clearance = apothem - max_proj
            outer_clearances.append(clearance)
    
    return pair_clearances, outer_clearances

def grid_based_ga(init_centers, time_limit=40.0):
    """
    Grid-based genetic algorithm using triangular lattice positions.
    Represents each hexagon by a lattice index and rotation angle.
    Uses vectorized collision detection.
    """
    start_time = time.time()
    
    # Define triangular lattice
    # Lattice spacing: hexagons of side 1 have diameter 2, so lattice spacing ~2*sqrt(3) for non-overlap
    # But we want to allow some overlap in the search space, so use finer grid
    # Use a hexagonal grid with spacing that allows exploration
    lattice_spacing = 0.5  # Fine grid for exploration
    
    # Grid extent
    grid_range = 12  # -12 to 12 in lattice units
    
    # Precompute lattice positions
    # For hexagonal lattice: two basis vectors at 60 degrees
    a1 = np.array([lattice_spacing, 0.0])
    a2 = np.array([lattice_spacing * 0.5, lattice_spacing * np.sqrt(3) / 2])
    
    # Generate all lattice points
    lattice_points = []
    for i in range(-grid_range, grid_range + 1):
        for j in range(-grid_range, grid_range + 1):
            pt = i * a1 + j * a2
            lattice_points.append(pt)
    
    lattice_points = np.array(lattice_points)
    n_lattice = len(lattice_points)
    
    # Discretize angles: 0, 30, 60 degrees (hexagon has 6-fold symmetry)
    # Actually, for a regular hexagon, rotations by 60 degrees give the same shape
    # So we only need angles in [0, 60)
    angle_options = np.array([0.0, 15.0, 30.0, 45.0, 60.0])  # 5 angle options
    
    n_angles = len(angle_options)
    
    # Population: each individual is (n_positions, n_angles)
    # n_positions: indices into lattice_points
    # n_angles: indices into angle_options
    
    pop_size = 200
    
    def init_population():
        """Initialize population from init_centers plus random."""
        pop = []
        
        # First individual: based on init_centers, snap to nearest lattice point
        ind = []
        for c in init_centers:
            # Find nearest lattice point
            dists = np.sum((lattice_points - np.array([c[0], c[1]])) ** 2, axis=1)
            pos_idx = np.argmin(dists)
            # Snap angle
            angle_idx = np.argmin(np.abs(angle_options - c[2]))
            ind.append([pos_idx, angle_idx])
        pop.append(np.array(ind))
        
        # Rest: random
        for _ in range(pop_size - 1):
            ind = []
            for _ in range(N):
                pos_idx = np.random.randint(n_lattice)
                angle_idx = np.random.randint(n_angles)
                ind.append([pos_idx, angle_idx])
            pop.append(np.array(ind))
        
        return np.array(pop)
    
    def evaluate_individual(ind):
        """Evaluate an individual: returns (required_side, min_clearance)."""
        centers = []
        for pos_idx, angle_idx in ind:
            px, py = lattice_points[pos_idx]
            pa = angle_options[angle_idx]
            centers.append([px, py, pa])
        
        # Check for overlaps
        min_pair_sep = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                s = sat_min_separation(centers[i], centers[j])
                if s < min_pair_sep:
                    min_pair_sep = s
        
        # Compute required outer side
        req_side = required_outer_side(centers, (0.0, 0.0), 0.0)
        
        # If there's overlap, heavily penalize
        if min_pair_sep < 0:
            return req_side + 100.0 * abs(min_pair_sep), min_pair_sep
        
        return req_side, min_pair_sep
    
    def population_fitness(pop):
        """Evaluate all individuals in population."""
        fitness = []
        min_clearances = []
        for ind in pop:
            fit, mc = evaluate_individual(ind)
            fitness.append(fit)
            min_clearances.append(mc)
        return np.array(fitness), np.array(min_clearances)
    
    def select_parents(fitness, n_select):
        """Tournament selection."""
        parents = []
        for _ in range(n_select):
            # Tournament of 3
            idxs = np.random.choice(len(fitness), 3, replace=False)
            best = idxs[np.argmin(fitness[idxs])]
            parents.append(pop[best])
        return np.array(parents)
    
    def crossover(parent1, parent2):
        """Uniform crossover."""
        mask = np.random.rand(N) > 0.5
        child = np.where(mask[:, None], parent1, parent2)
        return child
    
    def mutate(ind, rate=0.1):
        """Mutation: change position or angle."""
        ind = ind.copy()
        for i in range(N):
            if np.random.rand() < rate:
                # Mutate position: move to nearby lattice point
                if np.random.rand() < 0.5:
                    # Small move
                    delta_i = np.random.randint(-2, 3)
                    delta_j = np.random.randint(-2, 3)
                    # Decode current position
                    # This is approximate; just pick random nearby
                    new_pos = np.random.randint(max(0, ind[i, 0] - 5), min(n_lattice - 1, ind[i, 0] + 5))
                    ind[i, 0] = new_pos
                else:
                    # Mutate angle
                    ind[i, 1] = np.random.randint(n_angles)
        return ind
    
    # Initialize
    pop = init_population()
    fitness, min_clears = population_fitness(pop)
    
    best_fit = np.min(fitness)
    best_ind = pop[np.argmin(fitness)]
    best_centers = []
    for pos_idx, angle_idx in best_ind:
        px, py = lattice_points[pos_idx]
        pa = angle_options[angle_idx]
        best_centers.append([px, py, pa])
    
    # GA loop
    gen = 0
    while time.time() - start_time < time_limit:
        gen += 1
        
        # Elitism: keep top 2
        sorted_idx = np.argsort(fitness)
        new_pop = [pop[sorted_idx[0]].copy(), pop[sorted_idx[1]].copy()]
        
        # Generate rest via selection, crossover, mutation
        n_offspring = pop_size - 2
        while len(new_pop) < pop_size:
            p1_idx = np.random.randint(len(fitness))
            p2_idx = np.random.randint(len(fitness))
            # Weighted selection: better individuals more likely
            if np.random.rand() < 0.7:
                # Select from top half
                p1_idx = np.random.randint(len(fitness) // 2)
                p2_idx = np.random.randint(len(fitness) // 2)
            
            child = crossover(pop[p1_idx], pop[p2_idx])
            child = mutate(child, rate=0.15)
            new_pop.append(child)
        
        pop = np.array(new_pop)
        fitness, min_clears = population_fitness(pop)
        
        # Update best
        if np.min(fitness) < best_fit:
            best_fit = np.min(fitness)
            best_ind = pop[np.argmin(fitness)]
            best_centers = []
            for pos_idx, angle_idx in best_ind:
                px, py = lattice_points[pos_idx]
                pa = angle_options[angle_idx]
                best_centers.append([px, py, pa])
    
    return best_centers

def local_refinement(centers, time_limit=20.0):
    """
    Local refinement using Nelder-Mead on the continuous variables.
    Minimizes the maximum of (required_outer_side, -min_pair_clearance).
    """
    start_time = time.time()
    
    # Flatten parameters
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = centers[i][0]
        params[3*i+1] = centers[i][1]
        params[3*i+2] = centers[i][2]
    
    def objective(p):
        if time.time() - start_time > time_limit:
            return 1e6
        
        c = []
        for i in range(N):
            c.append([p[3*i], p[3*i+1], p[3*i+2]])
        
        # Check overlaps
        min_sep = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                s = sat_min_separation(c[i], c[j])
                if s < min_sep:
                    min_sep = s
        
        req_side = required_outer_side(c, (0.0, 0.0), 0.0)
        
        # Penalty for overlap
        penalty = max(0, -min_sep) * 50.0
        
        return req_side + penalty
    
    result = minimize(
        objective, params, method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
    )
    
    new_centers = []
    for i in range(N):
        new_centers.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2]])
    
    return new_centers

def initial_config_hexagonal():
    """Create a hexagonal lattice initial configuration."""
    centers = []
    dx = 2.0 * np.sqrt(3)
    dy = 3.0
    
    positions = [
        (0, 0),
        (dx, 0), (-dx, 0),
        (dx/2, dy*0.866), (-dx/2, dy*0.866),
        (dx/2, -dy*0.866), (-dx/2, -dy*0.866),
        (3*dx/2, dy*0.866), (-3*dx/2, dy*0.866),
        (3*dx/2, -dy*0.866), (-3*dx/2, -dy*0.866),
        (0, dy*1.732), (0, -dy*1.732)
    ]
    
    for (x, y) in positions:
        centers.append([x, y, 0.0])
    
    return centers

def initial_config_circular():
    """Create a circular arrangement."""
    centers = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        r = 2.5
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a = np.degrees(theta) % 60
        centers.append([x, y, a])
    return centers

def initial_config_random(seed=42):
    """Create a random initial configuration."""
    np.random.seed(seed)
    centers = []
    for i in range(N):
        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, 4)
        a = np.random.uniform(0, 60)
        centers.append([x, y, a])
    return centers

def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    configs = [
        ("hexagonal", initial_config_hexagonal()),
        ("circular", initial_config_circular()),
        ("random1", initial_config_random(42)),
        ("random2", initial_config_random(123)),
    ]
    
    for name, config in configs:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
        
        # Phase 1: Grid-based GA for global exploration
        ga_time = min(35.0, DEADLINE - elapsed - 15)
        if ga_time < 10:
            ga_time = 10
        
        try:
            ga_centers = grid_based_ga(config, time_limit=ga_time)
        except Exception:
            ga_centers = [list(c) for c in config]
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 12:
            break
        
        # Phase 2: Local refinement with Nelder-Mead
        req_side = required_outer_side(ga_centers, (0.0, 0.0), 0.0)
        refine_time = min(10.0, DEADLINE - elapsed - 3)
        
        try:
            refined = local_refinement(ga_centers, time_limit=refine_time)
        except Exception:
            refined = ga_centers
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 3:
            break
        
        # Compute final required side
        final_side = required_outer_side(refined, (0.0, 0.0), 0.0)
        
        # Validate: check no overlaps
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                s = sat_min_separation(refined[i], refined[j])
                if s < -1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and final_side < best_score:
            best_score = final_side
            best_result = (refined, (0.0, 0.0), final_side, 0.0)
    
    if best_result is not None:
        return best_result
    
    # Fallback
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    required_side = required_outer_side(inner_default, (0.0, 0.0), 0.0)
    return inner_default, (0.0, 0.0), required_side, 0.0
