"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 14

# EVOLVE-BLOCK-START
def _convex_hull_area(points):
    """Compute area of convex hull of points using monotone chain."""
    if len(points) < 3:
        return 0.0
    points = np.array(points)
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    
    # Shoelace formula
    x = hull[:, 0]
    y = hull[:, 1]
    area = 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    return area


def _min_triangle_area(points):
    """Compute minimum area of all triples of points."""
    n = len(points)
    if n < 3:
        return 0.0
    
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                xi, yi = points[i]
                xj, yj = points[j]
                xk, yk = points[k]
                area = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _differential_evolution(objective, bounds, max_iter=100, pop_size=50, 
                           F=0.8, CR=0.9, tol=1e-8, seed=42):
    """Simple differential evolution optimizer.
    
    Args:
        objective: function to minimize (we'll negate score to maximize)
        bounds: list of (min, max) for each dimension
        max_iter: maximum iterations
        pop_size: population size
        F: scaling factor
        CR: crossover rate
        tol: tolerance for convergence
        seed: random seed
    
    Returns:
        best_x, best_f
    """
    np.random.seed(seed)
    D = len(bounds)
    
    # Initialize population
    pop = np.zeros((pop_size, D))
    for i in range(D):
        pop[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], pop_size)
    
    # Evaluate initial population
    fitness = np.array([objective(x) for x in pop])
    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]
    
    for it in range(max_iter):
        new_pop = pop.copy()
        
        for i in range(pop_size):
            # Select three distinct random indices
            while True:
                a, b, c = np.random.choice(pop_size, 3, replace=False)
                if a != b and b != c and a != c:
                    break
            
            # Create trial vector
            v = pop[a] + F * (pop[b] - pop[c])
            
            # Crossover
            r = np.random.rand()
            jrand = np.random.randint(D)
            for j in range(D):
                if np.random.rand() < CR or j == jrand:
                    new_pop[i, j] = v[j]
                else:
                    new_pop[i, j] = pop[i, j]
            
            # Clip to bounds
            for j in range(D):
                new_pop[i, j] = np.clip(new_pop[i, j], bounds[j][0], bounds[j][1])
            
            # Evaluate
            new_f = objective(new_pop[i])
            if new_f < fitness[i]:
                fitness[i] = new_f
                pop[i] = new_pop[i]
                if new_f < best_f:
                    best_f = new_f
                    best_x = new_pop[i].copy()
        
        # Convergence check
        if it > 0:
            if abs(fitness[best_idx] - best_f) < tol and it > 10:
                break
    
    return best_x, best_f


def construct():
    """Construct 14 points to maximize minimum triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Represent points in polar coordinates: (r, theta) for each point
    # r in [0, 1], theta in [0, 2*pi)
    # We'll enforce increasing angular order to avoid self-intersection
    
    # Initial guess: points on a regular polygon
    def _initial_guess():
        thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N) * 0.95
        return np.column_stack([radii, thetas])
    
    # Bounds for polar coordinates
    # r in [0.1, 1.0], theta in [0, 2*pi]
    bounds = [(0.1, 1.0)] * N + [(0, 2 * np.pi)] * N
    
    def _decode(polar_params):
        """Convert polar params to Cartesian points with sorted angles."""
        radii = polar_params[:N]
        thetas = polar_params[N:]
        
        # Sort by angle
        sort_idx = np.argsort(thetas)
        radii_sorted = radii[sort_idx]
        thetas_sorted = thetas[sort_idx]
        
        # Ensure strictly increasing angles by adding small offsets if needed
        # Actually, just use them as is; sorting handles the order
        
        x = radii_sorted * np.cos(thetas_sorted)
        y = radii_sorted * np.sin(thetas_sorted)
        return np.column_stack([x, y])
    
    def _objective(polar_params):
        """Objective to minimize: negative of score."""
        points = _decode(polar_params)
        s = _score(points)
        return -s
    
    # Start with a good initial population
    best_score = 0.0
    best_points = None
    
    # Try several random restarts with differential evolution
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Create initial population
        pop_size = 40
        max_iter = 80
        
        # First individual: regular polygon
        init_polar = _initial_guess()
        
        # Create population with perturbations
        pop = np.zeros((pop_size, 2 * N))
        pop[0] = init_polar
        
        for i in range(1, pop_size):
            # Random perturbation around initial guess
            perturbation = np.random.normal(0, 0.1, 2 * N)
            pop[i] = init_polar + perturbation
            # Clip to bounds
            for j in range(N):
                pop[i, j] = np.clip(pop[i, j], bounds[j][0], bounds[j][1])
                pop[i, j + N] = np.clip(pop[i, j + N], bounds[j + N][0], bounds[j + N][1])
        
        # Evaluate initial best
        for i in range(pop_size):
            points = _decode(pop[i])
            s = _score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
        
        # Run differential evolution
        best_x, best_f = _differential_evolution(
            _objective, bounds, max_iter=max_iter, pop_size=pop_size,
            F=0.8, CR=0.9, tol=1e-10, seed=42 + restart
        )
        
        # Check result
        points = _decode(best_x)
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        
        # Check time
        if time.time() > deadline - 20.0:
            break
    
    if best_points is None:
        # Fallback: regular polygon
        thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N) * 0.95
        best_points = np.column_stack([radii * np.cos(thetas), radii * np.sin(thetas)])
    
    return best_points
# EVOLVE-BLOCK-END
