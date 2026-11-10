import numpy as np

def construct():
    N = 300
    
    # 1. Initialize with a hexagonal lattice
    # Generate a grid of points and select N points closest to origin
    # or use a spiral. Hex lattice is denser.
    s = 0.1
    k = int(1.0 / s) + 2
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            points.append((x, y))
    points = np.array(points)
    radii = np.hypot(points[:, 0], points[:, 1])
    # Keep points well inside
    points = points[radii < 0.9]
    
    if len(points) > N:
        # Sort by distance to origin to pick the densest central cluster
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)
        points = points[idx[:N]]
    elif len(points) < N:
        # Pad with random points if lattice is too sparse
        while len(points) < N:
            p = np.random.uniform(-0.8, 0.8, 2)
            if np.linalg.norm(p) < 0.9:
                points = np.vstack([points, p])

    # 2. Simulated Annealing
    # Objective: Maximize min_dist
    # Energy: -min_dist (we minimize energy)
    
    current_pos = points.copy()
    best_pos = current_pos.copy()
    best_score = -1.0
    
    # Calculate initial score
    def get_score(pos):
        # Vectorized pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        # Min distance between centers
        np.fill_diagonal(dist, np.inf)
        min_d = np.min(dist)
        
        # Min distance to boundary
        radii = np.linalg.norm(pos, axis=1)
        min_b = 1.0 - np.max(radii)
        
        # The radius is limited by both
        r = min(min_d / 2.0, min_b)
        return r

    current_score = get_score(current_pos)
    if current_score > best_score:
        best_score = current_score
        best_pos = current_pos.copy()

    T = 0.01 # Initial temperature
    T_min = 1e-5
    cooling = 0.999
    max_iter = 50000
    
    # Pre-allocate for speed? 
    # Loop
    for i in range(max_iter):
        # Generate a move: perturb a random subset of points
        n_perturb = np.random.randint(1, 5) # Perturb 1 to 4 points
        idx = np.random.choice(N, n_perturb, replace=False)
        
        new_pos = current_pos.copy()
        # Perturbation size should decrease with T
        step_size = T * 0.5
        new_pos[idx] += np.random.normal(0, step_size, (n_perturb, 2))
        
        # Enforce boundary constraint: project back inside
        radii = np.linalg.norm(new_pos, axis=1)
        too_far = radii > 1.0
        if np.any(too_far):
            new_pos[too_far] = new_pos[too_far] / radii[too_far, np.newaxis]
            
        new_score = get_score(new_pos)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_pos = new_pos
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pos = current_pos.copy()
                
        T *= cooling
        if T < T_min:
            break

    return best_pos
