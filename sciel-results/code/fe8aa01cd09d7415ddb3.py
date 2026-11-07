import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

def _fast_aco_simulate(distance_matrix, heuristic, n_ants=20, n_iterations=30, rho=0.9, seed=42):
    """
    Fast internal ACO simulation to evaluate heuristic quality.
    Implements directed pheromone updates as per contract.
    """
    rng = np.random.RandomState(seed)
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    best_tour_len = float('inf')
    
    for it in range(n_iterations):
        tour_lengths = []
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_len = 0.0
            tour_path = [0]
            
            for _ in range(n - 1):
                probs = tau[current, :] * heuristic[current, :]
                probs[visited] = 0.0
                
                total_prob = probs.sum()
                if total_prob < 1e-10:
                    unvisited_indices = np.where(~visited)[0]
                    next_node = unvisited_indices[rng.randint(len(unvisited_indices))]
                else:
                    probs = probs / total_prob
                    next_node = rng.choice(n, p=probs)
                
                tour_len += distance_matrix[current, next_node]
                visited[next_node] = True
                tour_path.append(next_node)
                current = next_node
            
            tour_len += distance_matrix[current, 0]
            tour_lengths.append(tour_len)
            
            deposit = 1.0 / tour_len
            # Vectorized pheromone update for the closed tour
            u = np.array(tour_path)
            v = np.array(tour_path[1:] + [tour_path[0]])
            tau[u, v] += deposit
            
        tau *= rho
        
        if tour_lengths:
            current_best = min(tour_lengths)
            if current_best < best_tour_len:
                best_tour_len = current_best
                
    return best_tour_len

def _nn_tour(distance_matrix, start=0):
    """Nearest neighbor tour construction."""
    n = distance_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    current = start
    tour = [start]
    
    for _ in range(n - 1):
        dists = distance_matrix[current].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    return tour

def _two_opt_improve(tour, distance_matrix, max_iter=100):
    """2-opt local search improvement."""
    n = len(tour)
    tour = list(tour)
    improved = True
    iterations = 0
    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j == n - 1 and i == 0:
                    continue
                # Calculate delta
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                old_len = distance_matrix[a, b] + distance_matrix[c, d]
                new_len = distance_matrix[a, c] + distance_matrix[b, d]
                if new_len < old_len - 1e-10:
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
    return tour

def _construct_backbone_tour(distance_matrix, n_restarts=5, seed=42):
    """Construct a strong backbone tour using NN+2-opt with multiple restarts."""
    rng = np.random.RandomState(seed)
    n = distance_matrix.shape[0]
    best_tour = None
    best_len = float('inf')
    
    for restart in range(n_restarts):
        # Random start node
        start = rng.randint(n)
        # NN tour from start
        tour = _nn_tour(distance_matrix, start)
        # 2-opt improvement
        tour = _two_opt_improve(tour, distance_matrix, max_iter=50)
        
        # Calculate tour length
        tour_len = sum(distance_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))
        
        if tour_len < best_len:
            best_len = tour_len
            best_tour = tour
    
    return best_tour, best_len

def _compute_delaunay_circumradii(distance_matrix):
    """
    Compute Delaunay triangulation and return a matrix of local circumradii for each edge.
    For edges in the Delaunay triangulation, the value is the circumradius of the triangle.
    For non-Delaunay edges, the value is infinity.
    """
    n = distance_matrix.shape[0]
    # Recover coordinates from distance matrix (up to rotation/translation)
    # We can use MDS or just assume the distance matrix comes from points.
    # Since we only have the distance matrix, we need to embed it.
    # However, for 50 points in 2D, we can use classical MDS.
    # Or, simpler: the task says "50 uniform random points in the unit square".
    # The distance matrix is derived from these points.
    # We can recover the points using MDS.
    
    # Classical MDS
    D = distance_matrix
    D_sq = D ** 2
    # Double centering
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D_sq @ J
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top 2 components
    coords = eigenvectors[:, :2] * np.sqrt(np.maximum(eigenvalues[:2], 0))
    
    # Perform Delaunay triangulation
    try:
        tri = Delaunay(coords)
    except Exception:
        # Fallback if triangulation fails (e.g., degenerate points)
        return np.full((n, n), np.inf), coords
        
    circumradii = np.full((n, n), np.inf)
    
    for simplex in tri.simplices:
        p1 = coords[simplex[0]]
        p2 = coords[simplex[1]]
        p3 = coords[simplex[2]]
        
        # Calculate circumradius
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        
        # Area using Heron's formula
        s = (a + b + c) / 2.0
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq <= 0:
            continue
        area = np.sqrt(area_sq)
        
        if area < 1e-10:
            continue
            
        R = (a * b * c) / (4.0 * area)
        
        # Update circumradii for the edges of this triangle
        # The circumradius is the same for all 3 edges of this triangle.
        # If an edge is shared by multiple triangles, we might want the minimum or average.
        # Minimum circumradius implies "tighter" local fit, which might be better.
        # Let's take the minimum circumradius for each edge (most local density).
        
        edges = [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]
        for u, v in edges:
            if R < circumradii[u, v]:
                circumradii[u, v] = R
            if R < circumradii[v, u]:
                circumradii[v, u] = R
                
    return circumradii, coords

def _compute_mst_edges(distance_matrix):
    """
    Compute the Minimum Spanning Tree edges using Prim's algorithm.
    Returns a boolean matrix where True indicates an MST edge.
    """
    n = distance_matrix.shape[0]
    mst_adj = np.zeros((n, n), dtype=bool)
    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    min_dist = distance_matrix[0].copy()
    min_dist[0] = np.inf
    
    for _ in range(n - 1):
        # Find the vertex with the minimum distance to the MST
        # np.argmin returns the first occurrence of the minimum value
        u = np.argmin(min_dist)
        # If min_dist is inf, we have a disconnected graph (shouldn't happen for TSP)
        if min_dist[u] == np.inf:
            break
            
        in_mst[u] = True
        # Update min_dist for neighbors
        # We need to find which vertex in the MST is connected to u with min_dist[u]
        # Actually, min_dist[u] stores the min distance from u to ANY vertex in MST.
        # To find the specific parent, we need to search.
        # However, for just marking edges, we can just record that u is added.
        # But we need the edge (parent, u).
        
        # Let's find the parent
        parent = -1
        for v in range(n):
            if in_mst[v] and v != u:
                if distance_matrix[v, u] == min_dist[u]:
                    parent = v
                    break
        if parent != -1:
            mst_adj[parent, u] = True
            mst_adj[u, parent] = True
        
        # Update min_dist for all non-MST vertices
        for v in range(n):
            if not in_mst[v]:
                if distance_matrix[u, v] < min_dist[v]:
                    min_dist[v] = distance_matrix[u, v]
                    
    return mst_adj

def heuristics(distance_matrix):
    start_time = time.time()
    # Safety margin: evaluator kills at 330s, we should finish well before.
    deadline = start_time + 280.0
    
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance (smooth continuous floor)
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    
    base_heur = 1.0 / safe_dist
    
    # Construct backbone tour using NN+2-opt with restarts
    if time.time() > deadline - 120.0:
        # Not enough time for backbone, use pure inverse distance
        final_heuristic = base_heur.copy()
        final_heuristic[final_heuristic <= 0] = 1e-9
        final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
        return final_heuristic
    
    backbone_tour, backbone_len = _construct_backbone_tour(distance_matrix, n_restarts=5, seed=42)
    
    # Create backbone edge mask (directed)
    backbone_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        u = backbone_tour[i]
        v = backbone_tour[(i + 1) % n]
        backbone_mask[u, v] = True
    
    # Search for boost factor and beta exponent
    # Heuristic = base_heur ** beta * (1 + boost * backbone_mask)
    # This is a smooth blend: backbone edges get a multiplicative boost
    
    def evaluate_params(beta, boost):
        if time.time() > deadline - 5.0:
            return float('inf')
        
        h = base_heur ** beta
        # Smooth blend: add boost on backbone edges
        h = h * (1.0 + boost * backbone_mask.astype(float))
        
        # Ensure validity
        h[h <= 0] = 1e-9
        h[~np.isfinite(h)] = 1e-9
        
        # Quick evaluation with 5 iterations
        score = _fast_aco_simulate(distance_matrix, h, n_ants=20, n_iterations=5, seed=42)
        return score

    best_beta = 2.0
    best_boost = 10.0
    best_score = float('inf')
    
    # Coarse search
    betas_coarse = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    boosts_coarse = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    for beta in betas_coarse:
        for boost in boosts_coarse:
            if time.time() > deadline - 60.0:
                break
            score = evaluate_params(beta, boost)
            if score < best_score:
                best_score = score
                best_beta = beta
                best_boost = boost
        if time.time() > deadline - 60.0:
            break
    
    # Fine search
    if time.time() < deadline - 30.0:
        lower_b = max(0.5, best_beta - 1.0)
        upper_b = min(6.0, best_beta + 1.0)
        lower_boost = max(0.1, best_boost * 0.5)
        upper_boost = min(200.0, best_boost * 2.0)
        
        betas_fine = np.linspace(lower_b, upper_b, 8)
        boosts_fine = np.linspace(lower_boost, upper_boost, 8)
        
        for beta in betas_fine:
            for boost in boosts_fine:
                if time.time() > deadline - 10.0:
                    break
                score = evaluate_params(beta, boost)
                if score < best_score:
                    best_score = score
                    best_beta = beta
                    best_boost = boost
            if time.time() > deadline - 10.0:
                break
    
    # Construct final heuristic
    final_heuristic = base_heur ** best_beta
    final_heuristic = final_heuristic * (1.0 + best_boost * backbone_mask.astype(float))
    
    # Ensure validity
    final_heuristic[final_heuristic <= 0] = 1e-9
    final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
    
    return final_heuristic
