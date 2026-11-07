import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
        n = distance_matrix.shape[0]
        dm = np.asarray(distance_matrix, dtype=np.float64)
        coords = np.asarray(coordinates, dtype=np.float64)
        dem = np.asarray(demands, dtype=np.float64)
        
        # Precompute angular differences
        depot = coords[0]
        dx = coords[:, 0] - depot[0]
        dy = coords[:, 1] - depot[1]
        angles = np.arctan2(dy, dx)
        
        # Vectorized angle difference matrix
        angle_diff = angles[:, np.newaxis] - angles[np.newaxis, :]
        angle_diff = (angle_diff + np.pi) % (2.0 * np.pi) - np.pi
        
        # Mask for customer-customer edges
        mask = np.ones((n, n), dtype=bool)
        mask[0, :] = False
        mask[:, 0] = False
        
        eps = 1e-9
        
        # Fast Greedy NNC Proxy Solver
        def greedy_cost(alpha, lambda_angle):
            # Compute heuristic for this parameter set
            # We only need the values for the greedy choice, 
            # but computing the full matrix is fast for N=50
            h = 1.0 / (dm + eps) ** alpha
            if lambda_angle > 0:
                ang_factor = np.exp(-lambda_angle * (angle_diff ** 2))
                h = h.copy()
                h[mask] *= ang_factor[mask]
            
            total_dist = 0.0
            unvisited = set(range(1, n))
            current = 0
            load = 0.0
            
            while unvisited:
                # Find candidates that fit
                # Optimization: Instead of filtering list every time, 
                # we can just check capacity in the loop. N is small.
                best_node = -1
                best_score = -1.0
                
                # Iterate over unvisited
                # To speed up, we can use numpy if we keep a boolean array
                # But for N=50, python loop is fine and simpler to implement correctly
                # Let's use a boolean array for unvisited to allow numpy argmax
                # Actually, let's stick to the set/list for clarity and low N
                candidates = []
                for j in unvisited:
                    if dem[j] <= capacity - load:
                        candidates.append(j)
                
                if not candidates:
                    # Return to depot
                    total_dist += dm[current, 0]
                    current = 0
                    load = 0.0
                    continue
                
                # Get scores
                # h[current, j]
                # Find max
                max_score = -1.0
                max_idx = -1
                for j in candidates:
                    score = h[current, j]
                    if score > max_score:
                        max_score = score
                        max_idx = j
                
                if max_idx == -1:
                    # Should not happen if candidates is not empty
                    break
                
                total_dist += dm[current, max_idx]
                load += dem[max_idx]
                unvisited.remove(max_idx)
                current = max_idx
                
            # Return to depot
            total_dist += dm[current, 0]
            return total_dist

        # Grid Search
        best_alpha = 2.0
        best_lambda = 4.0
        min_cost = float('inf')
        
        # Refined grid
        alphas = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
        lambdas = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
        
        for a in alphas:
            for l in lambdas:
                cost = greedy_cost(a, l)
                if cost < min_cost:
                    min_cost = cost
                    best_alpha = a
                    best_lambda = l
        
        # Final Heuristic Computation with best params
        final_h = 1.0 / (dm + eps) ** best_alpha
        if best_lambda > 0:
            ang_factor = np.exp(-best_lambda * (angle_diff ** 2))
            final_h = final_h.copy()
            final_h[mask] *= ang_factor[mask]
            
        # Sanitize
        final_h[~np.isfinite(final_h)] = 1e-9
        final_h[final_h <= 0] = 1e-9
        
        return final_h
