        if len(points) > N:
            np.random.seed(trial)
            selected_indices = []
            
            # min_dist[i] is the min distance from points[i] to any selected point
            min_dist = np.inf * np.ones(len(points))
            
            # Start with a random point
            idx = np.random.choice(len(points))
            selected_indices.append(idx)
            min_dist[idx] = 0 # Distance to self is 0
            
            for _ in range(N-1):
                # Update min_dist for all points based on the newly selected idx
                # dist from all points to idx
                dists_to_idx = np.linalg.norm(points - points[idx], axis=1)
                min_dist = np.minimum(min_dist, dists_to_idx)
                
                # Select the point with the largest min_dist
                # Exclude already selected
                for idx_sel in selected_indices:
                    min_dist[idx_sel] = -1
                    
                idx = np.argmax(min_dist)
                selected_indices.append(idx)
                
            points = points[selected_indices]
