    def facet_gradient_step(points, alpha, radii, circumcenters, faces, beta=20.0):
        """
        Move each vertex toward the circumcenters of facets it belongs to,
        weighted by a soft-max over facet radii to smooth the max.
        """
        n_points = len(points)
        n_faces = len(faces)
        
        if n_faces == 0:
            return points
        
        # Soft-max weights: w_i = exp(beta * r_i) / sum(exp(beta * r_j))
        # This makes facets with larger radii get more weight
        max_r = np.max(radii)
        weights = np.exp(beta * (radii - max_r))
        weights /= np.sum(weights)
        
        # Vectorized force calculation
        # faces: (F, 3)
        # weights: (F,)
        # circumcenters: (F, 3)
        # points: (N, 3)
        
        # Get points for each face vertex
        # face_points: (F, 3, 3)
        face_points = points[faces]
        
        # Force direction for each face-vertex pair: cc - p
        # cc: (F, 1, 3) to broadcast
        forces_raw = circumcenters[:, np.newaxis, :] - face_points # (F, 3, 3)
        
        # Project to tangent plane: force - dot(force, p) * p
        # dot_force_p: (F, 3)
        dot_fp = np.sum(forces_raw * face_points, axis=2)
        
        # projected_force: (F, 3, 3)
        projected_forces = forces_raw - dot_fp[:, :, np.newaxis] * face_points
        
        # Weighted forces: (F, 3, 3)
        weighted_forces = projected_forces * weights[:, np.newaxis, :]
        
        # Scatter to points
        forces = np.zeros((n_points, 3))
        np.add.at(forces, faces, weighted_forces)
        
        points = points + alpha * forces
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        return points
