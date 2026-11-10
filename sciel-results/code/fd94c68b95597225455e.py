<<<<<<< SEARCH
    def check_separation(v1, v2, tol=1e-6):
        """Check if two hexagons are separated using SAT."""
        # Get edge normals for both hexagons
        for verts in [v1, v2]:
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                normal = normal / np.linalg.norm(normal)
                
                # Project all vertices onto normal
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Check if projections are separated
                if max1 < min2 - tol or max2 < min1 - tol:
                    return False  # Not separated (overlap)
        
        return True  # Separated
=======
    def check_separation(v1, v2, tol=1e-6):
        """Check if two hexagons are separated using SAT. Returns True if separated."""
        # Get edge normals for both hexagons
        for verts in [v1, v2]:
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal = normal / norm
                
                # Project all vertices onto normal
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Check if projections are separated (gap > tol)
                if max1 < min2 - tol or max2 < min1 - tol:
                    return True  # Separated along this axis
        
        return False  # Not separated (overlap or touching)
>>>>>>> REPLACE

