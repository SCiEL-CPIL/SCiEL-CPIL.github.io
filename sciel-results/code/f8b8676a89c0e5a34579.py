<<<<<<< SEARCH
def _project_to_triangle(points):
    """Project points onto the triangle (clip to boundary if outside)."""
    for i in range(N):
        px, py = points[i]
        # Compute barycentric coordinates
        l1 = 1 - px - py / np.sqrt(3)
        l2 = px - py / (2 * np.sqrt(3))
        l3 = py / np.sqrt(3)
        
        # If any barycentric coordinate is negative, project to nearest edge
        if l1 < 0 or l2 < 0 or l3 < 0:
            # Simple projection: find closest point on triangle boundary
            # For simplicity, just clamp and re-normalize
            l = np.array([l1, l2, l3])
            l = np.maximum(l, 0)
            l_sum = l.sum()
            if l_sum > 0:
                l = l / l_sum
            else:
                l = np.array([1/3, 1/3, 1/3])
            points[i] = l[0] * V[0] + l[1] * V[1] + l[2] * V[2]
    return points
=======
def _project_to_triangle(points):
    """Project points onto the triangle (clip to boundary if outside)."""
    # Vertices
    A = V[0]
    B = V[1]
    C = V[2]
    
    def _closest_point_on_segment(p, a, b):
        """Find the closest point to p on the line segment ab."""
        ab = b - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq == 0:
            return a
        t = np.dot(p - a, ab) / ab_len_sq
        t = np.clip(t, 0, 1)
        return a + t * ab

    for i in range(N):
        p = points[i]
        # Check if p is inside using barycentric coordinates
        l1 = 1 - p[0] - p[1] / np.sqrt(3)
        l2 = p[0] - p[1] / (2 * np.sqrt(3))
        l3 = p[1] / np.sqrt(3)
        
        if l1 < -1e-9 or l2 < -1e-9 or l3 < -1e-9:
            # Point is outside. Find closest point on the boundary.
            p_ab = _closest_point_on_segment(p, A, B)
            p_bc = _closest_point_on_segment(p, B, C)
            p_ca = _closest_point_on_segment(p, C, A)
            
            d_ab = np.linalg.norm(p - p_ab)
            d_bc = np.linalg.norm(p - p_bc)
            d_ca = np.linalg.norm(p - p_ca)
            
            min_d = min(d_ab, d_bc, d_ca)
            if min_d == d_ab:
                points[i] = p_ab
            elif min_d == d_bc:
                points[i] = p_bc
            else:
                points[i] = p_ca
    return points
>>>>>>> REPLACE

