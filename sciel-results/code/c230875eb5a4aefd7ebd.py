def _polish(points, max_iters=2000, deadline=None):
    """
    High-precision polish pass.
    """
    points = points.copy()
    best_radius = _covering_radius_and_worst_facets(points)[0]
    step = 0.005
    top_k = 5
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 3.0:
            break
        
        radius, top_facets = _covering_radius_and_worst_facets(points, top_k=top_k)
        
        if not top_facets:
            break
        
        if radius < best_radius - 1e-14:
            best_radius = radius
            step *= 0.98
        else:
            step *= 0.5
        
        if step < 1e-12:
            break
        
        grad = np.zeros_like(points)
        for facet, circ in top_facets:
            for idx in facet:
                direction = circ - points[idx]
                grad[idx] += direction
        
        for i in range(points.shape[0]):
            grad[i] = grad[i] - np.dot(grad[i], points[i]) * points[i]
        
        norms = np.linalg.norm(grad, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        grad = grad / norms
        
        # Single step for polish
        new_pts = points + step * grad
        new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        new_radius, _ = _covering_radius_and_worst_facets(new_pts)
        
        if new_radius < best_radius:
            points = new_pts
            best_radius = new_radius
        else:
            # If no improvement, try smaller step
            step *= 0.5
            if step < 1e-12:
                break

    return points
