def min_triangle_area_fast(points):
    n = len(points)
    # Create indices
    i, j, k = np.indices((n, n, n))
    # Mask i < j < k
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    # Calculate areas
    # area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    cross = (p2[:,0]-p1[:,0])*(p3[:,1]-p1[:,1]) - (p3[:,0]-p1[:,0])*(p2[:,1]-p1[:,1])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)
