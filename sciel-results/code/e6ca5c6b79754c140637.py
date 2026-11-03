"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _angular_distance_matrix(P):
    """Compute angular distances between all pairs of points on the sphere."""
    # P is (N, 3)
    dots = P @ P.T
    # Clip to [-1, 1] for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    return np.arccos(dots)


def _covering_radius_from_points(P):
    """Compute the covering radius given a set of points on the sphere.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest chosen point. This is equivalent to half the maximum edge length
    of the convex hull's facets (for points on a sphere, the covering radius is
    determined by the largest empty spherical cap, which corresponds to the circumradius
    of the largest facet of the convex hull).
    
    For a set of points on the unit sphere, the covering radius can be computed as:
    - Find the convex hull of the points
    - For each facet (triangle) of the hull, compute the circumradius (the angular 
      distance from the facet's circumcenter to its vertices)
    - The covering radius is the maximum of these circumradii
    
    However, a simpler approach: the covering radius is half the maximum distance
    between any two points that are "adjacent" in the sense that no other point 
    is closer to the midpoint. Actually, the most robust way is:
    
    The covering radius equals the maximum over all points y on the sphere of 
    min_i angular_dist(y, p_i). This is achieved at the circumcenters of the 
    Delaunay triangulation on the sphere, which correspond to the facets of the 
    convex hull.
    
    For a triangle with vertices a, b, c on the unit sphere, the circumradius (angular)
    can be computed. The circumcenter of the triangle (on the sphere) is the point
    equidistant from a, b, c. The angular distance from this circumcenter to any vertex
    is the circumradius.
    
    Actually, for the purpose of this problem, the covering radius is the maximum 
    circumradius of the facets of the convex hull. Each facet is a triangle.
    For a triangle with side lengths (angular) a, b, c, the circumradius R satisfies:
    R = a / (2 * sin(A)) where A is the angle opposite to side a.
    
    Alternatively, we can compute it directly. Given three unit vectors u, v, w,
    the circumradius of the spherical triangle they form is the angular distance 
    from the circumcenter to any vertex. The circumcenter can be found as the 
    intersection of the perpendicular bisectors, or more simply, it's the point
    on the sphere that maximizes the minimum distance to the three vertices... 
    no, it's the point equidistant from all three.
    
    A cleaner approach: the covering radius is the largest value r such that there 
    exists a point on the sphere at distance r from all three vertices of some facet.
    This is the circumradius of the spherical triangle.
    
    For three unit vectors a, b, c, the circumradius R of the spherical triangle 
    can be computed as follows:
    - The circumcenter (on the sphere) is proportional to a cross b + b cross c + c cross a
      (normalized). Actually, that's not quite right for spherical geometry.
    
    Let me use a different, more reliable method: 
    The covering radius is half the maximum edge length of the nearest-neighbor graph?
    No, that's not correct either.
    
    The most reliable method: the covering radius is the maximum over all facets 
    (triangles) of the convex hull of the circumradius of that triangle.
    
    For a triangle with vertices A, B, C (unit vectors), the circumradius R is given by:
    R = arcsin(|A cross B| * |B cross C| * |C cross A| / (2 * |A cross B cross C|))
    Wait, that's not right either.
    
    Let me think again. For a planar triangle with side lengths a, b, c, the circumradius
    is R = abc / (4 * area). For a spherical triangle, the formula is different.
    
    Actually, the simplest robust method: for each facet (triangle) of the convex hull,
    find the point on the sphere that is equidistant from all three vertices. This is
    the circumcenter. The angular distance from this circumcenter to any vertex is the
    circumradius.
    
    To find the circumcenter: it's the intersection of the sphere with the line that
    is perpendicular to the plane of the triangle and passes through the origin? No,
    that's not right for spherical geometry.
    
    Actually, for three points on a sphere, the circumcenter (on the sphere) is the
    point that lies on the great circle perpendicular to the chord connecting... 
    this is getting complicated.
    
    Let me use a simpler approach that's known to work: 
    The covering radius is the maximum over all pairs of points that are adjacent 
    in the convex hull of half their angular distance. No, that's not right either.
    
    OK, let me just use the definition directly with a fine grid, but that's too slow.
    
    Actually, the correct and efficient method: the covering radius is the maximum 
    circumradius of the facets. For a triangle with vertices a, b, c (unit vectors),
    the circumradius R satisfies:
    sin(R) = |a cross b| * |b cross c| * |c cross a| / (2 * |a cross b cross c|)
    Hmm, I'm not confident in this formula.
    
    Let me use a different approach: for each facet, compute the circumradius by 
    solving for the point equidistant from all three vertices. The circumcenter 
    (in 3D, on the sphere) is proportional to the cross product of the edges... 
    Actually, the circumcenter of a spherical triangle with vertices a, b, c is 
    given by the normalized vector: (a cross b) * |c| + (b cross c) * |a| + (c cross a) * |b|
    Since all are unit vectors, this is just (a cross b) + (b cross c) + (c cross a).
    Wait, that's the formula for the circumcenter of a planar triangle, not spherical.
    
    For spherical geometry, the circumcenter is the point p on the sphere such that
    p · a = p · b = p · c. This means p is orthogonal to (a - b) and (b - c), so
    p is proportional to (a - b) cross (b - c).
    
    Let me verify: if p = (a - b) cross (b - c), then p · a = p · b? 
    p · a = [(a-b) cross (b-c)] · a = (a-b) · [(b-c) cross a] (using scalar triple product)
    p · b = [(a-b) cross (b-c)] · b = (a-b) · [(b-c) cross b]
    These are not obviously equal.
    
    Actually, the condition p · a = p · b = p · c means p is orthogonal to (a-b) and (a-c).
    So p is proportional to (a-b) cross (a-c).
    
    Let me use this: circumcenter_direction = (a - b) cross (a - c)
    Then circumcenter = circumcenter_direction / |circumcenter_direction|
    And the circumradius is arccos(circumcenter · a) (or · b or · c, they should be equal).
    
    Let me test this with an equilateral triangle: a = (1,0,0), b = (0,1,0), c = (0,0,1).
    a - b = (1,-1,0), a - c = (1,0,-1)
    (a-b) cross (a-c) = |i  j  k; 1 -1 0; 1 0 -1| = i(1-0) - j(-1-0) + k(0-(-1)) = (1, 1, 1)
    circumcenter = (1,1,1)/sqrt(3)
    circumcenter · a = 1/sqrt(3)
    circumradius = arccos(1/sqrt(3)) ≈ 54.74 degrees
    
    Is this correct? The three points are at 90 degrees from each other. The circumcenter
    should be equidistant from all three. arccos(1/sqrt(3)) ≈ 54.74 degrees. 
    And indeed, the point (1,1,1)/sqrt(3) is at arccos(1/sqrt(3)) from each of (1,0,0), 
    (0,1,0), (0,0,1). This seems correct.
    
    So the algorithm is:
    1. Compute the convex hull of the N points.
    2. For each triangular facet, compute the circumradius using the formula above.
    3. The covering radius is the maximum circumradius over all facets.
    
    This should be efficient and accurate.
    """
    hull = ConvexHull(P)
    max_circumradius = 0.0
    for simplex in hull.simplices:
        a = P[simplex[0]]
        b = P[simplex[1]]
        c = P[simplex[2]]
        # Circumcenter direction
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm = np.linalg.norm(cross)
        if norm < 1e-12:
            # Degenerate triangle, skip
            continue
        circumcenter = cross / norm
        # Angular distance from circumcenter to any vertex
        dot = np.clip(np.dot(circumcenter, a), -1.0, 1.0)
        circumradius = np.arccos(dot)
        if circumradius > max_circumradius:
            max_circumradius = circumradius
    return np.degrees(max_circumradius)


def _objective_and_gradient(x_flat, beta):
    """Compute the smooth approximation of the covering radius and its gradient.
    
    We use a soft-max formulation: instead of max_i f_i(x), we use 
    (1/beta) * log(sum_i exp(beta * f_i(x)))
    where f_i(x) are the circumradii of the facets.
    
    However, the facet structure changes as points move, which makes this tricky.
    A simpler smooth surrogate: use the maximum pairwise angular distance divided by 2,
    but that's not the covering radius.
    
    Actually, a better smooth surrogate: the covering radius is the maximum over 
    all points y of min_i angular_dist(y, p_i). We can approximate this by considering
    a set of sample points y_j and computing max_j min_i angular_dist(y_j, p_i),
    then smooth this with a soft-max.
    
    But this requires many sample points and is slow.
    
    Alternative: use the fact that the covering radius is determined by the facets.
    We can track the facets and use their circumradii. But the facet structure is 
    discrete, so we need to handle this carefully.
    
    For now, let's use a simpler approach: gradient descent on the points, using 
    the gradient of the maximum circumradius. The gradient of the maximum of a set 
    of smooth functions is the gradient of the function that achieves the maximum.
    """
    P = x_flat.reshape(-1, 3)
    # Normalize
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    P = P / norms
    
    # Compute covering radius and identify the worst facet
    hull = ConvexHull(P)
    max_circumradius = 0.0
    worst_simplex = None
    for simplex in hull.simplices:
        a = P[simplex[0]]
        b = P[simplex[1]]
        c = P[simplex[2]]
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm = np.linalg.norm(cross)
        if norm < 1e-12:
            continue
        circumcenter = cross / norm
        dot = np.clip(np.dot(circumcenter, a), -1.0, 1.0)
        circumradius = np.arccos(dot)
        if circumradius > max_circumradius:
            max_circumradius = circumradius
            worst_simplex = simplex
    
    if worst_simplex is None:
        return 0.0, np.zeros_like(x_flat)
    
    # Gradient of the circumradius with respect to the points
    # This is complex; for now, use a finite difference or a simpler heuristic
    # For the purpose of this implementation, we'll use a simple gradient estimate
    grad = np.zeros_like(x_flat)
    # The gradient of arccos(p · a) with respect to a is -p / sqrt(1 - (p·a)^2)
    # where p is the circumcenter and a is the vertex
    a = P[worst_simplex[0]]
    b = P[worst_simplex[1]]
    c = P[worst_simplex[2]]
    v1 = a - b
    v2 = a - c
    cross = np.cross(v1, v2)
    norm = np.linalg.norm(cross)
    circumcenter = cross / norm
    
    dot_a = np.clip(np.dot(circumcenter, a), -1.0, 1.0)
    sin_theta = np.sqrt(1.0 - dot_a * dot_a)
    if sin_theta > 1e-12:
        # Gradient of arccos(circumcenter · a) with respect to a
        # d/dx arccos(f(x)) = -f'(x) / sqrt(1 - f(x)^2)
        # Here f(x) = circumcenter · a, but circumcenter also depends on a, b, c
        # This is getting very complex. For simplicity, use a heuristic:
        # Move the worst facet's vertices towards the circumcenter
        for idx in worst_simplex:
            # Move point towards the circumcenter to reduce the circumradius
            grad[idx * 3:idx * 3 + 3] -= circumcenter
    
    return np.degrees(max_circumradius), grad


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial points: jittered Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add small jitter to break symmetry
    np.random.seed(42)
    P += np.random.normal(0, 0.01, P.shape)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    P = P / norms
    
    current_radius = _covering_radius_from_points(P)
    best_P = P.copy()
    best_radius = current_radius
    
    # Optimization loop
    lr = 0.01
    for iteration in range(500):
        if time.time() > deadline - 5.0:
            break
        
        # Compute gradient using finite differences for the worst facet
        P_flat = P.flatten()
        hull = ConvexHull(P)
        max_circumradius = 0.0
        worst_simplex = None
        for simplex in hull.simplices:
            a = P[simplex[0]]
            b = P[simplex[1]]
            c = P[simplex[2]]
            v1 = a - b
            v2 = a - c
            cross = np.cross(v1, v2)
            norm = np.linalg.norm(cross)
            if norm < 1e-12:
                continue
            circumcenter = cross / norm
            dot = np.clip(np.dot(circumcenter, a), -1.0, 1.0)
            circumradius = np.arccos(dot)
            if circumradius > max_circumradius:
                max_circumradius = circumradius
                worst_simplex = simplex
        
        if worst_simplex is None:
            break
        
        # Move the worst facet's vertices towards the circumcenter
        a = P[worst_simplex[0]]
        b = P[worst_simplex[1]]
        c = P[worst_simplex[2]]
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm = np.linalg.norm(cross)
        circumcenter = cross / norm
        
        # Update points
        for idx in worst_simplex:
            P[idx] += lr * circumcenter
            # Renormalize
            P[idx] /= np.linalg.norm(P[idx])
        
        # Compute new radius
        new_radius = _covering_radius_from_points(P)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_P = P.copy()
        
        # Decay learning rate
        lr *= 0.99
    
    return best_P
# EVOLVE-BLOCK-END
