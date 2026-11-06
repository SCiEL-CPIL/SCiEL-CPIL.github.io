"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
from math import gcd
from math import isqrt

# EVOLVE-BLOCK-START

def _is_squarefree(n):
    if n < 1:
        return False
    if n == 1:
        return True
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True

def _check_no_three_collinear(points):
    """Check that no three points are collinear using exact rational arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Points are (x_i, y_i*sqrt(D)), (x_j, y_j*sqrt(D)), (x_k, y_k*sqrt(D))
                # Collinearity condition: (x_j - x_i)(y_k*sqrt(D) - y_i*sqrt(D)) == (x_k - x_i)(y_j*sqrt(D) - y_i*sqrt(D))
                # This simplifies to: (x_j - x_i)(y_k - y_i) == (x_k - x_i)(y_j - y_i)
                x_i, y_i = points[i]
                x_j, y_j = points[j]
                x_k, y_k = points[k]
                if (x_j - x_i) * (y_k - y_i) == (x_k - x_i) * (y_j - y_i):
                    return False
    return True

def _check_no_four_concyclic(points, D):
    """Check that no four points are concyclic using exact rational arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    # Four points (x_a, y_a*sqrt(D)), etc. are concyclic iff the determinant is zero
                    # Circle equation: x^2 + y^2* D + A*x + B*y*sqrt(D) + C = 0
                    # For point (x_i, y_i*sqrt(D)): x_i^2 + D*y_i^2 + A*x_i + B*y_i*sqrt(D) + C = 0
                    # This must hold for 4 points. We can eliminate B by considering the system.
                    # Actually, let's use the standard concyclicity test.
                    # Four points (x1,y1), (x2,y2), (x3,y3), (x4,y4) in the plane are concyclic iff:
                    # |x1^2+y1^2  x1  y1  1|
                    # |x2^2+y2^2  x2  y2  1| = 0
                    # |x3^2+y3^2  x3  y3  1|
                    # |x4^2+y4^2  x4  y4  1|
                    # But here our points are (x_i, y_i*sqrt(D)) in the Euclidean plane.
                    # So the squared distance from origin is x_i^2 + D*y_i^2.
                    # The concyclicity condition for points (x_i, y_i*sqrt(D)) is:
                    # The determinant of the matrix with rows [x_i^2 + D*y_i^2, x_i, y_i*sqrt(D), 1] is zero.
                    # However, since y_i*sqrt(D) involves sqrt(D), we need to handle this carefully.
                    # Let's write the circle equation as: x^2 + D*y^2 + A*x + B*y*sqrt(D) + C = 0
                    # For each point i: x_i^2 + D*y_i^2 + A*x_i + B*y_i*sqrt(D) + C = 0
                    # This is a linear system in A, B*sqrt(D), C. Let's denote B' = B*sqrt(D).
                    # Then: x_i^2 + D*y_i^2 + A*x_i + B'*y_i + C = 0
                    # We have 4 equations and 3 unknowns (A, B', C), so the system is consistent iff the 4x4 determinant is zero.
                    # The matrix is:
                    # [x_i^2 + D*y_i^2, x_i, y_i, 1] for i in {i,j,k,l}
                    # Wait, this is not quite right because B' is a single coefficient.
                    # Let me reconsider. The general circle equation in the (x, y*sqrt(D)) plane is:
                    # x^2 + (y*sqrt(D))^2 + A*x + B*(y*sqrt(D)) + C = 0
                    # x^2 + D*y^2 + A*x + B*sqrt(D)*y + C = 0
                    # Let C' = C, A' = A, B' = B*sqrt(D). Then:
                    # x^2 + D*y^2 + A'*x + B'*y + C' = 0
                    # For each point (x_i, y_i): x_i^2 + D*y_i^2 + A'*x_i + B'*y_i + C' = 0
                    # This is linear in A', B', C'. For 4 points, the system has a non-trivial solution iff the determinant is zero.
                    # The determinant is of the matrix:
                    # [x_i^2 + D*y_i^2, x_i, y_i, 1]
                    # [x_j^2 + D*y_j^2, x_j, y_j, 1]
                    # [x_k^2 + D*y_k^2, x_k, y_k, 1]
                    # [x_l^2 + D*y_l^2, x_l, y_l, 1]
                    # We need this determinant to be non-zero for all quadruples.
                    # Compute the determinant using exact rational arithmetic.
                    a1, b1 = points[i]
                    a2, b2 = points[j]
                    a3, b3 = points[k]
                    a4, b4 = points[l]
                    # Compute x^2 + D*y^2 for each point
                    v1 = a1 * a1 + D * b1 * b1
                    v2 = a2 * a2 + D * b2 * b2
                    v3 = a3 * a3 + D * b3 * b3
                    v4 = a4 * a4 + D * b4 * b4
                    # Compute the 4x4 determinant
                    # M = [[v1, a1, b1, 1], [v2, a2, b2, 1], [v3, a3, b3, 1], [v4, a4, b4, 1]]
                    # det(M) = v1*(a2*(b3*1 - 1*b4) - a3*(b2*1 - 1*b4) + a4*(b2*1 - b3*1))
                    #         - a1*(v2*(b3*1 - 1*b4) - v3*(b2*1 - 1*b4) + v4*(b2*1 - b3*1))
                    #         + b1*(v2*(a3*1 - 1*a4) - v3*(a2*1 - 1*a4) + v4*(a2*1 - a3*1))
                    #         - 1*(v2*(a3*b4 - b3*a4) - v3*(a2*b4 - b2*a4) + v4*(a2*b3 - b2*a3))
                    # Let's compute this step by step.
                    term1 = v1 * (a2 * (b3 - b4) - a3 * (b2 - b4) + a4 * (b2 - b3))
                    term2 = -a1 * (v2 * (b3 - b4) - v3 * (b2 - b4) + v4 * (b2 - b3))
                    term3 = b1 * (v2 * (a3 - a4) - v3 * (a2 - a4) + v4 * (a2 - a3))
                    term4 = -(v2 * (a3 * b4 - b3 * a4) - v3 * (a2 * b4 - b2 * a4) + v4 * (a2 * b3 - b2 * a3))
                    det = term1 + term2 + term3 + term4
                    if det == 0:
                        return False
    return True

def _is_perfect_square(n):
    """Check if n is a perfect square using integer arithmetic."""
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _check_all_pairwise_distances_rational(points, D):
    """Check that all pairwise distances are rational."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x_i, y_i = points[i]
            x_j, y_j = points[j]
            dx = x_i - x_j
            dy = y_i - y_j
            dist_sq = dx * dx + D * dy * dy
            if dist_sq <= 0:
                return False
            num = dist_sq.numerator
            den = dist_sq.denominator
            if not _is_perfect_square(num):
                return False
            if not _is_perfect_square(den):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 135s deadline, 15s margin
    
    def format_result(points, D):
        result_points = []
        for x, y in points:
            result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
        return {'D': D, 'points': result_points}
    
    # Try multiple D values and starting configurations
    best_result = None
    best_count = 0
    
    # Known configurations with D=1
    configs = [
        (1, [(Fraction(0), Fraction(0)),
             (Fraction(-56), Fraction(33)),
             (Fraction(-16), Fraction(63)),
             (Fraction(0), Fraction(33)),
             (Fraction(16), Fraction(63))]),
        (1, [(Fraction(0), Fraction(0)),
             (Fraction(3), Fraction(0)),
             (Fraction(0), Fraction(4)),
             (Fraction(3), Fraction(4))]),
        (1, [(Fraction(0), Fraction(0)),
             (Fraction(1), Fraction(0)),
             (Fraction(0), Fraction(1)),
             (Fraction(1), Fraction(1)),
             (Fraction(2), Fraction(0))]),
    ]
    
    # Try D=2,3,5,7,11,13,17,19,23,29,31 for potential improvements
    squarefree_Ds = [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31]
    
    for D in squarefree_Ds:
        if time.time() > deadline - 5:
            break
        
        # Generate base configurations for this D
        # For D>1, we need points where distances work out
        # Try simple configurations
        if D == 1:
            for d_val, pts in configs:
                if d_val != D:
                    continue
                if not _check_no_three_collinear(pts):
                    continue
                if not _check_no_four_concyclic(pts, D):
                    continue
                if not _check_all_pairwise_distances_rational(pts, D):
                    continue
                if len(pts) > best_count:
                    best_count = len(pts)
                    best_result = format_result(pts, D)
        else:
            # For D>1, try to find configurations
            # Simple approach: try points on a grid scaled appropriately
            # Point (x, y*sqrt(D)) - distance squared = (x1-x2)^2 + D*(y1-y2)^2
            # For this to be rational square, need specific relationships
            # Try small integer coordinates
            base_pts = [(Fraction(0), Fraction(0))]
            # Add points that are at rational distances from origin
            # (a, b*sqrt(D)) has distance sqrt(a^2 + D*b^2) - need this rational
            # So a^2 + D*b^2 must be a perfect square
            # For D=2: a^2 + 2b^2 = c^2 -> (a+b*sqrt(2))(a-b*sqrt(2))=c^2
            # Try to find some points
            found_pts = [(Fraction(0), Fraction(0))]
            for a in range(1, 50):
                for b in range(1, 50):
                    dist_sq = a*a + D*b*b
                    if _is_perfect_square(dist_sq):
                        pt = (Fraction(a), Fraction(b))
                        if all(pt[0] != p[0] or pt[1] != p[1] for p in found_pts):
                            found_pts.append(pt)
                        if len(found_pts) >= 3:
                            break
                if len(found_pts) >= 3:
                    break
            
            if len(found_pts) >= 3:
                if _check_no_three_collinear(found_pts) and \
                   _check_no_four_concyclic(found_pts, D) and \
                   _check_all_pairwise_distances_rational(found_pts, D):
                    if len(found_pts) > best_count:
                        best_count = len(found_pts)
                        best_result = format_result(found_pts, D)
    
    if best_result is None:
        # Fallback: return the 5-point config
        base_points = [(Fraction(0), Fraction(0)),
                       (Fraction(-56), Fraction(33)),
                       (Fraction(-16), Fraction(63)),
                       (Fraction(0), Fraction(33)),
                       (Fraction(16), Fraction(63))]
        return format_result(base_points, 1)
    
    best_points = None
    best_D = best_result['D']
    # Reconstruct points from best_result
    for x_pair, y_pair in best_result['points']:
        pass
    
    # Now try to extend the best configuration
    # Parse back the points
    best_points = [(Fraction(xp[0], xp[1]), Fraction(yp[0], yp[1])) for xp, yp in zip(
        [p[0] for p in best_result['points']], [p[1] for p in best_result['points']])
    ]
    
    D = best_D
    
    def try_add_point(current_points, D, candidate):
        """Try adding a candidate point to the current set, with fast rejection."""
        # Fast check: new point must not be collinear with any pair
        for i in range(len(current_points)):
            for j in range(i + 1, len(current_points)):
                xi, yi = current_points[i]
                xj, yj = current_points[j]
                xc, yc = candidate
                if (xj - xi) * (yc - yi) == (xc - xi) * (yj - yi):
                    return None
        # Check distances from candidate to all existing points
        for i in range(len(current_points)):
            xi, yi = current_points[i]
            xc, yc = candidate
            dx = xi - xc
            dy = yi - yc
            dist_sq = dx * dx + D * dy * dy
            if dist_sq <= 0:
                return None
            if not _is_perfect_square(dist_sq.numerator):
                return None
            if not _is_perfect_square(dist_sq.denominator):
                return None
        # Full checks
        new_points = current_points + [candidate]
        if not _check_no_three_collinear(new_points):
            return None
        if not _check_no_four_concyclic(new_points, D):
            return None
        if not _check_all_pairwise_distances_rational(new_points, D):
            return None
        return new_points
    
    # Generate candidates more efficiently
    # Use a wider range but with smarter ordering
    # Prioritize points that are "compatible" with existing ones
    
    max_coord = 500
    candidates = []
    
    # First, generate candidates near existing points (more likely to work)
    for cx in range(-max_coord, max_coord + 1):
        for cy in range(-max_coord, max_coord + 1):
            candidates.append((Fraction(cx), Fraction(cy)))
    
    # Sort by distance to centroid of existing points
    if best_points:
        cx_avg = sum(p[0] for p in best_points) / len(best_points)
        cy_avg = sum(p[1] for p in best_points) / len(best_points)
        candidates.sort(key=lambda c: (c[0] - cx_avg)**2 + (c[1] - cy_avg)**2)
    
    # Try to add points
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for candidate in candidates:
            if time.time() > deadline - 5:
                break
            if any(candidate[0] == p[0] and candidate[1] == p[1] for p in best_points):
                continue
            new_points = try_add_point(best_points, D, candidate)
            if new_points is not None:
                best_points = new_points
                improved = True
                break
    
    return format_result(best_points, D)

# EVOLVE-BLOCK-END
