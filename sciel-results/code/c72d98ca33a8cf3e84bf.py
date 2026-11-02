"""Rewritten program for Heilbronn triangles using SLP in barycentric coordinates."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import itertools

def construct():
    """
    Place 11 points in the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    Maximize the minimum area of any triangle formed by three points.
    Uses SLP in barycentric coordinates.
    """
    sqrt3 = np.sqrt(3)
    h = sqrt3 / 2  # height of the triangle
    
    # Area of the equilateral triangle
    area_unit = h * 1.0  # base=1, height=sqrt(3)/2
    
    def bary_to_cart(a, b, c):
        """Convert barycentric coords (a,b,c) to Cartesian (x,y)."""
        x = b + 0.5 * c
        y = h * c
        return x, y
    
    def cart_to_bary(x, y):
        """Convert Cartesian (x,y) to barycentric coords (a,b,c)."""
        c = y / h
        b = x - 0.5 * c
        a = 1.0 - b - c
        return a, b, c
    
    # Generate all triples
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)
    
    def compute_min_area_cart(pts):
        """Compute minimum triangle area from Cartesian points."""
        min_a = float('inf')
        for i, j, k in triples:
            pi, pj, pk = pts[i], pts[j], pts[k]
            area = 0.5 * abs((pj[0] - pi[0]) * (pk[1] - pi[1]) - (pk[0] - pi[0]) * (pj[1] - pi[1]))
            if area < min_a:
                min_a = area
        return min_a
    
    # Initial configuration: use a symmetric grid-like arrangement in barycentric coords
    # Vertices, edge midpoints, and interior points
    initial_bary = np.array([
        [0.85, 0.10, 0.05],   # near A
        [0.05, 0.85, 0.10],   # near B
        [0.05, 0.10, 0.85],   # near C
        [0.50, 0.25, 0.25],   # centroid
        [0.25, 0.50, 0.25],   # AB edge area
        [0.25, 0.25, 0.50],   # AC edge area
        [0.50, 0.05, 0.45],   # BC edge area
        [0.40, 0.30, 0.30],   # interior
        [0.30, 0.40, 0.30],   # interior
        [0.30, 0.30, 0.40],   # interior
        [0.35, 0.35, 0.30],   # interior
    ])
    
    # Ensure valid barycentric coords
    for i in range(N):
        if initial_bary[i, 0] < 0: initial_bary[i, 0] = 0.01
        if initial_bary[i, 1] < 0: initial_bary[i, 1] = 0.01
        if initial_bary[i, 2] < 0: initial_bary[i, 2] = 0.01
        s = initial_bary[i].sum()
        initial_bary[i] /= s
    
    best_bary = initial_bary.copy()
    best_pts = np.array([bary_to_cart(a, b, c) for a, b, c in best_bary])
    best_min = compute_min_area_cart(best_pts)
    
    # SLP iterations
    max_iters = 100
    current_bary = best_bary.copy()
    
    for iteration in range(max_iters):
        current_pts = np.array([bary_to_cart(a, b, c) for a, b, c in current_bary])
        
        # Compute signed areas and signs at current configuration
        # Area of triangle with barycentric points can be computed via determinant
        # For points with barycentric coords, the area ratio is the absolute value of the determinant
        # of the matrix of barycentric coordinates
        
        signed_areas = np.zeros(n_triples)
        signs = np.ones(n_triples)
        
        for idx, (i, j, k) in enumerate(triples):
            # The signed area of triangle formed by three points with barycentric coords
            # is proportional to the determinant of the 3x3 matrix of their barycentric coords
            M = np.array([current_bary[i], current_bary[j], current_bary[k]])
            det = np.linalg.det(M)
            signed_areas[idx] = det
            if det < 0:
                signs[idx] = -1
        
        # Build LP in barycentric space
        # Variables: a_0,b_0,c_0, a_1,b_1,c_1, ..., a_10,b_10,c_10, t
        # 3*N + 1 = 34 variables
        n_vars = 3 * N + 1
        
        c_vec = np.zeros(n_vars)
        c_vec[-1] = -1  # maximize t
        
        A_ub = []
        b_ub = []
        
        # Barycentric constraints: a_i >= 0, b_i >= 0, c_i >= 0
        # These are handled by bounds, but let's add a_i + b_i + c_i = 1
        # We'll use equality constraints for sum=1
        
        A_eq = []
        b_eq = []
        
        for i in range(N):
            row = np.zeros(n_vars)
            row[3*i] = 1
            row[3*i+1] = 1
            row[3*i+2] = 1
            A_eq.append(row)
            b_eq.append(1.0)
        
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        
        # Triangle area constraints (linearized)
        # The area of triangle (i,j,k) in barycentric coords is |det(M_ijk)| * area_unit
        # where M_ijk is the 3x3 matrix of barycentric coordinates
        # 
        # det(M) = a_i(b_j c_k - c_j b_k) - b_i(a_j c_k - c_j a_k) + c_i(a_j b_k - b_j a_k)
        #
        # Linearize around current point:
        # det_lin = det_cur + sum over all barycentric coords of (partial_det/partial_coord) * delta_coord
        #
        # Constraint: sign * det_lin >= t / area_unit
        # => sign * det_cur + sign * sum(grad * delta) - t/area_unit >= 0
        # => -sign * det_cur - sign * sum(grad * delta) + t/area_unit <= 0
        
        t_scale = 1.0 / area_unit  # convert t (area) to det scale
        
        for idx, (i, j, k) in enumerate(triples):
            det_cur = signed_areas[idx]
            sgn = signs[idx]
            
            row = np.zeros(n_vars)
            b_val = -sgn * det_cur
            
            # Compute gradient of det with respect to each barycentric coordinate
            # det = a_i*b_j*c_k - a_i*c_j*b_k - b_i*a_j*c_k + b_i*c_j*a_k + c_i*a_j*b_k - c_i*b_j*a_k
            
            # d(det)/d(a_i) = b_j*c_k - c_j*b_k
            row[3*i] = -sgn * (current_bary[j,1]*current_bary[k,2] - current_bary[j,2]*current_bary[k,1])
            # d(det)/d(b_i) = -(a_j*c_k - c_j*a_k) = -a_j*c_k + c_j*a_k
            row[3*i+1] = -sgn * (-current_bary[j,0]*current_bary[k,2] + current_bary[j,2]*current_bary[k,0])
            # d(det)/d(c_i) = a_j*b_k - b_j*a_k
            row[3*i+2] = -sgn * (current_bary[j,0]*current_bary[k,1] - current_bary[j,1]*current_bary[k,0])
            
            # d(det)/d(a_j) = -(b_i*c_k - c_i*b_k) = -b_i*c_k + c_i*b_k
            row[3*j] = -sgn * (-current_bary[i,1]*current_bary[k,2] + current_bary[i,2]*current_bary[k,1])
            # d(det)/d(b_j) = a_i*c_k - c_i*a_k
            row[3*j+1] = -sgn * (current_bary[i,0]*current_bary[k,2] - current_bary[i,2]*current_bary[k,0])
            # d(det)/d(c_j) = -(a_i*b_k - b_i*a_k) = -a_i*b_k + b_i*a_k
            row[3*j+2] = -sgn * (-current_bary[i,0]*current_bary[k,1] + current_bary[i,1]*current_bary[k,0])
            
            # d(det)/d(a_k) = b_i*c_j - c_i*b_j
            row[3*k] = -sgn * (current_bary[i,1]*current_bary[j,2] - current_bary[i,2]*current_bary[j,1])
            # d(det)/d(b_k) = -(a_i*c_j - c_i*a_j) = -a_i*c_j + c_i*a_j
            row[3*k+1] = -sgn * (-current_bary[i,0]*current_bary[j,2] + current_bary[i,2]*current_bary[j,0])
            # d(det)/d(c_k) = a_i*b_j - b_i*a_j
            row[3*k+2] = -sgn * (current_bary[i,0]*current_bary[j,1] - current_bary[i,1]*current_bary[j,0])
            
            # Coefficient of t
            row[-1] = t_scale
            
            A_ub.append(row)
            b_ub.append(b_val)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds: barycentric coords in [0, 1], t in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 1))
        bounds.append((0, 0.5))
        
        # Solve LP
        try:
            result = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                           bounds=bounds, method='highs')
            if result.success:
                new_bary = result.x[:3*N].reshape(N, 3)
                new_t = result.x[-1]
                
                # Validate
                valid = True
                for i in range(N):
                    if new_bary[i,0] < -1e-9 or new_bary[i,1] < -1e-9 or new_bary[i,2] < -1e-9:
                        valid = False
                        break
                    s = new_bary[i].sum()
                    if abs(s - 1.0) > 1e-6:
                        valid = False
                        break
                
                if valid:
                    # Clip small negatives
                    new_bary = np.maximum(new_bary, 0)
                    new_bary /= new_bary.sum(axis=1, keepdims=True)
                    
                    new_pts = np.array([bary_to_cart(a, b, c) for a, b, c in new_bary])
                    new_min = compute_min_area_cart(new_pts)
                    
                    if new_min > best_min - 1e-12:
                        best_bary = new_bary.copy()
                        best_pts = new_pts.copy()
                        best_min = new_min
                        current_bary = new_bary
                    else:
                        break
                else:
                    break
            else:
                break
        except Exception:
            break
    
    # Local refinement using scipy.optimize.minimize
    from scipy.optimize import minimize
    
    def objective(bary_flat):
        bary = bary_flat.reshape(N, 3)
        # Project to valid barycentric
        bary = np.maximum(bary, 0)
        bary /= bary.sum(axis=1, keepdims=True)
        pts = np.array([bary_to_cart(a, b, c) for a, b, c in bary])
        min_a = float('inf')
        for i, j, k in triples:
            pi, pj, pk = pts[i], pts[j], pts[k]
            area = 0.5 * abs((pj[0] - pi[0]) * (pk[1] - pi[1]) - (pk[0] - pi[0]) * (pj[1] - pi[1]))
            if area < min_a:
                min_a = area
        return -min_a
    
    # Try local refinement from current best
    try:
        result = minimize(objective, best_bary.flatten(), method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-12})
        if result.fun < -best_min:
            refined_bary = result.x.reshape(N, 3)
            refined_bary = np.maximum(refined_bary, 0)
            refined_bary /= refined_bary.sum(axis=1, keepdims=True)
            refined_pts = np.array([bary_to_cart(a, b, c) for a, b, c in refined_bary])
            refined_min = -result.fun
            if refined_min > best_min:
                best_bary = refined_bary
                best_pts = refined_pts
                best_min = refined_min
    except Exception:
        pass
    
    return best_pts

# EVOLVE-BLOCK-END
