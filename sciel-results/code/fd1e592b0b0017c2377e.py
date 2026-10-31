import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin

    sqrt3 = np.sqrt(3)
    h = sqrt3 / 2.0
    
    # Bounds for x and y for each point
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0, 1), (0, h)] * N

    def project_to_triangle(points):
        """
        Projects points to the interior/boundary of the equilateral triangle
        defined by (0,0), (1,0), (0.5, sqrt(3)/2).
        """
        pts = points.copy()
        # Clip to bounding box first
        pts[:, 0] = np.clip(pts[:, 0], 0, 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h)
        
        # Check constraint 1: y <= sqrt(3) * x
        # If violated, project onto y = sqrt(3)*x
        # The line is sqrt(3)x - y = 0. Normal vector n = (sqrt(3), -1).
        # Distance d = (sqrt(3)x - y) / 2.
        # If sqrt(3)x - y < 0, we are "above/left" of the line? 
        # Let's check: Point (0, 0.1). sqrt(3)*0 - 0.1 = -0.1 < 0. Violation.
        # We want to move it towards the triangle interior (down/right).
        # Actually, the interior is where sqrt(3)x - y >= 0.
        # So if val = sqrt(3)*x - y < 0, we need to increase x or decrease y.
        # Closest point on line sqrt(3)x - y = 0 from (x,y) is:
        # x' = x + sqrt(3)/2 * val
        # y' = y - 1/2 * val
        # Wait, let's use the standard projection formula.
        # Line: Ax + By + C = 0. Here A=sqrt(3), B=-1, C=0.
        # d = (Ax+By+C)/sqrt(A^2+B^2) = (sqrt(3)x - y)/2.
        # If d < 0, we are on the wrong side.
        # P' = P - d * (A, B) / sqrt(A^2+B^2)
        # P' = (x,y) - d * (sqrt(3), -1) / 2
        # x' = x - d*sqrt(3)/2
        # y' = y - d*(-1)/2 = y + d/2
        
        val1 = sqrt3 * pts[:, 0] - pts[:, 1]
        mask1 = val1 < 0
        if np.any(mask1):
            d1 = val1[mask1] / 2.0
            pts[mask1, 0] -= d1 * sqrt3
            pts[mask1, 1] += d1
            
        # Check constraint 2: y <= sqrt(3) * (1 - x)
        # Line: sqrt(3)(1-x) - y = 0 => -sqrt(3)x - y + sqrt(3) = 0
        # A = -sqrt(3), B = -1, C = sqrt(3)
        # d = (-sqrt(3)x - y + sqrt(3)) / 2
        # If d < 0, we are outside (above/right of the line).
        val2 = -sqrt3 * pts[:, 0] - pts[:, 1] + sqrt3
        mask2 = val2 < 0
        if np.any(mask2):
            d2 = val2[mask2] / 2.0
            # P' = P - d * (A, B) / 2
            # x' = x - d * (-sqrt(3)) / 2 = x + d*sqrt(3)/2
            # y' = y - d * (-1) / 2 = y + d/2
            pts[mask2, 0] += d2 * sqrt3
            pts[mask2, 1] += d2
            
        # Final safety clip (projections might slightly overshoot due to floating point or corner interactions)
        pts[:, 0] = np.clip(pts[:, 0], 0, 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h)
        
        # One more pass to ensure strict validity if projections interacted badly
        # It's rare but possible. A simple iterative clipping helps.
        for _ in range(5):
            val1 = sqrt3 * pts[:, 0] - pts[:, 1]
            mask1 = val1 < -1e-12
            if np.any(mask1):
                d1 = val1[mask1] / 2.0
                pts[mask1, 0] -= d1 * sqrt3
                pts[mask1, 1] += d1
                
            val2 = -sqrt3 * pts[:, 0] - pts[:, 1] + sqrt3
            mask2 = val2 < -1e-12
            if np.any(mask2):
                d2 = val2[mask2] / 2.0
                pts[mask2, 0] += d2 * sqrt3
                pts[mask2, 1] += d2
                
            pts[:, 0] = np.clip(pts[:, 0], 0, 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h)
            
        return pts

    def objective(params):
        points = params.reshape(N, 2)
        
        # Calculate all triangle areas
        # Number of combinations is C(11, 3) = 165
        # We can vectorize this somewhat, but for 11 points, a simple loop is fine
        min_area = float('inf')
        
        # Indices
        idx = np.arange(N)
        for i in range(N-2):
            for j in range(i+1, N-1):
                for k in range(j+1, N):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    
                    # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                    area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                    if area < min_area:
                        min_area = area
                        
        # Use a soft-min to smooth the objective for the optimizer
        # We want to maximize the minimum area.
        # Let areas be A. We want to maximize min(A).
        # This is equivalent to minimizing -min(A).
        # -min(A) = logsumexp(-A) approx? No.
        # min(A) = - logsumexp(-A) / beta ?
        # Actually, min(A) = - (1/beta) * log(sum(exp(-beta * A)))
        # So -min(A) = (1/beta) * log(sum(exp(-beta * A)))
        # Let's compute the areas vector first to use logsumexp.
        
        # Re-calculate areas into a vector for soft-min
        areas = []
        for i in range(N-2):
            for j in range(i+1, N-1):
                for k in range(j+1, N):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                    areas.append(area)
        
        areas = np.array(areas)
        
        # Soft min parameter. Higher beta makes it closer to hard min.
        # 500 is a good value for balancing smoothness and fidelity.
        beta = 500.0
        # To avoid overflow/underflow in exp, subtract max
        max_neg_area = np.max(-areas)
        log_sum_exp = max_neg_area + np.log(np.sum(np.exp(-areas - max_neg_area)))
        soft_min_neg = log_sum_exp / beta
        
        return soft_min_neg

    # Initial run with Differential Evolution
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10,
            tol=1e-4,
            polish=False # We will polish manually
        )
    except Exception as e:
        # Fallback to random if DE fails
        points = np.random.rand(N, 2)
        points[:, 0] *= 1.0
        points[:, 1] *= h
        points = project_to_triangle(points)
        return points

    best_params = res.x
    best_points = project_to_triangle(best_params.reshape(N, 2))
    
    # Refine with Nelder-Mead
    if time.time() < deadline - 5.0:
        try:
            res_local = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_local.fun < res.fun: # Compare raw objective values
                best_params = res_local.x
                best_points = project_to_triangle(best_params.reshape(N, 2))
        except Exception:
            pass

    return best_points
