import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14
k = N // 2  # 7

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin against 150s kill

    def get_points(params):
        v = params.reshape(k, 2)
        P = np.vstack([v, -v])
        return P

    def score(points):
        # Compute convex hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return -1e10

        if hull_area < 1e-12:
            return -1e10

        # Compute min triangle area
        n = points.shape[0]
        min_area = float('inf')

        for i in range(n):
            xi, yi = points[i, 0], points[i, 1]
            for j in range(i + 1, n):
                dx1 = points[j, 0] - xi
                dy1 = points[j, 1] - yi
                for l in range(j + 1, n):
                    dx2 = points[l, 0] - xi
                    dy2 = points[l, 1] - yi
                    area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                    if area < min_area:
                        min_area = area
                        if min_area < 1e-15:
                            return -1e10

        return min_area / hull_area

    def objective(params):
        P = get_points(params)
        s = score(P)
        return -s

    def objective_soft(params, beta=100.0):
        P = get_points(params)
        try:
            hull = ConvexHull(P)
            hull_area = hull.volume
        except Exception:
            return 1e10

        if hull_area < 1e-12:
            return 1e10

        n = P.shape[0]
        areas = []
        for i in range(n):
            xi, yi = P[i, 0], P[i, 1]
            for j in range(i + 1, n):
                dx1 = P[j, 0] - xi
                dy1 = P[j, 1] - yi
                for l in range(j + 1, n):
                    dx2 = P[l, 0] - xi
                    dy2 = P[l, 1] - yi
                    area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                    areas.append(area)
        
        areas = np.array(areas)
        max_a = np.max(areas)
        soft_min = -np.log(np.sum(np.exp(-beta * (areas - max_a)))) / beta + max_a
        
        return -soft_min / hull_area

    # Support function parameterization:
    # We define a convex body by its support function h(θ) = sum of cosines with frequencies
    # For centrally symmetric bodies, h(θ) = sum_{m=1}^{M} a_m * cos(2m*θ)
    # The boundary points can be approximated from this.
    
    def support_function_to_points(a, num_points=N):
        """
        Convert support function coefficients to boundary points.
        a[0] is the base radius, a[1], a[2], ... are cosine coefficients.
        h(θ) = a[0] + sum_{m=1}^{len(a)-1} a[m] * cos(2*m*θ)
        """
        if a[0] < 0.1:
            a = a.copy()
            a[0] = 0.1 + abs(a[0]) * 0.1
        
        theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        
        # Compute support function values
        h_vals = np.zeros(num_points)
        for m in range(len(a)):
            if m == 0:
                h_vals += a[0]
            else:
                h_vals += a[m] * np.cos(2 * m * theta)
        
        # The support function gives the distance from origin to tangent line.
        # The point on the boundary in direction θ is given by:
        # x = h(θ) * cos(θ) - h'(θ) * sin(θ)
        # y = h(θ) * sin(θ) + h'(θ) * cos(θ)
        
        # Compute derivative h'(θ)
        h_prime = np.zeros(num_points)
        for m in range(1, len(a)):
            h_prime += a[m] * (-2 * m) * np.sin(2 * m * theta)
        
        x = h_vals * np.cos(theta) - h_prime * np.sin(theta)
        y = h_vals * np.sin(theta) + h_prime * np.cos(theta)
        
        P = np.column_stack([x, y])
        
        # Ensure the points are valid (hull area > 0)
        return P

    def objective_support(params):
        P = support_function_to_points(params)
        s = score(P)
        return -s

    def objective_support_soft(params, beta=100.0):
        P = support_function_to_points(params)
        try:
            hull = ConvexHull(P)
            hull_area = hull.volume
        except Exception:
            return 1e10

        if hull_area < 1e-12:
            return 1e10

        n = P.shape[0]
        areas = []
        for i in range(n):
            xi, yi = P[i, 0], P[i, 1]
            for j in range(i + 1, n):
                dx1 = P[j, 0] - xi
                dy1 = P[j, 1] - yi
                for l in range(j + 1, n):
                    dx2 = P[l, 0] - xi
                    dy2 = P[l, 1] - yi
                    area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                    areas.append(area)
        
        areas = np.array(areas)
        max_a = np.max(areas)
        soft_min = -np.log(np.sum(np.exp(-beta * (areas - max_a)))) / beta + max_a
        
        return -soft_min / hull_area

    best_params = None
    best_score = -np.inf
    best_params_support = None
    best_score_support = -np.inf

    # Phase 1: Symmetric parameterization optimization
    angles = np.linspace(0, np.pi, k, endpoint=False)
    v_init = np.column_stack([np.cos(angles), np.sin(angles)])
    x0 = v_init.flatten()
    
    best_params = x0.copy()
    best_score = -objective(x0)

    # Phase 1a: Soft objective optimization
    if time.time() - start_time < time_limit - 30:
        try:
            for beta in [10.0, 50.0, 100.0]:
                if time.time() - start_time > time_limit - 20:
                    break
                res = minimize(objective_soft, best_params, method='Powell', 
                               args=(beta,),
                               options={'maxiter': 2000, 'xtol': 1e-8, 'ftol': 1e-8})
                if res.fun < -best_score:
                    best_params = res.x
                    best_score = -res.fun
        except Exception:
            pass

    # Phase 1b: Hard objective optimization
    if time.time() - start_time < time_limit - 15:
        try:
            res = minimize(objective, best_params, method='Powell', 
                           options={'maxiter': 3000, 'xtol': 1e-10, 'ftol': 1e-10})
            if res.fun < -best_score:
                best_params = res.x
                best_score = -res.fun
        except Exception:
            pass

    # Phase 1c: Multi-start local refinement
    if time.time() - start_time < time_limit - 10:
        rng = np.random.RandomState(42)
        for _ in range(5):
            if time.time() - start_time > time_limit - 5:
                break
            perturbation = rng.normal(0, 0.1, size=best_params.shape)
            x_pert = best_params + perturbation
            
            try:
                res = minimize(objective, x_pert, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < -best_score:
                    best_params = res.x
                    best_score = -res.fun
            except Exception:
                pass

    # Phase 2: Support function parameterization optimization
    if time.time() - start_time < time_limit - 5:
        # Initialize support function coefficients
        # a[0] is base radius, a[1..5] are cosine coefficients
        a_init = np.zeros(6)
        a_init[0] = 1.0  # base radius
        
        # Try a few different initializations
        initializations = [
            a_init.copy(),
            np.array([1.0, 0.1, 0.05, 0.02, 0.01, 0.005]),
            np.array([1.0, -0.1, 0.05, -0.02, 0.01, -0.005]),
            np.array([1.0, 0.2, 0.1, 0.05, 0.02, 0.01]),
            np.array([1.0, 0.05, -0.02, 0.01, -0.005, 0.002]),
        ]
        
        for a0 in initializations:
            if time.time() - start_time > time_limit - 3:
                break
            
            try:
                # Soft objective first
                res = minimize(objective_support_soft, a0, method='Powell', 
                               args=(50.0,),
                               options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-8})
                if res.fun < -best_score_support:
                    best_params_support = res.x
                    best_score_support = -res.fun
                
                # Hard objective
                res = minimize(objective_support, a0, method='Powell', 
                               options={'maxiter': 1500, 'xtol': 1e-10, 'ftol': 1e-10})
                if res.fun < -best_score_support:
                    best_params_support = res.x
                    best_score_support = -res.fun
            except Exception:
                pass
        
        # Compare with symmetric parameterization
        if best_score_support > best_score and best_params_support is not None:
            best_score = best_score_support
            # We'll use the support function result
            final_points = support_function_to_points(best_params_support)
        elif best_params is not None:
            final_points = get_points(best_params)
        else:
            # Fallback: regular 14-gon
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            final_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return final_points
