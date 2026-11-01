import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

def construct():
        start_time = time.time()
        deadline = start_time + 110.0 # Safety margin
        
        best_score = np.inf
        best_points = None
        
        # Helper to calculate score
        def get_score(points):
            # Check distinctness
            dists = cdist(points, points)
            np.fill_diagonal(dists, np.inf)
            min_d = np.min(dists)
            if min_d < 1e-10:
                return np.inf
            max_d = np.max(dists)
            return (max_d / min_d) ** 2
        
        # Model 1: Gen Cuboctahedron + Poles
        # Vars: x, z, h
        # Points:
        # 4 at (±x, ±x, 0)
        # 4 at (±x, 0, ±z)
        # 4 at (0, ±x, ±z)
        # 2 at (0, 0, ±h)
        
        def model1(params):
            x, z, h = params
            if x <= 0 or z <= 0 or h <= 0:
                return np.inf
            p1 = np.array([[x, x, 0], [x, -x, 0], [-x, x, 0], [-x, -x, 0]])
            p2 = np.array([[x, 0, z], [x, 0, -z], [-x, 0, z], [-x, 0, -z]])
            p3 = np.array([[0, x, z], [0, x, -z], [0, -x, z], [0, -x, -z]])
            p4 = np.array([[0, 0, h], [0, 0, -h]])
            points = np.vstack([p1, p2, p3, p4])
            return points

        def obj1(params):
            if time.time() > deadline:
                return np.inf
            points = model1(params)
            return get_score(points)

        # Model 2: Two Rings of 7
        # Vars: r1, z1, r2, z2, phi
        def model2(params):
            r1, z1, r2, z2, phi = params
            if r1 <= 0 or r2 <= 0:
                return np.inf
            angles1 = np.linspace(0, 2*np.pi, 7, endpoint=False)
            angles2 = angles1 + phi
            p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1), np.full(7, z1)])
            p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2), np.full(7, z2)])
            return np.vstack([p1, p2])

        def obj2(params):
            if time.time() > deadline:
                return np.inf
            points = model2(params)
            return get_score(points)

        # Model 3: Ring 6, Ring 6, 2 Poles
        # Vars: r1, z1, r2, z2, phi
        def model3(params):
            r1, z1, r2, z2, phi = params
            if r1 <= 0 or r2 <= 0:
                return np.inf
            angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
            angles2 = angles1 + phi
            p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1), np.full(6, z1)])
            p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2), np.full(6, z2)])
            p3 = np.array([[0, 0, 1.5], [0, 0, -1.5]]) # Scale poles? No, let's add h as var?
            # Better: 6 vars: r1, z1, r2, z2, phi, h
            # Let's stick to 5 vars for simplicity, fix h relative to z?
            # Or just make it 6 vars.
            return np.vstack([p1, p2]) # This only has 12 points. Need 2 more.
            # Let's add poles at (0,0,±h). So 6 vars.
            
        # Let's refine Model 3 to 6 vars: r1, z1, r2, z2, phi, h
        def model3_full(params):
            r1, z1, r2, z2, phi, h = params
            if r1 <= 0 or r2 <= 0 or h <= 0:
                return np.inf
            angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
            angles2 = angles1 + phi
            p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1), np.full(6, z1)])
            p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2), np.full(6, z2)])
            p3 = np.array([[0, 0, h], [0, 0, -h]])
            return np.vstack([p1, p2, p3])

        def obj3(params):
            if time.time() > deadline:
                return np.inf
            points = model3_full(params)
            return get_score(points)

        # Optimization Loop
        # We will run Nelder-Mead on each model with several initial guesses.
        
        # Initial guesses for Model 1 (x, z, h)
        # Cuboctahedron is x=1, z=1. Poles h=sqrt(2)?
        # Let's try a grid of initial values.
        inits1 = []
        for x in [0.5, 1.0, 1.5]:
            for z in [0.5, 1.0, 1.5]:
                for h in [0.5, 1.0, 1.5, 2.0]:
                    inits1.append([x, z, h])
        
        # Initial guesses for Model 2 (r1, z1, r2, z2, phi)
        inits2 = []
        for r1 in [0.5, 1.0, 1.5]:
            for z1 in [-1.0, 0.0, 1.0]:
                for r2 in [0.5, 1.0, 1.5]:
                    for z2 in [-1.0, 0.0, 1.0]:
                        for phi in [0, np.pi/7, np.pi/3]:
                            inits2.append([r1, z1, r2, z2, phi])
                            
        # Initial guesses for Model 3 (r1, z1, r2, z2, phi, h)
        inits3 = []
        for r1 in [0.5, 1.0]:
            for z1 in [-1.0, 0.0, 1.0]:
                for r2 in [0.5, 1.0]:
                    for z2 in [-1.0, 0.0, 1.0]:
                        for phi in [0, np.pi/6]:
                            for h in [1.0, 1.5, 2.0]:
                                inits3.append([r1, z1, r2, z2, phi, h])

        def run_opt(obj_func, inits, model_func):
            nonlocal best_score, best_points
            for init in inits:
                if time.time() > deadline:
                    break
                try:
                    res = minimize(obj_func, init, method='Nelder-Mead', 
                                   options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                    if res.fun < best_score:
                        best_score = res.fun
                        best_points = model_func(res.x)
                except Exception:
                    pass

        run_opt(obj1, inits1, model1)
        run_opt(obj2, inits2, model2)
        run_opt(obj3, inits3, model3_full)
        
        if best_points is None:
            # Fallback: Random points? Or seed?
            # Let's return a known decent configuration (e.g. from Model 1 default)
            best_points = model1([1.0, 1.0, 1.4])
            best_score = get_score(best_points)
            
        return best_points
