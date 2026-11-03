import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
        start_time = time.time()
        deadline = 110.0 # Leave 40s margin for safety
        
        best_score = -np.inf
        best_x = None
        
        # Define the objective function
        # x = [x1..xn, y1..yn, r1..rn, w, h]
        def objective(v):
            # Check time
            if time.time() > deadline:
                return np.inf
            
            xs = v[:N]
            ys = v[N:2*N]
            rs = v[2*N:3*N]
            w = v[3*N]
            h = v[3*N+1]
            
            # Penalty for perimeter constraint w + h <= 2
            perim_pen = 0.0
            if w + h > 2.0:
                perim_pen = (w + h - 2.0)**2
            
            # Penalty for circles being outside rectangle
            bound_pen = 0.0
            for i in range(N):
                if xs[i] < rs[i]:
                    bound_pen += (rs[i] - xs[i])**2
                if xs[i] > w - rs[i]:
                    bound_pen += (xs[i] - (w - rs[i]))**2
                if ys[i] < rs[i]:
                    bound_pen += (rs[i] - ys[i])**2
                if ys[i] > h - rs[i]:
                    bound_pen += (ys[i] - (h - rs[i]))**2
            
            # Penalty for overlapping circles
            overlap_pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs[i] + rs[j]
                    if dist < min_dist:
                        overlap_pen += (min_dist - dist)**2
            
            # Objective: Maximize sum of radii, minimize penalties
            # We want to minimize -sum(r) + penalties
            val = -np.sum(rs) + 100.0 * (perim_pen + bound_pen + overlap_pen)
            
            return val

        # Initial configurations
        configs = []
        
        # 1. Grid
        side = int(np.ceil(np.sqrt(N)))
        w0, h0 = 1.0, 1.0
        xs = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)])[:N]
        ys = np.array([(j + 0.5) / side for _ in range(side) for j in range(side)])[:N]
        rs = np.full(N, 0.5 / side - 0.01)
        configs.append(np.concatenate([xs, ys, rs, [w0, h0]]))
        
        # 2. Random
        for _ in range(5):
            w0, h0 = 1.0, 1.0
            xs = np.random.uniform(0.1, 0.9, N)
            ys = np.random.uniform(0.1, 0.9, N)
            rs = np.full(N, 0.05)
            configs.append(np.concatenate([xs, ys, rs, [w0, h0]]))

        # 3. Hexagonal-ish
        # Just random with slightly larger radii
        for _ in range(5):
            w0, h0 = 1.0, 1.0
            xs = np.random.uniform(0.1, 0.9, N)
            ys = np.random.uniform(0.1, 0.9, N)
            rs = np.full(N, 0.1)
            configs.append(np.concatenate([xs, ys, rs, [w0, h0]]))

        for init_x in configs:
            if time.time() > deadline:
                break
                
            try:
                res = minimize(objective, init_x, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
                
                # Extract result
                xs = res.x[:N]
                ys = res.x[N:2*N]
                rs = res.x[2*N:3*N]
                w = res.x[3*N]
                h = res.x[3*N+1]
                
                # Check validity
                if w + h > 2.0 + 1e-5:
                    continue
                valid = True
                for i in range(N):
                    if xs[i] < rs[i] - 1e-5 or xs[i] > w - rs[i] + 1e-5:
                        valid = False
                        break
                    if ys[i] < rs[i] - 1e-5 or ys[i] > h - rs[i] + 1e-5:
                        valid = False
                        break
                if not valid:
                    continue
                
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        if dist < rs[i] + rs[j] - 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                score = np.sum(rs)
                if score > best_score:
                    best_score = score
                    best_x = res.x
                    
            except Exception:
                continue

        if best_x is None:
            # Fallback to seed
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
            return np.array(cells[:N])
            
        # Construct output array
        xs = best_x[:N]
        ys = best_x[N:2*N]
        rs = best_x[2*N:3*N]
        
        # Scale to fit perimeter 4 exactly?
        # The optimizer maximizes sum of radii subject to w+h<=2.
        # It might leave w+h < 2. We can scale the whole configuration to increase radii?
        # No, if w+h < 2, we can increase w and h proportionally?
        # If we scale coordinates and radii by factor k, w' = kw, h' = kh.
        # Perimeter 2(k(w+h)) <= 4 => k(w+h) <= 2.
        # If current w+h < 2, we can scale up until w+h=2.
        # Let k = 2 / (w + h).
        # New xs = k * xs, new ys = k * ys, new rs = k * rs.
        # This is valid because relative positions and overlaps are preserved, and boundary constraints are preserved if they were satisfied (strictly inside or on boundary).
        # Wait, if a circle is touching the boundary, scaling keeps it touching.
        # So we should always scale to the maximum possible perimeter.
        
        w = best_x[3*N]
        h = best_x[3*N+1]
        k = 2.0 / (w + h)
        
        xs = k * xs
        ys = k * ys
        rs = k * rs
        
        result = np.column_stack((xs, ys, rs))
        return result
