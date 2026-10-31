import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    deadline = time.time() + 115.0
    best_S = float('inf')
    best_params = None

    apothem_factor = math.sqrt(3) / 2

    def get_vertices(x, y, theta):
        verts = []
        for k in range(6):
            angle = theta + math.radians(60 * k)
            verts.append((x + math.cos(angle), y + math.sin(angle)))
        return verts

    def hex_radius(phi, theta):
        rel = phi - theta
        delta = (rel - math.pi/6) % (math.pi/3)
        if delta > math.pi/6:
            delta = math.pi/3 - delta
        return apothem_factor / math.cos(delta)

    def objective_penalty(params, lam):
        # params: [x1, y1, t1, ..., x11, y11, t11, tout]
        theta_out = params[33]
        
        # Calculate S_req
        S_req = 0.0
        outer_normals = []
        for k in range(6):
            angle = theta_out + math.radians(30 + 60 * k)
            outer_normals.append((math.cos(angle), math.sin(angle)))
            
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            t = params[3*i+2]
            verts = get_vertices(x, y, t)
            for nx, ny in outer_normals:
                for vx, vy in verts:
                    proj = vx*nx + vy*ny
                    if proj > S_req:
                        S_req = proj
        S_req = (2.0 / math.sqrt(3)) * S_req
        
        # Calculate Penalty
        penalty = 0.0
        centers = []
        thetas = []
        for i in range(N):
            centers.append((params[3*i], params[3*i+1]))
            thetas.append(params[3*i+2])
            
        for i in range(N):
            for j in range(i+1, N):
                cx1, cy1 = centers[i]
                cx2, cy2 = centers[j]
                dx = cx2 - cx1
                dy = cy2 - cy1
                dist_sq = dx*dx + dy*dy
                if dist_sq < 1e-12:
                    dist = 0.0
                    phi = 0.0
                else:
                    dist = math.sqrt(dist_sq)
                    phi = math.atan2(dy, dx)
                
                r1 = hex_radius(phi, thetas[i])
                r2 = hex_radius(phi + math.pi, thetas[j])
                g = dist - (r1 + r2)
                if g < 0:
                    penalty += g*g
                    
        return S_req + lam * penalty

    # Initial configurations
    starts = []
    
    # Lattice
    sp = math.sqrt(3)
    lattice = []
    lattice.append([0, 0, 0])
    for k in range(6):
        ang = math.radians(60*k)
        lattice.append([sp*math.cos(ang), sp*math.sin(ang), 0])
    for k in [0, 1, 4, 5]:
        ang = math.radians(60*k + 30)
        lattice.append([2*sp*math.cos(ang), 2*sp*math.sin(ang), 0])
    while len(lattice) < N:
        lattice.append([0,0,0])
    lattice_flat = [item for sublist in lattice[:N] for item in sublist]
    lattice_flat.append(0.0) # tout
    starts.append(lattice_flat)

    # Random starts
    random.seed(42)
    for _ in range(15):
        rand_start = []
        for _ in range(N):
            r = random.uniform(0, 4)
            a = random.uniform(0, 2*math.pi)
            t = random.choice([0, math.pi/6, math.pi/3])
            rand_start.extend([r*math.cos(a), r*math.sin(a), t])
        rand_start.append(random.uniform(0, math.pi/3)) # tout
        starts.append(rand_start)

    # Optimization loop
    for start in starts:
        if time.time() > deadline:
            break
        x0 = np.array(start)
        
        # Multi-stage penalty
        current_x = x0
        for lam in [1.0, 10.0, 100.0, 1000.0]:
            if time.time() > deadline:
                break
            try:
                res = minimize(objective_penalty, current_x, args=(lam,), method='Nelder-Mead', 
                               options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
                current_x = res.x
            except Exception:
                pass
        
        # Final evaluation with high lambda to ensure validity
        if time.time() > deadline:
            break
        try:
            res_final = minimize(objective_penalty, current_x, args=(10000.0,), method='Nelder-Mead',
                                 options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7})
            final_x = res_final.x
            S_val = objective_penalty(final_x, 0.0) # Just S_req
            
            # Check validity (penalty should be ~0)
            # We can check min constraint g_ij
            # If penalty is small, it's valid.
            # Let's verify S_req is lower than best
            if S_val < best_S:
                # Double check disjointness
                is_valid = True
                centers = []
                thetas = []
                for i in range(N):
                    centers.append((final_x[3*i], final_x[3*i+1]))
                    thetas.append(final_x[3*i+2])
                for i in range(N):
                    for j in range(i+1, N):
                        cx1, cy1 = centers[i]
                        cx2, cy2 = centers[j]
                        dx = cx2 - cx1
                        dy = cy2 - cy1
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 1e-9:
                            phi = 0
                        else:
                            phi = math.atan2(dy, dx)
                        r1 = hex_radius(phi, thetas[i])
                        r2 = hex_radius(phi + math.pi, thetas[j])
                        if dist < r1 + r2 - 1e-6:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if is_valid:
                    best_S = S_val
                    best_params = final_x
        except Exception:
            pass

    if best_params is None:
        return [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N-N//2)], (0.0, 0.0), 40.0, 0.0

    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        t = best_params[3*i+2]
        t_deg = math.degrees(t)
        inner.append([x, y, t_deg])
    
    tout_deg = math.degrees(best_params[33])
    return inner, (0.0, 0.0), best_S, tout_deg

