"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    sqrt3 = math.sqrt(3)

    def support_unit(n_vec):
        """Support function of a unit regular hexagon (pointy-top by default, angle 0).
        n_vec is a 2D numpy array. Returns max dot product with vertices."""
        # Vertices at angles 0, 60, 120, 180, 240, 300
        # x = cos, y = sin
        cos_a = np.cos([0, 60, 120, 180, 240, 300] * np.pi / 180)
        sin_a = np.sin([0, 60, 120, 180, 240, 300] * np.pi / 180)
        return np.max(cos_a * n_vec[0] + sin_a * n_vec[1])

    def calc_S(positions, alpha_deg, beta_deg, cx=0.0, cy=0.0):
        """Calculate minimum side length S of outer hexagon centered at (0,0) with rotation beta_deg
        containing inner hexagons at positions + (cx,cy) with rotation alpha_deg."""
        alpha_rad = math.radians(alpha_deg)
        beta_rad = math.radians(beta_deg)
        pos_shifted = positions + np.array([cx, cy])
        
        max_proj = 0.0
        # Outer hexagon normals are at 30 + beta + k*60
        # We check 6 normals. Due to symmetry, checking 3 might suffice if we consider +/-, 
        # but let's check all 6 to be safe and simple.
        for k in range(6):
            normal_angle = math.radians(30) + beta_rad + k * math.radians(60)
            n = np.array([math.cos(normal_angle), math.sin(normal_angle)])
            
            # Support of inner hexagon i in direction n:
            # (P_i + d) . n + h_unit(R_-alpha n)
            # where d is the center offset of the inner hexagon? No, P_i is the center.
            # The vertices are P_i + R_alpha * v_j.
            # Max dot product is P_i . n + max_j (R_alpha v_j) . n = P_i . n + max_j v_j . (R_-alpha n)
            
            c = math.cos(-alpha_rad)
            s = math.sin(-alpha_rad)
            n_rot = np.array([c*n[0] - s*n[1], s*n[0] + c*n[1]])
            
            h_unit_val = support_unit(n_rot)
            
            # Vectorized projection
            projs = pos_shifted @ n + h_unit_val
            max_p_proj = np.max(projs)
            
            if max_p_proj > max_proj:
                max_proj = max_p_proj
        
        # The width of the projection is 2 * max_proj (since outer hex is centered at 0 and convex/symmetric)
        # Wait, is the outer hexagon centered at 0? Yes.
        # The distance from center to side is the apothem.
        # For a regular hexagon with side S, apothem a = S * sqrt(3) / 2.
        # So S = 2 * a / sqrt(3).
        # The projection max_proj is the distance from center to the furthest point in direction n.
        # This must be <= apothem? No.
        # The vertices of the outer hexagon project to S in directions 0, 60... 
        # The sides are at 30, 90... 
        # The support function of the outer hexagon in direction n (which is a normal direction) is exactly the apothem.
        # So max_proj must be <= apothem.
        # Thus apothem >= max_proj.
        # S * sqrt(3) / 2 >= max_proj  => S >= 2 * max_proj / sqrt(3).
        
        return max_proj * 2.0 / sqrt3

    def get_positions(topo, scale=2.0):
        pts = []
        for q, r in topo:
            x = scale * (q + r/2.0)
            y = scale * (r * sqrt3/2.0)
            pts.append((x, y))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        return np.array([(x-cx, y-cy) for x, y in pts])

    # Topologies
    topologies = []
    # 3x4
    topologies.append([(q, r) for q in range(3) for r in range(4)])
    # 4x3
    topologies.append([(q, r) for q in range(4) for r in range(3)])
    # 2x6
    topologies.append([(q, r) for q in range(2) for r in range(6)])
    # 6x2
    topologies.append([(q, r) for q in range(6) for r in range(2)])
    # Cross-like
    t5 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (-2,0), (0,2), (0,-2), (2,-1)
    ]
    if len(set(t5)) == 12:
        topologies.append(t5)
    # Another
    t6 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    if len(set(t6)) == 12:
        topologies.append(t6)

    best_S = float('inf')
    best_params = None
    best_topology_idx = 0

    # Strategy:
    # 1. Coarse grid search over alpha, beta for each topology.
    # 2. Refine top candidates with Nelder-Mead on (cx, cy, alpha, beta).
    
    for t_idx, topo in enumerate(topologies):
        if time.time() > deadline - 15:
            break
            
        positions = get_positions(topo, scale=2.0)
        
        candidates = []
        step = 2.0  # Start with slightly coarser grid to save time, then refine
        
        # We can exploit symmetry: alpha and beta are modulo 60.
        # Also, alpha=0 and alpha=30 might be equivalent under rotation?
        # Let's just search 0 to 60.
        
        for a in np.arange(0, 60, step):
            if time.time() > deadline - 10:
                break
            for b in np.arange(0, 60, step):
                S = calc_S(positions, a, b)
                candidates.append((S, a, b))
        
        if not candidates:
            continue
            
        candidates.sort(key=lambda x: x[0])
        
        # Refine top 3 candidates
        for S_cand, a_cand, b_cand in candidates[:3]:
            if time.time() > deadline - 5:
                break
                
            def obj_all(params):
                cx, cy, alpha, beta = params
                return calc_S(positions, alpha, beta, cx, cy)
            
            try:
                res = minimize(obj_all, [0.0, 0.0, a_cand, b_cand], method='Nelder-Mead', 
                               options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < S_cand:
                    S_cand = res.fun
            except:
                pass
            
            if S_cand < best_S:
                best_S = S_cand
                best_params = (positions, res.x[2], res.x[3], res.x[0], res.x[1])
                best_topology_idx = t_idx

    # Fallback
    if best_params is None:
        positions = get_positions(topologies[0], scale=2.0)
        best_S = calc_S(positions, 0, 0)
        best_params = (positions, 0.0, 0.0, 0.0, 0.0)
        
    positions, alpha, beta, cx, cy = best_params
    
    # Final polish
    if time.time() < deadline - 2:
        def obj_all(params):
            cx, cy, alpha, beta = params
            return calc_S(positions, alpha, beta, cx, cy)
            
        try:
            res = minimize(obj_all, [cx, cy, alpha, beta], method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_S:
                best_S = res.fun
                cx, cy, alpha, beta = res.x
        except:
            pass
            
    final_positions = positions + np.array([cx, cy])
    
    inner = []
    for x, y in final_positions:
        inner.append([x, y, alpha])
        
    return inner, (0.0, 0.0), best_S, beta

# EVOLVE-BLOCK-END
