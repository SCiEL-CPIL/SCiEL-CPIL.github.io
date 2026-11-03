"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3)

    # Precompute unit hexagon vertices
    unit_verts = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])

    def calc_S(positions, alphas, beta_deg, scale=1.0):
        # positions: (N, 2) array of centers
        # alphas: array of N angles in degrees (rotation of each inner hexagon)
        # beta_deg: rotation of outer hexagon
        # scale: side length of inner hexagons (should be 1.0 for this task)
        # Returns minimum side length of outer hexagon centered at origin
        
        beta_rad = math.radians(beta_deg)
        alpha_rads = np.radians(alphas)
        
        # Vertices of the i-th hexagon
        # v = P + scale * R(alpha) @ unit_verts
        # We need to find the minimal S such that all vertices lie inside
        # the outer hexagon of side S, centered at origin, rotated by beta.
        
        # Outer hexagon: intersection of 6 half-planes.
        # Normals are at angles: 30 + beta + k*60
        # Equation: dot(v, n_k) <= S / sqrt(3)
        
        # So for each hex i and vertex j:
        # dot(P_i + scale * R(alpha_i) @ unit_verts_j, n_k) <= S / sqrt(3)
        # S >= sqrt(3) * dot(P_i + scale * R(alpha_i) @ unit_verts_j, n_k)
        
        # We take max over all i, j, k.
        
        # Vectorization:
        # For each k, we need max over i, j of dot(P_i, n_k) + scale * dot(R(alpha_i) @ unit_verts_j, n_k)
        # dot(R(alpha_i) @ v, n) = dot(v, R(-alpha_i) @ n)
        # So term = dot(P_i, n_k) + scale * dot(unit_verts_j, R(-alpha_i) @ n_k)
        
        # Let M_i = R(-alpha_i) @ n_k (shape (N, 2) if we fix k)
        # Then dot(unit_verts_j, M_i) is a (N, 6) matrix (dotting 6 verts with N vectors? No)
        # dot(unit_verts (6,2), M_i.T (2,N)) -> (6, N). We want max over j and i.
        
        # Let's compute for each k:
        # 1. n_k (2,)
        # 2. P_i @ n_k -> (N,)
        # 3. n_rot_i = R(-alpha_i) @ n_k -> (N, 2)
        # 4. dots = unit_verts @ n_rot_i.T -> (6, N)
        # 5. max_dot = np.max(dots)
        # 6. val_k = np.max(P_i @ n_k) + scale * max_dot ?
        #    NO. The max is over the sum: max_{i,j} (A_i + B_{i,j}).
        #    max_{i,j} (A_i + B_{i,j}) = max_i (A_i + max_j B_{i,j}).
        
        max_constraint = 0.0
        
        for k in range(6):
            normal_angle = math.radians(30) + beta_rad + k * math.radians(60)
            n = np.array([math.cos(normal_angle), math.sin(normal_angle)])
            
            # A_i = dot(P_i, n)
            A = positions @ n  # (N,)
            
            # B_{i,j} = dot(unit_verts_j, R(-alpha_i) @ n)
            # R(-alpha_i) @ n
            c = np.cos(-alpha_rads)
            s = np.sin(-alpha_rads)
            # n_rot_i = [c*n[0] - s*n[1], s*n[0] + c*n[1]]
            n_rot_x = c * n[0] - s * n[1]
            n_rot_y = s * n[0] + c * n[1]
            
            # dots_{i,j} = unit_verts[j,0]*n_rot_x[i] + unit_verts[j,1]*n_rot_y[i]
            # (6, N)
            dots = unit_verts[:, 0] * n_rot_x[None, :] + unit_verts[:, 1] * n_rot_y[None, :]
            
            # max_j dots_{i,j}
            max_B_per_i = np.max(dots, axis=0)  # (N,)
            
            # max_i (A_i + scale * max_B_per_i)
            val = np.max(A + scale * max_B_per_i)
            
            if val > max_constraint:
                max_constraint = val
                
        return max_constraint * sqrt3

    # Topologies for positions
    # We use a scaled lattice.
    # Standard hexagonal grid spacing for tangent hexagons (side 1) is sqrt(3).
    
    def get_positions(topo, scale=sqrt3):
        pts = []
        for q, r in topo:
            x = scale * (q + r/2.0)
            y = scale * (r * sqrt3/2.0)
            pts.append((x, y))
        # Center
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        return np.array([(x-cx, y-cy) for x, y in pts])
        
    topologies = []
    
    # Topo 1: 3x4 Parallelogram
    t1 = [(q, r) for q in range(3) for r in range(4)]
    topologies.append(t1)
    
    # Topo 2: 4x3 Parallelogram
    t2 = [(q, r) for q in range(4) for r in range(3)]
    topologies.append(t2)
    
    # Topo 3: 2x6 Parallelogram
    t3 = [(q, r) for q in range(2) for r in range(6)]
    topologies.append(t3)
    
    # Topo 4: 6x2 Parallelogram
    t4 = [(q, r) for q in range(6) for r in range(2)]
    topologies.append(t4)
    
    # Topo 5: Hex-like (1 center, 6 ring, 5 outer)
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Outer: (2,0), (-2,0), (0,2), (0,-2), (2,-1)
    t5 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (-2,0), (0,2), (0,-2), (2,-1)
    ]
    if len(set(t5)) == 12:
        topologies.append(t5)
        
    # Topo 6: Another variation
    t6 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    if len(set(t6)) == 12:
        topologies.append(t6)

    best_S = float('inf')
    best_params = None
    
    # Strategy:
    # 1. For each topology, optimize global rotation alpha (same for all), beta (outer), and translation cx, cy.
    #    This is a 4-parameter optimization.
    #    We assume all inner hexagons have the same rotation alpha. This is a strong assumption but valid for lattice packings.
    #    If we want individual rotations, the parameter space is 12+1+2 = 15D, which is harder.
    #    Given the "STAGNATION" directive, using Differential Evolution on 4D is a good structural change from Nelder-Mead on fixed topology/angles.
    
    #    Actually, let's allow individual rotations in the objective function, but optimize a single global alpha first to guide DE, or just let DE handle 15 params?
    #    15 params in DE with pop=50, iter=500 is 25k evals.
    #    Each eval is O(N*6) = 72 ops. Very fast.
    #    Let's try individual rotations!
    
    for t_idx, topo in enumerate(topologies):
        if time.time() > deadline - 20:
            break
            
        base_positions = get_positions(topo)
        
        # Objective function for DE
        # Params: alpha_0..alpha_11 (12), beta (1), cx, cy (2) -> 15 params
        # Note: alpha is periodic with 60 deg.
        
        def objective(params):
            if time.time() > deadline - 5:
                return 1e6
            alphas = params[:12]
            beta = params[12]
            cx = params[13]
            cy = params[14]
            
            positions = base_positions + np.array([cx, cy])
            return calc_S(positions, alphas, beta, scale=1.0)
            
        # Bounds
        # Alphas: 0 to 60
        # Beta: 0 to 60
        # cx, cy: -10 to 10 (arbitrary large enough)
        bounds = [(-30, 30)] * 12 + [(0, 60), (-5, 5), (-5, 5)]
        
        # DE parameters
        # popsize=50, maxiter=300, tol=1e-8
        # This is a "medium" search.
        
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=200, 
                popsize=30, 
                tol=1e-8, 
                seed=42 + t_idx,
                init='latinhypercube',
                polish=False  # Skip polish to save time, we can do a final NM if needed
            )
            if res.fun < best_S:
                best_S = res.fun
                best_params = (base_positions, res.x[:12], res.x[12], res.x[13], res.x[14])
        except Exception as e:
            pass

    if best_params is None:
        # Fallback
        base_positions = get_positions(topologies[0])
        alphas = np.zeros(12)
        beta = 0.0
        cx, cy = 0.0, 0.0
        best_S = calc_S(base_positions, alphas, beta)
        best_params = (base_positions, alphas, beta, cx, cy)
        
    base_positions, alphas, beta, cx, cy = best_params
    final_positions = base_positions + np.array([cx, cy])
    
    inner = []
    for i in range(N):
        x, y = final_positions[i]
        inner.append([x, y, alphas[i]])
        
    return inner, (0.0, 0.0), best_S, beta

# EVOLVE-BLOCK-END
