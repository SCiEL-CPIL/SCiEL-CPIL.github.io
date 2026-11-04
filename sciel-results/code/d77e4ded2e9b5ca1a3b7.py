import numpy as np
import time
from scipy.spatial import ConvexHull
import sys

START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # Safety margin for 150s limit

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_outer_side_exact(inner_positions, inner_angles, outer_center, outer_angle_deg):
    """
    Compute the minimum side length S of a regular hexagon centered at outer_center 
    with rotation outer_angle_deg such that all vertices of all inner hexagons are inside.
    
    For a regular hexagon centered at origin with side S and rotation 0:
    The boundary is defined by max(|x|, |x + y/sqrt(3)|, |x - y/sqrt(3)|) <= S.
    For a rotated hexagon, we rotate the point into the hexagon's local frame.
    """
    cx, cy = outer_center
    pos = np.asarray(inner_positions, dtype=np.float64)
    ang = np.radians(np.asarray(inner_angles, dtype=np.float64))
    
    # Generate all vertices of all inner hexagons
    # Inner hexagons have side length 1.0
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    
    all_verts_x = verts_x.ravel()
    all_verts_y = verts_y.ravel()
    
    # Translate to outer center
    dx = all_verts_x - cx
    dy = all_verts_y - cy
    
    # Rotate into outer hexagon's local frame
    a_outer = np.radians(outer_angle_deg)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    # Constraint: max(|lx|, |lx + ly/sqrt3|, |lx - ly/sqrt3|) <= S
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-5):
    """
    Check if packing is valid:
    1. No overlaps between inner hexagons.
    2. All inner hexagons are inside the outer hexagon.
    """
    cx, cy = outer_center
    verts_list = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        verts_list.append(verts)
    
    # 1. Check overlaps using SAT (Separating Axis Theorem)
    def sat_overlap(v1, v2):
        axes = []
        for v in [v1, v2]:
            for i in range(6):
                p1 = v[i]
                p2 = v[(i+1)%6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    axes.append(normal / norm)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            # If there is a separating axis, they don't overlap
            if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                return False
        return True

    for i in range(len(verts_list)):
        for j in range(i+1, len(verts_list)):
            if sat_overlap(verts_list[i], verts_list[j]):
                return False
            
    # 2. Check containment
    outer_verts = _hex_vertices(cx, cy, outer_side, outer_angle)
    
    def point_in_poly(pt, poly):
        x, y = pt
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    for verts in verts_list:
        for v in verts:
            if not point_in_poly(v, outer_verts):
                return False
    return True

def _evaluate_solution(params):
    """
    Evaluate a candidate solution.
    params: [x1, y1, a1, ..., x11, y11, a11, cx, cy, A_outer]
    Returns (score, is_valid). Lower score is better.
    """
    N = 11
    pos = params[:2*N].reshape(N, 2)
    ang = params[2*N:3*N]
    outer_center = params[3*N:3*N+2]
    outer_angle = params[3*N+2]
    
    # 1. Check validity (overlaps and containment)
    inner_list = [[float(pos[i,0]), float(pos[i,1]), float(ang[i])] for i in range(N)]
    
    # Quick check: compute required outer side to contain all
    # If the "required" side is less than a theoretical minimum, it might be invalid due to overlaps?
    # No, containment check is separate.
    
    # To be fast, we first compute the side length required for containment.
    # If this is huge, it's likely invalid or just bad.
    # But we must check overlaps first.
    
    # Overlap check is O(N^2 * 12) approx. Fast enough for DE.
    if not _check_validity(inner_list, tuple(outer_center), 100.0, outer_angle):
         # If we use a huge outer hexagon for validity check, we only check overlaps.
         # But _check_validity also checks containment. 
         # We need a dedicated overlap check or modify _check_validity.
         # Let's refine: _check_validity checks both. 
         # If we pass a huge outer_side, containment is trivially true (unless points are inf).
         # So this checks overlaps.
         return np.inf, False
    
    # If no overlaps, compute the minimum outer side length required.
    side = _min_outer_side_exact(pos, ang, outer_center, outer_angle)
    
    # The score is the side length.
    # We should also verify that with this exact side length, the packing is still valid (containment).
    # _min_outer_side_exact computes the *minimum* side to contain. 
    # So if we use that side, containment is satisfied by definition (boundary touches).
    # Overlaps are already checked.
    
    return side, True

def construct():
    global START_TIME, DEADLINE
    
    N = 11
    best_score = np.inf
    best_params = None
    
    # Initial population for Differential Evolution
    # Strategy: Mix of random and structured starts
    # Structured: The known 1+6+4 lattice
    sqrt3 = np.sqrt(3)
    pos_lattice = np.zeros((N, 2))
    ang_lattice = np.zeros(N)
    pos_lattice[0] = [0, 0]
    for i in range(6):
        a = i * np.pi / 3
        pos_lattice[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/11, 11*np.pi/6] # Slight variation? No, keep standard
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        a = ring2_angles[i]
        pos_lattice[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
    
    lattice_params = np.concatenate([pos_lattice.ravel(), ang_lattice, [0, 0], [0]])
    
    # Define bounds for DE
    # Positions: [-10, 10]
    # Angles: [0, 360]
    # Outer Center: [-5, 5]
    # Outer Angle: [0, 60] (Hexagon symmetry)
    
    bounds = []
    for _ in range(2*N):
        bounds.append((-10.0, 10.0))
    for _ in range(N):
        bounds.append((0.0, 360.0))
    for _ in range(2):
        bounds.append((-5.0, 5.0))
    bounds.append((0.0, 60.0))
    
    # Initial Population
    pop_size = 50
    population = []
    
    # Add lattice solution
    population.append(lattice_params.copy())
    
    # Add random solutions
    for i in range(pop_size - 1):
        if time.time() > DEADLINE - 20.0:
            break
        p = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        # Enforce symmetry? No, let DE find it.
        # But we can reduce search space by fixing outer center to (0,0) initially?
        # No, the optimal outer center might not be (0,0) if the cluster is asymmetric.
        # However, for a symmetric cluster, it is.
        # Let's allow it to move.
        population.append(p)
    
    # Differential Evolution Parameters
    F = 0.8
    CR = 0.9
    tol = 1e-6
    max_iter = 200
    
    # DE Loop
    for gen in range(max_iter):
        if time.time() > DEADLINE - 10.0:
            break
            
        # Sort population by score
        # We need to evaluate the population first if not done
        # In this simple loop, we evaluate candidates generated in previous step
        # Let's use a standard DE loop structure
        
        # 1. Evaluate current population if needed (simplification: evaluate all at start of gen)
        # Actually, standard DE:
        # 1. Generate trial vectors
        # 2. Evaluate trial vectors
        # 3. Select
        
        # We need scores for the current population.
        # Let's maintain a list of (params, score)
        
        if gen == 0:
            scored_pop = []
            for p in population:
                score, valid = _evaluate_solution(p)
                scored_pop.append((p, score))
            scored_pop.sort(key=lambda x: x[1])
            best_params, best_score = scored_pop[0]
            
        else:
            # Generate trial vectors
            new_scored_pop = []
            # Keep best
            new_scored_pop.append(scored_pop[0])
            
            for i in range(1, pop_size):
                # Select 3 distinct random indices
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = scored_pop[r1][0] + F * (scored_pop[r2][0] - scored_pop[r3][0])
                
                # Crossover
                trial = np.zeros_like(v)
                j_rand = np.random.randint(0, len(v))
                for j in range(len(v)):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = scored_pop[i][0][j]
                
                # Bounds check
                for j in range(len(trial)):
                    if trial[j] < bounds[j][0]:
                        trial[j] = bounds[j][0]
                    elif trial[j] > bounds[j][1]:
                        trial[j] = bounds[j][1]
                
                # Evaluate trial
                score, valid = _evaluate_solution(trial)
                if not valid:
                    score = np.inf
                
                # Selection
                if score < scored_pop[i][1]:
                    new_scored_pop.append((trial, score))
                else:
                    new_scored_pop.append(scored_pop[i])
            
            scored_pop = new_scored_pop
            scored_pop.sort(key=lambda x: x[1])
            best_params, best_score = scored_pop[0]
            
            # Convergence check
            if gen > 10:
                if abs(scored_pop[0][1] - scored_pop[-1][1]) < 1e-8:
                    # If best and worst are similar, we might be stuck
                    # Or if best hasn't improved in many generations
                    pass # Simple check omitted for brevity, DE usually handles this
    
    # Final Refinement: Use L-BFGS-B on the best DE solution to polish
    if time.time() < DEADLINE - 5.0 and best_params is not None:
        def _objective_refine(params):
            score, valid = _evaluate_solution(params)
            if not valid:
                return 1e6 + np.sum((params - best_params)**2)
            return score
        
        try:
            from scipy.optimize import minimize
            res = minimize(_objective_refine, best_params, method='L-BFGS-B',
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Extract best solution
    if best_params is None:
        # Fallback to lattice
        best_params = lattice_params
        
    N = 11
    pos_best = best_params[:2*N].reshape(N, 2)
    ang_best = best_params[2*N:3*N]
    outer_center = best_params[3*N:3*N+2]
    outer_angle = best_params[3*N+2]
    
    inner_list = [[float(pos_best[i,0]), float(pos_best[i,1]), float(ang_best[i])] for i in range(N)]
    final_side = _min_outer_side_exact(pos_best, ang_best, outer_center, outer_angle)
    
    # Final validity check with the computed side
    if not _check_validity(inner_list, tuple(outer_center), final_side, outer_angle):
        # Fallback to lattice if invalid (should not happen if DE was careful)
        pos_f, ang_f = _initial_configuration()
        inner_list = [[float(pos_f[i,0]), float(pos_f[i,1]), float(ang_f[i])] for i in range(N)]
        final_side = _min_outer_side_exact(pos_f, ang_f, (0.0, 0.0), 0.0)
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        
    return inner_list, tuple(outer_center), float(final_side), float(outer_angle)

def _initial_configuration():
    """
    Generate a specific initial configuration for 11 hexagons.
    Topology:
    - Center: (0,0)
    - Ring 1: 6 hexagons at distance 2*sqrt(3) from center.
    - Ring 2: 4 hexagons at distance 6 from center, in the gaps.
    """
    sqrt3 = np.sqrt(3)
    pos = np.zeros((11, 2))
    ang = np.zeros(11)
    
    # 0: Center
    pos[0] = [0, 0]
    
    # 1-6: Ring 1
    for i in range(6):
        a = i * np.pi / 3
        pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        
    # 7-10: Ring 2
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        a = ring2_angles[i]
        pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        
    return pos, ang

if __name__ == "__main__":
    construct()
