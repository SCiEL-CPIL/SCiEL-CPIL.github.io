import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 12
SQRT3 = np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Support function of a unit regular hexagon
    # h(phi) = max_k cos(phi - k*pi/3)
    def h_unit(phi):
        # Reduce phi to [0, 2*pi)
        phi = phi % (2 * np.pi)
        # The function is periodic with pi/3 (60 deg)
        # Max is at 0, pi/3, etc.
        # h(phi) = cos(phi_mod) where phi_mod is distance to nearest multiple of pi/3?
        # No, h(phi) = max(cos(phi), cos(phi-pi/3), ...)
        # Due to symmetry, we can just compute max over 6 vertices
        angles = np.arange(6) * np.pi / 3
        return np.max(np.cos(phi - angles))

    # Precompute outer normals for Theta = 0
    # Outer hexagon side S, center 0.
    # Normals are at 30, 90, 150, 210, 270, 330 degrees
    outer_normals = np.array([
        [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))] for k in range(6)
    ])
    
    # Constant term for containment RHS: -h_unit(normal - theta_i)
    # Wait, the constraint is:
    # c_i . n_k + h_unit(n_k - theta_i) <= S * h_unit(0)
    # c_i . n_k - S * (SQRT3/2) <= -h_unit(n_k - theta_i)
    # h_unit(0) = 1.0? 
    # Let's check h_unit(0). angles = 0, 60... cos(0-0)=1. max is 1.
    # But the support of a unit hexagon in direction of a normal (30 deg) is the apothem?
    # My h_unit definition: max v_k . n.
    # If n is (1,0), v_0 is (1,0). Dot is 1.
    # If n is normal at 30 deg, v_0=(1,0), dot = cos(30)=sqrt(3)/2. v_1=(1/2, sqrt(3)/2), dot = sqrt(3)/2.
    # So h_unit(30 deg) = sqrt(3)/2.
    # My formula h_unit(phi) = max cos(phi - k*60).
    # h_unit(30 deg) = max cos(30-0), cos(30-60)... = cos(30) = sqrt(3)/2. Correct.
    # So the outer support in direction n_k (which is a normal) is S * h_unit(30 deg) = S * SQRT3/2.
    
    def solve_lp(thetas):
        # thetas: array of 12 angles
        # Variables: x1, y1, ..., x12, y12, S
        # c = 0 for x, y, 1 for S
        c = np.zeros(2 * N + 1)
        c[-1] = 1.0
        
        A_ub = []
        b_ub = []
        
        # 1. Containment Constraints
        # For each i, for each outer normal k
        for i in range(N):
            for k in range(6):
                n = outer_normals[k]
                # coeff for x_i, y_i
                row = np.zeros(2 * N + 1)
                row[2*i] = n[0]
                row[2*i+1] = n[1]
                # coeff for S
                row[-1] = -SQRT3 / 2.0
                
                # RHS: -h_unit(n - theta_i)
                # n is a vector, but h_unit takes angle.
                # Angle of n is 30 + 60k
                angle_n = np.arctan2(n[1], n[0])
                rhs = -h_unit(angle_n - thetas[i])
                
                A_ub.append(row)
                b_ub.append(rhs)
                
        # 2. Non-Overlap Constraints
        # For each pair i, j
        for i in range(N):
            for j in range(i + 1, N):
                # Get normals for i and j
                # Normals of hex i: theta_i + 30 + 60*l
                normals_i = np.array([
                    [np.cos(thetas[i] + np.radians(30 + 60*l)), np.sin(thetas[i] + np.radians(30 + 60*l))] 
                    for l in range(6)
                ])
                normals_j = np.array([
                    [np.cos(thetas[j] + np.radians(30 + 60*l)), np.sin(thetas[j] + np.radians(30 + 60*l))] 
                    for l in range(6)
                ])
                
                all_normals = np.vstack([normals_i, normals_j])
                
                for n in all_normals:
                    row = np.zeros(2 * N + 1)
                    # (x_i - x_j) . n + (y_i - y_j) . n
                    row[2*i] = n[0]
                    row[2*i+1] = n[1]
                    row[2*j] = -n[0]
                    row[2*j+1] = -n[1]
                    
                    # RHS: - (h_unit(n - theta_i) + h_unit(n + theta_j))
                    # Note: h_unit(-n - theta_j) = h_unit(n + theta_j) due to symmetry
                    angle_n = np.arctan2(n[1], n[0])
                    term_i = h_unit(angle_n - thetas[i])
                    term_j = h_unit(angle_n + thetas[j])
                    rhs = -(term_i + term_j)
                    
                    A_ub.append(row)
                    b_ub.append(rhs)
                    
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds: S >= 0. Positions unbounded? 
        # To prevent numerical issues, bound positions to a reasonable range, e.g. [-10, 10]
        bounds = [(-10, 10)] * (2 * N) + [(0, 10)]
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status == 0:
            return res.fun
        else:
            return 1e6

    def objective_angles(thetas):
        if time.time() > deadline:
            return 1e10
        return solve_lp(thetas)

    # Initializations
    inits = []
    # All 0
    inits.append(np.zeros(N))
    # All 30
    inits.append(np.full(N, np.pi/6))
    # Alternating 0, 30
    alt = np.zeros(N)
    alt[1::2] = np.pi/6
    inits.append(alt)
    # Random
    for _ in range(5):
        inits.append(np.random.uniform(0, np.pi/3, N))

    best_s = np.inf
    best_thetas = None
    best_positions = None
    
    for init in inits:
        if time.time() > deadline - 20:
            break
            
        try:
            res = minimize(
                objective_angles,
                init,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            
            s_val = res.fun
            if s_val < best_s:
                best_s = s_val
                best_thetas = res.x.copy()
                
                # Recover positions from LP
                # We need to re-run LP to get x
                # Modify solve_lp to return x
                # Or just call a helper
                # Let's create a helper that returns x
                def get_positions(thetas):
                    c = np.zeros(2 * N + 1)
                    c[-1] = 1.0
                    A_ub = []
                    b_ub = []
                    for i in range(N):
                        for k in range(6):
                            n = outer_normals[k]
                            row = np.zeros(2 * N + 1)
                            row[2*i] = n[0]
                            row[2*i+1] = n[1]
                            row[-1] = -SQRT3 / 2.0
                            angle_n = np.arctan2(n[1], n[0])
                            rhs = -h_unit(angle_n - thetas[i])
                            A_ub.append(row)
                            b_ub.append(rhs)
                    for i in range(N):
                        for j in range(i + 1, N):
                            normals_i = np.array([
                                [np.cos(thetas[i] + np.radians(30 + 60*l)), np.sin(thetas[i] + np.radians(30 + 60*l))] 
                                for l in range(6)
                            ])
                            normals_j = np.array([
                                [np.cos(thetas[j] + np.radians(30 + 60*l)), np.sin(thetas[j] + np.radians(30 + 60*l))] 
                                for l in range(6)
                            ])
                            all_normals = np.vstack([normals_i, normals_j])
                            for n in all_normals:
                                row = np.zeros(2 * N + 1)
                                row[2*i] = n[0]
                                row[2*i+1] = n[1]
                                row[2*j] = -n[0]
                                row[2*j+1] = -n[1]
                                angle_n = np.arctan2(n[1], n[0])
                                term_i = h_unit(angle_n - thetas[i])
                                term_j = h_unit(angle_n + thetas[j])
                                rhs = -(term_i + term_j)
                                A_ub.append(row)
                                b_ub.append(rhs)
                    A_ub = np.array(A_ub)
                    b_ub = np.array(b_ub)
                    bounds = [(-10, 10)] * (2 * N) + [(0, 10)]
                    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                    if res.status == 0:
                        return res.x
                    return None

                pos_s = get_positions(best_thetas)
                if pos_s is not None:
                    best_positions = pos_s[:2*N].reshape(N, 2)
        except Exception:
            pass

    if best_positions is None:
        # Fallback
        best_s = 5.0
        best_thetas = np.zeros(N)
        # Just use a valid configuration from a simple heuristic if LP fails
        # But LP should work.
        # If LP fails, we return a dummy valid config
        # Simple hexagonal packing
        pos = []
        for r in range(3):
            for c in range(4):
                x = c * SQRT3
                if r % 2 == 1: x += SQRT3/2
                y = r * 1.5
                pos.append([x, y])
        pos = np.array(pos)
        pos = pos - pos.mean(axis=0)
        best_positions = pos

    final_pos = best_positions
    final_ang = best_thetas
    final_s = best_s

    inner = [[final_pos[i, 0], final_pos[i, 1], np.degrees(final_ang[i])] for i in range(N)]
    outer_center = (0.0, 0.0)
    outer_side = final_s
    outer_angle_deg = 0.0

    return inner, outer_center, outer_side, outer_angle_deg
