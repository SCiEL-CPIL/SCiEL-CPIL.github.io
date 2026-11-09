"""
Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length.
Strategy: Simulated Annealing with hard constraints and rigid cluster moves.
"""
import math
import time
import numpy as np

N = 11
_DEADLINE = 145.0

def _hex_inradius(s):
    return s * math.cos(math.radians(30))

def _point_in_hex(px, py, cx, cy, s, rot):
    """Check if point (px, py) is inside a hexagon centered at (cx, cy) with side s and rotation rot."""
    dx = px - cx
    dy = py - cy
    ca, sa = math.cos(-rot), math.sin(-rot)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    r_in = _hex_inradius(s)
    c30 = math.cos(math.radians(30))
    d1 = abs(c30 * lx + 0.5 * ly)
    d2 = abs(ly)
    d3 = abs(-c30 * lx + 0.5 * ly)
    return max(d1, d2, d3) <= r_in + 1e-9

def _hex_verts(cx, cy, s, rot):
    vs = np.empty((6, 2))
    for k in range(6):
        a = math.radians(60.0 * k) + rot
        vs[k, 0] = cx + s * math.cos(a)
        vs[k, 1] = cy + s * math.sin(a)
    return vs

def _is_valid_config(inner, center, side, angle):
    """Strictly validate the configuration."""
    cx, cy = center
    s_out = side
    r_out = _hex_inradius(s_out)
    
    # 1. Check boundary containment
    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        verts = _hex_verts(xi, yi, 1.0, ai)
        for v in verts:
            if not _point_in_hex(v[0], v[1], cx, cy, s_out, angle):
                return False

    # 2. Check pairwise disjointness
    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        vi = _hex_verts(xi, yi, 1.0, ai)
        for j in range(i + 1, len(inner)):
            xj, yj, aj_deg = inner[j]
            aj = math.radians(aj_deg)
            vj = _hex_verts(xj, yj, 1.0, aj)
            
            # Quick reject based on center distance
            dc = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dc > 2.0:
                continue
            
            # Check if any vertex of i is in j
            overlap = False
            for v in vi:
                if _point_in_hex(v[0], v[1], xj, yj, 1.0, aj):
                    overlap = True
                    break
            if overlap:
                return False
            
            # Check if any vertex of j is in i
            for v in vj:
                if _point_in_hex(v[0], v[1], xi, yi, 1.0, ai):
                    overlap = True
                    break
            if overlap:
                return False
                
    return True

def _honeycomb_positions():
    """Generate honeycomb packing positions for 11 hexagons."""
    positions = []
    positions.append((0.0, 0.0, 0.0))
    for k in range(6):
        a = math.radians(60.0 * k + 30.0)
        positions.append((2.0*math.cos(a), 2.0*math.sin(a), 0.0))
    for k in range(4):
        a = math.radians(60.0 * k + 30.0)
        positions.append((4.0*math.cos(a), 4.0*math.sin(a), 0.0))
    return positions[:N]

def construct():
    t_start = time.time()
    n = N
    
    def _remaining():
        return _DEADLINE - (time.time() - t_start)

    # Initial Setup
    # We define the structure:
    # Hex 0: Center
    # Hex 1-6: Surrounding ring
    # Hex 7-10: Outer ring (4 of them)
    
    # Parameters:
    # x[0], y[0], a[0]
    # x[1..6], y[1..6], a[1..6]
    # x[7..10], y[7..10], a[7..10]
    # S: Outer side length
    
    # We will optimize S implicitly by checking validity and minimizing S.
    # Actually, for SA, it's better to fix S and try to fit, or fix positions and find min S.
    # Let's fix S and minimize "validity" (0 if valid, 1 if invalid) is not differentiable.
    # Better: Minimize S directly, but reject invalid moves.
    # Initial S estimate from honeycomb.
    
    init_pos = _honeycomb_positions()
    
    max_r = 0.0
    for (xi, yi, ai) in init_pos:
        verts = _hex_verts(xi, yi, 1.0, ai)
        for v in verts:
            r = math.sqrt(v[0]**2 + v[1]**2)
            if r > max_r:
                max_r = r
    
    S_init = max_r / math.cos(math.radians(30)) * 1.02
    
    # State representation:
    # state = [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10, S]
    
    state = np.zeros(3*n + 1)
    for i in range(n):
        state[3*i] = init_pos[i][0]
        state[3*i+1] = init_pos[i][1]
        state[3*i+2] = init_pos[i][2]
    state[3*n] = S_init
    
    best_state = state.copy()
    best_S = S_init
    
    # Helper to extract positions
    def get_positions(s):
        return [(s[3*i], s[3*i+1], math.degrees(s[3*i+2])) for i in range(n)]
    
    # Helper to check validity and return (is_valid, min_violation_metric)
    # For SA, we just need boolean validity.
    
    def is_valid(s):
        inner = get_positions(s)
        return _is_valid_config(inner, (0.0, 0.0), s[3*n], 0.0)
    
    # Initial validity check
    if not is_valid(state):
        # Expand S until valid
        while not is_valid(state) and state[3*n] < 10.0:
            state[3*n] += 0.05
        best_state = state.copy()
        best_S = state[3*n]
    
    # Simulated Annealing
    # Temperature schedule
    T_start = 0.5
    T_end = 0.001
    steps = 5000 # Adjust based on time
    # Actually, let's do fixed time steps.
    
    rng = np.random.RandomState(42)
    
    current_S = state[3*n]
    
    # Move strategies:
    # 1. Move single hexagon (x, y, angle)
    # 2. Move cluster (1-6) as rigid body (dx, dy, dtheta)
    # 3. Move outer ring (7-10) individually
    # 4. Shrink S
    
    for step in range(100000): # Max steps, break on time
        if _remaining() < 5.0:
            break
            
        # Generate new state
        new_state = state.copy()
        move_type = rng.randint(0, 10)
        
        if move_type < 3:
            # Move center hex
            i = 0
            new_state[3*i] += rng.randn() * 0.1
            new_state[3*i+1] += rng.randn() * 0.1
            new_state[3*i+2] += rng.randn() * 0.05
        elif move_type < 7:
            # Move one of the ring hexagons (1-6)
            i = rng.randint(1, 7)
            new_state[3*i] += rng.randn() * 0.1
            new_state[3*i+1] += rng.randn() * 0.1
            new_state[3*i+2] += rng.randn() * 0.05
        elif move_type < 9:
            # Move one of the outer hexagons (7-10)
            i = rng.randint(7, 11)
            new_state[3*i] += rng.randn() * 0.2
            new_state[3*i+1] += rng.randn() * 0.2
            new_state[3*i+2] += rng.randn() * 0.1
        else:
            # Rigid body move for ring 1-6
            # Rotate around center (0,0) and translate
            dx = rng.randn() * 0.1
            dy = rng.randn() * 0.1
            dtheta = rng.randn() * 0.02
            
            for i in range(1, 7):
                x, y, a = new_state[3*i], new_state[3*i+1], new_state[3*i+2]
                # Translate
                x += dx
                y += dy
                # Rotate around origin
                ca, sa = math.cos(dtheta), math.sin(dtheta)
                nx = ca * x - sa * y
                ny = sa * x + ca * y
                new_state[3*i] = nx
                new_state[3*i+1] = ny
                new_state[3*i+2] = a + dtheta
        
        # Try to shrink S slightly
        if rng.rand() < 0.2:
            new_state[3*n] = max(1.5, new_state[3*n] * (1.0 - 0.001 * rng.rand()))
        
        # Check validity
        if is_valid(new_state):
            # Accept if S is smaller, or with probability exp(-dE/T)
            # Here "Energy" is S. We want to minimize S.
            dE = new_state[3*n] - state[3*n]
            if dE < 0:
                state = new_state
                if state[3*n] < best_S:
                    best_S = state[3*n]
                    best_state = state.copy()
            else:
                # Accept bad S change if it helps configuration? 
                # No, S is the objective. If S increases, we only accept if it allows future shrinking.
                # But since we reject invalid, and valid configs with larger S are worse, 
                # we should rarely accept S increase. 
                # However, sometimes a "bad" S increase allows a much better packing that allows larger S decrease later.
                # Let's accept with probability exp(-dE / T)
                T = T_start * (T_end / T_start) ** (step / 10000.0)
                if rng.rand() < math.exp(-dE / T):
                    state = new_state
        else:
            # Invalid move: reject
            pass
            
    # Final polish: Try to minimize S from best_state
    # Use binary search on S? No, positions depend on S.
    # Just do a few more random shuffles with strict S minimization.
    
    final_state = best_state.copy()
    for i in range(1000):
        if _remaining() < 2.0:
            break
        new_state = final_state.copy()
        # Random perturbation
        idx = rng.randint(0, 3*n)
        new_state[idx] += rng.randn() * 0.05
        # Try to shrink S
        new_state[3*n] = final_state[3*n] * 0.999
        
        if is_valid(new_state):
            final_state = new_state
        else:
            # If invalid, maybe just shrink S a tiny bit more from current valid state?
            # Actually, if we shrink S and it becomes invalid, we can't keep it.
            # So we only update if valid.
            pass
            
    # Ensure final result is valid
    inner = get_positions(final_state)
    S_final = final_state[3*n]
    
    if _is_valid_config(inner, (0.0, 0.0), S_final, 0.0):
        return inner, (0.0, 0.0), float(S_final), 0.0
    else:
        # Fallback to best_state if final_state somehow invalid (should not happen)
        inner = get_positions(best_state)
        S_final = best_state[3*n]
        if _is_valid_config(inner, (0.0, 0.0), S_final, 0.0):
            return inner, (0.0, 0.0), float(S_final), 0.0
        
        # Absolute fallback: Honeycomb with large margin
        inner_init = [[float(p[0]), float(p[1]), float(math.degrees(p[2]))] for p in init_pos]
        return inner_init, (0.0, 0.0), float(S_init * 1.1), 0.0
