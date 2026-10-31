"""Circle packing in perimeter-4 rectangle: Topological Contact Graph Solver."""
import numpy as np
import time
import itertools
from scipy.optimize import fsolve

N = 21

def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_value = 0.0
    best_result = None

    def solve_contact_graph(x_init, y_init, r_init, w_init, graph, wall_contacts):
        """
        Solve for exact positions and radii given a fixed contact graph and wall contacts.
        Variables: x, y, r, w.
        Equations:
        1. d_ij = r_i + r_j for each edge (i,j) in graph
        2. x_i = r_i if 'L' in wall_contacts[i]
        3. x_i = w - r_i if 'R' in wall_contacts[i]
        4. y_i = r_i if 'B' in wall_contacts[i]
        5. y_i = h - r_i if 'T' in wall_contacts[i]
        
        We need exactly 3N + 1 equations for 3N + 1 variables.
        If we have fewer constraints, the system is underdetermined.
        We fix the aspect ratio w to a specific value to close the system, 
        or we can add a "dummy" constraint that minimizes energy? 
        No, fsolve needs a square system.
        
        Strategy: 
        We fix w to a value. 
        We need 3N - (#wall_constraints) equations from the graph.
        Total variables: 3N + 1 (x, y, r, w).
        If we fix w, we have 3N variables.
        We need 3N equations.
        #wall_constraints + #graph_edges = 3N.
        
        If #graph_edges < 3N - #wall_constraints, we are underdetermined.
        We can add "soft" constraints by fixing some radii or positions?
        Better: Use a penalty method inside fsolve? No, fsolve is for roots.
        
        Alternative: Use scipy.optimize.least_squares if underdetermined, but we want exact tangency.
        Actually, for a "rigid" packing, the number of contacts is typically 3N - 3 - C_boundary.
        For 21 circles, we need roughly 3*21 = 63 equations.
        A Delaunay triangulation of 21 points has ~3*21 - 3 - h edges, where h is hull size.
        If h ~ 6, edges ~ 63 - 3 - 6 = 54.
        Plus wall contacts.
        
        Let's assume the graph is "rigid" enough. If not, we can add constraints that 
        specific circles touch specific walls (which we can infer or try).
        
        To make it robust:
        1. Build graph from current positions (Delaunay-like: connect if close).
        2. Determine wall contacts from current positions.
        3. If equations < variables, add constraints:
           - Fix w to w_init.
           - Fix some radii? No, we want to maximize sum(r).
           
        Actually, a better approach for "exact" solving in an underdetermined system:
        We can solve for the configuration that satisfies the contact constraints AND minimizes 
        the deviation from the initial guess? No, that's not exact tangency.
        
        Let's stick to: 
        1. Identify contacts from a "relaxed" state.
        2. Solve the system. If it fails, relax the graph.
        
        For this implementation, I will:
        1. Run a quick force relaxation to get a good initial guess and identify a stable contact graph.
        2. Extract the graph and wall contacts.
        3. Solve the non-linear system.
        4. If the solution is valid, use it.
        """
        
        # We need to form the residual function
        # Variables: z = [x_1...x_N, y_1...y_N, r_1...r_N, w]
        # Size: 3N + 1
        
        def residual(z):
            x = z[0:N]
            y = z[N:2*N]
            r = z[2*N:3*N]
            w = z[3*N]
            h = 2.0 - w
            
            res = []
            
            # Wall constraints
            for i in range(N):
                wc = wall_contacts[i]
                if 'L' in wc:
                    res.append(x[i] - r[i])
                if 'R' in wc:
                    res.append(x[i] + r[i] - w)
                if 'B' in wc:
                    res.append(y[i] - r[i])
                if 'T' in wc:
                    res.append(y[i] + r[i] - h)
            
            # Graph constraints
            for i, j in graph:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                res.append(d - (r[i] + r[j]))
                
            # If we have too few constraints, we need more.
            # We can add constraints that non-contacting circles are "far enough"? 
            # No, that's an inequality.
            # We can add constraints that fix the aspect ratio w?
            # If len(res) < 3N+1, we are underdetermined.
            # We can add "anchor" constraints: e.g., fix x_0=0, y_0=0? 
            # But we want to find the optimal w.
            
            # To handle underdetermination, we will add constraints that 
            # the circles are as "spread out" as possible? 
            # Actually, if the graph is rigid, it should be determined.
            # If not, we can fix w to a value.
            
            # For now, if underdetermined, we append zeros? No, fsolve will fail or find a trivial solution.
            # We must ensure we have enough constraints.
            
            # Let's assume the graph + walls provide enough constraints.
            # If not, we return a large penalty? No, fsolve doesn't work that way.
            
            # We will return the residual as is. If it's underdetermined, fsolve might not converge.
            # We will check the count outside.
            
            return np.array(res)
        
        # Initial guess
        z0 = np.concatenate([x_init, y_init, r_init, [w_init]])
        
        # Check constraint count
        num_wall = sum(len(wc) for wc in wall_contacts)
        num_graph = len(graph)
        total_constraints = num_wall + num_graph
        total_vars = 3 * N + 1
        
        if total_constraints < total_vars:
            # Underdetermined. We need to add constraints.
            # We can fix w to w_init.
            # And fix some positions?
            # Let's just return None if underdetermined, as it's risky.
            # Or, we can add constraints that r_i > 0? No.
            # Let's try to add "dummy" constraints that are always 0? No.
            # 
            # Actually, if we fix w, we have 3N variables.
            # We need 3N constraints.
            # If num_wall + num_graph < 3N, we are still underdetermined.
            # 
            # Strategy: If underdetermined, we don't use fsolve. 
            # We use the force relaxation result directly.
            return None
            
        # Solve
        sol, info, ier, msg = fsolve(residual, z0, full_output=True, xtol=1e-10, maxfev=1000)
        
        if ier != 1:
            return None
            
        x_sol = sol[0:N]
        y_sol = sol[N:2*N]
        r_sol = sol[2*N:3*N]
        w_sol = sol[3*N]
        h_sol = 2.0 - w_sol
        
        # Check validity
        if not _is_valid(x_sol, y_sol, r_sol, w_sol, h_sol):
            return None
            
        val = float(np.sum(r_sol))
        return x_sol, y_sol, r_sol, w_sol, val

    def extract_graph_and_walls(x, y, r, w, h, tol=0.05):
        """Extract contact graph and wall contacts from a valid packing."""
        graph = []
        wall_contacts = [set() for _ in range(N)]
        
        # Wall contacts
        for i in range(N):
            if x[i] - r[i] < tol:
                wall_contacts[i].add('L')
            if x[i] + r[i] > w - tol:
                wall_contacts[i].add('R')
            if y[i] - r[i] < tol:
                wall_contacts[i].add('B')
            if y[i] + r[i] > h - tol:
                wall_contacts[i].add('T')
                
        # Graph contacts
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < (r[i] + r[j]) + tol:
                    graph.append((i, j))
                    
        return graph, wall_contacts

    def run_force_relaxation(z0, time_budget):
        """Standard force relaxation to get a good initial guess."""
        x = z0[0::3][:N].copy()
        y = z0[1::3][:N].copy()
        r = z0[2::3][:N].copy()
        w = z0[-1]
        
        vx = np.zeros(N)
        vy = np.zeros(N)
        vr = np.zeros(N)
        vw = 0.0
        
        dt = 0.002
        damping = 0.96
        k_rep = 300.0
        k_wall = 400.0
        k_attract = 0.00005
        k_aspect = 0.05
        r_min = 0.01
        r_max = 0.5
        w_min = 0.3
        w_max = 1.7
        
        max_iter = int(time_budget * 1000)
        
        for it in range(max_iter):
            if time.time() > time_budget:
                break
                
            h = 2.0 - w
            if h < 0.1 or w < 0.1:
                break
                
            fx = np.zeros(N)
            fy = np.zeros(N)
            fr = np.zeros(N)
            fw = 0.0
            
            # Pairwise repulsion
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            d = np.sqrt(np.maximum(d_sq, 1e-12))
            r_sum = r[:, None] + r[None, :]
            
            margin = 0.1
            active = d < (r_sum + margin)
            d_safe = np.where(d > 1e-6, d, 1e-6)
            f_mag = k_rep / (d_safe * d_safe)
            f_mag = np.where(active, f_mag, 0.0)
            
            ux = dx / d_safe
            uy = dy / d_safe
            
            fx = np.sum(f_mag * ux, axis=1)
            fy = np.sum(f_mag * uy, axis=1)
            
            # Wall repulsion
            viol = np.maximum(r - x, 0.0)
            fx -= viol * k_wall
            viol = np.maximum(x + r - w, 0.0)
            fx += viol * k_wall
            viol = np.maximum(r - y, 0.0)
            fy -= viol * k_wall
            viol = np.maximum(y + r - h, 0.0)
            fy += viol * k_wall
            
            # Attraction to grow radii
            fr[:] = k_attract
            
            # Aspect ratio penalty
            fw += k_aspect * (1.0 - w)
            
            # Integrate
            vx = vx * damping + fx * dt
            vy = vy * damping + fy * dt
            vr = vr * damping + fr * dt
            vw = vw * damping + fw * dt
            
            x += vx * dt
            y += vy * dt
            r += vr * dt
            w += vw * dt
            
            # Clamp
            x = np.clip(x, 0.0, 2.0)
            y = np.clip(y, 0.0, 2.0)
            r = np.clip(r, r_min, r_max)
            w = np.clip(w, w_min, w_max)
            
            h = 2.0 - w
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)
            
            if it % 500 == 0 and it > 1000:
                if np.max(np.abs(vx)) < 1e-5 and np.max(np.abs(vy)) < 1e-5:
                    rng = np.random.default_rng(it)
                    x += rng.uniform(-0.001, 0.001, N)
                    y += rng.uniform(-0.001, 0.001, N)
                    r += rng.uniform(-0.0005, 0.0005, N)
                    x = np.clip(x, r, w - r)
                    y = np.clip(y, r, h - r)
                    
        if not _is_valid(x, y, r, w, h):
            for scale in [0.99, 0.98, 0.95, 0.90]:
                r_trial = r * scale
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
                    break
            else:
                return None
                
        if not _is_valid(x, y, r, w, h):
            return None
            
        val = float(np.sum(r))
        return x, y, r, w, val

    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))
        
        # Structured starts
        aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.4
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)
            
        # Random starts
        rng = np.random.default_rng(123)
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.03, 0.1)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)
            
        return guesses

    guesses = get_initial_guesses()
    
    for i, z0 in enumerate(guesses):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
            
        remaining_guesses = len(guesses) - i
        budget = (deadline - 2.0 - elapsed) / remaining_guesses
        budget = max(2.0, min(budget, 12.0))
        
        try:
            # 1. Force relaxation
            res = run_force_relaxation(z0, budget)
            if res is not None:
                x, y, r, w, val = res
                h = 2.0 - w
                
                # 2. Extract graph and walls
                graph, wall_contacts = extract_graph_and_walls(x, y, r, w, h, tol=0.05)
                
                # 3. Try to solve exact contact system
                sol_res = solve_contact_graph(x, y, r, w, graph, wall_contacts)
                
                if sol_res is not None:
                    x_sol, y_sol, r_sol, w_sol, val_sol = sol_res
                    if val_sol > best_value:
                        best_value = val_sol
                        best_result = (x_sol.copy(), y_sol.copy(), r_sol.copy(), w_sol)
                else:
                    # Fallback to force relaxation result
                    if val > best_value:
                        best_value = val
                        best_result = (x.copy(), y.copy(), r.copy(), w)
                        
        except Exception:
            continue

    # Final polish on best result
    if best_result is not None and time.time() < deadline - 10:
        x, y, r, w = best_result
        h = 2.0 - w
        
        # Per-circle expansion
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                r_max = r_wall
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_allowed = d - r[j]
                    if r_allowed < r_max:
                        r_max = r_allowed
                r_max = max(0.0, r_max)
                if r_max > r[i] + 1e-6:
                    r[i] = r_max
                    improved = True
            if not improved:
                break
                
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w)

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w = best_result
    return np.column_stack((x, y, r))
