import numpy as np
import time
import math

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 75  # Safety margin for 90s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    e_count = len(edges)
    
    # Precompute edge structure
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle_edges.append(i)
        elif u >= m and v >= m:
            bottom_cycle_edges.append(i)
        else:
            matching_edges.append(i)
    
    def compute_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))
    
    def compute_smooth_objective(signs):
        """Compute sum of squared eigenvalues (smooth proxy for spectral radius)."""
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        eigvals = np.linalg.eigvalsh(A)
        return float(np.sum(eigvals ** 2))
    
    def is_valid_lift(signs):
        """Check if the lift is connected using cycle product method."""
        # The lift is connected iff the product of signs around each
        # "wrapping" cycle is -1 (i.e., odd number of -1 signs).
        # For prism graphs: top cycle, bottom cycle, and any matching+cycle path.
        
        # Top cycle product
        top_prod = 1
        for i in top_cycle_edges:
            top_prod *= signs[i]
        
        # Bottom cycle product
        bottom_prod = 1
        for i in bottom_cycle_edges:
            bottom_prod *= signs[i]
        
        # For the prism, the lift is connected iff:
        # - The product around the top cycle is -1 (odd # of -1s), OR
        # - The product around the bottom cycle is -1, OR  
        # - Both are +1 but there's an odd number of -1s on matching edges
        # Actually, the correct condition: the lift is connected iff the 
        # "winding number" is odd. For a prism with two m-cycles and matching:
        # The lift is connected iff the product of signs along any cycle that
        # wraps around the "torus" direction is -1.
        
        # More precisely: the 2-lift of a graph is connected iff the graph
        # has at least one cycle with an odd number of -1 edges.
        # For prism: top cycle and bottom cycle are the two independent cycles
        # (plus the matching). The lift is connected iff at least one of the
        # fundamental cycles has an odd number of -1 signs.
        
        # Top cycle: odd number of -1s
        top_odd = (top_prod == -1)
        # Bottom cycle: odd number of -1s  
        bottom_odd = (bottom_prod == -1)
        
        # If either cycle has odd # of -1s, the lift is connected
        if top_odd or bottom_odd:
            return True
        
        # If both cycles have even # of -1s, check if the combined structure
        # still gives connectivity. The matching edges connect the two cycles.
        # The lift is connected iff the total "winding" is non-trivial.
        # With both cycles even, we need to check the product around a path
        # that goes: top cycle -> matching -> bottom cycle -> matching back.
        # This is equivalent to checking if the product of all matching signs
        # combined with the cycle parities gives an odd total.
        
        # Actually for a prism graph (two cycles + perfect matching), the
        # fundamental cycles are: top cycle, bottom cycle, and for each i:
        # (i, i+1) on top, (i+1, i+1+m) matching, (i+1+m, i+m) on bottom,
        # (i+m, i) matching. The product around this "rectangle" cycle is:
        # sign(top_i) * sign(match_i+1) * sign(bottom_i) * sign(match_i)
        
        # The lift is connected iff at least one fundamental cycle has odd product.
        # We already checked top and bottom cycles. Now check one rectangle cycle:
        if len(matching_edges) >= 1 and len(top_cycle_edges) >= 1 and len(bottom_cycle_edges) >= 1:
            # Find matching edges indexed by their top vertex
            match_by_top = {}
            for i in matching_edges:
                u, v = edges[i]
                if u < m:
                    match_by_top[u] = i
                else:
                    match_by_top[v] = i  # v < m
            
            # Check rectangle cycles for a few positions
            for k in range(min(m, len(match_by_top))):
                if k not in match_by_top or (k + 1) % m not in match_by_top:
                    continue
                # Top edge from k to (k+1)%m
                top_edge_idx = None
                for i in top_cycle_edges:
                    u, v = edges[i]
                    if (u == k and v == (k+1) % m) or (u == (k+1) % m and v == k):
                        top_edge_idx = i
                        break
                if top_edge_idx is None:
                    continue
                
                # Bottom edge from k+m to (k+1+m)%n
                bot_edge_idx = None
                for i in bottom_cycle_edges:
                    u, v = edges[i]
                    u_mod = u - m if u >= m else u
                    v_mod = v - m if v >= m else v
                    if (u_mod == k and v_mod == (k+1) % m) or (u_mod == (k+1) % m and v_mod == k):
                        bot_edge_idx = i
                        break
                if bot_edge_idx is None:
                    continue
                
                rect_prod = signs[top_edge_idx] * signs[match_by_top[(k+1) % m]] * signs[bot_edge_idx] * signs[match_by_top[k]]
                if rect_prod == -1:
                    return True
        
        # If we get here, all fundamental cycles have even product -> disconnected
        return False
    
    def generate_phase_signs(phases):
        """Convert complex unit phases to edge signs."""
        signs = np.zeros(e_count, dtype=int)
        for i, (u, v) in enumerate(edges):
            val = phases[u] * np.conj(phases[v])
            real_part = np.real(val)
            signs[i] = 1 if real_part >= 0 else -1
        return signs
    
    def optimize_phases(init_phases, max_iters=3000, lr=0.05):
        """Gradient descent on smooth objective using phase parameterization."""
        phases = init_phases.copy()
        best_obj = float('inf')
        best_phases = phases.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute current signs and smooth objective
            signs = generate_phase_signs(np.exp(1j * phases))
            obj = compute_smooth_objective(signs)
            
            if obj < best_obj:
                best_obj = obj
                best_phases = phases.copy()
            
            # Compute gradient via finite differences
            grad = np.zeros(n)
            eps = 0.05
            for v in range(n):
                phases_plus = phases.copy()
                phases_plus[v] += eps
                signs_plus = generate_phase_signs(np.exp(1j * phases_plus))
                obj_plus = compute_smooth_objective(signs_plus)
                grad[v] = (obj_plus - obj) / eps
            
            # Gradient descent step
            phases -= lr * grad
            # Wrap phases to [0, 2*pi)
            phases = phases % (2 * math.pi)
            
            # Adaptive learning rate
            if it % 200 == 0:
                lr *= 0.92
        
        return best_phases, best_obj
    
    # Generate diverse initial phase configurations
    initial_phases_list = []
    
    # 1. All zeros (all +1 signs)
    initial_phases_list.append(np.zeros(n))
    
    # 2. Alternating phases
    phases_alt = np.zeros(n)
    for v in range(n):
        phases_alt[v] = math.pi if v % 2 == 1 else 0
    initial_phases_list.append(phases_alt)
    
    # 3. Top/bottom split
    phases_split = np.zeros(n)
    for v in range(m, 2 * m):
        phases_split[v] = math.pi
    initial_phases_list.append(phases_split)
    
    # 4. Random phases
    rng = np.random.default_rng(42)
    for _ in range(8):
        initial_phases_list.append(rng.uniform(0, 2 * math.pi, n))
    
    # 5. Structured phases based on cycle position
    phases_cycle = np.zeros(n)
    for v in range(m):
        phases_cycle[v] = 2 * math.pi * v / m
        phases_cycle[v + m] = 2 * math.pi * (v + 1) / m
    initial_phases_list.append(phases_cycle)
    
    # 6. Phases that encourage odd cycles
    phases_odd = np.zeros(n)
    for v in range(m):
        phases_odd[v] = math.pi * v / m
    initial_phases_list.append(phases_odd)
    
    # Run phase optimization from each initialization
    best_signs = np.ones(e_count, dtype=int)
    best_radius = float('inf')
    
    for init_phases in initial_phases_list:
        if time.time() > deadline:
            break
        
        opt_phases, opt_obj = optimize_phases(init_phases, max_iters=800, lr=0.05)
        
        signs_from_phases = generate_phase_signs(np.exp(1j * opt_phases))
        
        # Only consider valid lifts
        if is_valid_lift(signs_from_phases):
            r = compute_radius(signs_from_phases)
            if r < best_radius:
                best_radius = r
                best_signs = signs_from_phases
    
    # If no valid lift found, try to find one
    if not is_valid_lift(best_signs):
        # Try random valid configurations
        for attempt in range(200):
            if time.time() > deadline:
                break
            random_signs = rng.choice([-1, 1], size=e_count)
            if is_valid_lift(random_signs):
                r = compute_radius(random_signs)
                if r < best_radius:
                    best_radius = r
                    best_signs = random_signs.copy()
    
    # Discrete local search to polish using exact spectral radius
    current_signs = best_signs.copy()
    current_radius = best_radius
    no_improve_count = 0
    
    while time.time() < deadline and no_improve_count < 50:
        improved = False
        
        for edge_idx in range(e_count):
            if time.time() > deadline:
                break
            
            new_signs = current_signs.copy()
            new_signs[edge_idx] = -new_signs[edge_idx]
            
            if not is_valid_lift(new_signs):
                continue
                
            try:
                r = compute_radius(new_signs)
                if r < current_radius - 1e-10:
                    current_signs = new_signs
                    current_radius = r
                    best_signs = new_signs.copy()
                    best_radius = r
                    improved = True
                    no_improve_count = 0
                    break
            except Exception:
                pass
        
        if not improved:
            no_improve_count += 1
            # Try 2-flip moves periodically
            if no_improve_count % 3 == 0:
                for i in range(e_count):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, e_count):
                        if time.time() > deadline:
                            break
                        new_signs = current_signs.copy()
                        new_signs[i] = -new_signs[i]
                        new_signs[j] = -new_signs[j]
                        
                        if not is_valid_lift(new_signs):
                            continue
                            
                        try:
                            r = compute_radius(new_signs)
                            if r < current_radius - 1e-10:
                                current_signs = new_signs
                                current_radius = r
                                best_signs = new_signs.copy()
                                best_radius = r
                                improved = True
                                no_improve_count = 0
                                break
                        except Exception:
                            pass
                    if improved:
                        break
    
    return best_signs
