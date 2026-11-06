"""
K2,3-free graph construction on 20 vertices.
Combines MILP-based global optimization (with y=x_uw*x_vw linearization)
with bitmask-based fast common-neighbor tracking and simulated annealing
refinement using remove-and-refill moves.
"""
import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    total_pairs = len(all_pairs)
    
    # Precompute bit masks
    bit = [1 << i for i in range(n)]
    
    def popcount(x):
        return bin(x).count('1')
    
    def count_edges(adj):
        c = 0
        for i in range(n):
            c += popcount(adj[i])
        return c // 2
    
    def common_count(adj, u, v):
        """Count common neighbors of u and v."""
        return popcount(adj[u] & adj[v])
    
    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u,v) maintains K2,3-freeness.
        """
        # Check 1: u and v can't have >2 common neighbors
        if common_count(adj, u, v) > 2:
            return False
        
        # For each w in N(v), w != u: common_count(adj, u, w) <= 1
        nv = adj[v]
        while nv:
            low = nv & (-nv)
            w = low.bit_length() - 1
            if w != u:
                if common_count(adj, u, w) > 1:
                    return False
            nv ^= low
        
        # For each w in N(u), w != v: common_count(adj, v, w) <= 1
        nu = adj[u]
        while nu:
            low = nu & (-nu)
            w = low.bit_length() - 1
            if w != v:
                if common_count(adj, v, w) > 1:
                    return False
            nu ^= low
        
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= bit[v]
        adj[v] |= bit[u]
    
    def remove_edge(adj, u, v):
        adj[u] &= ~bit[v]
        adj[v] &= ~bit[u]
    
    def get_edges_list(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append((i, j))
        return edges
    
    def get_missing_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not ((adj[i] >> j) & 1):
                    edges.append((i, j))
        return edges
    
    def greedy_fill(adj, order=None):
        """Add all valid edges from the given order (or all missing)."""
        if order is None:
            order = get_missing_edges(adj)
        for u, v in order:
            if not ((adj[u] >> v) & 1):
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
    
    def adj_to_matrix(adj):
        """Convert bitmask adjacency list to numpy matrix."""
        A = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if (adj[i] >> j) & 1:
                    A[i, j] = 1
        return A
    
    def matrix_to_adj(A):
        """Convert numpy adjacency matrix to bitmask adjacency list."""
        adj = [0] * n
        for i in range(n):
            for j in range(n):
                if A[i, j] == 1:
                    adj[i] |= bit[j]
        return adj
    
    def validate_adj(adj):
        """Verify that the graph is K2,3-free."""
        for u in range(n):
            for v in range(u+1, n):
                if common_count(adj, u, v) > 2:
                    return False
        return True
    
    def solve_milp(warm_adj, time_limit):
        """Solve the linearized MILP for the densest K2,3-free graph.
        Variables: x_{uv} (edge, u<v) and y_{u,v,w} (= x_{uw}*x_{vw}).
        Constraints: y <= x_{uw}, y <= x_{vw}, y >= x_{uw}+x_{vw}-1,
                     sum_w y_{u,v,w} <= 2 for each pair (u,v).
        Returns adjacency list (bitmask) or None.
        """
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
        except Exception:
            return None
        
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        idx = {(u, v): i for i, (u, v) in enumerate(pairs)}
        num_x = len(pairs)  # 190
        
        # Auxiliary y variables: for each pair (u,v) and each w not in {u,v}
        y_info = []  # list of (y_index, u, v, w, x_uw_index, x_vw_index)
        y_index = 0
        for (u, v) in pairs:
            for w in range(n):
                if w == u or w == v:
                    continue
                y_info.append((y_index, u, v, w, idx[(min(u, w), max(u, w))], idx[(min(v, w), max(v, w))]))
                y_index += 1
        num_y = y_index
        total_vars = num_x + num_y
        
        # Objective: maximize sum x  ->  minimize -sum x
        c = np.zeros(total_vars)
        c[:num_x] = -1.0
        
        lb = np.zeros(total_vars)
        ub = np.ones(total_vars)
        
        # Build constraint matrix A_ub (inequalities)
        rows_lo = []
        rows_hi = []
        rows_vec = []
        
        def add_row(lo, hi, vec):
            rows_lo.append(lo)
            rows_hi.append(hi)
            rows_vec.append(vec)
        
        # y <= x_uw  =>  y - x_uw <= 0
        # y <= x_vw  =>  y - x_vw <= 0
        # y >= x_uw + x_vw - 1  =>  -y + x_uw + x_vw <= 1
        # sum_w y_{u,v,w} <= 2  =>  sum y <= 2
        for (yi, u, v, w, i_uw, i_vw) in y_info:
            yv = num_x + yi
            vec = np.zeros(total_vars)
            vec[yv] = 1.0
            vec[i_uw] = -1.0
            add_row(-np.inf, 0.0, vec)
            
            vec = np.zeros(total_vars)
            vec[yv] = 1.0
            vec[i_vw] = -1.0
            add_row(-np.inf, 0.0, vec)
            
            vec = np.zeros(total_vars)
            vec[yv] = -1.0
            vec[i_uw] = 1.0
            vec[i_vw] = 1.0
            add_row(-np.inf, 1.0, vec)
        
        # sum_w y_{u,v,w} <= 2 for each pair (u,v)
        for (u, v) in pairs:
            vec = np.zeros(total_vars)
            for (yi, uu, vv, w, i_uw, i_vw) in y_info:
                if uu == u and vv == v:
                    vec[num_x + yi] = 1.0
            add_row(-np.inf, 2.0, vec)
        
        A_ub = np.array(rows_vec, dtype=float)
        lb_arr = np.array(rows_lo, dtype=float)
        ub_arr = np.array(rows_hi, dtype=float)
        
        constraints = LinearConstraint(A_ub, lb_arr, ub_arr)
        bounds = Bounds(lb, ub)
        
        # Warm start: x from warm_adj, y = x_uw * x_vw
        x0 = np.zeros(total_vars)
        warm_A = adj_to_matrix(warm_adj)
        for (u, v) in pairs:
            x0[idx[(u, v)]] = warm_A[u, v]
        for (yi, u, v, w, i_uw, i_vw) in y_info:
            x0[num_x + yi] = warm_A[u, w] * warm_A[v, w]
        
        try:
            res = milp(c=c, constraints=constraints, bounds=bounds,
                       integrality=np.ones(total_vars), x0=x0,
                       options={"time_limit": time_limit, "mip_gap": 0.0,
                                "threads": 4, "presolve": "default"})
        except Exception:
            return None
        
        if res.x is None:
            return None
        
        x_sol = res.x[:num_x]
        # Round to nearest integer and rebuild adjacency
        adj = [0] * n
        for (u, v) in pairs:
            val = int(round(x_sol[idx[(u, v)]]))
            if val > 0:
                adj[u] |= bit[v]
                adj[v] |= bit[u]
        
        # Validate and fix if needed
        if not validate_adj(adj):
            # Try to repair by removing problematic edges
            changed = True
            while changed and time.time() < start_time + deadline:
                changed = False
                for u in range(n):
                    for v in range(u+1, n):
                        if common_count(adj, u, v) > 2:
                            # Remove edge (u,v)
                            remove_edge(adj, u, v)
                            changed = True
                            break
                    if changed:
                        break
            if not validate_adj(adj):
                return None
        
        return adj
    
    def simulated_annealing(adj, max_steps=20000, deadline=deadline):
        """
        Simulated annealing with remove-and-refill moves.
        Move types:
        1. Add a random missing edge if valid
        2. Remove a random edge, then greedily refill
        3. Swap: remove one edge, add another if valid, then refill
        """
        current_adj = adj[:]
        current_count = count_edges(current_adj)
        best_adj = current_adj[:]
        best_count = current_count
        
        T_start = 5.0
        T_end = 0.01
        total_steps = max_steps
        
        for step in range(total_steps):
            if time.time() > deadline - 3:
                break
            
            # Temperature schedule (geometric cooling)
            progress = step / total_steps
            T = T_start * (T_end / T_start) ** progress
            
            cur_edges = get_edges_list(current_adj)
            missing = get_missing_edges(current_adj)
            
            if not cur_edges:
                break
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Type 1: Try to add a random missing edge
                if not missing:
                    continue
                u, v = missing[random.randint(0, len(missing) - 1)]
                if is_valid_add(current_adj, u, v):
                    add_edge(current_adj, u, v)
                    new_count = current_count + 1
                    # Always accept improving moves
                    current_count = new_count
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = current_adj[:]
                # If not valid, skip (no penalty)
                
            elif move_type < 0.7:
                # Type 2: Remove a random edge, then greedily refill
                u_rem, v_rem = cur_edges[random.randint(0, len(cur_edges) - 1)]
                backup = current_adj[:]
                remove_edge(current_adj, u_rem, v_rem)
                old_count = current_count - 1
                
                # Greedy refill
                refill_missing = get_missing_edges(current_adj)
                random.shuffle(refill_missing)
                for u, v in refill_missing:
                    if is_valid_add(current_adj, u, v):
                        add_edge(current_adj, u, v)
                
                new_count = count_edges(current_adj)
                delta = new_count - current_count
                
                if delta >= 0 or random.random() < math.exp(delta / max(T, 1e-10)):
                    current_count = new_count
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = current_adj[:]
                else:
                    current_adj = backup[:]
                    
            else:
                # Type 3: Swap - remove one edge, add another, then refill
                if not missing:
                    continue
                u_rem, v_rem = cur_edges[random.randint(0, len(cur_edges) - 1)]
                u_add, v_add = missing[random.randint(0, len(missing) - 1)]
                
                backup = current_adj[:]
                remove_edge(current_adj, u_rem, v_rem)
                
                if is_valid_add(current_adj, u_add, v_add):
                    add_edge(current_adj, u_add, v_add)
                    
                    # Greedy refill
                    refill_missing = get_missing_edges(current_adj)
                    random.shuffle(refill_missing)
                    for u, v in refill_missing:
                        if is_valid_add(current_adj, u, v):
                            add_edge(current_adj, u, v)
                    
                    new_count = count_edges(current_adj)
                    delta = new_count - current_count
                    
                    if delta >= 0 or random.random() < math.exp(delta / max(T, 1e-10)):
                        current_count = new_count
                        if new_count > best_count:
                            best_count = new_count
                            best_adj = current_adj[:]
                    else:
                        current_adj = backup[:]
                else:
                    # Can't add the edge, just keep the removal
                    new_count = current_count - 1
                    delta = new_count - current_count
                    if delta >= 0 or random.random() < math.exp(delta / max(T, 1e-10)):
                        current_count = new_count
                    else:
                        current_adj = backup[:]
        
        return best_adj, best_count
    
    # Main strategy:
    # 1. Build initial solutions via multiple random restarts with greedy
    # 2. Use MILP to escape local optima
    # 3. Use SA for final refinement
    
    best_adj_overall = None
    best_count_overall = 0
    
    # Phase 1: Build diverse initial solutions
    max_restarts = 15
    for r in range(max_restarts):
        if time.time() > deadline - 60:
            break
        
        adj = [0] * n
        
        # Different ordering strategies
        strategy = r % 5
        order = all_pairs[:]
        
        if strategy == 0:
            random.shuffle(order)
        elif strategy == 1:
            # Sort by sum of indices
            order.sort(key=lambda x: x[0] + x[1])
            random.shuffle(order)
        elif strategy == 2:
            # Sort by difference
            order.sort(key=lambda x: abs(x[1] - x[0]))
            random.shuffle(order)
        elif strategy == 3:
            # Random with bias toward smaller indices
            order.sort(key=lambda x: x[0])
            random.shuffle(order)
        else:
            # Fully random
            random.shuffle(order)
        
        # Greedy construct
        greedy_fill(adj, order)
        
        # Quick SA refinement
        adj, cnt = simulated_annealing(adj, max_steps=5000)
        
        if cnt > best_count_overall:
            best_count_overall = cnt
            best_adj_overall = adj[:]
        
        # Early exit if we hit a high count
        if best_count_overall >= 65:
            break
    
    # Phase 2: Use MILP to escape local optima
    if best_adj_overall is not None and time.time() < deadline - 40:
        milp_time_limit = min(30, max(10, int(deadline - time.time() - 20)))
        milp_result = solve_milp(best_adj_overall, milp_time_limit)
        
        if milp_result is not None:
            milp_count = count_edges(milp_result)
            if milp_count > best_count_overall:
                best_count_overall = milp_count
                best_adj_overall = milp_result[:]
    
    # Phase 3: Final intensive SA on best result
    if best_adj_overall is not None and time.time() < deadline - 5:
        adj_final, cnt_final = simulated_annealing(best_adj_overall, max_steps=50000)
        if cnt_final > best_count_overall:
            best_count_overall = cnt_final
            best_adj_overall = adj_final[:]
    
    # Convert best_adj to edge list
    if best_adj_overall is None:
        return []
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj_overall[i] >> j) & 1:
                edges.append([i, j])
    
    return edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
