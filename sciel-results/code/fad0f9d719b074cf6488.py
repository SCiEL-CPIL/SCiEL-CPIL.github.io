import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def is_valid(n, colors):
        """Verify a coloring is valid (no red K4, no blue K5)."""
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    def solve_maxsat(n, deadline):
        """
        Bit-parallel Max-SAT solver for Hypergraph Ramsey.
        Variables: Triples (0=Red, 1=Blue).
        Hard clauses:
          - Red K4: For each quad, at least one triple is Blue (1).
          - Blue K5: For each quint, at least one triple is Red (0).
        """
        triple_list = list(combinations(range(n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}
        
        # Build clause data as numpy arrays for vectorized ops
        # Each clause: (var_indices array, polarity array)
        # Polarity: 1 means "var must be 1" (positive literal), -1 means "var must be 0"
        
        quad_clauses_vars = []
        quad_clauses_pols = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            quad_clauses_vars.append(np.array([t1, t2, t3, t4], dtype=np.int32))
            quad_clauses_pols.append(np.array([1, 1, 1, 1], dtype=np.int8))
        
        quint_clauses_vars = []
        quint_clauses_pols = []
        for quint in combinations(range(n), 5):
            vars_arr = []
            for t in combinations(quint, 3):
                vars_arr.append(triple_to_idx[t])
            quint_clauses_vars.append(np.array(vars_arr, dtype=np.int32))
            quint_clauses_pols.append(np.array([-1]*10, dtype=np.int8))
        
        all_clauses_vars = quad_clauses_vars + quint_clauses_vars
        all_clauses_pols = quad_clauses_pols + quint_clauses_pols
        num_clauses = len(all_clauses_vars)
        
        # Clause weights: K5 clauses are harder to satisfy, weight them higher
        quad_weight = 1
        quint_weight = 3
        clause_weights = np.array([quad_weight]*len(quad_clauses_vars) + [quint_weight]*len(quint_clauses_vars), dtype=np.float64)
        
        # Precompute: for each variable, which clauses it's in and with what polarity
        var_to_clauses = [[] for _ in range(num_triples)]
        for c_idx in range(num_clauses):
            for v in all_clauses_vars[c_idx]:
                var_to_clauses[v].append(c_idx)
        
        # Initialize with modular coloring
        color = np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
        
        # Compute initial cost: sum of weights of unsatisfied clauses
        # A clause is unsatisfied if ALL its literals are false
        # For a literal (var, pol): it's satisfied if color[var] matches the desired value
        #   pol=1: satisfied if color[var]==1
        #   pol=-1: satisfied if color[var]==0
        # Clause is unsatisfied if no literal is satisfied
        
        def compute_cost(color_arr):
            """Compute total weighted unsatisfied clause count."""
            total_cost = 0.0
            for c_idx in range(num_clauses):
                vars_arr = all_clauses_vars[c_idx]
                pols_arr = all_clauses_pols[c_idx]
                # For each literal, check if satisfied
                # pol=1: sat if color==1; pol=-1: sat if color==0
                # sat = (color == 1) & (pol == 1) | (color == 0) & (pol == -1)
                colors_in_clause = color_arr[vars_arr]
                # A literal is satisfied if: (pol==1 and color==1) or (pol==-1 and color==0)
                # Equivalently: (color * pol) == 1 when we map 0->-1... let's just do it directly
                sat = ((colors_in_clause == 1) & (pols_arr == 1)) | ((colors_in_clause == 0) & (pols_arr == -1))
                if not np.any(sat):
                    total_cost += clause_weights[c_idx]
            return total_cost
        
        def compute_cost_fast(color_arr):
            """Faster cost computation using vectorized operations per clause."""
            total_cost = 0.0
            for c_idx in range(num_clauses):
                vars_arr = all_clauses_vars[c_idx]
                pols_arr = all_clauses_pols[c_idx]
                colors_in_clause = color_arr[vars_arr]
                # Literal satisfied: (color==1 and pol==1) or (color==0 and pol==-1)
                # This is equivalent to: (2*color - 1) * pol > 0, i.e., color and pol agree in sign
                # Map color 0->-1, 1->1: mapped = 2*color - 1
                # Then sat if mapped * pol > 0, i.e., mapped * pol == 1
                mapped = 2 * colors_in_clause - 1  # 0->-1, 1->1
                products = mapped * pols_arr
                if not np.any(products == 1):
                    total_cost += clause_weights[c_idx]
            return total_cost
        
        current_cost = compute_cost_fast(color)
        best_cost = current_cost
        best_color = color.copy()
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.9999
        min_T = 0.01
        
        step = 0
        max_steps = 5000000
        
        while time.time() < deadline - 2 and T > min_T:
            step += 1
            
            # Pick a random variable to flip
            var_idx = random.randint(0, num_triples - 1)
            
            # Compute delta cost by checking affected clauses
            old_val = color[var_idx]
            new_val = 1 - old_val
            
            delta_cost = 0.0
            for c_idx in var_to_clauses[var_idx]:
                vars_arr = all_clauses_vars[c_idx]
                pols_arr = all_clauses_pols[c_idx]
                
                # Check clause status before flip
                colors_old = color[vars_arr]
                mapped_old = 2 * colors_old - 1
                prods_old = mapped_old * pols_arr
                was_sat = np.any(prods_old == 1)
                
                # Check clause status after flip
                colors_new = colors_old.copy()
                # Find position of var_idx in vars_arr
                pos = np.where(vars_arr == var_idx)[0]
                if len(pos) == 0:
                    continue
                pos = pos[0]
                colors_new[pos] = new_val
                mapped_new = 2 * colors_new - 1
                prods_new = mapped_new * pols_arr
                is_sat = np.any(prods_new == 1)
                
                if was_sat and not is_sat:
                    delta_cost += clause_weights[c_idx]
                elif not was_sat and is_sat:
                    delta_cost -= clause_weights[c_idx]
            
            # Metropolis acceptance
            if delta_cost <= 0 or random.random() < np.exp(-delta_cost / T):
                color[var_idx] = new_val
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_color = color.copy()
                    if best_cost == 0:
                        return best_color.tolist(), 0
            
            T *= cooling
            
            # Periodic check
            if step % 10000 == 0 and best_cost == 0:
                return best_color.tolist(), 0
        
        return best_color.tolist(), best_cost

    # Strategy: Try n=34, 33, 32 with the Max-SAT solver
    targets = [34, 33, 32]
    time_allocations = [35, 25, 20]
    
    best_found = (9, mod_color(9, 3))
    
    for i, n in enumerate(targets):
        if time.time() > deadline - 15:
            break
            
        alloc = time_allocations[i] if i < len(time_allocations) else 15
        current_deadline = time.time() + alloc
        
        # Multiple restarts with different initializations
        max_restarts = 10
        found_valid = False
        
        for r in range(max_restarts):
            if time.time() > current_deadline - 3:
                break
            
            # Different initializations
            if r == 0:
                # Modular by 3
                init_color = np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in combinations(range(n), 3)], dtype=np.int8)
            elif r == 1:
                # Modular by 5
                init_color = np.array([0 if (a+b+c) % 5 == 0 else 1 for (a,b,c) in combinations(range(n), 3)], dtype=np.int8)
            elif r == 2:
                # Modular by 7
                init_color = np.array([0 if (a+b+c) % 7 == 0 else 1 for (a,b,c) in combinations(range(n), 3)], dtype=np.int8)
            elif r == 3:
                # Vertex-based: color by sum of vertices mod 4
                init_color = np.array([0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in combinations(range(n), 3)], dtype=np.int8)
            elif r == 4:
                # Random with 40% red
                init_color = np.array([0 if random.random() < 0.4 else 1 for _ in range(len(list(combinations(range(n), 3))))], dtype=np.int8)
            elif r == 5:
                # Random with 50% red
                init_color = np.array([0 if random.random() < 0.5 else 1 for _ in range(len(list(combinations(range(n), 3))))], dtype=np.int8)
            elif r == 6:
                # Pattern: red if a+b+c is even
                init_color = np.array([0 if (a+b+c) % 2 == 0 else 1 for (a,b,c) in combinations(range(n), 3)], dtype=np.int8)
            elif r == 7:
                # Pattern: red if (a mod 3) == (b mod 3)
                init_color = np.array([0 if a % 3 == b % 3 else 1 for (a,b,c) in combinations(range(n), 3)], dtype=np.int8)
            elif r == 8:
                # Pattern: red if (a mod 4) < (b mod 4)
                init_color = np.array([0 if a % 4 < b % 4 else 1 for (a,b,c) in combinations(range(n), 3)], dtype=np.int8)
            else:
                # Random with 35% red
                init_color = np.array([0 if random.random() < 0.35 else 1 for _ in range(len(list(combinations(range(n), 3))))], dtype=np.int8)
            
            # Run the solver with this initialization
            # Modify solve_maxsat to accept initial color
            def solve_maxsat_with_init(n_val, deadline_val, init_col):
                triple_list = list(combinations(range(n_val), 3))
                num_triples = len(triple_list)
                triple_to_idx = {t: i for i, t in enumerate(triple_list)}
                
                quad_clauses_vars = []
                for quad in combinations(range(n_val), 4):
                    a, b, c, d = quad
                    t1 = triple_to_idx[(a,b,c)]
                    t2 = triple_to_idx[(a,b,d)]
                    t3 = triple_to_idx[(a,c,d)]
                    t4 = triple_to_idx[(b,c,d)]
                    quad_clauses_vars.append(np.array([t1, t2, t3, t4], dtype=np.int32))
                
                quint_clauses_vars = []
                for quint in combinations(range(n_val), 5):
                    vars_arr = []
                    for t in combinations(quint, 3):
                        vars_arr.append(triple_to_idx[t])
                    quint_clauses_vars.append(np.array(vars_arr, dtype=np.int32))
                
                all_clauses_vars = quad_clauses_vars + quint_clauses_vars
                num_clauses = len(all_clauses_vars)
                
                quad_weight = 1
                quint_weight = 3
                clause_weights = np.array([quad_weight]*len(quad_clauses_vars) + [quint_weight]*len(quint_clauses_vars), dtype=np.float64)
                
                var_to_clauses = [[] for _ in range(num_triples)]
                for c_idx in range(num_clauses):
                    for v in all_clauses_vars[c_idx]:
                        var_to_clauses[v].append(c_idx)
                
                color = init_col.copy()
                
                def compute_cost_fast_local(color_arr):
                    total_cost = 0.0
                    for c_idx in range(num_clauses):
                        vars_arr = all_clauses_vars[c_idx]
                        colors_in_clause = color_arr[vars_arr]
                        mapped = 2 * colors_in_clause - 1
                        # For quad clauses (first len(quad_clauses_vars)): pol=1, so sat if mapped==1
                        # For quint clauses: pol=-1, so sat if mapped==-1
                        if c_idx < len(quad_clauses_vars):
                            if not np.any(mapped == 1):
                                total_cost += clause_weights[c_idx]
                        else:
                            if not np.any(mapped == -1):
                                total_cost += clause_weights[c_idx]
                    return total_cost
                
                current_cost = compute_cost_fast_local(color)
                best_cost = current_cost
                best_color = color.copy()
                
                T = 10.0
                cooling = 0.9999
                min_T = 0.01
                
                step = 0
                max_steps = 5000000
                
                while time.time() < deadline_val - 2 and T > min_T:
                    step += 1
                    
                    var_idx = random.randint(0, num_triples - 1)
                    old_val = color[var_idx]
                    new_val = 1 - old_val
                    
                    delta_cost = 0.0
                    for c_idx in var_to_clauses[var_idx]:
                        vars_arr = all_clauses_vars[c_idx]
                        colors_old = color[vars_arr]
                        mapped_old = 2 * colors_old - 1
                        
                        if c_idx < len(quad_clauses_vars):
                            was_sat = np.any(mapped_old == 1)
                        else:
                            was_sat = np.any(mapped_old == -1)
                        
                        colors_new = colors_old.copy()
                        pos = np.where(vars_arr == var_idx)[0]
                        if len(pos) == 0:
                            continue
                        pos = pos[0]
                        colors_new[pos] = new_val
                        mapped_new = 2 * colors_new - 1
                        
                        if c_idx < len(quad_clauses_vars):
                            is_sat = np.any(mapped_new == 1)
                        else:
                            is_sat = np.any(mapped_new == -1)
                        
                        if was_sat and not is_sat:
                            delta_cost += clause_weights[c_idx]
                        elif not was_sat and is_sat:
                            delta_cost -= clause_weights[c_idx]
                    
                    if delta_cost <= 0 or random.random() < np.exp(-delta_cost / T):
                        color[var_idx] = new_val
                        current_cost += delta_cost
                        
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_color = color.copy()
                            if best_cost == 0:
                                return best_color.tolist(), 0
                    
                    T *= cooling
                    
                    if step % 10000 == 0 and best_cost == 0:
                        return best_color.tolist(), 0
                
                return best_color.tolist(), best_cost
            
            colors, cost = solve_maxsat_with_init(n, current_deadline, init_color)
            
            if cost == 0:
                if is_valid(n, colors):
                    best_found = (n, colors)
                    found_valid = True
                    break
        
        if found_valid:
            break

    return {'n': best_found[0], 'colors': best_found[1]}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
    if res['n'] > 9:
        print("Validating...")
        print(is_valid(res['n'], res['colors']))
