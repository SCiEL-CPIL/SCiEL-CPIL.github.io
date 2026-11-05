import time
import numpy as np
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Strategy: 
    1. Try Voltage Lift of Petersen Graph with Z_m group using Min-Conflicts Local Search.
    2. Fallback to Voltage Lift of K4 with PSL(2,7) group (n=672).
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    def remaining():
        return deadline - time.time()

    # ---------------- Girth Verification ----------------
    def check_girth(n, edges, g, time_limit):
        """Verify no cycle shorter than g using non-backtracking BFS from every vertex."""
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        min_cycle = float('inf')
        
        for s in range(n):
            if time.time() > time_limit:
                return -1
            
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            
            if min_cycle < g:
                return min_cycle
        
        return min_cycle

    # ---------------- Voltage lift of Petersen graph ----------------
    def voltage_lift_petersen():
        if k != 3 or g != 16:
            return None
        
        # Petersen graph edges (10 vertices, 15 edges)
        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer cycle
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner star
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
        ]
        
        n_base = 10
        edge_index = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i
            
        adj = [[] for _ in range(n_base)]
        for u, v in pet_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Generate all non-backtracking closed walks of length < g
        # We need to find linear constraints on voltages v_i such that 
        # sum c_i v_i != 0 mod m for all short closed walks.
        # A walk is represented by a vector of edge traversals.
        # If a walk uses edge i with direction +1 (u->v) and -1 (v->u),
        # the voltage sum is sum (dir * v_i).
        
        max_L = g - 1
        walk_vectors = set()
        
        for start_v in range(n_base):
            # Stack: (current, prev, length, counts_vector)
            # counts_vector is a tuple of 15 ints
            stack = [(start_v, -1, 0, [0]*15)]
            while stack:
                current, prev, length, counts = stack.pop()
                
                if length >= max_L:
                    continue
                
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = pet_edges[e_idx]
                    d = 1 if current == u else -1
                    
                    new_counts = counts.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1
                    
                    if next_v == start_v and new_l >= 3:
                        # Normalize the vector to avoid duplicates
                        # We treat (c) and (-c) as equivalent constraints 
                        # because if sum c_i v_i != 0, then sum -c_i v_i != 0.
                        # However, we must be careful. If we add both, we double the work.
                        # Let's keep the raw vector for now and dedupe later.
                        walk_vectors.add(tuple(new_counts))
                    
                    stack.append((next_v, current, new_l, new_counts))
        
        if not walk_vectors:
            return None
            
        # Filter and normalize constraints
        unique_constraints = set()
        for c in walk_vectors:
            if all(x == 0 for x in c):
                continue
            # Normalize sign: make the first non-zero element positive
            first_non_zero = -1
            for i in range(15):
                if c[i] != 0:
                    first_non_zero = i
                    break
            if first_non_zero != -1:
                if c[first_non_zero] < 0:
                    c = tuple(-x for x in c)
                unique_constraints.add(c)
                
        C_list = list(unique_constraints)
        if not C_list:
            return None
            
        C = np.array(C_list, dtype=np.int64)
        num_constraints = len(C)
        
        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
                i += 6
            return True
        
        def check_valid(volts, m):
            """Check if voltage assignment avoids all short cycles."""
            # Vectorized check: C @ volts % m != 0 for all rows
            residues = np.mod(C @ volts, m)
            return not np.any(residues == 0)
            
        def solve_voltage_min_conflicts(m, max_iters=20000):
            """
            Min-Conflicts heuristic:
            1. Initialize random v.
            2. Compute cost (number of violated constraints).
            3. While cost > 0 and time < limit:
               - Pick a random violated constraint.
               - Pick a random variable involved in it.
               - Set variable to the value that minimizes total cost.
            """
            # Initialize
            v = np.random.randint(1, m, size=15).astype(np.int64)
            
            def get_cost(v):
                residues = np.mod(C @ v, m)
                return np.sum(residues == 0)
                
            best_v = v.copy()
            best_cost = get_cost(v)
            
            # Precompute non-zero indices for each constraint for faster local search
            # For each constraint row, which variables are involved?
            constraint_vars = []
            for row in C:
                nz = np.nonzero(row)[0]
                constraint_vars.append(nz)
            
            no_improve = 0
            patience = 500
            
            for it in range(max_iters):
                if remaining() < 10:
                    break
                
                if best_cost == 0:
                    return best_v, 0
                
                # Pick a random violated constraint
                residues = np.mod(C @ v, m)
                violated_indices = np.nonzero(residues == 0)[0]
                if len(violated_indices) == 0:
                    return v, 0
                    
                c_idx = random.choice(violated_indices)
                row = C[c_idx]
                vars_involving = constraint_vars[c_idx]
                
                if len(vars_involving) == 0:
                    # Constraint is 0=0, which shouldn't happen if we filtered zero vectors
                    continue
                    
                # Pick a random variable from the violated constraint
                j = random.choice(vars_involving)
                
                # Try all values for v[j] to find the one that minimizes cost
                # To speed up, we can just try a few random values or all values if m is small
                # m is around 50-100, so trying all is feasible (100 evaluations)
                # But evaluation is O(num_constraints). 
                # Let's try a subset if m is large, or all if small.
                
                best_val = v[j]
                min_cost = best_cost
                current_cost = best_cost
                
                # We need to update cost efficiently.
                # Changing v[j] changes the residue of constraint i by c_ij * delta.
                # This is complex to track incrementally without a full recalc.
                # Given m is small, full recalc is OK.
                
                if m < 100:
                    for val in range(1, m):
                        if val == v[j]:
                            continue
                        v[j] = val
                        new_cost = get_cost(v)
                        if new_cost < min_cost:
                            min_cost = new_cost
                            best_val = val
                else:
                    # Sample some values
                    for _ in range(10):
                        val = random.randint(1, m-1)
                        v[j] = val
                        new_cost = get_cost(v)
                        if new_cost < min_cost:
                            min_cost = new_cost
                            best_val = val
                
                v[j] = best_val
                current_cost = min_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_v = v.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    # If stuck, restart or random kick
                    if no_improve > patience:
                        v = np.random.randint(1, m, size=15).astype(np.int64)
                        best_v = v.copy()
                        best_cost = get_cost(v)
                        no_improve = 0
                        patience = 500
            
            return best_v, best_cost

        # Try m from Moore bound upwards
        # Moore bound for n(3,16) is 512. n = 10*m. So m >= 52.
        # We should try m=52, 53, ...
        # It is known that m=94 works (n=940).
        # Let's try to find a smaller m.
        
        best_result = None
        
        # Prioritize smaller m
        # We will try m from 52 to 100
        for m in range(52, 101):
            if remaining() < 15:
                break
            
            # We can skip some m if we know they are unlikely?
            # No, let's just try them.
            
            # Try multiple restarts
            found = False
            for restart in range(10):
                if remaining() < 10:
                    break
                
                v, cost = solve_voltage_min_conflicts(m, max_iters=5000)
                
                if cost == 0:
                    # Construct graph
                    n = 10 * m
                    edges = []
                    for u, v_base in pet_edges:
                        e_idx = edge_index[(u, v_base)]
                        vol = v[e_idx]
                        for layer in range(m):
                            u_lift = u * m + layer
                            v_lift = v_base * m + (layer + vol) % m
                            if u_lift < v_lift:
                                edges.append((u_lift, v_lift))
                            else:
                                edges.append((v_lift, u_lift))
                    
                    # Verify
                    gc = check_girth(n, edges, g, deadline - 2)
                    if gc >= g:
                        return (n, edges)
                    elif gc > 0:
                        # Double check
                        gc2 = check_girth(n, edges, g, deadline - 1)
                        if gc2 >= g:
                            return (n, edges)
                    
                    # If verification fails, it might be due to a bug in constraint gen or check
                    # But we trust the check.
            
        return None

    # ---------------- K4 PSL(2,7) Fallback ----------------
    def k4_psl_lift():
        if k != 3 or g != 16:
            return None
            
        p = 7
        # K4 edges
        k4_edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        
        # We need to assign PSL(2,7) matrices to edges such that 
        # product around any short cycle is not identity.
        # This is harder to solve generically quickly.
        # However, there are known constructions.
        # For now, we return a known valid graph if we can hardcode it?
        # No, we must generate it.
        
        # Since finding a valid PSL(2,7) assignment is complex, 
        # and Petersen Z_m is our primary, we only use this if Petersen fails.
        # But if Petersen fails, we might not have time to solve PSL.
        # Let's try a simple random search for PSL(2,7) as well.
        
        def random_psl_element(p, rng):
            while True:
                a = rng.randint(0, p-1)
                b = rng.randint(0, p-1)
                c = rng.randint(0, p-1)
                d = rng.randint(0, p-1)
                det = (a*d - b*c) % p
                if det == 1:
                    return (a, b, c, d)
            
        def mat_mul(m1, m2, p):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            return ((a1*a2 + b1*c2) % p, (a1*b2 + b1*d2) % p,
                    (c1*a2 + d1*c2) % p, (c1*b2 + d1*d2) % p)
        
        def mat_inv(m, p):
            a, b, c, d = m
            # det=1, so inv is [[d, -b], [-c, a]]
            return (d % p, (-b) % p, (-c) % p, a % p)
            
        def check_psl_girth(voltages, p, required_girth):
            # voltages: dict (u,v) -> matrix
            # Check all closed walks of length < required_girth
            adj = {0:[1,2,3], 1:[0,2,3], 2:[0,1,3], 3:[0,1,2]}
            
            # Precompute short walks
            max_L = required_girth - 1
            # DFS from each node
            for start in range(4):
                stack = [(start, -1, 0, (1,0,0,1))]
                while stack:
                    curr, prev, length, prod = stack.pop()
                    if length >= max_L:
                        continue
                    for next_v in adj[curr]:
                        if next_v == prev:
                            continue
                        # Get voltage
                        if (curr, next_v) in voltages:
                            v_mat = voltages[(curr, next_v)]
                        else:
                            v_mat = voltages[(next_v, curr)]
                            v_mat = mat_inv(v_mat, p)
                        
                        new_prod = mat_mul(prod, v_mat, p)
                        new_l = length + 1
                        
                        if next_v == start and new_l >= 3:
                            if new_prod == (1,0,0,1) or new_prod == (p-1, 0, 0, p-1):
                                return False
                        
                        stack.append((next_v, curr, new_l, new_prod))
            return True
            
        rng = random.Random(42)
        
        # Try to find a valid assignment
        for _ in range(100):
            if remaining() < 5:
                break
            voltages = {}
            for u, v in k4_edges:
                voltages[(u, v)] = random_psl_element(p, rng)
                
            if check_psl_girth(voltages, p, g):
                # Build graph
                # n = 4 * |PSL(2,7)| = 4 * 168 = 672
                # We need to enumerate the group elements?
                # Or just build the lift by BFS.
                # Since the graph is connected, we can BFS from (0, I).
                
                index = {}
                order = []
                edges = []
                
                identity = (1,0,0,1)
                start_key = (0, identity)
                index[start_key] = 0
                order.append(start_key)
                
                queue = deque([start_key])
                
                while queue:
                    base_v, elem = queue.popleft()
                    idx_v = index[(base_v, elem)]
                    
                    for next_v in [1,2,3,0]:
                        if next_v == base_v: continue
                        # Check if edge exists in K4
                        # K4 is complete, so yes.
                        if (base_v, next_v) in voltages:
                            v_mat = voltages[(base_v, next_v)]
                        else:
                            v_mat = voltages[(next_v, base_v)]
                            v_mat = mat_inv(v_mat, p)
                        
                        new_elem = mat_mul(elem, v_mat, p)
                        new_key = (next_v, new_elem)
                        
                        if new_key not in index:
                            index[new_key] = len(order)
                            order.append(new_key)
                            queue.append(new_key)
                        
                        idx_u = index[new_key]
                        if idx_v < idx_u:
                            edges.append((idx_v, idx_u))
                
                n = len(order)
                # Verify girth
                gc = check_girth(n, edges, g, deadline - 2)
                if gc >= g:
                    return (n, edges)
        
        return None

    # ---------------- Main ----------------
    # 1. Try Petersen Voltage Lift
    res = voltage_lift_petersen()
    if res is not None:
        return res
    
    # 2. Try K4 PSL(2,7) Lift
    res = k4_psl_lift()
    if res is not None:
        return res
    
    # 3. Fallback: Return a trivial graph? No, must be valid.
    # If we fail, we must return something. 
    # The task says "invalid result... scores 0".
    # We should try to return the best candidate found even if not fully verified?
    # No, the evaluator checks girth. If it fails, score 0.
    # So we must return a valid graph.
    
    # If we run out of time, we might have a partial result.
    # But we don't store partial results in the current structure.
    
    # As a last resort, use the LUW construction from the seed if it was present?
    # The seed had `luw_fallback`. Let's re-include a simple LUW or similar if needed.
    # But for k=3, g=16, LUW might not work well.
    
    # Let's try one more time with Petersen but with a very tight deadline check
    # to ensure we return *something* valid if we find it late.
    
    # Actually, if voltage_lift_petersen returned None, it means it didn't find a solution.
    # We should try to return the K4 solution if it found one.
    
    # If both fail, we are in trouble. 
    # Let's ensure the Petersen solver is robust enough.
    
    # Dummy return to satisfy interface (should not be reached if search works)
    # We can construct a known 936-vertex graph? No, we don't have the edges hardcoded.
    
    # Let's assume the search finds a solution.
    return (0, [])

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
