import time
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Specialized for k=5, g=9
    if k == 5 and g == 9:
        deadline = time.time() + 110  # 10s margin

        # Base graph K_6
        base_n = 6
        base_edges = []
        edge_idx = {}
        for u in range(base_n):
            for v in range(u + 1, base_n):
                edge_idx[(u, v)] = len(base_edges)
                base_edges.append((u, v))
        
        num_edges = len(base_edges)  # 15

        # Generate non-backtracking closed walks of length 3..8
        # We represent each walk by its coefficient vector in Z^E
        # where E is the set of edges of K_6.
        # The condition for no short cycles in the lift is:
        # For all walks W of length L < 9, sum_{e in W} s_e * phi_e != 0 mod m
        # where s_e is +1 if traversed in canonical direction, -1 otherwise.
        
        constraints = []
        
        def get_edge_id(u, v):
            if u > v: u, v = v, u
            return edge_idx.get((u, v), -1)
        
        def gen_walks(length):
            # Recursive generation of non-backtracking closed walks in K_6
            # v[0] = 0, v[length] = 0
            # v[i] != v[i-1] and v[i] != v[i-2] for all i (with cyclic boundary)
            
            def backtrack(v_idx, current, prev, path):
                if v_idx == length:
                    # Must close to 0
                    if current != 0 and prev != 0:
                        # Check edge (current, 0) exists (it does in K_6)
                        # Check non-backtracking at closure: path[1] != path[length-1]
                        if path[1] != path[length-1]:
                            # Valid walk
                            coeffs = [0] * num_edges
                            for i in range(length):
                                u = path[i]
                                v = path[(i+1)%length]
                                eid = get_edge_id(u, v)
                                if eid == -1: continue
                                # Sign
                                if u < v:
                                    sign = 1
                                else:
                                    sign = -1
                                coeffs[eid] += sign
                            constraints.append(tuple(coeffs))
                    return
                
                if v_idx == length - 1:
                    # Must go to 0
                    if current != 0 and prev != 0:
                        path.append(0)
                        # Check non-backtracking at closure: path[1] != path[length-1]
                        if path[1] != path[length-1]:
                            coeffs = [0] * num_edges
                            for i in range(length):
                                u = path[i]
                                v = path[(i+1)%length]
                                eid = get_edge_id(u, v)
                                if eid == -1: continue
                                if u < v: sign = 1
                                else: sign = -1
                                coeffs[eid] += sign
                            constraints.append(tuple(coeffs))
                        path.pop()
                    return

                for next_v in range(base_n):
                    if next_v == current or next_v == prev:
                        continue
                    path.append(next_v)
                    backtrack(v_idx + 1, next_v, current, path)
                    path.pop()
            
            # Start
            # v0=0, v1 in 1..5
            for v1 in range(1, base_n):
                backtrack(1, v1, 0, [0, v1])

        for L in range(3, 9):
            gen_walks(L)
            
        # Deduplicate constraints
        constraints = list(set(constraints))
        
        # Precompute which constraints involve which edge for faster updates
        # constraint_to_edges[c] = list of (edge_idx, coeff)
        # edge_to_constraints[e] = list of constraint indices
        
        # For local search, we want to quickly compute the score.
        # score(volts) = number of constraints c such that sum(c_i * volts_i) % m == 0
        
        best_n = 1250  # Fallback upper bound
        best_edges = None
        best_voltages = None
        
        # Try m from small to large
        # We want smallest n = 6*m.
        # Theoretical lower bound for m: girth 9 requires m to be large enough.
        # For K_6, the girth of the lift is at least 9 if m is large enough.
        # Let's try m up to 200.
        
        # To speed up, we can precompute the dot products?
        # Or just use a simple local search.
        
        # Heuristic: start with random voltages, then hill climb.
        
        # Let's try a few m values in parallel or sequentially.
        # Since we have a deadline, we can allocate time per m.
        
        # A better approach: use simulated annealing or random restarts.
        
        # Let's try m from 1 to 100, with a time budget.
        
        # Actually, for K_6, the number of constraints is large.
        # For m=1, score is always 0? No, 0 mod 1 is 0. So score = num_constraints.
        # We want score = 0.
        
        # Let's try to find a valid voltage assignment for a given m.
        
        def find_voltages(m, time_budget):
            start_time = time.time()
            
            # Random initialization
            best_score = float('inf')
            best_volts = None
            
            num_restarts = 100
            steps_per_restart = 1000
            
            for r in range(num_restarts):
                if time.time() - start_time > time_budget:
                    break
                
                volts = [random.randint(0, m-1) for _ in range(num_edges)]
                current_score = 0
                
                # Compute initial score
                for c in constraints:
                    s = 0
                    for i in range(num_edges):
                        if c[i] != 0:
                            s += c[i] * volts[i]
                    if s % m == 0:
                        current_score += 1
                
                if current_score == 0:
                    return volts
                
                # Local search
                for step in range(steps_per_restart):
                    if time.time() - start_time > time_budget:
                        break
                    if current_score == 0:
                        return volts
                    
                    # Pick a random edge
                    idx = random.randint(0, num_edges - 1)
                    old_val = volts[idx]
                    
                    # Try a new value
                    # To be efficient, we can try all values and pick the best,
                    # but that's O(m * constraints).
                    # Instead, pick a random new value.
                    new_val = random.randint(0, m-1)
                    if new_val == old_val:
                        continue
                    
                    # Compute delta score
                    # Only constraints involving idx change.
                    # We need to know which constraints involve idx.
                    # Let's precompute this.
                    
                    # For now, recompute full score? Too slow.
                    # Let's precompute edge_to_constraints.
                    
                    delta = 0
                    for c_idx in edge_to_constraints[idx]:
                        c = constraints[c_idx]
                        old_s = sum(c[i] * volts[i] for i in range(num_edges)) % m
                        # Update
                        volts[idx] = new_val
                        new_s = sum(c[i] * volts[i] for i in range(num_edges)) % m
                        # Revert
                        volts[idx] = old_val
                        
                        if old_s == 0 and new_s != 0:
                            delta -= 1
                        elif old_s != 0 and new_s == 0:
                            delta += 1
                    
                    new_score = current_score + delta
                    if new_score < current_score:
                        volts[idx] = new_val
                        current_score = new_score
                    elif new_score == current_score and random.random() < 0.1:
                        # Accept equal with small probability to escape local minima
                        volts[idx] = new_val
                        current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_volts = volts[:]
            
            if best_score == 0:
                return best_volts
            return None

        # Precompute edge_to_constraints
        edge_to_constraints = [[] for _ in range(num_edges)]
        for c_idx, c in enumerate(constraints):
            for i in range(num_edges):
                if c[i] != 0:
                    edge_to_constraints[i].append(c_idx)
        
        # Optimize score calculation:
        # We can maintain a count of satisfied constraints.
        # But for simplicity, let's just try a few m values.
        
        # Actually, the local search above is O(num_constraints * num_edges) per step.
        # That's too slow.
        
        # Better: maintain the sum for each constraint.
        # sums[c_idx] = sum(c[i] * volts[i]) % m
        
        def find_voltages_fast(m, time_budget):
            start_time = time.time()
            
            num_restarts = 50
            steps_per_restart = 2000
            
            best_score = float('inf')
            best_volts = None
            
            for r in range(num_restarts):
                if time.time() - start_time > time_budget:
                    break
                
                volts = [random.randint(0, m-1) for _ in range(num_edges)]
                
                # Compute sums
                sums = [0] * len(constraints)
                for c_idx, c in enumerate(constraints):
                    s = 0
                    for i in range(num_edges):
                        if c[i] != 0:
                            s += c[i] * volts[i]
                    sums[c_idx] = s % m
                
                current_score = sum(1 for s in sums if s == 0)
                
                if current_score == 0:
                    return volts
                
                for step in range(steps_per_restart):
                    if time.time() - start_time > time_budget:
                        break
                    if current_score == 0:
                        return volts
                    
                    idx = random.randint(0, num_edges - 1)
                    old_val = volts[idx]
                    new_val = random.randint(0, m-1)
                    if new_val == old_val:
                        continue
                    
                    delta_score = 0
                    for c_idx in edge_to_constraints[idx]:
                        old_s = sums[c_idx]
                        c = constraints[c_idx]
                        coeff = c[idx]
                        new_s = (old_s + coeff * (new_val - old_val)) % m
                        if old_s == 0 and new_s != 0:
                            delta_score -= 1
                        elif old_s != 0 and new_s == 0:
                            delta_score += 1
                        sums[c_idx] = new_s
                    
                    new_score = current_score + delta_score
                    if new_score < current_score:
                        volts[idx] = new_val
                        current_score = new_score
                    elif new_score == current_score and random.random() < 0.2:
                        volts[idx] = new_val
                        current_score = new_score
                    else:
                        # Revert
                        for c_idx in edge_to_constraints[idx]:
                            c = constraints[c_idx]
                            coeff = c[idx]
                            sums[c_idx] = (sums[c_idx] - coeff * (new_val - old_val)) % m
                
                if current_score < best_score:
                    best_score = current_score
                    best_volts = volts[:]
            
            if best_score == 0:
                return best_volts
            return None

        # Try m from 1 to 200
        # Allocate time: 110s total.
        # Let's try m in increasing order.
        
        # For small m, it's unlikely to work.
        # Let's start from m=10 and go up.
        
        # Actually, let's try a range of m values.
        
        # We can also try non-abelian groups, but that's complex.
        # Let's stick to abelian Z_m.
        
        # Let's try m = 10, 20, 30, ..., 100.
        
        # To be safe, let's try all m from 1 to 100, but with a small time budget each.
        
        # Actually, the number of constraints is large, so local search might be slow.
        # Let's estimate the number of constraints.
        # For K_6, number of non-backtracking closed walks of length L.
        # L=3: 20
        # L=4: 45
        # L=5: 120
        # L=6: 330
        # L=7: 900
        # L=8: 2400
        # Total ~ 3800 constraints.
        
        # With 15 edges, local search step is O(15 * num_constraints_involving_edge).
        # num_constraints_involving_edge ~ 3800/15 ~ 250.
        # So each step is ~ 250 ops.
        # 2000 steps * 50 restarts * 100 m values = 10^7 steps * 250 = 2.5 * 10^9 ops.
        # Too slow.
        
        # We need to be smarter.
        
        # Let's try only a few m values, with more time each.
        
        # Let's try m = 100, 50, 20, 10.
        
        # Or, let's use a better heuristic: start with a random assignment,
        # and only accept moves that reduce the score.
        
        # Actually, the previous code was already doing this.
        # Let's just try a few m values.
        
        m_values = [10, 20, 30, 50, 100, 150]
        
        for m in m_values:
            if time.time() > deadline - 10:
                break
            
            time_budget = 10
            volts = find_voltages_fast(m, time_budget)
            
            if volts is not None:
                n = 6 * m
                # Build edges
                edges = []
                for u in range(base_n):
                    for i in range(m):
                        for v in range(base_n):
                            if u == v: continue
                            eid = get_edge_id(u, v)
                            if eid == -1: continue
                            # Determine voltage
                            if u < v:
                                phi = volts[eid]
                            else:
                                phi = -volts[eid]
                            
                            j = (i + phi) % m
                            # Edge (u, i) to (v, j)
                            id1 = u * m + i
                            id2 = v * m + j
                            # Only add if id1 < id2 to avoid duplicates
                            if id1 < id2:
                                edges.append((id1, id2))
                
                # Verify degree
                # Should be 5.
                
                return n, edges
        
        # If no solution found in time, return fallback
        return luw_construct(k, g)
        
    else:
        return luw_construct(k, g)

# EVOLVE-BLOCK-END
