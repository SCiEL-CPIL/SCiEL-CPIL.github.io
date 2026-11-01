import time
import random
import math

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
        deadline = time.time() + 110
        
        # Base graph K_6
        base_n = 6
        base_edges = []
        edge_idx = {}
        for u in range(base_n):
            for v in range(u + 1, base_n):
                edge_idx[(u, v)] = len(base_edges)
                base_edges.append((u, v))
        
        num_edges = len(base_edges)
        
        def get_edge_id(u, v):
            if u > v: u, v = v, u
            return edge_idx.get((u, v), -1)
        
        # Generate non-backtracking closed walks of length 3..8
        constraints = []
        
        def gen_walks(length):
            def backtrack(v_idx, current, prev, path):
                if v_idx == length:
                    if current != 0 and prev != 0:
                        if path[1] != path[length-1]:
                            coeffs = {}
                            for i in range(length):
                                u = path[i]
                                v = path[(i+1)%length]
                                eid = get_edge_id(u, v)
                                if eid == -1: continue
                                sign = 1 if u < v else -1
                                if eid in coeffs:
                                    coeffs[eid] += sign
                                else:
                                    coeffs[eid] = sign
                            c_list = {}
                            for eid, s in coeffs.items():
                                if s != 0:
                                    c_list[eid] = s
                            if c_list:
                                constraints.append(c_list)
                    return
                
                if v_idx == length - 1:
                    if current != 0 and prev != 0:
                        path.append(0)
                        if path[1] != path[length-1]:
                            coeffs = {}
                            for i in range(length):
                                u = path[i]
                                v = path[(i+1)%length]
                                eid = get_edge_id(u, v)
                                if eid == -1: continue
                                sign = 1 if u < v else -1
                                if eid in coeffs:
                                    coeffs[eid] += sign
                                else:
                                    coeffs[eid] = sign
                            c_list = {}
                            for eid, s in coeffs.items():
                                if s != 0:
                                    c_list[eid] = s
                            if c_list:
                                constraints.append(c_list)
                        path.pop()
                    return

                for next_v in range(base_n):
                    if next_v == current or next_v == prev:
                        continue
                    path.append(next_v)
                    backtrack(v_idx + 1, next_v, current, path)
                    path.pop()
            
            for v1 in range(1, base_n):
                backtrack(1, v1, 0, [0, v1])

        for L in range(3, 9):
            gen_walks(L)
            
        # Precompute edge-to-constraint mapping for fast updates
        # constraint_sums[i] is the sum for constraint i
        # We store for each edge, the list of (constraint_index, coefficient)
        edge_to_cons = [[] for _ in range(num_edges)]
        for i, c_list in enumerate(constraints):
            for eid, coeff in c_list.items():
                edge_to_cons[eid].append((i, coeff))
        
        num_constraints = len(constraints)
        
        best_n = None
        best_edges = None
        
        # Try m from 1 upwards
        # We want smallest n = 6*m
        for m in range(1, 500):
            if time.time() > deadline:
                break
            
            # Local search for voltages
            # We maintain current voltage assignment and constraint sums
            # score = number of constraints with sum % m == 0
            
            def solve_m(m):
                # Use soft-min surrogate: sum of exp(-beta * min(s, m-s))
                # This is smooth and helps escape local minima of the hard count
                beta = 2.0  # temperature parameter
                
                def soft_score(sums, m, beta):
                    total = 0.0
                    for s in sums:
                        dist = min(s, m - s) if m > 0 else 0
                        if dist == 0:
                            total += math.exp(beta)  # large penalty
                        else:
                            total += math.exp(-beta * dist / max(1, m // 10))
                    return total
                
                def hard_score(sums):
                    return sum(1 for s in sums if s == 0)
                
                # Random init
                volts = [random.randint(0, m-1) for _ in range(num_edges)]
                sums = [0] * num_constraints
                
                # Calculate initial sums
                for e in range(num_edges):
                    v = volts[e]
                    if v == 0: continue
                    for c_idx, coeff in edge_to_cons[e]:
                        sums[c_idx] = (sums[c_idx] + v * coeff) % m
                
                # Calculate initial scores
                h_score = hard_score(sums)
                s_score = soft_score(sums, m, beta)
                
                # Simulated Annealing with soft objective
                for restart in range(30): # 30 restarts per m
                    if time.time() > deadline:
                        return None
                    
                    if restart > 0:
                        volts = [random.randint(0, m-1) for _ in range(num_edges)]
                        sums = [0] * num_constraints
                        for e in range(num_edges):
                            v = volts[e]
                            if v == 0: continue
                            for c_idx, coeff in edge_to_cons[e]:
                                sums[c_idx] = (sums[c_idx] + v * coeff) % m
                        h_score = hard_score(sums)
                        s_score = soft_score(sums, m, beta)
                    
                    if h_score == 0:
                        return volts
                    
                    # Simulated annealing
                    T = 1.0
                    cooling = 0.95
                    no_improve = 0
                    max_no_improve = 200
                    
                    while time.time() < deadline and no_improve < max_no_improve:
                        # Pick a random edge to change
                        e = random.randint(0, num_edges - 1)
                        old_v = volts[e]
                        new_v = random.randint(0, m-1)
                        if new_v == old_v:
                            continue
                        delta = (new_v - old_v) % m
                        if delta == 0:
                            continue
                        
                        # Compute change in soft score
                        old_sums_affected = []
                        old_soft = 0.0
                        new_soft = 0.0
                        for c_idx, coeff in edge_to_cons[e]:
                            old_s = sums[c_idx]
                            new_s = (old_s + delta * coeff) % m
                            old_dist = min(old_s, m - old_s)
                            new_dist = min(new_s, m - new_s)
                            scale = max(1, m // 10)
                            old_contrib = math.exp(beta) if old_dist == 0 else math.exp(-beta * old_dist / scale)
                            new_contrib = math.exp(beta) if new_dist == 0 else math.exp(-beta * new_dist / scale)
                            old_soft += old_contrib
                            new_soft += new_contrib
                            old_sums_affected.append((c_idx, old_s, new_s))
                        
                        delta_soft = new_soft - old_soft
                        
                        # Accept if improvement or with probability exp(-delta/T)
                        if delta_soft < 0 or random.random() < math.exp(-delta_soft / T):
                            # Apply change
                            for c_idx, old_s, new_s in old_sums_affected:
                                sums[c_idx] = new_s
                            volts[e] = new_v
                            s_score += delta_soft
                            h_score = hard_score(sums)
                            if h_score == 0:
                                return volts
                            no_improve = 0
                        else:
                            no_improve += 1
                        
                        T *= cooling
                        if T < 0.01:
                            T = 0.01
                    
                    if h_score == 0:
                        return volts
                        
                # Final polish: local search on hard objective
                # Try each edge, each value, accept if reduces hard score
                for polish_iter in range(100):
                    if time.time() > deadline:
                        break
                    improved = False
                    for e in range(num_edges):
                        old_v = volts[e]
                        for new_v in range(m):
                            if new_v == old_v:
                                continue
                            delta = (new_v - old_v) % m
                            if delta == 0:
                                continue
                            
                            # Check hard score change
                            old_zeros = 0
                            new_zeros = 0
                            affected = []
                            for c_idx, coeff in edge_to_cons[e]:
                                old_s = sums[c_idx]
                                new_s = (old_s + delta * coeff) % m
                                if old_s == 0:
                                    old_zeros += 1
                                if new_s == 0:
                                    new_zeros += 1
                                affected.append((c_idx, old_s, new_s))
                            
                            if new_zeros < old_zeros:
                                # Apply
                                for c_idx, old_s, new_s in affected:
                                    sums[c_idx] = new_s
                                volts[e] = new_v
                                h_score += new_zeros - old_zeros
                                improved = True
                                if h_score == 0:
                                    return volts
                                break
                            elif new_zeros == old_zeros and random.random() < 0.05:
                                # Accept neutral move with small probability
                                for c_idx, old_s, new_s in affected:
                                    sums[c_idx] = new_s
                                volts[e] = new_v
                                improved = True
                                break
                            else:
                                # No change, don't revert (we didn't apply)
                                pass
                    if not improved:
                        break
                    if h_score == 0:
                        return volts
                
                return None
            
            sol = solve_m(m)
            if sol:
                n = 6 * m
                edges = []
                for u in range(base_n):
                    for i in range(m):
                        for v in range(base_n):
                            if u == v: continue
                            eid = get_edge_id(u, v)
                            if eid == -1: continue
                            if u < v:
                                phi = sol[eid]
                            else:
                                phi = -sol[eid]
                            
                            j = (i + phi) % m
                            id1 = u * m + i
                            id2 = v * m + j
                            if id1 < id2:
                                edges.append((id1, id2))
                
                return n, edges

        # Fallback
        return luw_construct(k, g)
    else:
        return luw_construct(k, g)

# EVOLVE-BLOCK-END
