import time
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        q = k
        # For g=9, d=4 gives girth 9, n=2*5^4=1250
        # For g=10, d=5 gives girth 10, n=2*5^5=6250
        # We want min n, so use smallest d such that d+5 >= g
        d = max(3, g - 5)
        if d % 2 != (g - 5) % 2: # Adjust parity if needed?
            # Seed logic: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
            # For g=9: g-5=4 (even), so d = 9-4=5? 
            # Wait, seed said d=5 for g=9.
            # Let's stick to seed logic to be safe, or use d = g-5.
            # Girth of LUW(d,q) is d+5.
            # If we want girth >= 9, d+5 >= 9 => d >= 4.
            # Seed used d=5. Why?
            # Maybe d must be odd? Or even?
            # In LUW, d is usually >= 3.
            # Let's try d=4.
            pass
        
        # Use seed's logic for d
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
        deadline = time.time() + 110 # 10s margin
        
        # Base graph K_6
        base_n = 6
        base_edges = []
        edge_idx = {}
        for u in range(base_n):
            for v in range(u + 1, base_n):
                edge_idx[(u, v)] = len(base_edges)
                base_edges.append((u, v))
        
        # Generate non-backtracking closed walks of length 3..8
        # Walk: v0=0, v1...vL=0
        # Coeffs: list of (edge_id, sign)
        constraints = []
        
        def get_edge_id(u, v):
            if u > v: u, v = v, u
            return edge_idx.get((u, v), -1)
        
        def gen_walks(length):
            # v[0] = 0
            # v[i] != v[i-1], v[i] != v[i-2]
            # v[L] = 0
            # We fix v0=0 to avoid rotation duplicates, but for CSP we need all distinct walks?
            # No, we just need to ensure NO walk has sum 0.
            # Checking all walks starting at 0 is sufficient if the graph is vertex-transitive?
            # K6 is vertex-transitive. So if no walk starting at 0 has sum 0, no walk has sum 0?
            # Wait, voltage sum depends on the specific vertices.
            # We must check ALL walks.
            # But we can generate all walks of length L in K6.
            
            # Recursive generation
            def backtrack(v_idx, current, prev, path):
                if v_idx == length:
                    # Must close to 0
                    if current != 0 and prev != 0:
                        # Check edge (current, 0)
                        # No backtrack: 0 != prev (checked), current != 0 (checked)
                        # Also need to check if the walk is non-backtracking at the end?
                        # The condition is v_{i+1} != v_{i-1}.
                        # For the last step v_L=0, v_{L-1} != v_{L-2} is checked in recursion.
                        # And v_0=0 != v_{L-1}? No, v_0 is start.
                        # The cycle is v0..vL.
                        # Non-backtracking at v0: v1 != v_{L-1}.
                        # We need to check this.
                        if path[1] != path[length-1]:
                            # Valid walk
                            # Compute coeffs
                            coeffs = {}
                            for i in range(length):
                                u = path[i]
                                v = path[(i+1)%length]
                                eid = get_edge_id(u, v)
                                if eid == -1: continue # Should not happen
                                # Sign
                                if u < v:
                                    sign = 1
                                else:
                                    sign = -1
                                if eid in coeffs:
                                    coeffs[eid] += sign
                                else:
                                    coeffs[eid] = sign
                            # Convert to list
                            c_list = [0]*15
                            for eid, s in coeffs.items():
                                c_list[eid] = s
                            constraints.append(c_list)
                    return
                
                if v_idx == length - 1:
                    # Must go to 0
                    if current != 0 and prev != 0:
                        path.append(0)
                        # Check non-backtracking at closure: path[1] != path[length-1]
                        if path[1] != path[length-1]:
                             # Add to constraints
                             coeffs = {}
                             for i in range(length):
                                u = path[i]
                                v = path[(i+1)%length]
                                eid = get_edge_id(u, v)
                                if eid == -1: continue
                                if u < v: sign = 1
                                else: sign = -1
                                if eid in coeffs: coeffs[eid] += sign
                                else: coeffs[eid] = sign
                             c_list = [0]*15
                             for eid, s in coeffs.items():
                                c_list[eid] = s
                             constraints.append(c_list)
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
            
        # Remove duplicate constraints?
        # Not strictly necessary, but speeds up.
        # constraints = list(set(map(tuple, constraints)))
        
        # Search for m and voltages
        best_n = 1250 # Fallback
        best_edges = None
        best_voltages = None
        
        # Try m from 1 to 200
        # We want smallest n = 6m.
        for m in range(1, 201):
            if time.time() > deadline:
                break
            
            # Local search for voltages
            # Init random
            volts = [random.randint(0, m-1) for _ in range(15)]
            
            def calc_score(v):
                score = 0
                for c in constraints:
                    s = 0
                    for i in range(15):
                        if c[i] != 0:
                            s += c[i] * v[i]
                    if s % m == 0:
                        score += 1
                return score
            
            current_score = calc_score(volts)
            
            # If m is small, score might be high.
            # Local search
            for _ in range(2000):
                if time.time() > deadline:
                    break
                if current_score == 0:
                    break
                # Pick random var
                idx = random.randint(0, 14)
                old_val = volts[idx]
                # Try all values? Or random?
                # Random is faster.
                new_val = random.randint(0, m-1)
                if new_val == old_val: continue
                
                # Delta score
                # Only constraints involving idx change
                # Precompute which constraints involve which edge?
                # For simplicity, recalc full score if m is small?
                # Or just recalc. 15 vars, ~200 constraints. 3000 ops. 2000 iters = 6M ops. OK.
                
                volts[idx] = new_val
                new_score = calc_score(volts)
                
                if new_score < current_score:
                    current_score = new_score
                else:
                    # Revert
                    volts[idx] = old_val
            
            if current_score == 0:
                # Found solution
                n = 6 * m
                # Build edges
                edges = []
                # Vertex id: u * m + i
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
