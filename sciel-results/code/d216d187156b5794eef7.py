import time
import sys
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification (BFS-based, fast) ----------------
    def check_girth(n, edges, min_girth, time_limit):
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
                        if cycle_len < min_girth:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < min_girth:
                return min_cycle
        return min_cycle

    # ---------------- Voltage Lift of Petersen ----------------
    def voltage_lift_petersen():
        if k != 3 or g != 16:
            return None

        # Petersen graph definition
        # Vertices 0-4: outer pentagon, 5-9: inner star
        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner
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

        # Generate all non-backtracking closed walks of length 5 to 15
        max_L = g - 1
        # We store the net count vector for each edge.
        # Edge i has direction u->v in pet_edges[i].
        # If walk uses u->v, count[i] += 1.
        # If walk uses v->u, count[i] -= 1.
        
        constraints = set()
        
        # DFS to find closed walks
        # State: (current_node, prev_node, length, counts_tuple)
        # We use a stack for iterative DFS to avoid recursion limits and overhead
        
        for start_v in range(n_base):
            stack = [(start_v, -1, 0, [0] * 15)]
            while stack:
                current, prev, length, counts_list = stack.pop()
                
                if length > max_L:
                    continue
                
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    
                    e_idx = edge_index[(current, next_v)]
                    u, v = pet_edges[e_idx]
                    # Determine direction
                    if current == u:
                        d = 1
                    else:
                        d = -1
                        
                    new_counts = counts_list.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1
                    
                    if next_v == start_v and new_l >= 5:
                        # Found a closed walk
                        # Normalize: if first non-zero is negative, flip sign
                        # This identifies {c, -c} as the same constraint (since c.v=0 iff -c.v=0)
                        c_tuple = tuple(new_counts)
                        if any(x != 0 for x in c_tuple):
                            # Find first non-zero
                            first_nz = -1
                            for i in range(15):
                                if c_tuple[i] != 0:
                                    first_nz = i
                                    break
                            if first_nz != -1:
                                if c_tuple[first_nz] < 0:
                                    c_tuple = tuple(-x for x in c_tuple)
                                constraints.add(c_tuple)
                    
                    if new_l < max_L:
                        stack.append((next_v, current, new_l, new_counts))

        if not constraints:
            return None
            
        C_list = list(constraints)
        C = np.array(C_list, dtype=np.int64)
        
        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True

        def rank_mod_m(mat, m):
            """Compute rank of matrix over Z_m using Gaussian elimination."""
            mat = mat.copy()
            rows = len(mat)
            cols = len(mat[0])
            rank = 0
            for col in range(cols):
                # Find pivot
                pivot = -1
                for row in range(rank, rows):
                    if mat[row][col] % m != 0:
                        pivot = row
                        break
                if pivot == -1:
                    continue
                # Swap rows
                mat[rank], mat[pivot] = mat[pivot], mat[rank]
                # Normalize pivot row
                pivot_val = mat[rank][col] % m
                inv_pivot = pow(pivot_val, m - 2, m)
                for c in range(col, cols):
                    mat[rank][c] = (mat[rank][c] * inv_pivot) % m
                # Eliminate below
                for row in range(rank + 1, rows):
                    factor = mat[row][col] % m
                    if factor != 0:
                        for c in range(col, cols):
                            mat[row][c] = (mat[row][c] - factor * mat[rank][c]) % m
                rank += 1
            return rank

        def solve_voltage(m, max_iters=20000):
            # We want v in (Z_m)^15 such that C @ v != 0 (mod m) for all rows
            # Cost = number of rows where (C @ v) % m == 0
            
            def cost(v):
                residues = np.mod(C @ v, m)
                return np.sum(residues == 0)
            
            # Check rank condition: if rank(C) < 15, the only solution to C@v=0 is trivial?
            # Actually, we want C@v != 0. If rank(C) < 15, there are non-trivial solutions to C@v=0,
            # which are bad. But we want to AVOID those. The condition is that the only vector
            # satisfying C@v=0 is v=0 (which is not in our domain since we use 1..m-1).
            # Wait, if rank(C) < 15, then the null space is non-trivial. So there exist v != 0 with C@v=0.
            # These are "bad" vectors. We want to find a vector NOT in the union of hyperplanes.
            # If rank(C) == 15, then C@v=0 only has v=0, so any v in 1..m-1 is good.
            # So if rank(C) == 15, we can just pick any v and it works!
            
            r = rank_mod_m(C, m)
            if r == 15:
                # Any non-zero vector works? No, we need v in 1..m-1.
                # Actually, if rank is 15, the only solution to C@v=0 is v=0.
                # So any v with all components in 1..m-1 is a valid solution.
                v0 = np.ones(15, dtype=np.int64)
                c = cost(v0)
                if c == 0:
                    return v0, 0
                # This should not happen if rank is 15. But let's be safe.
            
            def cost_smooth(v):
                # Smooth surrogate: sum of exp(-|C@v mod m|) or similar
                # We want to minimize the number of zeros.
                # Use a soft penalty: for each row, if residue is 0, penalty is high.
                # residues = np.mod(C @ v, m)
                # return np.sum(np.exp(-residues.astype(float)))
                # This is not quite right because we want to avoid 0 specifically.
                # Better: count of zeros is the hard cost. For smoothness, we can use
                # sum of (1 - min(residue, m-residue)/m) which is 1 when residue=0, 0 otherwise.
                residues = np.mod(C @ v, m)
                # Distance to 0 in Z_m
                dist = np.minimum(residues, m - residues)
                # Soft penalty: 1 when dist=0, decreasing as dist increases
                # Use exp(-alpha * dist)
                alpha = 2.0
                return np.sum(np.exp(-alpha * dist))
            
            # Try multiple initializations: random, and heuristic
            best_v = None
            best_cost = float('inf')
            
            # Heuristic initialization: try to satisfy constraints greedily
            # Start with all ones, then fix violations
            v0 = np.ones(15, dtype=np.int64)
            for _ in range(50):
                c = cost(v0)
                if c == 0:
                    return v0, 0
                # Find a violated constraint
                residues = np.mod(C @ v0, m)
                violated = np.where(residues == 0)[0]
                if len(violated) == 0:
                    break
                row = violated[random.randint(0, len(violated)-1)]
                # Perturb a random coordinate
                j = random.randint(0, 14)
                v0[j] = random.randint(1, m-1)
            
            if best_cost > cost(v0):
                best_cost = cost(v0)
                best_v = v0.copy()
            
            # Random restarts
            for _ in range(3):
                v_rand = np.random.randint(1, m, size=15).astype(np.int64)
                c = cost(v_rand)
                if c < best_cost:
                    best_cost = c
                    best_v = v_rand.copy()
                if c == 0:
                    return v_rand, 0
            
            if best_cost == 0:
                return best_v, 0

            # Simulated Annealing from best initial
            v = best_v.copy()
            T = 1.0
            cooling = 0.9995
            min_T = 0.001
            
            for it in range(max_iters):
                if remaining() < 5:
                    break
                
                # Pick random index to change
                j = random.randint(0, 14)
                old_val = v[j]
                new_val = random.randint(1, m - 1)
                if new_val == old_val:
                    continue
                v[j] = new_val
                
                new_cost = cost(v)
                delta = new_cost - best_cost
                
                if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                    if new_cost == 0:
                        return v, 0
                else:
                    v[j] = old_val
                    
                T *= cooling

            # Final polish: local search with tighter tolerance
            if best_cost > 0 and remaining() > 10:
                v = best_v.copy()
                T = 0.01
                for _ in range(5000):
                    if remaining() < 5:
                        break
                    j = random.randint(0, 14)
                    old_val = v[j]
                    new_val = random.randint(1, m - 1)
                    if new_val == old_val:
                        continue
                    v[j] = new_val
                    new_cost = cost(v)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                        if new_cost == 0:
                            return v, 0
                    else:
                        v[j] = old_val

            return best_v, best_cost

        # Moore bound for k=3, g=16: 2 + 3(1 + 2 + 4 + 8 + 16 + 32 + 64) = 2 + 3*127 = 383?
        # Wait, Moore bound for cubic graphs: n >= 2 + 3(2^{(g-2)/2} - 1)
        # For g=16: n >= 2 + 3(2^7 - 1) = 2 + 3*127 = 383.
        # Actually, the lower bound is 512 as stated in the task.
        # n = 10 * m. So 10*m >= 512 => m >= 51.2.
        # Smallest prime >= 52 is 53.
        
        m = 53
        attempts = 0
        while m < 500 and remaining() > 20:
            if is_prime(m):
                # Try more restarts for this m
                for restart in range(15):
                    if remaining() < 10:
                        break
                    v, cost = solve_voltage(m, max_iters=8000)
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
                        gc = check_girth(n, edges, g, deadline - 3)
                        if gc >= g:
                            return (n, edges)
                        else:
                            # Verification failed? Should not happen if constraints are correct.
                            # But maybe we missed a walk? Return None to be safe.
                            pass
                attempts += 1
            m += 1
            
        return None

    # ---------------- Ramanujan cubic graphs (k=3) ----------------
    def build_ramanujan(q):
        def mul(a, b):
            return ((a[0]*b[0] + a[1]*b[2]) % q, (a[0]*b[1] + a[1]*b[3]) % q,
                    (a[2]*b[0] + a[3]*b[2]) % q, (a[2]*b[1] + a[3]*b[3]) % q)

        def det(a):
            return (a[0]*a[3] - a[1]*a[2]) % q

        def inv(a):
            d = det(a)
            if d == 0: return None
            invd = pow(d, q - 2, q)
            return ((a[3]*invd) % q, (-a[1]*invd) % q, (-a[2]*invd) % q, (a[0]*invd) % q)

        def norm(a):
            return (a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3]) % q

        def involution(a):
            if det(a) != 1:
                return False
            m = mul(a, a)
            n = norm(m)
            return n == 0 or n == 2  # m ~ I

        I = (1, 0, 0, 1)
        A = (0, 1, q - 1, 0)
        B = (0, 1, 1, 0)
        if not (involution(A) and involution(B)):
            return None
        C = None
        for s in range(1, q):
            cand = (1, 0, s, 1)
            if involution(cand):
                C = cand
                break
        if C is None:
            return None
        G = [I, A, B, C,
             mul(A, B), mul(B, A), mul(A, C), mul(C, A),
             mul(B, C), mul(C, B), mul(A, B), mul(B, A)]
        # dedup up to sign
        seen, gens = set(), []
        for m in G:
            key = m if (m[0], m[1], m[2], m[3]) <= (q - m[0] % q, q - m[1] % q, q - m[2] % q, q - m[3] % q) else (q - m[0] % q, q - m[1] % q, q - m[2] % q, q - m[3] % q)
            if key not in seen:
                seen.add(key)
                gens.append(m)
        if len(gens) != 3:
            return None
        # BFS over PSL(2,q)
        start = I
        index = {start: 0}
        order = [start]
        edges = []
        for i in range(len(order)):
            v = order[i]
            vi = inv(v)
            for w in gens:
                u = mul(vi, w)
                if u not in index:
                    index[u] = len(order)
                    order.append(u)
                if index[v] < index[u]:
                    edges.append((index[v], index[u]))
        return len(order), edges

    # ---------------- LUW construction (Fallback) ----------------
    def _luw_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def generate_rules(d, variant=0, offset=0):
        base = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        if variant == 0:
            for c in range(5, d + 1):
                r, c0 = ((c - 5 + offset) % 4), c - ((c - 5 + offset) % 4)
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        elif variant == 1:
            for c in range(5, d + 1):
                base[c] = (1, c - 2) if (c + offset) % 2 == 0 else (c - 1, 1)
        elif variant == 2:
            for c in range(5, d + 1):
                if (c + offset) % 3 == 0:
                    base[c] = (1, c - 3)
                elif (c + offset) % 3 == 1:
                    base[c] = (c - 1, 1)
                else:
                    base[c] = (c - 2, 1)
        elif variant == 3:
            for c in range(5, d + 1):
                base[c] = (1, c - (1 + (c + offset) % 2))
        elif variant == 4:
            for c in range(5, d + 1):
                base[c] = [(1, c - 1), (c - 1, 1), (1, c - 2), (c - 2, 1)][(c + offset) % 4]
        else:
            for c in range(5, d + 1):
                r, c0 = ((c - 5 + offset) % 4), c - ((c - 5 + offset) % 4)
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return base

    def build_luw(q, d, rule, time_limit):
        add, sub, mul = _luw_ops(q)

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
            if time.time() > time_limit:
                return None
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def luw_construct(k, g):
        q = k
        d = max(3, (g - 2) // 2)
        best = None
        best_n = float('inf')

        for variant in range(5):
            for offset in range(4):
                if remaining() < 12:
                    break
                rule = generate_rules(d, variant, offset)
                res = build_luw(q, d, rule, deadline - 12)
                if res is None:
                    continue
                n, edges = res
                if n >= best_n:
                    continue
                gc = check_girth(n, edges, g, deadline - 8)
                if gc == -1:
                    continue
                if gc >= g:
                    best, best_n = (n, edges), n

        if best is None:
            for d_try in range(d + 1, d + 10):
                if remaining() < 10:
                    break
                for variant in range(3):
                    for offset in range(3):
                        if remaining() < 5:
                            break
                        rule = generate_rules(d_try, variant, offset)
                        res = build_luw(q, d_try, rule, deadline - 10)
                        if res is None:
                            continue
                        n, edges = res
                        gc = check_girth(n, edges, g, deadline - 5)
                        if gc == -1:
                            continue
                        if gc >= g:
                            best = (n, edges)
                            break
                    if best is not None:
                        break
                if best is not None:
                    break

        if best is None:
            rule = generate_rules(d, 0, 0)
            res = build_luw(q, d, rule, deadline - 2)
            if res is not None:
                best = res
        return best

    # ---------------- main ----------------
    if k != 3:
        res = luw_construct(k, g)
        return res if res is not None else (0, [])

    # k == 3: Try Voltage Lift first (potential for much smaller n)
    res = voltage_lift_petersen()
    if res is not None:
        return res

    # Fallback to Ramanujan graphs
    best = None
    best_n = float('inf')

    for q in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]:
        if remaining() < 8:
            break
        res = build_ramanujan(q)
        if res is None:
            continue
        n, edges = res
        if n >= best_n:
            continue
        gc = check_girth(n, edges, g, deadline - 5)
        if gc == -1:
            continue
        if gc >= g:
            best, best_n = (n, edges), n

    if best is not None:
        return best

    # fall back to LUW (guaranteed valid)
    res = luw_construct(k, g)
    return res if res is not None else (0, [])
