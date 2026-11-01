import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit, leaving 10s margin
    
    # Fallback to LUW if Tinhofer fails or takes too long
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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

    # Tinhofer Construction with optimized girth check
    def tinhofer_construct(k, g):
        m = k // 2
        has_involution = (k % 2 == 1)
        
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        
        best_result = None
        
        for p in primes:
            if time.time() > deadline:
                break
                
            import math
            target_size = k * ((k - 1) ** ((g - 1) // 2))
            if target_size < 10:
                d_min = 1
            else:
                d_min = max(1, int(math.log(target_size / 2, p)) - 1)
                
            for d in range(d_min, d_min + 4):
                if time.time() > deadline:
                    break
                    
                n_vertices = 2 * (p ** d)
                
                for attempt in range(30):
                    if time.time() > deadline:
                        break
                    
                    rng = random.Random(12345 + attempt * 1000 + p * 100 + d)
                    
                    # Generate m random non-zero vectors with better conflict checking
                    vectors = []
                    used_lines = set()  # Store normalized lines to avoid ±v conflicts
                    valid_vectors = True
                    while len(vectors) < m:
                        v = tuple(rng.randint(0, p - 1) for _ in range(d))
                        if all(x == 0 for x in v):
                            continue
                        # Normalize: find first non-zero, make it 1 (or keep as is for p=2)
                        # For conflict check, store both v and -v in a canonical form
                        neg_v = tuple((-x) % p for x in v)
                        # Use lexicographically smaller of v, neg_v as canonical
                        canon = v if v < neg_v else neg_v
                        if canon in used_lines:
                            continue
                        used_lines.add(canon)
                        vectors.append(v)
                    
                    if len(vectors) < m:
                        continue
                    
                    # Build generators
                    generators = []
                    for v in vectors:
                        neg_v = tuple((-x) % p for x in v)
                        generators.append((0, v))
                        generators.append((0, neg_v))
                    if has_involution:
                        generators.append((1, tuple(0 for _ in range(d))))
                    
                    # Fast girth check using BFS with early termination
                    # We only need to check if there's a cycle of length < g
                    # In a Cayley graph, a cycle of length < g means there's a relation
                    # among generators of length < g
                    # For girth >= g, we need no non-trivial relation of length < g
                    # This is equivalent to: the ball of radius (g-1)//2 around identity has no collisions
                    # except for the trivial backtracking ones
                
                    max_depth = (g - 1) // 2
                    if max_depth < 1:
                        max_depth = 1
                    
                    # BFS from (0, 0)
                    start_state = (0, tuple(0 for _ in range(d)))
                    visited = {start_state: 0}  # state: depth
                    queue = deque([(start_state, 0)])
                    girth_found = False
                    
                    while queue:
                        curr, depth = queue.popleft()
                        
                        if depth >= max_depth:
                            continue
                            
                        for gen in generators:
                            gb, gv = gen
                            cb, cv = curr
                            nb = (cb + gb) % 2
                            nv = tuple((cv[i] + gv[i]) % p for i in range(d))
                            nstate = (nb, nv)
                            
                            if nstate not in visited:
                                visited[nstate] = depth + 1
                                queue.append((nstate, depth + 1))
                            else:
                                # Found a cycle
                                # The path from start to curr has length depth
                                # The path from start to nstate has length visited[nstate]
                                # The edge from curr to nstate closes a cycle
                                # Cycle length = depth + 1 + visited[nstate]
                                # But we need to be careful about backtracking
                                # In a non-backtracking sense, we should exclude the case
                                # where nstate is the parent of curr
                                # For simplicity, if cycle_len < g, mark as found
                                cycle_len = depth + 1 + visited[nstate]
                                if cycle_len < g:
                                    girth_found = True
                                    break
                        if girth_found:
                            break
                    
                    if not girth_found:
                        # Valid graph - construct edges
                        edges = []
                        # Map state to index
                        index_map = {}
                        idx = 0
                        for b in range(2):
                            for i in range(p ** d):
                                digits = []
                                temp = i
                                for _ in range(d):
                                    digits.append(temp % p)
                                    temp //= p
                                coord = tuple(digits)
                                state = (b, coord)
                                index_map[state] = idx
                                idx += 1
                                
                        # Add edges
                        for b in range(2):
                            for i in range(p ** d):
                                digits = []
                                temp = i
                                for _ in range(d):
                                    digits.append(temp % p)
                                    temp //= p
                                coord = tuple(digits)
                                state = (b, coord)
                                u_idx = index_map[state]
                                
                                for gen in generators:
                                    gb, gv = gen
                                    nb = (b + gb) % 2
                                    nv = tuple((coord[i] + gv[i]) % p for i in range(d))
                                    nstate = (nb, nv)
                                    v_idx = index_map[nstate]
                                    if u_idx < v_idx:
                                        edges.append((u_idx, v_idx))
                        
                        if best_result is None or n_vertices < best_result[0]:
                            best_result = (n_vertices, edges)
                        
                        return best_result

        return best_result

    try:
        result = tinhofer_construct(k, g)
        if result is not None:
            return result
    except Exception as e:
        pass

    # Fallback
    return luw_construct(k, g)
