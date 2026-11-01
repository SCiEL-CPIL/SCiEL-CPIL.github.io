import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100.0  # 100s limit, leaving 20s margin
    
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

    # Tinhofer Construction
    def tinhofer_construct(k, g):
        # m pairs of vectors, and 1 involution if k is odd
        m = k // 2
        has_involution = (k % 2 == 1)
        
        # We search for prime p and dimension d
        # Estimate target size
        # Moore bound approx: k * (k-1)^((g-1)/2)
        # We want 2 * p^d to be close to this
        
        # List of primes to try
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        
        best_result = None
        
        for p in primes:
            if time.time() > deadline:
                break
                
            # Estimate d
            # 2 * p^d ~ k * (k-1)^((g-1)/2)
            # d ~ log_p( k * (k-1)^((g-1)/2) / 2 )
            import math
            target_size = k * ((k - 1) ** ((g - 1) // 2))
            if target_size < 10:
                d_min = 1
            else:
                d_min = max(1, int(math.log(target_size / 2, p)) - 1)
                
            for d in range(d_min, d_min + 5): # Try a few dimensions
                if time.time() > deadline:
                    break
                    
                n_vertices = 2 * (p ** d)
                
                # Try a few random sets of vectors
                for attempt in range(20):
                    if time.time() > deadline:
                        break
                    
                    rng = random.Random(12345 + attempt * p + d)
                    
                    # Generate m random non-zero vectors
                    vectors = []
                    valid_vectors = True
                    while len(vectors) < m:
                        v = tuple(rng.randint(0, p - 1) for _ in range(d))
                        if all(x == 0 for x in v):
                            continue
                        # Check if v is already in vectors or -v
                        conflict = False
                        for u in vectors:
                            # v == u or v == -u (mod p)
                            if all((v[i] == u[i] or (v[i] + u[i]) % p == 0) for i in range(d)):
                                conflict = True
                                break
                        if not conflict:
                            vectors.append(v)
                    
                    if len(vectors) < m:
                        continue
                    
                    # Check girth using BFS on the Cayley graph
                    # Generators: (0, v), (0, -v) for v in vectors, and (1, 0) if has_involution
                    generators = []
                    for v in vectors:
                        neg_v = tuple((-x) % p for x in v)
                        generators.append((0, v))
                        generators.append((0, neg_v))
                    if has_involution:
                        generators.append((1, tuple(0 for _ in range(d))))
                    
                    # BFS from (0, 0)
                    # State: (bit, coord)
                    start_state = (0, tuple(0 for _ in range(d)))
                    visited = {start_state: (0, None)} # state: (depth, parent_state)
                    queue = deque([start_state])
                    girth_found = False
                    max_depth = (g - 1) // 2
                    
                    while queue:
                        curr = queue.popleft()
                        depth, parent = visited[curr]
                        
                        if depth > max_depth:
                            continue
                            
                        for gen in generators:
                            # Next state
                            b, v = gen
                            cb, cv = curr
                            nb = (cb + b) % 2
                            nv = tuple((cv[i] + v[i]) % p for i in range(d))
                            nstate = (nb, nv)
                            
                            if nstate not in visited:
                                visited[nstate] = (depth + 1, curr)
                                queue.append(nstate)
                            else:
                                # Found a cycle
                                # The cycle length is depth + 1 + depth_parent
                                # We need to be careful: the edge from curr to nstate closes the cycle
                                # The length of the path from start to curr is depth.
                                # The length of the path from start to nstate is depth_parent.
                                # The total cycle length is depth + 1 + depth_parent.
                                # However, this might not be a simple cycle if the paths share vertices.
                                # For a Cayley graph, if we find a collision in BFS, the cycle length is at most depth + 1 + depth_parent.
                                # We just need to know if it's < g.
                                cycle_len = depth + 1 + visited[nstate][0]
                                if cycle_len < g:
                                    girth_found = True
                                    break
                        if girth_found:
                            break
                            
                    if not girth_found:
                        # Valid graph
                        # Construct edges
                        edges = []
                        # Map state to index
                        index_map = {}
                        idx = 0
                        for b in range(2):
                            for i in range(p ** d):
                                # Convert i to tuple in Z_p^d
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
                        
                        # Check if this is better than best_result
                        if best_result is None or n_vertices < best_result[0]:
                            best_result = (n_vertices, edges)
                            
                        # Return the first valid one found to save time?
                        # Or keep searching for smaller?
                        # The problem asks for "as few vertices as possible".
                        # Smaller p and d usually give smaller n.
                        # We iterate p in increasing order, d in increasing order.
                        # So the first valid one is likely the smallest.
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
