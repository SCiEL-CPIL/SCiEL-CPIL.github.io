import time
import math

def construct(k, g):
    deadline = time.time() + 115.0  # Safety margin

    if k != 5 or g != 9:
        return construct_luw(k, g)

    # Search for circulant graphs C_n({-b, -a, a, b, n/2})
    # n must be even.
    # Girth check: BFS from 0 on Z_n.
    # If 0 is not reached in steps 1..g-1, girth >= g.
    
    def check_circulant_girth(n, a, b, min_girth):
        # Generators: a, n-a, b, n-b, n//2
        # Note: n-a is -a mod n.
        # We use BFS.
        # dist[v] = shortest distance from 0 to v.
        # If dist[0] (which is 0) is the only 0, we are safe until we hit 0 again?
        # No, BFS finds shortest paths. If the shortest path from 0 to 0 is > 0,
        # that's a cycle.
        # We start at 0.
        # We expand neighbors.
        # If we reach a node already visited, we might form a cycle.
        # But for girth, we specifically want the shortest cycle through 0.
        # In a vertex-transitive graph (Cayley), the girth is the length of the
        # shortest cycle through any vertex.
        # So BFS from 0:
        # Level 0: 0
        # Level 1: gens
        # ...
        # If 0 appears in Level L, girth is L.
        # We need girth >= 9. So 0 must NOT appear in Levels 1..8.
        
        # Since 0 is the start, we don't "visit" it again in the sense of a tree,
        # but we check if any neighbor of a node at level d-1 is 0.
        
        # Efficient implementation:
        # visited = [-1] * n
        # visited[0] = 0
        # queue = [0]
        # for d in range(1, min_girth):
        #     new_queue = []
        #     for u in queue:
        #         for g in gens:
        #             v = (u + g) % n
        #             if v == 0:
        #                 return False # Cycle of length d
        #             if visited[v] == -1:
        #                 visited[v] = d
        #                 new_queue.append(v)
        #     queue = new_queue
        #     if not queue:
        #         break # Graph is a tree? Impossible for regular graph unless n is small.
        # return True
        
        # Optimization:
        # The set of generators is fixed.
        # We can use a list for queue.
        # We can use a list for visited.
        
        gens = (a, n-a, b, n-b, n//2)
        visited = [-1] * n
        visited[0] = 0
        current_level = [0]
        
        for d in range(1, min_girth):
            next_level = []
            for u in current_level:
                # Unroll loop for speed?
                v = (u + a) % n
                if v == 0: return False
                if visited[v] == -1:
                    visited[v] = d
                    next_level.append(v)
                    
                v = (u + (n-a)) % n
                if v == 0: return False
                if visited[v] == -1:
                    visited[v] = d
                    next_level.append(v)
                    
                v = (u + b) % n
                if v == 0: return False
                if visited[v] == -1:
                    visited[v] = d
                    next_level.append(v)
                    
                v = (u + (n-b)) % n
                if v == 0: return False
                if visited[v] == -1:
                    visited[v] = d
                    next_level.append(v)
                    
                v = (u + (n//2)) % n
                if v == 0: return False
                if visited[v] == -1:
                    visited[v] = d
                    next_level.append(v)
            
            if not next_level:
                # If we exhausted all nodes before reaching min_girth,
                # and 0 was not hit, then girth is > min_girth-1?
                # Actually, if the graph is connected and we visited all nodes,
                # and 0 was not hit, it means there is no cycle of length < min_girth?
                # Wait, if we visited all nodes, we must have hit 0 eventually if there's a cycle.
                # If the graph is a tree, it's not regular (except small cases).
                # A 5-regular graph on n nodes has cycles.
                # If we break here, it means we didn't find a cycle < min_girth.
                return True
            
            current_level = next_level
            
        return True

    def build_circulant(n, a, b):
        edges = []
        for i in range(n):
            for g in (a, n-a, b, n-b, n//2):
                j = (i + g) % n
                if i < j:
                    edges.append((i, j))
        return n, edges

    # Search strategy:
    # n must be even.
    # Lower bound 428.
    # We iterate n. For each n, iterate a, b.
    # To avoid duplicates, we can assume 1 <= a < b < n/2.
    # Also, due to symmetry, we might restrict a,b further?
    # For now, just iterate.
    
    # Optimization: Skip n if it's not a multiple of something?
    # No obvious constraint.
    
    # Time management:
    # We will spend most of the time in check_circulant_girth.
    # We will break if time > deadline.
    
    # Let's try n from 428 to 6250 (or less, since LUW is 6250).
    # If we find one with n < 6250, we return it.
    
    # To speed up, we can check smaller n first.
    # However, small n are more likely to have short cycles.
    # We can randomize the order of (a,b) for a given n?
    # Or just iterate.
    
    # Let's limit the search to n < 6250.
    # If we don't find one, we return LUW.
    
    # We can also try to prune (a,b) pairs.
    # If a == b, or a == n/2, etc., we skip.
    # We assume a < b < n/2 and a != b.
    # Also a != n/2, b != n/2.
    
    # Let's run the search.
    # We will do a quick check for n up to, say, 1000 first, then expand.
    # But we have 115 seconds.
    # Let's just iterate n from 428 to 6248.
    
    # To be safe, we'll check time frequently.
    
    # Note: For n=428, number of (a,b) pairs is ~ 428^2/8 ~ 22k.
    # Check cost ~ 1000 ops.
    # Total ~ 22M ops. Fast.
    # For n=6000, pairs ~ 6000^2/8 ~ 4.5M.
    # Check cost ~ 6000 ops.
    # Total ~ 27B ops. Too slow.
    
    # So we can't check all (a,b) for large n.
    # We need a heuristic.
    # Maybe check only a few random (a,b) for large n?
    # Or stop searching large n if we don't find small ones?
    
    # Strategy:
    # 1. Exhaustive search for n in [428, 2000].
    # 2. Randomized search for n in [2000, 6250].
    
    # Let's implement this.
    
    # Phase 1: Exhaustive for small n
    for n in range(428, 2000, 2):
        if time.time() > deadline - 10: # Keep 10s for fallback
            break
        max_a = n // 2
        for a in range(1, max_a):
            if time.time() > deadline - 10:
                break
            for b in range(a + 1, max_a):
                # Skip if a or b is n/2? n/2 is max_a, so b < max_a ensures b != n/2.
                # a < max_a ensures a != n/2.
                if check_circulant_girth(n, a, b, 9):
                    return build_circulant(n, a, b)
                    
    # Phase 2: Randomized for larger n
    # We'll try a few random (a,b) for n from 2000 to 6248.
    import random
    random.seed(42) # Deterministic
    
    # We'll sample n values.
    # Let's try 100 random n values in [2000, 6248].
    # For each n, try 100 random (a,b) pairs.
    
    n_candidates = random.sample(range(2000, 6248, 2), 100)
    # Sort by n to try smaller ones first?
    n_candidates.sort()
    
    for n in n_candidates:
        if time.time() > deadline - 5:
            break
        max_a = n // 2
        # Try 200 random pairs
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            a = random.randint(1, max_a - 1)
            b = random.randint(1, max_a - 1)
            if a == b:
                continue
            # Ensure a < b for consistency? Not strictly needed for check.
            if check_circulant_girth(n, a, b, 9):
                return build_circulant(n, a, b)
                
    # Fallback to LUW
    return construct_luw(5, 9)

def construct_luw(k, g):
    q = k
    min_d = max(3, g - 5)
    if min_d % 2 == 0:
        min_d += 1
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    def get_rule(d):
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return rule
    
    def build_luw(d):
        rule = get_rule(d)
        
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
    
    return build_luw(min_d)
