import time
import itertools

def construct(k, g):
    deadline = time.time() + 115.0  # Safety margin
    
    # If k and g are not 5 and 9, fall back to LUW
    if k != 5 or g != 9:
        return construct_luw(k, g)

    # Search for a circulant graph C_n({-b, -a, a, b, n/2})
    # n must be even
    # We want girth >= 9
    
    def check_girth_circulant(n, a, b, min_girth):
        # Generators: a, n-a, b, n-b, n/2
        # Note: n-a is -a mod n
        gens = [a, n-a, b, n-b, n//2]
        
        # Check for cycles of length < min_girth
        # Using BFS from each vertex is O(n * k^depth)
        # For n~500, k=5, depth=4, this is ~500*625 = 312500 ops.
        # Number of (a,b) pairs is ~ n^2/4 = 62500.
        # Total ops ~ 2 * 10^10. Too slow.
        
        # Optimized check:
        # Precompute all "bad" combinations of generators that sum to 0 mod n
        # for lengths 3 to min_girth-1.
        
        # Actually, we can just check if any linear combination u*a + v*b + w*(n/2) == 0 mod n
        # where u, v, w are small integers derived from the path.
        
        # Let's use a simpler check:
        # For each vertex 0, do a non-backtracking walk up to depth (min_girth-1)//2.
        # If we return to 0, we found a cycle.
        # Wait, returning to 0 is not the only cycle.
        # We need to check all vertices.
        
        # However, for circulant graphs, if there is a cycle, it is "symmetric".
        # We can just check from vertex 0.
        # Is that true?
        # A cycle in a circulant graph is a sequence of generators.
        # The existence of such a sequence is independent of the starting vertex.
        # So if there is a cycle of length L, there is a sequence of L generators summing to 0 mod n.
        # So we just need to check if there is a sequence of generators of length < min_girth
        # that sums to 0 mod n, with no partial sum 0 mod n.
        
        # Let's enumerate all sequences of generators of length L < min_girth.
        # There are 5^L such sequences.
        # For L=8, 5^8 = 390625.
        # For each sequence, check if sum == 0 mod n and partial sums != 0 mod n.
        # This is fast.
        
        for L in range(3, min_girth):
            # Generate all sequences of length L
            # We can do this recursively or with product
            # But we can prune: if partial sum is 0, skip.
            
            # Use a set of visited partial sums to avoid duplicates?
            # No, we just need to know if ANY sequence works.
            
            # We can just iterate through all 5^L combinations.
            # 390625 * 8 (lengths) = ~3M checks.
            # Each check is O(L) to compute sum.
            # Total ~ 25M ops per (a,b) pair.
            # 62500 pairs * 25M = 1.5 * 10^12. Still too slow.
            
            pass
        
        return True

    # The above check is too slow.
    # We need a faster girth check for circulant graphs.
    
    # Alternative: Use the LUW construction but try to find a smaller one?
    # No, LUW is fixed.
    
    # Let's try a different strategy:
    # Use a known small (5,9) cage if available?
    # The task says "smallest known such graph has 1116 vertices".
    # We can hardcode a 1116 vertex graph if we can find its edge list?
    # We don't have it.
    
    # Let's try a randomized greedy construction.
    # Start with n vertices, try to add edges to make it 5-regular and girth >= 9.
    
    # Or, use the LUW graph but with a different q?
    # q must be prime or 4. k=5, so q=5.
    
    # Let's go back to the circulant search but optimize the check.
    # We can precompute the set of "bad" sums for a given n?
    # No, the sums depend on a, b.
    
    # What if we iterate over possible "cycle lengths" and solve for a, b?
    # For a cycle of length L, we have u*a + v*b + w*(n/2) = m*n.
    # This is a linear Diophantine equation.
    # We want to AVOID such solutions.
    
    # This is getting complicated.
    
    # Let's just return the LUW graph for now, but maybe try a smaller d?
    # d=5 gives n=6250.
    # Is there a d=4 construction?
    # LUW(d,q) for d even has girth >= d+4.
    # d=4 => girth >= 8. Not enough.
    
    # What if we use a different construction?
    # **Generalized Petersen Graphs**?
    # G(n, k) has 2n vertices, degree 3.
    # We need degree 5.
    # **Blow-up** of a (5,9) cage?
    
    # Let's try the **Möbius Ladder** or **Prism** graphs.
    # They are cubic.
    
    # I will stick with the LUW graph as the primary candidate, but I will add a check for a **smaller** LUW-like graph if possible.
    # Actually, the LUW graph is the standard explicit construction.
    
    # Let's try to improve the LUW construction by using **q=5, d=5** but checking if we can **prune** the graph?
    # No, it's a cage.
    
    # I will return the LUW graph. It is valid and score 1.0.
    # To beat 1.0, I need n < 6250.
    # I will attempt a **circulant search** with a tighter time limit and hope to find a small one.
    
    def build_circulant(n, a, b):
        edges = []
        for i in range(n):
            for g in [a, n-a, b, n-b, n//2]:
                j = (i + g) % n
                if i < j:
                    edges.append((i, j))
        return n, edges

    def is_valid_circulant(n, a, b, min_girth):
        # Fast check:
        # Check if any sequence of generators of length < min_girth sums to 0 mod n
        # with no partial sum 0.
        
        # We can precompute the set of all possible (u, v, w) coefficients for length L.
        # u = count(a) - count(-a)
        # v = count(b) - count(-b)
        # w = count(n/2)
        # Sum = u*a + v*b + w*(n/2)
        
        # For L < 9, u, v, w are small.
        # We can iterate over all possible (u, v, w) that can be formed by a walk of length L.
        # The number of such (u, v, w) is small.
        
        # For a given (u, v, w), we check if u*a + v*b + w*(n/2) == 0 mod n.
        # And we need to ensure that the walk is simple (no partial sum 0).
        # This part is hard to check analytically.
        
        # However, if we find a solution to the sum equation, it's likely a cycle.
        # We can just check the sum equation first. If no solution, it's valid.
        # If solution exists, we do a BFS check.
        
        # Generate all (u, v, w) for L < min_girth.
        # u, v in [-L, L], w in [0, L].
        # u+v+w <= L.
        
        bad_coeffs = set()
        for L in range(3, min_girth):
            for u in range(-L, L+1):
                for v in range(-L, L+1):
                    for w in range(0, L+1):
                        if abs(u) + abs(v) + w <= L:
                            # Check if a walk with these counts is possible
                            # This requires that the number of steps is L.
                            # Steps: u_pos - u_neg = u, so u_pos + u_neg >= |u| and parity matches.
                            # Similarly for v.
                            # w steps are n/2.
                            # Total steps = u_pos + u_neg + v_pos + v_neg + w = L.
                            # Min steps for given u, v, w is |u| + |v| + w.
                            # If L - (|u| + |v| + w) is even, it's possible.
                            if (L - (abs(u) + abs(v) + w)) % 2 == 0:
                                bad_coeffs.add((u, v, w))
        
        # Check if any (u, v, w) in bad_coeffs gives sum 0 mod n
        for u, v, w in bad_coeffs:
            s = (u * a + v * b + w * (n // 2)) % n
            if s == 0:
                # Potential cycle. Do a full BFS check to confirm.
                return full_bfs_check(n, a, b, min_girth)
        
        return True

    def full_bfs_check(n, a, b, min_girth):
        gens = [a, n-a, b, n-b, n//2]
        # BFS from 0
        visited = {0: 0}
        queue = [(0, 0)]
        while queue:
            node, dist = queue.pop(0)
            if dist >= (min_girth - 1) // 2:
                continue
            for g in gens:
                nb = (node + g) % n
                if nb == 0:
                    return False # Cycle found
                if nb not in visited:
                    visited[nb] = dist + 1
                    queue.append((nb, dist + 1))
        return True

    # Search
    for n in range(426, 1500, 2):
        if time.time() > deadline:
            break
        for a in range(1, n//2):
            if time.time() > deadline:
                break
            for b in range(a+1, n//2):
                if is_valid_circulant(n, a, b, g):
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
