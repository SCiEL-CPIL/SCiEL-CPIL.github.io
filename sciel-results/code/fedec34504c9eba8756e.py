import itertools
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Moore bound for k=4, g=10 is 161.
    # We search for circulant graphs C_n({+/-a, +/-b})
    # k must be 4.
    if k != 4:
        # Fallback to LUW for other k
        return _luw_construct(k, g)
        
    moore_bound = 1 + k * sum((k-1)**i for i in range((g-2)//2))
    if (g-2) % 2 != 0:
        # For odd girth, Moore bound is different?
        # Girth 10 is even.
        pass
        
    # Search n from moore_bound upwards
    # We need to check girth >= g.
    
    def has_short_cycle(n, a, b, max_len):
        # Check for cycles of length 3 to max_len
        # A cycle is a sequence s1...sl in {+a, -a, +b, -b}
        # such that sum si = 0 mod n
        # and s_i != -s_{i-1} (no backtracking)
        # and s_1 != -s_l (closure not backtrack? Actually closure is just sum=0)
        # Also need to ensure it's a simple cycle (no repeated vertices).
        # For small lengths, no backtracking usually implies simple cycle unless it's a 2-cycle or 4-cycle with specific symmetries.
        # Actually, if s_i != -s_{i-1}, it's a non-backtracking walk.
        # A non-backtracking walk of length L that closes (sum=0) is a cycle of length <= L.
        # If L < girth, it must be a cycle.
        
        steps = [a, -a, b, -b]
        
        # Length 3
        for s1 in steps:
            for s2 in steps:
                if s2 == -s1: continue
                for s3 in steps:
                    if s3 == -s2: continue
                    if (s1 + s2 + s3) % n == 0:
                        return True
        if max_len < 4: return False
        
        # Length 4
        for s1 in steps:
            for s2 in steps:
                if s2 == -s1: continue
                for s3 in steps:
                    if s3 == -s2: continue
                    for s4 in steps:
                        if s4 == -s3: continue
                        if (s1 + s2 + s3 + s4) % n == 0:
                            # Check if it's a 2-cycle disguised?
                            # s1=s3, s2=s4, s1+s2=0? No, s2 != -s1.
                            # s1=s4?
                            # If it's a simple 4-cycle, vertices are 0, s1, s1+s2, s1+s2+s3.
                            # Distinctness:
                            # s1 != 0
                            # s1+s2 != 0 => s2 != -s1 (checked)
                            # s1+s2+s3 != 0 => s3 != -(s1+s2)
                            # s1+s2+s3+s4 = 0
                            # Also s1 != s1+s2+s3? => s2+s3 != 0 => s3 != -s2 (checked)
                            # s1+s2 != s1+s2+s3? => s3 != 0 (checked)
                            # So non-backtracking implies simple for L=4?
                            # Yes, unless s1+s2+s3 = s1 => s2+s3=0 => s3=-s2 (checked).
                            return True
        if max_len < 5: return False
        
        # For longer lengths, we can use a BFS-like check or just solve the sum equation.
        # Given the small number of steps (4), we can enumerate.
        # But 4^9 is 262k. Times n? No, just check sum=0.
        # We can prune.
        
        # Actually, for girth 10, we need to check up to 9.
        # We can just iterate all non-backtracking walks of length L and check sum=0.
        
        for L in range(5, max_len + 1):
            # Recursive generation with pruning
            def gen(prefix, current_sum, last_step):
                if len(prefix) == L:
                    if current_sum % n == 0:
                        # Check simple cycle condition?
                        # Non-backtracking walks of length L that close are cycles.
                        # Are they simple?
                        # A non-backtracking walk can revisit a vertex only if it forms a cycle.
                        # If L < girth, it can't form a cycle.
                        # So if we find a closed non-backtracking walk of length L < girth,
                        # it MUST be a simple cycle of length L (or a divisor of L).
                        # Since we are checking L < girth, it's a cycle.
                        return True
                return False
                
            # Optimization:
            # We only need to check if there exists a sequence.
            # We can use DP or just brute force for L<=9.
            # 4 * 3^8 = 26244. Very fast.
            
            def brute_force_L(L):
                for s1 in steps:
                    for s2 in steps:
                        if s2 == -s1: continue
                        for s3 in steps:
                            if s3 == -s2: continue
                            if L == 3:
                                if (s1+s2+s3) % n == 0: return True
                            else:
                                for s4 in steps:
                                    if s4 == -s3: continue
                                    if L == 4:
                                        if (s1+s2+s3+s4) % n == 0: return True
                                    else:
                                        for s5 in steps:
                                            if s5 == -s4: continue
                                            if L == 5:
                                                if (s1+s2+s3+s4+s5) % n == 0: return True
                                            else:
                                                for s6 in steps:
                                                    if s6 == -s5: continue
                                                    if L == 6:
                                                        if (s1+s2+s3+s4+s5+s6) % n == 0: return True
                                                    else:
                                                        for s7 in steps:
                                                            if s7 == -s6: continue
                                                            if L == 7:
                                                                if (s1+s2+s3+s4+s5+s6+s7) % n == 0: return True
                                                            else:
                                                                for s8 in steps:
                                                                    if s8 == -s7: continue
                                                                    if L == 8:
                                                                        if (s1+s2+s3+s4+s5+s6+s7+s8) % n == 0: return True
                                                                    else:
                                                                        for s9 in steps:
                                                                            if s9 == -s8: continue
                                                                            if L == 9:
                                                                                if (s1+s2+s3+s4+s5+s6+s7+s8+s9) % n == 0: return True
                return False
            
            if brute_force_L(L):
                return True
                
    for n in range(moore_bound, 1000): # Upper limit
        if time.time() > deadline:
            break
        # Find a, b
        # 1 <= a < b <= n//2
        # a != b, a != n-b
        # 2a != 0, 2b != 0
        for a in range(1, n//2 + 1):
            if 2*a % n == 0: continue
            for b in range(a+1, n//2 + 1):
                if 2*b % n == 0: continue
                if b == n-a: continue # a and -a are same step? No, a and n-a are distinct in steps?
                # Steps are +a, -a, +b, -b.
                # If b = n-a, then +b = -a. So steps are {a, -a, -a, a}. Duplicates.
                # So we need a, -a, b, -b to be distinct.
                if b == a or b == n-a or n-b == a or n-b == n-a:
                    continue
                
                if has_short_cycle(n, a, b, g-1):
                    continue
                
                # Valid graph found
                edges = []
                for i in range(n):
                    for step in [a, -a, b, -b]:
                        j = (i + step) % n
                        if i < j:
                            edges.append((i, j))
                        elif i > j:
                            # Ensure we don't add duplicates
                            # Since graph is undirected, (i,j) is same as (j,i)
                            # We added (j,i) when i was j?
                            # No, we iterate i.
                            # If i > j, we skip?
                            # Wait, if we iterate all i, we will add (i, i+a) and (i+a, i) when i becomes i+a?
                            # No, (i+a, i) is (i+a, i+a-a).
                            # We want unique edges.
                            # Just add if i < j.
                            pass
                # Re-do edge generation properly
                edges = []
                for i in range(n):
                    for step in [a, b]: # Only positive steps to avoid duplicates?
                        j1 = (i + step) % n
                        j2 = (i - step) % n
                        # This adds 4 edges per node?
                        # No, +a and -a are symmetric.
                        # Edge (i, i+a) is same as (i+a, i).
                        # If we add (i, i+a) for all i, we add each edge twice.
                        # So we can just add (i, (i+step)%n) for i in range(n) and deduplicate?
                        # Or add if i < (i+step)%n?
                        pass
                
                edges = []
                for i in range(n):
                    for step in [a, b]:
                        j = (i + step) % n
                        if i < j:
                            edges.append((i, j))
                        # Also need to check -step?
                        # (i, i-a) is (i-a, i).
                        # When i' = i-a, j' = i. i' < j'?
                        # So it will be added when i' is processed.
                        # So iterating steps [a, b] and adding if i<j is sufficient?
                        # Let's check.
                        # Edge between u and v exists if v-u is in {a, -a, b, -b}.
                        # If v = u+a, then u < v? Not necessarily.
                        # If we add (u, u+a) for all u, we get all edges.
                        # But (u, u+a) and (u+a, u) are the same edge.
                        # So we should just add (min(u, u+a), max(u, u+a)).
                        
                edges = []
                for i in range(n):
                    for step in [a, b]:
                        j = (i + step) % n
                        if i < j:
                            edges.append((i, j))
                        else:
                            edges.append((j, i))
                
                # Deduplicate
                edges = list(set(edges))
                
                # Verify degree
                deg = [0]*n
                for u, v in edges:
                    deg[u] += 1
                    deg[v] += 1
                if all(d == 4 for d in deg):
                    return n, edges

    # Fallback to LUW if no circulant found
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # Original seed code logic
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
