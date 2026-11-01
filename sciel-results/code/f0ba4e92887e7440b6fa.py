import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = 115.0
    
    # Moore bound for k=3, g=16 is 766.
    # Best known is 936.
    
    # We will search for a circulant graph C_n(s1, s2, s3).
    
    def check_short_cycles(n, steps):
        # Check for cycles of length 4 and 6.
        # Length 4: s_i +/- s_j = 0
        # Length 6: s_i +/- s_j +/- s_k = 0
        s1, s2, s3 = steps
        
        # 4-cycles
        pairs = [(s1, s2), (s1, s3), (s2, s3)]
        for a, b in pairs:
            if (a + b) % n == 0 or (a - b) % n == 0:
                return False
            if (a + (-b)) % n == 0: # Same as a-b
                pass
        # Also s1+s1=0? No, simple graph.
        
        # 6-cycles
        # Sum of 3 terms from {+s, -s} is 0.
        # There are 2^3 * 3! / (perms) combinations?
        # Just check all 8 sign combinations for (s1, s2, s3) and permutations?
        # Actually, just check all sums of 3 steps with signs.
        # s1 + s2 + s3
        # s1 + s2 - s3
        # s1 - s2 + s3
        # s1 - s2 - s3
        # And permutations of indices?
        # Since addition is commutative, the set of sums is the same.
        # So we just check these 4 base sums for the specific set {s1, s2, s3}.
        # Wait, s1+s2+s3, s1+s2-s3, etc. cover all sign combos for the specific assignment.
        # But we can permute which step is which.
        # However, the set of values {s1, s2, s3} is fixed.
        # The possible sums of 3 elements from {+-s1, +-s2, +-s3} are:
        # +-s1 +-s2 +-s3
        # There are 8 such sums.
        sums = [
            (s1 + s2 + s3) % n,
            (s1 + s2 - s3) % n,
            (s1 - s2 + s3) % n,
            (s1 - s2 - s3) % n,
            (-s1 + s2 + s3) % n,
            (-s1 + s2 - s3) % n,
            (-s1 - s2 + s3) % n,
            (-s1 - s2 - s3) % n,
        ]
        if any(s == 0 for s in sums):
            return False
            
        return True

    def get_girth_circulant(n, steps):
        # BFS from 0
        dist = [-1] * n
        dist[0] = 0
        queue = [(0, 0)]
        
        while queue:
            node, d = queue.pop(0)
            if d >= g // 2:
                return g # Girth is at least g
            
            for s in steps:
                for sign in [1, -1]:
                    neighbor = (node + sign * s) % n
                    if dist[neighbor] == -1:
                        dist[neighbor] = d + 1
                        queue.append((neighbor, d + 1))
                    elif dist[neighbor] != d - 1:
                        cycle_len = d + dist[neighbor] + 1
                        if cycle_len < g:
                            return cycle_len
                            
        return g

    best_n = None
    best_edges = None
    
    # Search n from 766 to 1000
    for n in range(766, 1001):
        if time.time() > deadline:
            break
            
        # Try random steps
        # We fix s1=1
        # We try random s2, s3
        # Number of trials per n: 1000?
        
        found = False
        # Deterministic sweep might be better if we can prune well.
        # Let's try a limited deterministic sweep.
        
        # To ensure we find a solution, we should probably iterate s2, s3.
        # But 250k iterations is too many.
        # Let's try 1000 random pairs per n.
        
        for _ in range(2000):
            if time.time() > deadline:
                break
                
            s2 = random.randint(2, n // 2 - 1)
            s3 = random.randint(2, n // 2 - 1)
            if s2 == s3:
                continue
                
            steps = [1, s2, s3]
            
            if not check_short_cycles(n, steps):
                continue
                
            girth = get_girth_circulant(n, steps)
            if girth >= g:
                # Construct edges
                edges = []
                for i in range(n):
                    for s in steps:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            break
            
    if best_n is not None:
        return best_n, best_edges
        
    # Fallback: Use the seed's LUW construction
    # It is slow but valid.
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
