import time
import random
import math
from itertools import product

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Moore Bound calculation
    if g % 2 == 0:
        # g is even
        # n >= 1 + k * sum_{i=0}^{g/2 - 1} (k-1)^i
        # For k=4, g=10: 1 + 4 * (1 + 3 + 9 + 27 + 81) = 1 + 4 * 121 = 485?
        # Wait, Moore bound for even girth g:
        # n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
        # For k=4, g=10: 2 * (1 + 3 + 9 + 27 + 81) = 2 * 121 = 242.
        # So 243 is the integer bound.
        mb = 2 * sum((k-1)**i for i in range(g//2))
    else:
        # g is odd
        # n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        mb = 1 + k * sum((k-1)**i for i in range((g-1)//2))
        
    mb = max(mb, k + 1) # Ensure at least k+1? No, k-regular needs at least k+1 vertices? No, k=4 needs at least 5? No, K5 is 4-regular, girth 3.
    
    # 1. Try LPS construction for p=3 (if k=4)
    if k == 4:
        # LPS graph L(3, q)
        # Try q=7 first (n=336)
        # Then q=5 (n=120)
        for q in [7, 5, 11]:
            if time.time() > deadline:
                break
            n_lps = q * (q*q - 1)
            if n_lps < mb:
                continue
            # Check if LPS(3, q) has girth >= g
            # Implementing full LPS is complex. 
            # For now, skip if not confident, or use a simplified check?
            # Given time, I will skip LPS implementation to avoid bugs and stick to Circulant + LUW.
            pass

    # 2. Circulant Graph Search
    # Search for n from mb to mb + 200
    best_n = None
    best_edges = None
    
    # Pre-calculate max depth for girth check
    # Girth g means no cycles of length < g.
    # Check cycles of length 3 to g-1.
    
    def check_circulant_girth(n, S, target_girth):
        # S is a set of 4 integers
        # Check for non-backtracking closed walks of length L < target_girth
        # Starting from 0
        # We can use a set of visited states (node, last_step) to avoid backtracking?
        # Actually, we just enumerate walks.
        
        # Optimization: If there is a triangle (L=3), return False
        # a + b + c = 0 => c = -(a+b)
        # Check if -(a+b) in S or -S
        S_set = set(S)
        S_neg = set([-s for s in S])
        S_all = S_set | S_neg
        
        # L=3
        for a in S:
            for b in S:
                if b == -a: continue # Backtrack
                c = -(a + b)
                if c in S_all and c != -b:
                    return False
        
        # L=4
        # a + b + c + d = 0
        # d = -(a+b+c)
        # Check if d in S_all and d != -c
        for a in S:
            for b in S:
                if b == -a: continue
                for c in S:
                    if c == -b: continue
                    d = -(a + b + c)
                    if d in S_all and d != -c:
                        return False
        
        # L=5
        # a+b+c+d+e=0
        # e = -(a+b+c+d)
        # Check if e in S_all and e != -d
        for a in S:
            for b in S:
                if b == -a: continue
                for c in S:
                    if c == -b: continue
                    for d in S:
                        if d == -c: continue
                        e = -(a + b + c + d)
                        if e in S_all and e != -d:
                            return False
                            
        # For L > 5, we can do BFS
        # But if we pass 3,4,5, it's likely good enough for random search?
        # No, we need girth 10.
        # We must check up to 9.
        
        # BFS for L=6..9
        # State: (current_node, last_step)
        # Start: (0, None)
        # We want to reach (0, last_step) where last_step != None?
        # Actually, a closed walk of length L starts at 0 and ends at 0.
        # The last step must not be a backtrack of the second-to-last step.
        
        # We can just do a DFS/BFS to depth 9.
        # Number of nodes in BFS tree: 4 * 3^8 = 26244.
        # This is fast.
        
        # To avoid re-implementing, we can just use the walk enumeration for all L.
        # But the loops above are hardcoded for 3,4,5.
        # Let's generalize the check for L up to target_girth-1.
        
        # Actually, the hardcoded checks are faster for small L.
        # For L >= 6, use BFS.
        
        # BFS from 0
        # visited: dict (node, last_step) -> depth? No, just set of (node, last_step)
        # We need to find if we can return to 0 with a valid last_step.
        
        # Start: (0, None)
        # We are at depth 0.
        # We expand to depth 1, 2, ..., target_girth-1.
        # If at depth L, we are at node 0, and L < target_girth, return False.
        # Note: At depth 0, we are at 0. We don't count this as a cycle.
        
        # BFS Queue: (node, last_step)
        # last_step is the value of the last edge used (from S_all)
        
        queue = [(0, None)]
        # To prevent infinite loops or backtracking, we track (node, last_step)
        # Actually, we just need to ensure we don't immediately backtrack.
        # The state (node, last_step) is sufficient to prevent backtracking.
        
        # We want to find a path of length L < target_girth that ends at 0.
        # The first step can be anything.
        # The last step must not be - (second to last step).
        
        # Let's just check if 0 is reachable at any depth L in [3, target_girth-1].
        # If yes, there is a cycle of length L.
        
        # BFS
        # Depth 0: (0, None)
        # Depth 1: (s, s) for s in S_all
        # Depth 2: (s1+s2, s2) for s1 in S_all, s2 in S_all, s2 != -s1
        
        # We stop if we hit (0, s) where depth >= 3.
        
        visited = set()
        visited.add((0, None))
        
        for depth in range(1, target_girth):
            next_queue = []
            for node, last_step in queue:
                for step in S_all:
                    if step == -last_step:
                        continue
                    new_node = (node + step) % n
                    new_state = (new_node, step)
                    if new_state in visited:
                        continue
                    visited.add(new_state)
                    if new_node == 0 and depth + 1 >= 3:
                        return False
                    next_queue.append(new_state)
            queue = next_queue
            if not queue:
                break
                
        return True

    # Search Loop
    # We will search n from mb to mb + 100
    # For each n, try 100 random sets S
    
    random.seed(42)
    
    for n in range(mb, mb + 150):
        if time.time() > deadline:
            break
            
        # Check if n is valid for k-regular circulant
        # n must be such that we can pick 4 distinct steps
        # Steps must be in 1..n//2
        if n // 2 < 4:
            continue
            
        found = False
        for _ in range(200):
            if time.time() > deadline:
                break
            # Generate random S
            # Ensure no 0, no duplicates, no -x if x in S (unless n is even and x=n/2)
            # Simpler: pick 4 distinct from 1..n//2
            # If n is even, n/2 is its own inverse.
            # We must ensure the graph is simple and 4-regular.
            # If we pick s and n-s, they are the same edge.
            # So we pick from 1..n//2.
            # If we pick n/2, it's a single edge type.
            
            candidates = list(range(1, n//2 + 1))
            if len(candidates) < 4:
                continue
                
            S = random.sample(candidates, 4)
            
            # Check connectivity
            # gcd(n, S[0], S[1], S[2], S[3]) must be 1
            g_val = math.gcd(n, S[0], S[1], S[2], S[3])
            if g_val != 1:
                continue
                
            # Check girth
            if check_circulant_girth(n, S, g):
                # Construct edges
                edges = []
                for i in range(n):
                    for s in S:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                # Return
                return n, edges
                
    # 3. Fallback: LUW Construction
    # (Copy from seed program)
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
