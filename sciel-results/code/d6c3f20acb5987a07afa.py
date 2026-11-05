import random
import time
import sys
from collections import deque

def construct(k, g):
    # Fallback: The Lazebnik-Ustimenko-Woldar construction
    def l_u_w_construct(k, g):
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

    # New Approach: Backtracking with Constraint Propagation
    # We try to construct a k-regular graph of girth >= g with n vertices.
    # We build the graph vertex by vertex, assigning edges.
    
    def solve_backtracking(n, k, g, time_limit):
        start_time = time.time()
        adj = [set() for _ in range(n)]
        edges = []
        
        # Heuristic: Start with vertex 0.
        # We need to ensure the graph is connected and regular.
        # A common strategy is to build a tree of depth (g-1)//2 around each vertex, 
        # but for global regularity, we just backtrack on edge assignments.
        
        # To make it efficient, we process vertices in order 0 to n-1.
        # For vertex i, we must choose k neighbors.
        # To avoid small cycles, any neighbor j of i must be at distance >= g-1 from i in the current graph?
        # No, if we add edge (i, j), the cycle length is dist(i, j) + 1.
        # So we need dist(i, j) >= g - 1.
        
        # Optimization: 
        # 1. If i < g, we can't have neighbors < i that are too close.
        # 2. We can fix the first few edges to break symmetry.
        #    E.g., vertex 0 connects to 1, 2, ..., k.
        
        # Let's try a simpler backtracking:
        # We maintain a list of vertices that still need degree k.
        # We pick the vertex with the highest index that has degree < k? 
        # Or lowest? 
        # Let's stick to a structured approach:
        # We will fill the adjacency matrix.
        # Due to symmetry, we can fix the neighbors of vertex 0 to be 1, 2, ..., k.
        # This reduces search space.
        
        # However, a general backtracking is safer.
        
        def get_distance(u, v, max_dist):
            """Returns distance between u and v in current adj, or max_dist+1 if not found within max_dist."""
            if u == v: return 0
            if v in adj[u]: return 1
            
            visited = {u: 0}
            queue = deque([(u, 0)])
            while queue:
                curr, d = queue.popleft()
                if d >= max_dist: continue
                for nei in adj[curr]:
                    if nei == v:
                        return d + 1
                    if nei not in visited:
                        visited[nei] = d + 1
                        queue.append((nei, d + 1))
            return max_dist + 1

        def backtrack(i):
            """Try to assign k neighbors to vertex i."""
            if time.time() - start_time > time_limit:
                return False
            
            if i == n:
                # Check if all degrees are exactly k
                for v in range(n):
                    if len(adj[v]) != k:
                        return False
                return True
            
            # If vertex i already has k neighbors, move to next
            if len(adj[i]) == k:
                return backtrack(i + 1)
            
            # We need to add (k - len(adj[i])) more neighbors for vertex i.
            # To avoid duplicates, we only consider neighbors j > i? 
            # No, that would restrict the graph to be directed/acyclic in terms of indices, 
            # which prevents cycles. 
            # We must allow j < i. But to avoid double counting, 
            # we can say: we only add edge (i, j) if j > i OR if j < i and we haven't added it from j's side.
            # Actually, standard backtracking on undirected graphs is tricky.
            
            # Better strategy: 
            # We decide the entire neighborhood of vertex i.
            # But we must respect existing edges.
            
            # Let's use a different heuristic:
            # We pick the vertex with the most constraints (lowest available candidates).
            # But for simplicity and speed in Python, let's stick to i -> i+1 order,
            # but allow connecting to any j.
            
            # To prevent symmetry and speed up:
            # If i == 0, we MUST connect to 1, 2, ..., k. (If n > k)
            if i == 0:
                for j in range(1, k + 1):
                    if j < n:
                        adj[0].add(j)
                        adj[j].add(0)
                        edges.append((0, j))
                    else:
                        return False
                return backtrack(1)
            
            # General case for i > 0
            # We need to fill remaining degree of i.
            # Let's find all candidates j such that:
            # 1. j != i
            # 2. j not in adj[i]
            # 3. len(adj[j]) < k
            # 4. get_distance(i, j, g - 2) > g - 2  (i.e., dist >= g-1)
            
            # Wait, if we just add edge (i,j), the cycle length is dist(i,j)+1.
            # We need dist(i,j) >= g - 1.
            
            # To make this efficient, we should sort candidates.
            # Prefer j > i? No, that limits connectivity.
            # Prefer j with high degree? No, low degree (more freedom).
            
            # Let's try to add ONE edge at a time to vertex i.
            # This allows for more flexibility in ordering.
            
            # Find candidates for i
            candidates = []
            for j in range(n):
                if j == i: continue
                if j in adj[i]: continue
                if len(adj[j]) >= k: continue
                
                # Check distance
                # We only need to check if dist < g-1.
                # If dist >= g-1, it's safe.
                # BFS depth g-2.
                if get_distance(i, j, g - 2) < g - 1:
                    continue
                
                candidates.append(j)
            
            if not candidates:
                return False
            
            # Heuristic: Try j with highest index first? Or random?
            # Random with shuffling is often better to escape local minima.
            # But deterministic is better for reproducibility if we don't have time for random restarts.
            # Let's shuffle to break symmetry.
            random.shuffle(candidates)
            
            # Try adding edge (i, j)
            for j in candidates:
                # Add edge
                adj[i].add(j)
                adj[j].add(i)
                edges.append((i, j))
                
                # Recurse
                # We don't necessarily move to i+1. We might need to add more edges to i.
                # But if we just call backtrack(i), it will try to add more to i.
                # This works.
                
                if backtrack(i):
                    return True
                
                # Undo
                adj[i].remove(j)
                adj[j].remove(i)
                edges.pop()
            
            return False

        # Initialize
        random.seed(42)
        
        # We need to estimate n.
        # Moore bound is a lower bound.
        # Let's try a few n values starting from a small offset above Moore bound.
        
        # Calculate Moore Bound
        if g % 2 == 1:
            r = (g - 3) // 2
            mb = 0
            for i in range(r):
                mb += (k - 1) ** i
            mb *= k
        else:
            r = g // 2 - 1
            mb = 0
            for i in range(r + 1):
                mb += (k - 1) ** i
            mb *= 2
        
        # Try n values
        # For k=5, g=9, MB=425. Known best 1116.
        # Backtracking might find smaller ones if they exist, or fail.
        # Let's try a range.
        
        n_candidates = []
        # Start slightly above MB, step up.
        # If k=5, g=9, maybe try 425, 426... up to some limit?
        # But 1116 is known. Maybe we can find 1000?
        # Let's try a few specific values.
        
        if k == 5 and g == 9:
            # Try around known best, but lower first.
            n_candidates = [500, 600, 700, 800, 900, 1000, 1100]
        else:
            # Generic
            n_candidates = [mb + 10, mb + 50, mb + 100, mb + 200]
        
        # Sort by n
        n_candidates.sort()
        
        for n in n_candidates:
            if n < 2 * k: continue
            # Time budget per n
            time_per_n = 10.0
            # Clear state
            adj = [set() for _ in range(n)]
            edges = []
            
            # Run backtracking
            # We need to pass adj and edges to the function or close over them.
            # The function above closes over adj and edges.
            
            # We need to reset adj and edges for each n.
            # Let's refactor solve_backtracking to take n and return result.
            
            def run_solver(n_val):
                adj_local = [set() for _ in range(n_val)]
                edges_local = []
                
                def get_dist(u, v, max_d):
                    if u == v: return 0
                    if v in adj_local[u]: return 1
                    visited = {u: 0}
                    q = deque([(u, 0)])
                    while q:
                        c, d = q.popleft()
                        if d >= max_d: continue
                        for nei in adj_local[c]:
                            if nei == v: return d + 1
                            if nei not in visited:
                                visited[nei] = d + 1
                                q.append((nei, d + 1))
                    return max_d + 1

                def bt(i):
                    if time.time() - start_time > time_limit:
                        return False
                    if i == n_val:
                        for v in range(n_val):
                            if len(adj_local[v]) != k:
                                return False
                        return True
                    
                    if len(adj_local[i]) == k:
                        return bt(i + 1)
                    
                    if i == 0:
                        for j in range(1, k + 1):
                            if j < n_val:
                                adj_local[0].add(j)
                                adj_local[j].add(0)
                                edges_local.append((0, j))
                            else:
                                return False
                        return bt(1)
                    
                    cands = []
                    for j in range(n_val):
                        if j == i: continue
                        if j in adj_local[i]: continue
                        if len(adj_local[j]) >= k: continue
                        if get_dist(i, j, g - 2) < g - 1:
                            continue
                        cands.append(j)
                    
                    if not cands:
                        return False
                    
                    random.shuffle(cands)
                    
                    for j in cands:
                        adj_local[i].add(j)
                        adj_local[j].add(i)
                        edges_local.append((i, j))
                        
                        if bt(i):
                            return True
                        
                        adj_local[i].remove(j)
                        adj_local[j].remove(i)
                        edges_local.pop()
                    
                    return False
                
                bt(0)
                # Check if success
                if all(len(adj_local[v]) == k for v in range(n_val)):
                    return n_val, edges_local
                return None

            res = run_solver(n)
            if res:
                return res

        return None

    # Main logic
    total_time = 110.0
    start_total = time.time()
    
    # Try the new backtracking solver
    res = solve_backtracking(k, g, time_limit=total_time - 10)
    if res:
        return res
    
    # Fallback
    return l_u_w_construct(k, g)
