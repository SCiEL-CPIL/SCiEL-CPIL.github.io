"""Construct small 4-regular graphs of girth 10 using local search (2-opt swaps)."""
import random
import time
import numpy as np

def _verify(n, edges, k, g):
    """Verify simple, k-regular, no cycle < g via non-backtracking BFS from every vertex."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # Non-backtracking walk check
    for v0 in range(n):
        for v1 in adj[v0]:
            visited = set()
            visited.add((v0, v1))
            queue = [(v0, v1, 1)]
            while queue:
                prev, curr, dist = queue.pop(0)
                if dist >= g:
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    new_state = (curr, nxt)
                    if new_state in visited:
                        continue
                    visited.add(new_state)
                    if nxt == v0 and dist + 1 < g:
                        return False
                    queue.append((curr, nxt, dist + 1))
    return True

def construct(k, g):
    if k != 4 or g != 10:
        # Fallback for other parameters, though task specifies k=4, g=10
        # Return a trivial valid graph if possible, else raise
        # For now, we only implement the specific case
        pass

    N = 320
    n_half = N // 2
    start_time = time.time()
    time_limit = 110  # Leave 10s for verification and overhead
    
    # Generate initial random 4-regular bipartite graph
    # U: 0..n_half-1, V: n_half..N-1
    # Use 4 random permutations to define edges
    while True:
        perms = [random.sample(range(n_half), n_half) for _ in range(4)]
        adj_u = [set() for _ in range(n_half)]
        adj_v = [set() for _ in range(n_half)]
        edges = []
        valid = True
        for u in range(n_half):
            for p in range(4):
                v = perms[p][u]
                if v in adj_u[u]:
                    valid = False
                    break
                adj_u[u].add(v)
                adj_v[v].add(u)
                edges.append((u, n_half + v))
        if valid and all(len(adj_v[v]) == 4 for v in range(n_half)):
            break

    # Local Search: 2-opt swaps
    # Represent graph as list of neighbors for each vertex
    # neighbors[u] is list of 4 neighbors in V
    # neighbors[n_half+v] is list of 4 neighbors in U
    neighbors = [None] * N
    for u in range(n_half):
        neighbors[u] = list(adj_u[u])
    for v in range(n_half):
        neighbors[n_half + v] = [u for u in adj_v[v]]
    
    def count_short_cycles():
        """Count cycles of length 4, 6, 8."""
        # This is expensive to do from scratch every time.
        # We can update incrementally or just do a fast check.
        # For simplicity in this heuristic, we'll do a quick BFS for short cycles.
        # Or better, we can just check if girth is < 10.
        # But we need a cost function.
        # Let's count 4-cycles first, as they are the most restrictive.
        count = 0
        # 4-cycle: u1-v1-u2-v2-u1
        # u1, u2 in U; v1, v2 in V
        # u1 connected to v1, v2
        # u2 connected to v1, v2
        # This is equivalent to: for each pair of U nodes, count common neighbors.
        # If common neighbors > 1, we have 4-cycles.
        for i in range(n_half):
            for j in range(i+1, n_half):
                common = len(adj_u[i].intersection(adj_u[j]))
                if common > 1:
                    # Number of 4-cycles involving this pair is C(common, 2)
                    count += common * (common - 1) // 2
        # This only counts 4-cycles. We need 6 and 8 too.
        # A full girth check is O(N * 4^4).
        # Let's just return the number of 4-cycles as a primary cost,
        # and if 0, check for 6-cycles, etc.
        return count

    # Better cost: Just check if girth < 10.
    # If yes, return a penalty. If no, return 0.
    # To guide the search, we want to minimize the number of short cycles.
    
    def get_cost():
        # Fast check for 4-cycles
        c4 = 0
        for i in range(n_half):
            ni = adj_u[i]
            for j in range(i+1, n_half):
                nj = adj_u[j]
                common = len(ni & nj)
                if common > 1:
                    c4 += common * (common - 1) // 2
        if c4 > 0:
            return c4 * 1000
        
        # Check for 6-cycles
        # u1-v1-u2-v2-u3-v3-u1
        # This is harder.
        # Let's just use the _verify function's logic but stop early.
        # Actually, let's just try to eliminate 4-cycles first.
        # If c4 == 0, we check for 6-cycles.
        
        # For 6-cycles, we can look at pairs of edges.
        # This is getting complex. Let's use a simpler heuristic:
        # If _verify fails, return a high cost.
        # But _verify is slow.
        
        # Let's stick to minimizing 4-cycles first.
        # Once 4-cycles are gone, we check for 6-cycles.
        
        # Simple 6-cycle check:
        # For each u1, u2 in U, if they share a neighbor v, 
        # then for each other neighbor v1 of u1 and v2 of u2,
        # if v1 and v2 share a neighbor u3, and u3 != u1, u2, and v1 != v2...
        # This is O(N^2 * k^3).
        
        return c4

    current_cost = get_cost()
    
    # Define swap
    def swap(u1, v1, u2, v2):
        # u1, u2 are indices in U (0..n_half-1)
        # v1, v2 are indices in V (0..n_half-1)
        # Current edges: (u1, v1) and (u2, v2)
        # New edges: (u1, v2) and (u2, v1)
        # Check if new edges already exist
        if v2 in adj_u[u1] or v1 in adj_u[u2]:
            return False
        # Perform swap
        adj_u[u1].remove(v1)
        adj_u[u1].add(v2)
        adj_u[u2].remove(v2)
        adj_u[u2].add(v1)
        
        adj_v[v1].remove(u1)
        adj_v[v1].add(u2)
        adj_v[v2].remove(u2)
        adj_v[v2].add(u1)
        
        # Update neighbors lists
        neighbors[u1] = list(adj_u[u1])
        neighbors[u2] = list(adj_u[u2])
        neighbors[n_half + v1] = list(adj_v[v1])
        neighbors[n_half + v2] = list(adj_v[v2])
        
        return True

    def undo_swap(u1, v1, u2, v2):
        # Swap back
        adj_u[u1].remove(v2)
        adj_u[u1].add(v1)
        adj_u[u2].remove(v1)
        adj_u[u2].add(v2)
        
        adj_v[v1].remove(u2)
        adj_v[v1].add(u1)
        adj_v[v2].remove(u1)
        adj_v[v2].add(u2)
        
        neighbors[u1] = list(adj_u[u1])
        neighbors[u2] = list(adj_u[u2])
        neighbors[n_half + v1] = list(adj_v[v1])
        neighbors[n_half + v2] = list(adj_v[v2])

    # Local Search Loop
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Randomly pick a pair of edges to swap
        # To be efficient, we can focus on edges involved in 4-cycles.
        # But for simplicity, random sampling.
        for _ in range(1000):
            if time.time() - start_time > time_limit:
                break
            u1 = random.randint(0, n_half - 1)
            u2 = random.randint(0, n_half - 1)
            if u1 == u2:
                continue
            v1 = random.choice(list(adj_u[u1]))
            v2 = random.choice(list(adj_u[u2]))
            
            if v1 == v2:
                continue
                
            # Check if swap is valid (no duplicate edges)
            if v2 in adj_u[u1] or v1 in adj_u[u2]:
                continue
                
            # Perform swap
            swap(u1, v1, u2, v2)
            new_cost = get_cost()
            
            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
            else:
                undo_swap(u1, v1, u2, v2)
                
            # If cost is 0, check full girth
            if current_cost == 0:
                # Build edge list for verification
                edges = []
                for u in range(n_half):
                    for v in adj_u[u]:
                        edges.append((u, n_half + v))
                if _verify(N, edges, k, g):
                    return N, edges
                # If verification fails, there are 6 or 8 cycles.
                # We need to break those.
                # The cost function didn't capture them.
                # We can add a penalty for 6-cycles if we detect them.
                # For now, we continue swapping hoping to break them.
                # To avoid getting stuck, we can randomize more.
    
    # If we didn't find a perfect graph, return the best we have?
    # No, must be valid.
    # If we fail, we return a larger valid graph?
    # The seed fallback is L-U-W.
    # Let's try to just return the graph if it happens to be valid, 
    # otherwise we must ensure we return *some* valid graph.
    # The local search might not finish in time.
    
    # Fallback: Return the L-U-W graph from the original seed logic to ensure validity.
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
