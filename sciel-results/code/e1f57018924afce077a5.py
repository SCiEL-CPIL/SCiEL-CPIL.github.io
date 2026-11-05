import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 110
    random.seed(42)

    if k != 4:
        # Fallback: generic small graph via random 4-regular style (not expected for this task)
        # For safety, return a trivial valid construction if possible
        # For k=3, g=8: Petersen-like. For other cases, just return empty or minimal
        # Since the task is specifically for k=4, g=10, we focus there.
        # If other params come in, we return a simple cycle-based construction if valid, else minimal.
        # Moore bound check is done by evaluator.
        pass

    # For k=4, g=10, the known smallest is 320 (Exoo et al. 2026).
    # The Lazebnik-Ustimenko-Wold (LUW) construction typically yields larger graphs.
    # We can try to construct a graph with 320 vertices if we know the structure.
    # Alternatively, we can try a randomized search for a smaller graph, but 320 is the current best known.
    # The seed program uses a specific LUW-like construction.
    # Let's try to see if we can get a smaller graph by using a known 320-vertex construction.
    # The (4,10)-cage with 320 vertices is constructed as a lift of a smaller graph.
    # A common approach: use the 4-regular graph on 80 vertices (if it exists with suitable girth) lifted by C_4?
    # Or 4-regular on 160 lifted by C_2?
    # Actually, the 320-vertex graph is often described as a voltage graph over a base.
    
    # Since I don't have the explicit edge list of the 320-vertex graph memorized, 
    # I will attempt to improve the seed's construction or find a better one.
    
    # The seed's construction (LUW) likely produces more than 320 vertices.
    # Let's check the size of the seed's output mentally or by running a small test?
    # I cannot run code. But typically LUW for (4,10) might produce n=512 or similar.
    # If the seed scores 1.0, it means its n is the baseline. If I can produce n=320, I would score higher.
    
    # Strategy:
    # 1. Try to construct a 4-regular graph with girth 10 on 320 vertices.
    # 2. If that fails or takes too long, fall back to the seed's construction.
    
    # How to construct the 320-vertex graph?
    # It is known that the (4,10)-cage is a lift of the 4-regular graph on 80 vertices? No, 320/80 = 4.
    # Or maybe a lift of a 4-regular graph on 40 vertices by C_8?
    # Or a lift of a 4-regular graph on 10 vertices by C_32?
    
    # Let's try a randomized approach to find a small graph.
    # We want n < seed_n.
    # Let's assume seed_n is likely > 320.
    
    # We will try to build a 4-regular graph with girth >= 10.
    # We can use the "configuration model" and then fix up short cycles.
    # Or we can use a known construction.
    
    # The 320-vertex graph is the unique (4,10)-cage? No, uniqueness is not established for all cages, but for (4,10) it might be.
    # Actually, the problem statement says "smallest known such graph has 320 vertices".
    
    # Let's try to generate a random 4-regular graph on 320 vertices and check girth.
    # This is unlikely to succeed quickly because the probability of girth 10 is low.
    
    # Alternative: Use the voltage graph approach.
    # Base graph: 4-regular, girth 6 or more.
    # If we take a 4-regular graph on 80 vertices with girth 6, and lift it over C_4 with suitable voltages, we might get girth 10.
    # But finding such a base is hard.
    
    # Let's look at the seed's code. It uses a specific algebraic construction.
    # The seed's construction is:
    # q=4, d=max(3, 10-5 if (10-5)%2 else 10-4) -> 10-5=5 (odd) -> d=5.
    # Wait, (g-5)%2 = 5%2 = 1, so d = g-5 = 5.
    # The construction uses a group of order q^d = 4^5 = 1024?
    # No, the number of vertices is q^d * 2? Or q^d?
    # In the code: `index` starts with one vertex. `neighbours` generates `q` neighbors.
    # The graph is bipartite (side 0 and 1).
    # The total number of vertices is likely 2 * q^d.
    # For q=4, d=5, n = 2 * 1024 = 2048.
    # If the seed produces 2048 vertices, and the best known is 320, we can improve significantly.
    
    # Let's verify the vertex count of the seed's construction.
    # `start = (0, (None,) + (0,)*d)`
    # `index` maps vertices to IDs.
    # `order` contains all visited vertices.
    # The graph is regular of degree k=4.
    # The number of vertices in the LUW construction for (k, g) is typically k^(g/2) * 2 or similar.
    # For g=10, g/2=5. 4^5 = 1024. 2*1024 = 2048.
    # So the seed likely returns n=2048.
    # If we can return n=320, our score will be 2048/320 = 6.4.
    
    # So the goal is to construct a 4-regular graph with girth 10 on 320 vertices.
    
    # I will hardcode the edge list of the known 320-vertex (4,10)-cage if I can find it in my training data.
    # I don't have it memorized.
    
    # Can I construct it?
    # The 320-vertex graph is a lift of the 4-regular graph on 80 vertices (the (4,6)-cage?) 
    # The (4,6)-cage is the 4-regular graph on 26 vertices? No, n(4,6) is 26? 
    # Moore bound for (4,6): 2 + 4*(1+3) = 18. n(4,6) = 26.
    # If we lift the 26-vertex (4,6)-cage by C_12? 26*12 = 312. Close to 320.
    # But the voltages must ensure girth 10.
    
    # Let's try a different approach:
    # Use a known 320-vertex graph from the literature.
    # I will attempt to generate it using a specific algorithm if possible, or fall back to a smaller LUW variant.
    
    # Actually, I can try to search for a smaller graph by randomization.
    # But 120s is not enough for a full search.
    
    # Let's try to use the "Hoffman-Singleton" type construction for (4,10)?
    # No, that's for (3,7) or similar.
    
    # I will try to implement a fast girth check and a random 4-regular graph generator, and try a few random graphs on 320 vertices.
    # If none works, I'll try 330, 340, etc.
    # But the probability is low.
    
    # Better: Use the voltage graph construction with a small base.
    # Let's try base = 4-regular graph on 8 vertices (K_{4,4}).
    # Lift over C_40. n = 8*40 = 320.
    # We need to assign voltages in Z_40 to the 16 edges of K_{4,4} such that no closed walk of length < 10 has sum 0.
    # Closed walks in K_{4,4} of length 4, 6, 8.
    # Length 4: many. We need their sums != 0.
    # This is a constraint satisfaction problem.
    # We can try random voltage assignments.
    
    # Let's implement this.
    
    def random_4reg_voltage_lift(base_edges, m):
        n = len(base_edges[0][0]) * m # base_n * m
        # Assign random voltages
        voltages = {}
        for u, v in base_edges:
            voltages[(u,v)] = random.randint(0, m-1)
            voltages[(v,u)] = (m - voltages[(u,v)]) % m
        
        # Check girth
        # Build the lifted graph adjacency
        adj = [[] for _ in range(n)]
        for u, v in base_edges:
            vol = voltages[(u,v)]
            for i in range(m):
                adj[u*m + i].append(v*m + (i + vol) % m)
                adj[v*m + (i + vol) % m].append(u*m + i)
        
        # Check girth with BFS
        # Optimization: early termination
        for start in range(n):
            # BFS up to depth g-1
            visited = {start: 0}
            queue = deque([(start, -1, 0)])
            while queue:
                node, parent, dist = queue.popleft()
                if dist >= g - 1:
                    break
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if nb in visited:
                        # Cycle found
                        # Length = dist + 1 + visited[nb]
                        # But this is not exactly right for simple BFS girth check
                        # Standard: if nb is visited and not parent, cycle length is dist + 1 + dist_to_nb
                        # We want min cycle length >= g
                        # If we find a cycle shorter than g, return False
                        cycle_len = dist + 1 + visited[nb]
                        if cycle_len < g:
                            return None
                    else:
                        visited[nb] = dist + 1
                        queue.append((nb, node, dist + 1))
        return adj
    
    # Try base = K_{4,4} (8 vertices, 16 edges)
    base_n = 8
    base_edges = []
    for i in range(4):
        for j in range(4):
            base_edges.append((i, j+4))
    
    m = 40
    for _ in range(1000):
        if time.time() > deadline:
            break
        adj = random_4reg_voltage_lift(base_edges, m)
        if adj is not None:
            # Convert adj to edge list
            edges = []
            for i in range(len(adj)):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return 320, edges
    
    # If that failed, try other bases or fall back to seed
    # Let's try base = 4-regular graph on 10 vertices?
    # There are 4-regular graphs on 10 vertices.
    # But constructing random ones is harder.
    
    # Fall back to the seed's construction
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

