import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    if k == 4 and g == 10:
        result = _construct_4_10(deadline)
        if result is not None:
            return result
        return _luw_construct(k, g)
    
    return _luw_construct(k, g)


def _construct_4_10(deadline):
    """Construct a 4-regular graph with girth >= 10 on as few vertices as possible."""
    base_edges = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    m = 8
    l = 8
    n_lift = 5 * m * l  # 320
    
    def check_girth_nb(edges, n, girth_target):
        """Check girth using non-backtracking walks from every vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Non-backtracking: state is (current, previous)
        # We want to find the shortest cycle, which is the shortest non-backtracking
        # walk from (v, u) back to (u, v) for any edge (u,v)
        # Actually, girth = min over all directed edges (u->v) of shortest non-backtracking
        # path from v back to u (not immediately going back to v)
        
        # More efficient: for each vertex, do BFS in the non-backtracking graph
        # State: (node, prev_node) where prev_node is the node we came from
        # Start: for each neighbor v of start, state (v, start)
        # Goal: reach (start, v) for some neighbor v of start
        
        for start in range(n):
            # BFS in non-backtracking space
            # dist[(u, v)] = shortest non-backtracking path from start to u, 
            # where v is the node we came from (v -> u is the last step)
            # We want to find when we can reach start again
            # Actually, let's track: from start, we go to neighbor v (state (v, start)),
            # then from v we can go to w != start (state (w, v)), etc.
            # A cycle of length L through start exists if there's a non-backtracking
            # walk of length L-1 from some neighbor v of start back to start.
            
            # BFS: state is (current_node, prev_node)
            # Initialize: for each neighbor v of start: state (v, start) with dist=1
            # If we ever reach state (start, v) for any neighbor v, we have a cycle of length dist+1
            
            visited = set()
            queue = deque()
            for v in adj[start]:
                visited.add((v, start))
                queue.append((v, start, 1))
            
            while queue:
                u, prev, dist = queue.popleft()
                if u == start:
                    # We've returned to start, cycle length = dist + 1
                    # But wait, we started with dist=1 for (v, start)
                    # If we reach (start, v), the cycle is dist+1
                    # Actually the path is: start -> ... -> u=start, with dist steps
                    # The cycle length is dist (number of edges in the non-backtracking path)
                    # plus... let me think again.
                    # Path: start -> v1 -> v2 -> ... -> vk = start
                    # States: (v1, start) dist=1, (v2, v1) dist=2, ..., (start, v_{k-1}) dist=k
                    # So when we pop (start, v_{k-1}, k), cycle length = k
                    if dist < girth_target:
                        return False
                    continue
                for w in adj[u]:
                    if w == prev:
                        continue  # non-backtracking
                    if (w, u) not in visited:
                        visited.add((w, u))
                        queue.append((w, u, dist + 1))
        
        return True
    
    def build_lift(voltages):
        """voltages: list of (a,b) for each of the 10 base edges"""
        edges = []
        ml = m * l
        for idx, (u, v) in enumerate(base_edges):
            a, b = voltages[idx]
            for i in range(m):
                for j in range(l):
                    g2_i = (i + a) % m
                    g2_j = (j + b) % l
                    u_node = u * ml + i * l + j
                    v_node = v * ml + g2_i * l + g2_j
                    edges.append((u_node, v_node))
        return edges
    
    random.seed(42)
    
    # Try known good voltage assignments first
    # The Exoo et al. construction likely uses specific voltages
    # Let's try a structured approach: use different generators
    
    # First, try random search with fast girth check
    best_edges = None
    attempts = 0
    
    while time.time() < deadline - 15:
        voltages = [(random.randint(0, m-1), random.randint(0, l-1)) for _ in range(10)]
        edges = build_lift(voltages)
        
        if check_girth_nb(edges, n_lift, 10):
            return n_lift, edges
        
        attempts += 1
        
        if attempts % 100 == 0:
            if time.time() > deadline - 20:
                break
    
    # Try local search: start from a random assignment and mutate
    if time.time() < deadline - 20:
        voltages = [(random.randint(0, m-1), random.randint(0, l-1)) for _ in range(10)]
        edges = build_lift(voltages)
        
        # Evaluate girth (find shortest cycle)
        def find_shortest_cycle(edges, n, girth_limit=20):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            min_cycle = girth_limit + 1
            for start in range(n):
                visited = set()
                queue = deque()
                for v in adj[start]:
                    visited.add((v, start))
                    queue.append((v, start, 1))
                
                while queue:
                    u, prev, dist = queue.popleft()
                    if u == start:
                        if dist < min_cycle:
                            min_cycle = dist
                        continue
                    if dist >= min_cycle:
                        continue
                    for w in adj[u]:
                        if w == prev:
                            continue
                        if (w, u) not in visited:
                            visited.add((w, u))
                            queue.append((w, u, dist + 1))
            
            return min_cycle
        
        current_girth = find_shortest_cycle(edges, n_lift)
        
        for step in range(5000):
            if time.time() > deadline - 20:
                break
            
            # Mutate one voltage
            idx = random.randint(0, 9)
            old_a, old_b = voltages[idx]
            voltages[idx] = (random.randint(0, m-1), random.randint(0, l-1))
            
            new_edges = build_lift(voltages)
            new_girth = find_shortest_cycle(new_edges, n_lift)
            
            if new_girth >= current_girth:
                edges = new_edges
                current_girth = new_girth
                if current_girth >= 10:
                    return n_lift, edges
            else:
                voltages[idx] = (old_a, old_b)
        
        if current_girth >= 10:
            return n_lift, edges
    
    return None


def _luw_construct(k, g):
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
