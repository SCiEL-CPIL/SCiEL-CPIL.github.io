import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Parameters
    # Moore bound for (3, 16) is 510 (bipartite)
    # We try to find a graph with n < 1500 to ensure improvement over LUW (~354k)
    
    def is_valid_graph(n, adj, g):
        # Check regularity
        for i in range(n):
            if len(adj[i]) != k:
                return False
            if len(set(adj[i])) != k:
                return False
        
        # Check girth
        # BFS from each vertex
        for start in range(n):
            visited = {start: 0}
            queue = deque([(start, -1, 0)])
            while queue:
                node, parent, depth = queue.popleft()
                if depth >= (g // 2):
                    continue
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if neighbor in visited:
                        # Found a cycle
                        # Cycle length = depth + visited[neighbor] + 1?
                        # Actually, if we visit 'neighbor' which is already visited,
                        # and it's not the parent, we have a cycle.
                        # The length is depth + visited[neighbor] + 1.
                        # But we need to be careful with BFS levels.
                        # If neighbor is in visited, and neighbor != parent,
                        # the cycle length is depth + visited[neighbor] + 1.
                        # Wait, BFS depth is distance from start.
                        # If we find an edge (node, neighbor) where neighbor is visited,
                        # and neighbor is not parent of node,
                        # then we have a cycle of length depth(node) + depth(neighbor) + 1.
                        # However, this counts non-simple cycles if not careful.
                        # For girth check, we just need to know if ANY cycle < g exists.
                        # If depth + visited[neighbor] + 1 < g, return False.
                        if depth + visited[neighbor] + 1 < g:
                            return False
                    else:
                        visited[neighbor] = depth + 1
                        queue.append((neighbor, node, depth + 1))
        return True

    def count_short_cycles(adj, n, max_len=8):
        # Count 4-cycles and 6-cycles as a proxy
        # 4-cycles: For each vertex u, count pairs of common neighbors.
        # Each 4-cycle is counted 4 times (once from each vertex).
        four_cycles = 0
        for u in range(n):
            neighbors = adj[u]
            # For each pair of neighbors, check if they share a common neighbor other than u
            # This is O(d^3). d=3, so O(27) per vertex.
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    n1 = neighbors[i]
                    n2 = neighbors[j]
                    # Check if n1 and n2 have a common neighbor
                    # Intersection of adj[n1] and adj[n2]
                    # If intersection is non-empty, we have a 4-cycle u-n1-w-n2-u
                    # w cannot be u.
                    common = set(adj[n1]) & set(adj[n2])
                    if u in common:
                        common.remove(u)
                    if common:
                        four_cycles += 1
        four_cycles //= 4
        
        # 6-cycles are harder. Skip for speed, rely on 4-cycles being 0.
        return four_cycles

    def random_bipartite_3_regular(n):
        # n must be even
        n_half = n // 2
        # Left: 0..n_half-1, Right: n_half..n-1
        # Each left vertex has 3 edges to right.
        # Use a random permutation approach.
        # Create a list of right vertices, repeated 3 times.
        right_pool = []
        for i in range(n_half):
            right_pool.extend([n_half + i] * 3)
        random.shuffle(right_pool)
        
        adj = [[] for _ in range(n)]
        for i in range(n_half):
            # Assign 3 edges to left vertex i
            edges = right_pool[i*3 : (i+1)*3]
            for v in edges:
                adj[i].append(v)
                adj[v].append(i)
        
        # This might create multi-edges if the shuffle isn't careful?
        # No, right_pool has each right vertex exactly 3 times.
        # But a left vertex might get the same right vertex twice?
        # Yes, if right_pool has duplicates for the same left vertex slice.
        # We need to ensure no multi-edges.
        # Retry if multi-edge found.
        
        for i in range(n):
            if len(set(adj[i])) != 3:
                return random_bipartite_3_regular(n)
        
        return adj

    def simulated_annealing(adj, n, g, time_limit):
        current_adj = [list(x) for x in adj]
        current_energy = count_short_cycles(current_adj, n)
        best_adj = [list(x) for x in adj]
        best_energy = current_energy
        
        start = time.time()
        temperature = 1.0
        cooling = 0.999
        
        while time.time() - start < time_limit:
            # Move: Bipartite 2-opt
            # Pick two left vertices u1, u2
            n_half = n // 2
            u1 = random.randint(0, n_half - 1)
            u2 = random.randint(0, n_half - 1)
            if u1 == u2:
                continue
            
            # Pick a right neighbor v1 of u1 and v2 of u2
            v1 = random.choice(current_adj[u1])
            v2 = random.choice(current_adj[u2])
            
            # Ensure v1 != v2 and v1 != u2 (impossible in bipartite) and v2 != u1
            if v1 == v2:
                continue
                
            # Remove u1-v1 and u2-v2
            # Add u1-v2 and u2-v1
            
            # Check if u1-v2 and u2-v1 already exist
            if v2 in current_adj[u1] or v1 in current_adj[u2]:
                continue
                
            # Apply move
            current_adj[u1].remove(v1)
            current_adj[v1].remove(u1)
            current_adj[u2].remove(v2)
            current_adj[v2].remove(u2)
            
            current_adj[u1].append(v2)
            current_adj[v2].append(u1)
            current_adj[u2].append(v1)
            current_adj[v1].append(u2)
            
            new_energy = count_short_cycles(current_adj, n)
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_adj = [list(x) for x in current_adj]
                    if best_energy == 0:
                        break
            else:
                # Revert
                current_adj[u1].remove(v2)
                current_adj[v2].remove(u1)
                current_adj[u2].remove(v1)
                current_adj[v1].remove(u2)
                
                current_adj[u1].append(v1)
                current_adj[v1].append(u1)
                current_adj[u2].append(v2)
                current_adj[v2].append(u2)
                
            temperature *= cooling
            if temperature < 0.01:
                temperature = 0.01
                
        return best_adj if best_energy == 0 else None

    # Main Search Loop
    # Try sizes from 510 to 1000
    for n in range(510, 1000, 2):
        if time.time() > deadline - 20:
            break
            
        # Try a few restarts
        found = None
        for _ in range(3):
            if time.time() > deadline - 20:
                break
            adj = random_bipartite_3_regular(n)
            result = simulated_annealing(adj, n, g, time_limit=2)
            if result:
                # Verify
                if is_valid_graph(n, result, g):
                    found = result
                    break
        
        if found:
            edges = []
            for i in range(n):
                for j in found[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges

    # Fallback: LUW Construction
    # Optimized LUW for k=3, g=16
    q = k
    d = g - 5 if (g - 5) % 2 else g - 4
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Use integer encoding for speed
    # Vector y[1..d] in GF(q)^d
    # Index = y[1] * q^0 + y[2] * q^1 + ...
    
    def decode(idx):
        vec = [0] * (d + 1)
        for i in range(d):
            vec[i+1] = idx % q
            idx //= q
        return vec

    def encode(vec):
        idx = 0
        for i in range(d, 0, -1):
            idx = idx * q + vec[i]
        return idx

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
