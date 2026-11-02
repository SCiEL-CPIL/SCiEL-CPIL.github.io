import random
import time
from collections import deque

def construct(k, g):
    t_start = time.time()
    DEADLINE = t_start + 135

    if k != 4 or g != 10:
        # Fallback to original logic if not the specific case, 
        # but the task implies k=4, g=10.
        # We will just return the seed construction for other cases.
        def seed_construct():
            q = k
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
            if q == 4:
                table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
                add = lambda x, y: x ^ y
                sub = lambda x, y: x ^ y
                mul = lambda x, y: table[x][y]
            else:
                add = lambda x, y: (x + y) % q
                sub = lambda x, y: (x - y) % q
                mul = lambda x, y: x * y % q
            
            rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d + 1):
                r = (c - 5) % 4
                c0 = c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

            def neighbours(vertex):
                side, x = vertex
                for first in range(q):
                    y = [None, first] + [0] * (d - 1)
                    for c in range(2, d + 1):
                        a, b = rule[c]
                        if side == 0:
                            y[c] = add(x[c], mul(y[a], x[b]))
                        else:
                            y[c] = sub(x[c], mul(x[a], y[b]))
                    yield 1 - side, tuple(y)

            start = (0, (None,) + (0,) * d)
            index = {start: 0}
            order = [start]
            edges = []
            for vertex in order:
                for other in neighbours(vertex):
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    if vertex[0] == 0:
                        edges.append((index[vertex], index[other]))
            return len(order), edges
        return seed_construct()

    # Specific case k=4, g=10
    # Use Voltage Graph Method with K4 base and Z_m group
    # Vertices: 4 * m
    # Moore bound: 242. So 4*m >= 242 => m >= 61.
    
    def build_lift(m, x, y, z):
        """
        Builds the 4-regular graph from K4 base with voltages.
        Base vertices: 0, 1, 2, 3.
        Voltages (after gauge fixing 01,02,03 to 0):
        01: 0
        02: 0
        03: 0
        12: x
        13: y
        23: z
        """
        n = 4 * m
        edges = []
        # Map (base_node, group_elem) to index
        # index = base_node * m + group_elem
        
        # Edges in K4: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        # We add edges for each lift.
        # For edge (u,v) in base with voltage g:
        # Connect (u, a) to (v, a+g)
        
        # To avoid duplicates, we only add if u < v in base? 
        # No, the lift is undirected. 
        # If we iterate all base edges and all group elements, we might add duplicates or miss direction.
        # Better: for each base edge (u,v) with voltage g, 
        # for each a in Z_m:
        #   u_idx = u * m + a
        #   v_idx = v * m + (a + g) % m
        #   if u_idx < v_idx: add edge
        #   else: skip (will be added when processing v,u? No, base edges are undirected but we defined direction 0->1 etc.)
        
        # Let's define base edges with directions to cover all.
        # Base edges:
        # (0,1) volt 0
        # (0,2) volt 0
        # (0,3) volt 0
        # (1,2) volt x
        # (1,3) volt y
        # (2,3) volt z
        
        base_edges = [
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
            (1, 2, x),
            (1, 3, y),
            (2, 3, z)
        ]
        
        for u, v, g_val in base_edges:
            for a in range(m):
                u_idx = u * m + a
                v_idx = v * m + (a + g_val) % m
                if u_idx < v_idx:
                    edges.append((u_idx, v_idx))
                elif u_idx > v_idx:
                    # This edge is represented by (v_idx, u_idx)
                    # But wait, if we only add u<v, we might miss edges where the "natural" direction gives u>v?
                    # The set of edges is symmetric. 
                    # If we process (u,v) and (v,u) we double count.
                    # We only process each base edge once.
                    # So we should just add (min(u_idx, v_idx), max(u_idx, v_idx))
                    edges.append((min(u_idx, v_idx), max(u_idx, v_idx)))
        
        # Remove duplicates? 
        # Since base graph is simple and we process each edge once, and mapping is bijective for a fixed edge?
        # No, (u, a) -> (v, a+g) is a bijection on the set of lifted edges for that base edge.
        # So no duplicates within one base edge.
        # Duplicates across base edges? No, base edges are distinct.
        # So edges list should be clean.
        
        return n, edges

    def verify_girth(n, edges, required_girth):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False

        # BFS for girth
        # We need to check if there is a cycle of length < required_girth
        # Non-backtracking BFS from each vertex?
        # Standard BFS finds shortest path. 
        # Cycle length = dist[u] + dist[v] + 1 if edge (u,v) is a back edge.
        # This works for simple cycles.
        
        half_depth = (required_girth - 1) // 2
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            queue = deque([i])
            while queue:
                u = queue.popleft()
                if dist[u] >= half_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif v != i and dist[v] <= dist[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < required_girth:
                            return False
        return True

    # Search for m and voltages x, y, z
    # m starts from 61
    # We want to find the smallest m such that valid x,y,z exist.
    
    best_n = None
    best_edges = None
    
    for m in range(61, 100): # Try up to 100
        if time.time() > DEADLINE:
            break
            
        found = False
        # Random search for x, y, z
        # Constraints:
        # x,y,z != 0
        # x+y+z != 0
        # x+z != 0, y+z != 0, x+y != 0
        
        # There are m^3 possibilities. m=61 => 226k. 
        # We can iterate through all if fast enough, or random sample.
        # Let's try random sampling first, then exhaustive if needed.
        
        # Actually, we can just iterate. 226k iterations is fast in Python? 
        # Verification is O(n * E) which is slow. 
        # We should verify girth only after checking voltage sums for short walks.
        
        def check_voltage_sums(x, y, z, m):
            # Check all non-backtracking closed walks in K4 up to length 9
            # Walks are sequences of vertices v0..vk=v0
            # Voltage sum is sum of voltages of edges.
            # We need sum != 0 mod m.
            
            # Length 3
            if x % m == 0: return False
            if y % m == 0: return False
            if z % m == 0: return False
            if (x + y + z) % m == 0: return False
            
            # Length 4
            if (x + z) % m == 0: return False
            if (y + z) % m == 0: return False
            if (x + y) % m == 0: return False
            
            # Length 5
            # All length 5 non-backtracking closed walks in K4 seem to have sum x+y+z or similar.
            # Let's verify.
            # 0-1-2-3-1-0: 0+x+z+y+0 = x+y+z
            # 0-1-3-2-1-0: 0+y+z+x+0 = x+y+z
            # 0-2-1-3-2-0: 0+x+y+z+0 = x+y+z
            # 0-2-3-1-2-0: 0+z+y+x+0 = x+y+z
            # 0-1-2-1-3-0? No, backtrack.
            # Are there others?
            # 0-1-3-2-3-0? No.
            # It seems only x+y+z matters for length 5?
            # If so, already checked.
            
            # Length 6
            # 0-1-2-3-2-1-0? Backtrack.
            # 0-1-2-1-3-2-0? Backtrack.
            # Non-backtracking: 0-1-2-3-1-2-0?
            # Sum: 0+x+z+y+x+0 = 2x+y+z
            # 0-1-3-2-1-3-0?
            # Sum: 0+y+z+x+y+0 = x+2y+z
            # 0-2-1-3-2-3-0?
            # Sum: 0+x+y+z+z+0 = x+y+2z
            # 0-2-3-1-2-1-0?
            # Sum: 0+z+y+x+z+0 = x+y+2z (duplicate)
            # 0-1-2-3-2-? No.
            # 0-1-3-2-3-? No.
            # What about 0-1-2-0-? No.
            
            # Let's check 2x+y+z, x+2y+z, x+y+2z.
            if (2*x + y + z) % m == 0: return False
            if (x + 2*y + z) % m == 0: return False
            if (x + y + 2*z) % m == 0: return False
            
            # Are there other length 6 walks?
            # 0-1-2-3-1-0-? No, length 5.
            # 0-1-2-3-2-1-? No.
            # 0-1-3-2-1-0-? No.
            # 0-2-1-3-2-0-? No.
            # 0-2-3-1-2-0-? No.
            # 0-1-2-1-3-0-? No.
            
            # Length 7
            # 0-1-2-3-1-2-3-0?
            # Sum: 0+x+z+y+x+z+0 = 2x+2y+2z = 2(x+y+z)
            # If x+y+z != 0, this might be 0 if 2(x+y+z) = 0 mod m.
            # So check 2*(x+y+z) != 0.
            if (2*(x+y+z)) % m == 0: return False
            
            # Other length 7?
            # 0-1-2-3-2-1-3-0?
            # Sum: 0+x+z+z+x+y+0 = 2x+2z+y
            # 0-1-3-2-3-1-2-0?
            # Sum: 0+y+z+z+y+x+0 = 2y+2z+x
            # 0-2-1-3-1-2-3-0?
            # Sum: 0+x+y+y+x+z+0 = 2x+2y+z
            # 0-2-3-1-3-2-1-0?
            # Sum: 0+z+y+y+z+x+0 = 2z+2y+x (dup)
            # 0-1-2-3-1-3-2-0?
            # Sum: 0+x+z+y+y+z+0 = 2z+2y+x (dup)
            
            # Check these:
            if (2*x + 2*z + y) % m == 0: return False
            if (2*y + 2*z + x) % m == 0: return False
            if (2*x + 2*y + z) % m == 0: return False
            
            # Length 8
            # 0-1-2-3-1-2-3-1-0?
            # Sum: 0+x+z+y+x+z+y+0 = 2x+2y+2z (dup of L7 check if 2S=0)
            # Wait, L7 was 2S. L8 is 2S?
            # 0-1-2-3-1-2-3-0 is length 7?
            # 0-1 (1), 1-2 (2), 2-3 (3), 3-1 (4), 1-2 (5), 2-3 (6), 3-0 (7). Yes.
            # 0-1-2-3-1-2-3-1-0 is length 8.
            # Sum: x+z+y+x+z+y+x = 3x+2y+2z?
            # Let's trace:
            # 0->1: x
            # 1->2: x (wait, volt 12 is x)
            # 2->3: z
            # 3->1: y
            # 1->2: x
            # 2->3: z
            # 3->1: y
            # 1->0: x (volt 01 is 0? No, 1->0 is -0 = 0)
            # Sum: x+x+z+y+x+z+y+0 = 3x+2y+2z
            
            # This is getting complicated to enumerate manually.
            # Let's just do a BFS on the base graph K4 to find all non-backtracking closed walks up to length 9 and compute their voltage sums symbolically?
            # Or just iterate x,y,z and verify the lift for the first few candidates?
            # Given the small number of variables, we can just pick random x,y,z, build the graph, and verify girth.
            # Verification is O(n^2) roughly. n=244. 244^2 = 60k. Fast.
            # So we can just try random (x,y,z) and verify.
            
            return True

        # Random search
        rng = random.Random(42 + m)
        for _ in range(1000):
            x = rng.randint(1, m-1)
            y = rng.randint(1, m-1)
            z = rng.randint(1, m-1)
            
            # Quick checks
            if (x + y + z) % m == 0: continue
            if (x + y) % m == 0: continue
            if (x + z) % m == 0: continue
            if (y + z) % m == 0: continue
            
            n, edges = build_lift(m, x, y, z)
            
            if verify_girth(n, edges, g):
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            break
    
    if best_n is None:
        # Fallback to seed construction if no voltage graph found
        # This should not happen for m=61..100
        def seed_construct():
            q = k
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
            if q == 4:
                table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
                add = lambda x, y: x ^ y
                sub = lambda x, y: x ^ y
                mul = lambda x, y: table[x][y]
            else:
                add = lambda x, y: (x + y) % q
                sub = lambda x, y: (x - y) % q
                mul = lambda x, y: x * y % q
            
            rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d + 1):
                r = (c - 5) % 4
                c0 = c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

            def neighbours(vertex):
                side, x = vertex
                for first in range(q):
                    y = [None, first] + [0] * (d - 1)
                    for c in range(2, d + 1):
                        a, b = rule[c]
                        if side == 0:
                            y[c] = add(x[c], mul(y[a], x[b]))
                        else:
                            y[c] = sub(x[c], mul(x[a], y[b]))
                    yield 1 - side, tuple(y)

            start = (0, (None,) + (0,) * d)
            index = {start: 0}
            order = [start]
            edges = []
            for vertex in order:
                for other in neighbours(vertex):
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    if vertex[0] == 0:
                        edges.append((index[vertex], index[other]))
            return len(order), edges
        best_n, best_edges = seed_construct()

    return best_n, best_edges
