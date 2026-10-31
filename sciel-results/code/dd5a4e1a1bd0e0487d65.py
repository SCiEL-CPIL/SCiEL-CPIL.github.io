import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Fallback: L-U-W graph
    def luw_construct(k, g):
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

    # Helper: Generate random k-regular graph
    def generate_random_k_reg(n, k):
        # Use configuration model
        if (n * k) % 2 != 0:
            return None
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        # Randomly pair stubs
        # We need to avoid loops and multi-edges
        # Simple method: random permutation, then fix
        # This is tricky to guarantee simple graph directly.
        # Use a simpler approach for k=3:
        # Start with a cycle, then add chords?
        # Or use the "switching" method from a random multigraph.
        
        # Let's try a different approach:
        # Generate a random 3-regular graph by taking a random permutation P of 0..n-1
        # and connecting i to P(i) and P^{-1}(i)? No, that's 2-regular.
        # For 3-regular, we can take 3 permutations?
        
        # Actually, let's just use a greedy construction or backtracking for small n?
        # No, n can be 500+.
        
        # Let's use the method from "Random regular graphs" literature.
        # Or just use scipy? No, standard library + numpy.
        
        # Simple heuristic:
        # 1. Create list of vertices
        # 2. Connect in a ring (2-regular)
        # 3. Add chords to make 3-regular?
        # This is hard to make random.
        
        # Let's try the stub pairing with rejection.
        edges = []
        used = [set() for _ in range(n)]
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        
        # This simple pairing creates a multigraph.
        # We will then simplify it by swapping.
        
        adj = [set() for _ in range(n)]
        # Initial random pairing
        # This might fail to produce a simple graph easily.
        
        # Alternative: Use a known algorithm for random k-regular graphs.
        # For k=3, we can use the following:
        # Generate a random permutation p.
        # Edges: (i, (i+1)%n), (i, (i+2)%n) ... no.
        
        # Let's use a very simple method:
        # Start with a cycle.
        # Then randomly add edges and remove others to maintain degree 3.
        # This is essentially the local search we want anyway.
        
        # So, just generate a random 3-regular multigraph and then simplify.
        
        # Pairing:
        # To avoid loops, we can pair stubs from different vertices.
        # This is complex.
        
        # Let's just return None if we can't generate, and rely on L-U-W.
        # But we want to try.
        
        # Let's try a different generator:
        # 1. List of vertices 0..n-1
        # 2. Connect i to (i+1)%n and (i-1)%n (Cycle)
        # 3. We need 1 more edge per vertex.
        # 4. Randomly pair the remaining stubs.
        
        adj = [set() for _ in range(n)]
        for i in range(n):
            adj[i].add((i+1)%n)
            adj[i].add((i-1)%n)
            
        remaining_stubs = list(range(n))
        random.shuffle(remaining_stubs)
        
        # Pair them up
        # We need to ensure no loops and no multi-edges.
        # This is a perfect matching problem with constraints.
        # For now, let's just pair them and then fix.
        
        edges_to_add = []
        for i in range(0, n, 2):
            u = remaining_stubs[i]
            v = remaining_stubs[i+1]
            if u != v and v not in adj[u]:
                edges_to_add.append((u, v))
            else:
                # Rejection: try to fix later
                pass
                
        # This is getting complicated.
        # Let's just use the L-U-W graph if we can't generate a random one easily.
        # But the prompt asks for a structural change.
        # If I can't implement a robust random generator, I should stick to a simpler improvement.
        
        # What if I just use the L-U-W graph but with a smaller q?
        # No, q=k=3 is fixed.
        
        # What if I use a different explicit construction?
        # There are no other standard explicit constructions for 3-regular girth 16 that are small.
        
        # Let's try to implement the local search properly.
        # I will generate a random 3-regular graph using a simple algorithm:
        # 1. Create a list of 3N stubs.
        # 2. Randomly permute.
        # 3. Pair (0,1), (2,3), etc.
        # 4. This gives a multigraph.
        # 5. Simplify by swapping edges to remove loops and multi-edges.
        
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        adj = [set() for _ in range(n)]
        edges = []
        
        for i in range(0, n*k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                # Loop
                # Swap with another edge to fix
                # Find a random edge (x,y)
                # This is hard to do online.
                # Let's just restart generation if we get loops/multi-edges.
                return None
            if v in adj[u]:
                # Multi-edge
                return None
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
            
        return n, edges

    # Helper: Check girth
    def check_girth(n, adj, target_girth):
        # BFS from each vertex
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            queue = [start]
            visited[start] = 0
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if visited[u] >= target_girth // 2:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        # Length is visited[u] + visited[v] + 1?
                        # No, this is a back-edge in BFS tree.
                        # The cycle length is visited[u] + visited[v] + 1 only if v is ancestor.
                        # In undirected BFS, if v is visited and v != parent[u],
                        # then v is either in the same level or previous level.
                        # The cycle length is at least visited[u] + visited[v] + 1?
                        # Actually, if we find any back-edge, the girth is <= visited[u] + visited[v] + 1.
                        # If this is < target_girth, we return False.
                        if visited[u] + visited[v] + 1 < target_girth:
                            return False
        return True

    # Helper: Find a short cycle
    def find_short_cycle(n, adj, target_girth):
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            queue = [start]
            visited[start] = 0
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if visited[u] >= target_girth // 2:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        if visited[u] + visited[v] + 1 < target_girth:
                            # Reconstruct cycle
                            # This is complex. Just return True for now.
                            return True
        return False

    # Try local search
    best_n = None
    best_edges = None
    
    # Try a few sizes
    min_n = 512
    max_n = 1000
    
    for n in range(min_n, max_n, 2):
        if time.time() > deadline:
            break
            
        # Try to generate and optimize
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            n_gen, edges_gen = generate_random_k_reg(n, k)
            if n_gen is None:
                continue
                
            adj = [set() for _ in range(n_gen)]
            for u, v in edges_gen:
                adj[u].add(v)
                adj[v].add(u)
                
            # Local search
            for step in range(100):
                if time.time() > deadline:
                    break
                    
                if check_girth(n_gen, adj, g):
                    best_n = n_gen
                    best_edges = [(u, v) for u in range(n_gen) for v in adj[u] if u < v]
                    return best_n, best_edges
                    
                # Try a random swap
                # Pick two random edges
                e1 = random.choice(edges_gen)
                e2 = random.choice(edges_gen)
                if e1 == e2 or e1[0] == e2[0] or e1[0] == e2[1] or e1[1] == e2[0] or e1[1] == e2[1]:
                    continue
                
                u1, v1 = e1
                u2, v2 = e2
                
                # Try swap: (u1, u2) and (v1, v2)
                # Check if valid
                if u2 in adj[u1] or v2 in adj[v1] or u1 == u2 or v1 == v2:
                    continue
                    
                # Perform swap
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                adj[u2].remove(v2)
                adj[v2].remove(u2)
                
                adj[u1].add(u2)
                adj[u2].add(u1)
                adj[v1].add(v2)
                adj[v2].add(v1)
                
                edges_gen.remove(e1)
                edges_gen.remove(e2)
                edges_gen.append((u1, u2))
                edges_gen.append((v1, v2))
                
    if best_n is not None:
        return best_n, best_edges
        
    return luw_construct(k, g)
