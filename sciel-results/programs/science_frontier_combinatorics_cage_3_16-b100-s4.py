import random
import sys
from collections import deque

# GF(q) arithmetic for q=2^m
class GF:
    def __init__(self, m, poly):
        self.m = m
        self.q = 1 << m
        self.poly = poly
        self.mul_table = [[0]*self.q for _ in range(self.q)]
        self.inv_table = [0]*self.q
        self.add_table = [[0]*self.q for _ in range(self.q)]
        
        # Precompute
        for i in range(self.q):
            for j in range(self.q):
                self.add_table[i][j] = i ^ j
                # Mult
                res = 0
                a = i
                b = j
                while b:
                    if b & 1:
                        res ^= a
                    a <<= 1
                    if a >= self.q:
                        a ^= self.poly
                    b >>= 1
                self.mul_table[i][j] = res
        self.inv_table[1] = 1
        for i in range(2, self.q):
            # Find inverse
            for j in range(2, self.q):
                if self.mul_table[i][j] == 1:
                    self.inv_table[i] = j
                    self.inv_table[j] = i
                    break

    def add(self, a, b):
        return a ^ b

    def sub(self, a, b):
        return a ^ b

    def mul(self, a, b):
        return self.mul_table[a][b]

    def inv(self, a):
        return self.inv_table[a]

    def div(self, a, b):
        return self.mul_table[a][self.inv_table[b]]

def construct(k, g):
    # We aim for k=3, g=16
    # Using SL(2, 16)
    
    if k != 3:
        # If k is not 3, we might not have a good strategy.
        # But the task specifies k=3.
        # For robustness, if k != 3, return a simple error or try q=2^k?
        # Let's just focus on k=3.
        raise ValueError("Only k=3 supported in this implementation")

    # 1. Define GF(16)
    # x^4 + x + 1 = 19 (binary 10011)
    field = GF(4, 0b10011)
    q = field.q

    # 2. Generate SL(2, q)
    # Matrices [[a, b], [c, d]] with ad - bc = 1
    # In GF(2^m), sub is xor.
    elements = []
    det_func = lambda a,b,c,d: field.add(field.mul(a,d), field.mul(b,c)) # ad - bc = ad + bc
    # Wait, det = ad - bc. In GF(2), -bc = bc. So det = ad + bc.
    
    # We need det = 1.
    for a in range(q):
        for b in range(q):
            for c in range(q):
                # ad + bc = 1 => ad = 1 + bc => d = (1 + bc) / a
                if a == 0:
                    # Then bc = 1. c != 0, b != 0. d is free?
                    # No, if a=0, det = bc. So bc=1.
                    # d can be anything?
                    # Matrix [[0, b], [c, d]]. det = -bc = bc.
                    # If bc=1, d is free.
                    if field.mul(b, c) == 1:
                        for d in range(q):
                            elements.append((a,b,c,d))
                else:
                    bc = field.mul(b, c)
                    rhs = field.add(1, bc)
                    d = field.div(rhs, a)
                    elements.append((a,b,c,d))
                    
    n = len(elements)
    # n should be 4080 for q=16
    
    # Map elements to indices
    elem_to_idx = {e: i for i, e in enumerate(elements)}
    
    # 3. Find Involutions
    # M^2 = I => Trace(M) = 0 (in char 2) and M != I
    # Trace = a + d
    involutions = []
    for e in elements:
        a,b,c,d = e
        if field.add(a,d) == 0 and not (a==1 and b==0 and c==0 and d==1):
            involutions.append(e)
            
    # 4. Search for generators
    # We need 3 generators S.
    # Try random sets.
    random.seed(42)
    
    def get_girth(adj, limit):
        # Returns girth if < limit, else limit
        # BFS from each node
        n_nodes = len(adj)
        min_girth = float('inf')
        for start in range(n_nodes):
            dist = [-1] * n_nodes
            parent = [-1] * n_nodes
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found a cycle
                        # Length is dist[u] + dist[v] + 1
                        # Wait, this is standard BFS cycle finding.
                        # For undirected graph, if we see a visited neighbor not parent,
                        # cycle length = dist[u] + dist[v] + 1.
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        if min_girth < limit:
                            return min_girth
        return min_girth

    best_girth = 0
    best_set = None
    
    # Try many random sets of 3 involutions
    # To speed up, we can check girth locally first?
    # Just run full girth check. n=4080 is small.
    
    for _ in range(1000):
        S = random.sample(involutions, 3)
        # Ensure they generate the whole group? 
        # If not, graph is disconnected. 
        # Disconnected graph girth is min girth of components.
        # But we want a single graph.
        # Check connectivity implicitly by girth? 
        # If disconnected, BFS from 0 won't reach all.
        # So we should check if graph is connected.
        
        # Build adjacency
        adj = [[] for _ in range(n)]
        # Add edges
        is_connected = False
        # Actually, let's just build and check connectivity/girth
        for i, e in enumerate(elements):
            for s in S:
                # e * s
                a1,b1,c1,d1 = e
                a2,b2,c2,d2 = s
                # Mult
                aa = field.add(field.mul(a1,a2), field.mul(b1,c2))
                ab = field.add(field.mul(a1,b2), field.mul(b1,d2))
                ba = field.add(field.mul(c1,a2), field.mul(d1,c2))
                bb = field.add(field.mul(c1,b2), field.mul(d1,d2))
                f = (aa, ab, ba, bb)
                j = elem_to_idx[f]
                adj[i].append(j)
        
        # Remove duplicates (if s1=s2, but they are distinct)
        # Involutions are distinct.
        # But e*s might be same for different s? No, group is free on right.
        
        # Check connectivity and girth
        # BFS from 0
        visited = [False]*n
        q_d = deque([0])
        visited[0] = True
        count = 1
        while q_d:
            u = q_d.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    q_d.append(v)
        if count != n:
            continue
            
        # Girth
        girth = get_girth(adj, g)
        if girth >= g:
            # Found it
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
            
    # If not found, try q=32
    # (Fallback)
    raise RuntimeError("Failed to find valid graph")
