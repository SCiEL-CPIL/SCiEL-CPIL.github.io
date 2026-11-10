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
        raise ValueError("Only k=3 supported in this implementation")

    # 1. Define GF(16)
    field = GF(4, 0b10011)
    q = field.q

    # 2. Generate SL(2, q)
    elements = []
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0:
                    if field.mul(b, c) == 1:
                        for d in range(q):
                            elements.append((a,b,c,d))
                else:
                    bc = field.mul(b, c)
                    rhs = field.add(1, bc)
                    d = field.div(rhs, a)
                    elements.append((a,b,c,d))
                    
    n_total = len(elements)
    
    # Define subgroup H of order 4
    # H = { I, U1, U2, U1U2 }
    # U1 = [[1, 1], [0, 1]]
    # U2 = [[1, 0], [1, 1]]
    I = (1, 0, 0, 1)
    U1 = (1, 1, 0, 1)
    U2 = (1, 0, 1, 1)
    U1U2 = (0, 1, 1, 1) # [[1,1],[0,1]] * [[1,0],[1,1]] = [[1*1+1*1, 1*0+1*1], [0*1+1*1, 0*0+1*1]] = [[0, 1], [1, 1]]
    H = [I, U1, U2, U1U2]
    
    # Map elements to coset indices
    elem_to_coset = {}
    coset_reps = []
    visited = set()
    
    # Helper to multiply matrices
    def mat_mul(m1, m2):
        a1,b1,c1,d1 = m1
        a2,b2,c2,d2 = m2
        aa = field.add(field.mul(a1,a2), field.mul(b1,c2))
        ab = field.add(field.mul(a1,b2), field.mul(b1,d2))
        ba = field.add(field.mul(c1,a2), field.mul(d1,c2))
        bb = field.add(field.mul(c1,b2), field.mul(d1,d2))
        return (aa, ab, ba, bb)
        
    for e in elements:
        if e in visited:
            continue
        # Find all elements in eH
        coset = []
        for h in H:
            eh = mat_mul(e, h)
            coset.append(eh)
            visited.add(eh)
        idx = len(coset_reps)
        coset_reps.append(coset[0]) # Just store one rep
        for c in coset:
            elem_to_coset[c] = idx
            
    n = len(coset_reps)
    
    # Find Involutions outside H
    involutions = []
    for e in elements:
        a,b,c,d = e
        if field.add(a,d) == 0 and e != I:
            # Check if e is in H
            if e not in H:
                involutions.append(e)
                
    # Search for generators
    random.seed(42)
    
    def get_girth(adj, limit):
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
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        if min_girth < limit:
                            return min_girth
        return min_girth

    # Try random sets of 3 involutions
    for _ in range(500): # Reduced iterations since graph is smaller
        S = random.sample(involutions, 3)
        
        # Build adjacency
        adj = [[] for _ in range(n)]
        
        # Check if S generates the whole coset graph? 
        # And ensure S elements don't map a coset to itself (i.e., s not in H)
        # We already filtered H from involutions.
        
        for i, rep in enumerate(coset_reps):
            for s in S:
                v = mat_mul(s, rep)
                j = elem_to_coset[v]
                adj[i].append(j)
                
        # Remove duplicates
        for i in range(n):
            adj[i] = list(set(adj[i]))
            if len(adj[i]) != 3:
                break
            # Ensure no self loops
            if i in adj[i]:
                break
            # Ensure simple
            if len(set(adj[i])) != len(adj[i]):
                break
            # Check connectivity implicitly? 
            # If not connected, girth is min of components.
            # But we want a single graph.
            
        # Check connectivity
        visited_nodes = [False]*n
        q_d = deque([0])
        visited_nodes[0] = True
        count = 1
        while q_d:
            u = q_d.popleft()
            for v in adj[u]:
                if not visited_nodes[v]:
                    visited_nodes[v] = True
                    count += 1
                    q_d.append(v)
        if count != n:
            continue
            
        # Girth
        girth = get_girth(adj, g)
        if girth >= g:
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
            
    raise RuntimeError("Failed to find valid graph")
