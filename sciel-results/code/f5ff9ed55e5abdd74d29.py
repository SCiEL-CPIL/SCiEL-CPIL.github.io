import random
import sys
from collections import deque

class GF:
    def __init__(self, q):
        self.q = q
        self.is_prime = True
        # Check if q is a prime power of 2
        if q & (q - 1) == 0:
            self.is_prime = False
            self.m = q.bit_length() - 1
            # Find primitive polynomial for GF(2^m)
            # For m=4 (q=16), x^4 + x + 1 = 19
            # For m=3 (q=8), x^3 + x + 1 = 11
            # For m=5 (q=32), x^5 + x^2 + 1 = 37
            polys = {3: 0b1011, 4: 0b10011, 5: 0b100101, 6: 0b1000111}
            self.poly = polys.get(self.m, 0)
            self.mul_table = [[0]*q for _ in range(q)]
            self.inv_table = [0]*q
            for i in range(q):
                for j in range(q):
                    res = 0
                    a = i
                    b = j
                    while b:
                        if b & 1:
                            res ^= a
                        a <<= 1
                        if a >= q:
                            a ^= self.poly
                        b >>= 1
                    self.mul_table[i][j] = res
            self.inv_table[1] = 1
            for i in range(2, q):
                for j in range(2, q):
                    if self.mul_table[i][j] == 1:
                        self.inv_table[i] = j
                        self.inv_table[j] = i
                        break
        else:
            # Prime field
            self.mul_table = None
            self.inv_table = [0]*q
            self.inv_table[1] = 1
            for i in range(2, q):
                self.inv_table[i] = pow(i, q-2, q)

    def add(self, a, b):
        if self.is_prime:
            return (a + b) % self.q
        else:
            return a ^ b

    def sub(self, a, b):
        if self.is_prime:
            return (a - b) % self.q
        else:
            return a ^ b

    def mul(self, a, b):
        if self.is_prime:
            return (a * b) % self.q
        else:
            return self.mul_table[a][b]

    def inv(self, a):
        if self.is_prime:
            return self.inv_table[a]
        else:
            return self.inv_table[a]

    def div(self, a, b):
        return self.mul(a, self.inv(b))

def construct(k, g):
    if k != 3:
        raise ValueError("Only k=3 supported")
        
    # Try q = 11, 13, 16, 32...
    # q must be such that |SL(2, q)| >= Moore Bound
    # Moore Bound for k=3, g=16 is 510.
    # |SL(2, 11)| = 1320
    # |SL(2, 13)| = 2184
    # |SL(2, 16)| = 4080
    
    candidates_q = [11, 13, 16]
    
    for q in candidates_q:
        field = GF(q)
        # Generate SL(2, q)
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
                        # det = ad - bc = 1 => ad = 1 + bc
                        # In char 2, 1+bc = 1-bc. In prime field, 1-bc.
                        # Wait, det = ad - bc.
                        # We want ad - bc = 1 => ad = 1 + bc.
                        # d = (1 + bc) * a^{-1}
                        rhs = field.add(1, bc)
                        d = field.div(rhs, a)
                        elements.append((a,b,c,d))
                        
        n = len(elements)
        elem_to_idx = {e: i for i, e in enumerate(elements)}
        
        # Find Involutions
        involutions = []
        for e in elements:
            a,b,c,d = e
            # M^2 = I => Trace = 0 (char 2) or Trace = -2 (char != 2)?
            # In SL(2, q), M is involution iff M != I and Trace(M) = 0 (if char 2) or Trace(M) = -2 (if char != 2).
            # Actually, eigenvalues are 1, -1.
            # If char 2, -1 = 1. So M^2=I implies (M-I)^2=0.
            # For SL(2,2) (S3), involutions are transpositions.
            # General condition: M is involution iff M != I and det(M)=1 and trace(M) = 0 (char 2) or trace(M) = -2 (char != 2).
            # Wait, in SL(2, q) with q odd, involutions have trace -2.
            # In SL(2, q) with q even, involutions have trace 0.
            
            trace = field.add(a, d)
            is_identity = (a==1 and b==0 and c==0 and d==1)
            
            if field.is_prime:
                # q is odd
                target_trace = field.sub(0, 2) # -2 mod q
                if trace == target_trace and not is_identity:
                    involutions.append(e)
            else:
                # q is even
                if trace == 0 and not is_identity:
                    involutions.append(e)
                    
        if len(involutions) < 3:
            continue
            
        # Search for 3 involutions
        # Use random sampling with local search or just random sampling
        # For q=11, ~66 involutions. C(66,3) = 45000.
        # We can try all if fast enough, or random.
        
        # Precompute matrix multiplication for speed?
        # Let's just do it on the fly but optimized.
        
        def get_girth(adj, limit):
            n_nodes = len(adj)
            min_girth = limit
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

        random.seed(42)
        # Try random sets
        # To increase chance, we can bias towards generators that don't commute?
        # Just random for now.
        
        found = False
        # Limit iterations to avoid timeout
        max_iters = 5000
        for _ in range(max_iters):
            S = random.sample(involutions, 3)
            # Build adj
            adj = [[] for _ in range(n)]
            for i, e in enumerate(elements):
                a1,b1,c1,d1 = e
                for s in S:
                    a2,b2,c2,d2 = s
                    aa = field.add(field.mul(a1,a2), field.mul(b1,c2))
                    ab = field.add(field.mul(a1,b2), field.mul(b1,d2))
                    ba = field.add(field.mul(c1,a2), field.mul(d1,c2))
                    bb = field.add(field.mul(c1,b2), field.mul(d1,d2))
                    f = (aa, ab, ba, bb)
                    j = elem_to_idx[f]
                    adj[i].append(j)
                    
            # Check connectivity
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
                
            girth = get_girth(adj, g)
            if girth >= g:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
    raise RuntimeError("Failed to find valid graph")
