<<<<<<< SEARCH
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n = 20
    # M[i] is a bitmask of neighbors of i
    # M[i] & (1 << j) is 1 if i is connected to j
    M = [0] * n
    
    # Helper to check if adding edge (u, v) is valid
    def can_add(u, v):
        if u == v:
            return False
        if (M[u] >> v) & 1:
            return False # already exists
        
        # The new masks
        new_M_u = M[u] | (1 << v)
        new_M_v = M[v] | (1 << u)
        
        # Check all triples involving u or v
        # Triples {u, x, y} where x != u, y != u, x < y
        # We need to check if popcount(N(u) & N(x) & N(y)) > 2
        # With new edge, N(u) changes to new_M_u.
        
        # 1. Triples of form {u, v, w}
        for w in range(n):
            if w == u or w == v:
                continue
            # N(u) is new_M_u, N(v) is new_M_v, N(w) is M[w]
            # Intersection: new_M_u & new_M_v & M[w]
            if (new_M_u & new_M_v & M[w]).bit_count() > 2:
                return False
        
        # 2. Triples of form {u, x, y} where x, y != u, v
        # N(u) is new_M_u. N(x) is M[x], N(y) is M[y].
        # We need to check if new_M_u & M[x] & M[y] has > 2 bits.
        # Note: M[x] and M[y] are unchanged.
        # The only change is in N(u).
        # So we check if adding v to N(u) creates a 3rd common neighbor with x and y.
        # Basically, if M[x] & M[y] & (1 << v) is non-zero, then v is a common neighbor.
        # If M[x] & M[y] & M[u] already had 2 bits, and v is also in M[x] & M[y], then it becomes 3.
        
        # Efficient check:
        # For each pair (x, y) not involving u or v:
        # Let common_xy = M[x] & M[y]
        # If (common_xy & (1 << v)) != 0:
        #    Then v is a common neighbor of x and y.
        #    The new common neighbors of u, x, y are (new_M_u & common_xy).
        #    new_M_u = M[u] | (1 << v).
        #    So new intersection = (M[u] & common_xy) | (1 << v if v in common_xy else 0)
        #    We need popcount(new intersection) <= 2.
        
        # We can iterate x from 0 to n-1, y from x+1 to n-1
        # Skip if x==u or y==u or x==v or y==v
        
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                common_xy = M[x] & M[y]
                # If v is in common_xy, it adds to the count
                # current count for u,x,y is (M[u] & common_xy).bit_count()
                # new count is (new_M_u & common_xy).bit_count()
                # This is efficient enough for n=20.
                if (new_M_u & common_xy).bit_count() > 2:
                    return False
        
        # 3. Triples of form {v, x, y} where x, y != u, v
        # Symmetric to case 2
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                common_xy = M[x] & M[y]
                if (new_M_v & common_xy).bit_count() > 2:
                    return False
        
        return True
=======
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin for 150s limit

    n = 20
    M = [0] * n

    def can_add(u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free."""
        if u == v:
            return False
        if (M[u] >> v) & 1:
            return False

        # 1. Triples of form {u, v, w}
        # N(u)_new & N(v)_new & N(w) = M[u] & M[v] & M[w] (since no self-loops)
        common_uv = M[u] & M[v]
        for w in range(n):
            if w == u or w == v:
                continue
            if (common_uv & M[w]).bit_count() > 2:
                return False

        # 2. Triples of form {u, x, y} where x, y != u, v
        # 3. Triples of form {v, x, y} where x, y != u, v
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                common_xy = M[x] & M[y]
                # Check triple (u, x, y)
                cu = (M[u] & common_xy).bit_count()
                if cu > 2:
                    return False
                if cu == 2 and (common_xy >> v) & 1:
                    return False
                # Check triple (v, x, y)
                cv = (M[v] & common_xy).bit_count()
                if cv > 2:
                    return False
                if cv == 2 and (common_xy >> u) & 1:
                    return False

        return True
>>>>>>> REPLACE

