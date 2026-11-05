import time
import sys

def construct():
    deadline = time.time() + 115  # Safety margin under 150s kill
    
    def to_edges(n, A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (A[j] >> i) & 1:
                    edges.append([i, j])
        return edges
    
    def has_k4(n, A):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                for c in range(b + 1, n):
                    if not ((A[c] >> a) & 1 and (A[c] >> b) & 1):
                        continue
                    for d in range(c + 1, n):
                        if (A[d] >> a) & 1 and (A[d] >> b) & 1 and (A[d] >> c) & 1:
                            return True
        return False
    
    def has_ind6(n, A):
        """Check if graph has an independent set of size 6 using Bron-Kerbosch on complement."""
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
        
        found = [False]
        
        def bk(R, P, X):
            nonlocal found
            if found[0]:
                return
            if len(R) == 6:
                found[0] = True
                return
            if not P:
                return
            if len(R) + bin(P).count("1") < 6:
                return
            
            # Pivot selection for efficiency
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                max_deg = -1
                u = -1
                p_temp = P
                while p_temp:
                    v_bit = p_temp & -p_temp
                    v = v_bit.bit_length() - 1
                    deg = bin(comp[v] & P).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        u = v
                    p_temp ^= v_bit
                if u < 0:
                    candidates = P
                else:
                    candidates = P
            
            c = candidates
            while c and not found[0]:
                low = c & -c
                v = low.bit_length() - 1
                R.append(v)
                bk(R, P & comp[v], X & comp[v])
                R.pop()
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], all_mask, 0)
        return found[0]
    
    def valid(n, A):
        if has_k4(n, A):
            return False
        if has_ind6(n, A):
            return False
        return True
    
    def make_paley(q):
        """Create Paley graph of order q (q must be prime ≡ 1 mod 4)."""
        if q < 5:
            return None
        for i in range(2, int(q**0.5) + 1):
            if q % i == 0:
                return None
        if q % 4 != 1:
            return None
        squares = {x * x % q for x in range(1, q)}
        n = q
        A = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and (j - i) % q in squares:
                    A[i] |= (1 << j)
        return A
    
    def make_cyclic(n, D):
        """Create circulant graph on Z_n with difference set D."""
        A = [0] * n
        for i in range(n):
            for d in D:
                j = (i + d) % n
                if j != i:
                    A[i] |= (1 << j)
        return A
    
    def make_cyclic_symmetric(n, D_half):
        """Create circulant graph on Z_n with symmetric difference set.
        D_half contains elements in 1..n//2, and we add both d and n-d."""
        D = set()
        for d in D_half:
            D.add(d % n)
            D.add((n - d) % n)
        D.discard(0)
        A = [0] * n
        for i in range(n):
            for d in D:
                j = (i + d) % n
                if j != i:
                    A[i] |= (1 << j)
        return A
    
    best_n = 0
    best_A = []
    
    # Try Paley graphs
    for q in [5, 13, 17, 29, 37]:
        if time.time() > deadline - 30:
            break
        A = make_paley(q)
        if A is not None:
            n = len(A)
            if valid(n, A):
                if n > best_n:
                    best_n = n
                    best_A = A[:]
    
    # Try cyclic constructions on Z_36
    if time.time() < deadline - 20:
        n = 36
        # Try various difference sets
        # Strategy: use quadratic residues mod 36 (but 36 is not prime)
        # Instead, try structured sets
        
        # Set 1: Based on quadratic residues mod 17 and mod 19 (Chinese remainder)
        # But let's try simpler structured sets
        
        # Try: D = {1,2,4,5,8,10,11,13,16,17} (10 elements, symmetric gives 20 edges per vertex)
        D_sets = [
            {1,2,4,5,8,10,11,13,16,17},
            {1,2,3,4,7,8,10,11,13,14,15,16},
            {1,2,4,8,10,12,14,16},
            {1,2,4,5,8,9,10,13,14,16},
            {1,3,4,5,7,8,10,11,13,15,16},
            {1,2,4,6,8,10,12,14,16},
            {1,2,3,5,7,8,10,11,13,14,16},
            {1,2,4,5,7,9,10,11,13,15,16},
            {1,2,3,4,5,7,8,10,11,13,14,15,16},
            {1,2,4,5,8,10,11,12,13,14,15,16},
        ]
        
        for D_half in D_sets:
            if time.time() > deadline - 10:
                break
            A = make_cyclic_symmetric(36, D_half)
            if valid(36, A):
                if 36 > best_n:
                    best_n = 36
                    best_A = A[:]
                break
    
    # Try cyclic constructions on Z_35
    if time.time() < deadline - 20 and best_n < 36:
        n = 35
        D_sets_35 = [
            {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17},
            {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
            {1,2,3,4,5,6,7,8,9,10,11,12,13,14},
            {1,2,3,4,5,6,7,8,9,10,11,12,13},
            {1,2,3,4,5,6,7,8,9,10,11,12},
            {1,2,3,4,5,6,7,8,9,10,11},
            {1,2,3,4,5,6,7,8,9,10},
            {1,2,3,4,5,6,7,8,9},
            {1,2,3,4,5,6,7,8},
        ]
        
        for D_half in D_sets_35:
            if time.time() > deadline - 10:
                break
            A = make_cyclic_symmetric(35, D_half)
            if valid(35, A):
                if 35 > best_n:
                    best_n = 35
                    best_A = A[:]
                break
    
    # If we haven't found n>=35, try more systematic search
    if best_n < 35 and time.time() < deadline - 15:
        # Try Paley(29) complement or other constructions
        # Actually, let's try a different approach: start from a valid smaller graph and grow
        
        import random
        rng = random.Random(42)
        
        def random_graph(n, p, rng):
            A = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < p:
                        A[i] |= (1 << j)
                        A[j] |= (1 << i)
            return A
        
        def repair_graph(n, A, rng, deadline):
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Find and remove K4s
                k4 = None
                for a in range(n):
                    for b in range(a + 1, n):
                        if not ((A[b] >> a) & 1):
                            continue
                        for c in range(b + 1, n):
                            if not ((A[c] >> a) & 1 and (A[c] >> b) & 1):
                                continue
                            for d in range(c + 1, n):
                                if (A[d] >> a) & 1 and (A[d] >> b) & 1 and (A[d] >> c) & 1:
                                    k4 = (a, b, c, d)
                                    break
                            if k4:
                                break
                        if k4:
                            break
                    if k4:
                        break
                
                if k4:
                    a, b, c, d = k4
                    edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                    u, v = rng.choice(edges)
                    A[u] &= ~(1 << v)
                    A[v] &= ~(1 << u)
                    continue
                
                # Check for independent set of size 6
                if has_ind6(n, A):
                    # Add a random edge to break it
                    # Find an independent set
                    comp = [0] * n
                    all_mask = (1 << n) - 1
                    for i in range(n):
                        comp[i] = all_mask & ~(A[i] | (1 << i))
                    
                    found = [None]
                    
                    def bk(R, P, X):
                        nonlocal found
                        if found[0] is not None:
                            return
                        if len(R) == 6:
                            found[0] = tuple(R)
                            return
                        if not P:
                            return
                        if len(R) + bin(P).count("1") < 6:
                            return
                        
                        if X:
                            u_bit = X & -X
                            u = u_bit.bit_length() - 1
                            candidates = P & ~comp[u]
                        else:
                            candidates = P
                        
                        c = candidates
                        while c and found[0] is None:
                            low = c & -c
                            v = low.bit_length() - 1
                            R.append(v)
                            bk(R, P & comp[v], X & comp[v])
                            R.pop()
                            P = P & ~low
                            X = X | low
                            c = c & ~low
                    
                    bk([], all_mask, 0)
                    
                    if found[0]:
                        verts = list(found[0])
                        u, v = rng.sample(verts, 2)
                        A[u] |= (1 << v)
                        A[v] |= (1 << u)
                        continue
                
                break
            
            return A
        
        # Try random restarts
        for n_try in [35, 34, 33, 32]:
            if time.time() > deadline - 10:
                break
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                A = random_graph(n_try, 0.3, rng)
                A = repair_graph(n_try, A, rng, deadline)
                if valid(n_try, A):
                    if n_try > best_n:
                        best_n = n_try
                        best_A = A[:]
                    break
    
    # Final safety: if best_n is 0, return a trivial valid graph
    if best_n == 0:
        # Return a small valid graph (e.g., C5 or K5)
        n = 5
        A = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and abs(i-j) <= 2 and abs(i-j) >= 1:
                    # This is C5
                    pass
        # Just use C5: edges (0,1),(1,2),(2,3),(3,4),(4,0)
        A = [0]*5
        edges_c5 = [(0,1),(1,2),(2,3),(3,4),(4,0)]
        for u,v in edges_c5:
            A[u] |= (1<<v)
            A[v] |= (1<<u)
        best_n = 5
        best_A = A
    
    return {'n': best_n, 'edges': to_edges(best_n, best_A)}
