import random
import time

def construct():
    deadline = time.time() + 110
    rng = random.Random(42)
    n_target = 35
    best_n = 17
    best_A = None
    
    # Helper to construct Paley(17) as fallback
    def make_paley_17():
        q = 17
        squares = {x * x % q for x in range(1, q)}
        A = [0] * q
        for i in range(q):
            for j in range(q):
                if i != j and (j - i) % q in squares:
                    A[i] |= (1 << j)
        return A

    best_A = make_paley_17()
    best_n = 17

    def find_k4(A, n):
        """Returns (a,b,c,d) if K4 exists, else None."""
        for a in range(n):
            for b in range(a + 1, n):
                if not (A[b] >> a) & 1:
                    continue
                # Check if there are common neighbors
                common = A[a] & A[b] & ~((1 << a) | (1 << b))
                c_bit = common & -common
                if not c_bit:
                    continue
                c = c_bit.bit_length() - 1
                # Need a 4th vertex d connected to a,b,c
                # d must be in A[a] & A[b] & A[c]
                # We already know c is in A[a]&A[b].
                # So check if A[a] & A[b] & A[c] has any bit set other than a,b,c
                # Actually, we just need ANY d.
                # But we can just check if (A[a] & A[b] & A[c]) has more than just the bits for a,b,c?
                # No, d must be distinct.
                # Let's iterate through candidates for d.
                # Candidates are in A[a] & A[b] & A[c]
                # Since a,b are connected to c, a and b are in A[c].
                # So A[a] & A[b] & A[c] contains at least a,b.
                # If it contains anything else, that's d.
                mask = A[a] & A[b] & A[c]
                # Mask out a, b, c
                mask &= ~((1 << a) | (1 << b) | (1 << c))
                if mask:
                    d_bit = mask & -mask
                    d = d_bit.bit_length() - 1
                    return (a, b, c, d)
        return None

    def find_ind6(A, n):
        """Returns 6 vertices if Ind6 exists, else None.
        Uses Bron-Kerbosch on complement to find a 6-clique in complement."""
        # Complement adjacency
        full_mask = (1 << n) - 1
        comp = [0] * n
        for i in range(n):
            comp[i] = full_mask & ~(A[i] | (1 << i))
        
        # Bron-Kerbosch to find just one clique of size 6
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
            # Pivot selection
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

        bk([], full_mask, 0)
        return found[0]

    def valid(A, n):
        return find_k4(A, n) is None and find_ind6(A, n) is None

    def to_edges(n, A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (A[j] >> i) & 1:
                    edges.append([i, j])
        return edges

    # Main Loop: Try to find a valid 35-vertex graph
    # Strategy: Start with random graph, repair.
    
    for seed in range(100):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(seed)
        A = [0] * n_target
        # Initialize with p ~ 0.5
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if rng.random() < 0.5:
                    A[i] |= (1 << j)
                    A[j] |= (1 << i)
        
        # Local Repair Loop
        # Max iterations to avoid infinite loops in a single seed attempt
        max_repair_steps = 2000
        for _ in range(max_repair_steps):
            if time.time() > deadline - 5:
                break
            
            k4 = find_k4(A, n_target)
            if k4:
                a, b, c, d = k4
                # Remove a random edge from the K4
                edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                u, v = rng.choice(edges)
                A[u] &= ~(1 << v)
                A[v] &= ~(1 << u)
                continue
            
            ind6 = find_ind6(A, n_target)
            if ind6:
                # Add a random edge within the independent set
                verts = list(ind6)
                u, v = rng.sample(verts, 2)
                A[u] |= (1 << v)
                A[v] |= (1 << u)
                continue
            
            # No violations found
            if valid(A, n_target):
                best_n = n_target
                best_A = A
                break
        
        if best_n == n_target:
            break
            
        # If we failed, we might try a perturbation strategy for the next seed
        # But random restarts are usually sufficient for this density

    # Fallback: If we didn't find 35, try 34, 33...
    # But likely we found 35 or 17 is the best.
    # Given the known existence, random restart repair should find it.
    # If not, return the best found (likely 17).
    
    # Ensure we return the best A corresponding to best_n
    # If best_n is still 17, best_A is Paley(17).
    # If best_n is 35, best_A is the repaired one.
    
    return {'n': best_n, 'edges': to_edges(best_n, best_A)}
