import time
import random
import sys

def construct():
    deadline = time.time() + 110  # Safety margin under 120s kill
    rng = random.Random(42)
    
    # Helper: Paley 17 initialization
    def _paley_17():
        n = 17
        squares = {x * x % 17 for x in range(1, 17)}
        A = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 17 in squares:
                    A[i] |= (1 << j)
                    A[j] |= (1 << i)
        return A

    # Helper: Exact K4 count
    def count_k4(n, A):
        count = 0
        for a in range(n):
            # Neighbors of a greater than a
            na = A[a] & ~((1 << (a + 1)) - 1)
            mask_b = na
            while mask_b:
                b_bit = mask_b & -mask_b
                b = b_bit.bit_length() - 1
                mask_b ^= b_bit
                
                # Neighbors of b greater than b, also neighbors of a
                nab = A[b] & A[a] & ~((1 << (b + 1)) - 1)
                mask_c = nab
                while mask_c:
                    c_bit = mask_c & -mask_c
                    c = c_bit.bit_length() - 1
                    mask_c ^= c_bit
                    
                    # Neighbors of c greater than c, also neighbors of a and b
                    nc = A[c] & A[a] & A[b] & ~((1 << (c + 1)) - 1)
                    count += bin(nc).count('1')
        return count

    # Helper: Find a specific K4
    def find_k4(n, A):
        for a in range(n):
            na = A[a] & ~((1 << (a + 1)) - 1)
            mask_b = na
            while mask_b:
                b_bit = mask_b & -mask_b
                b = b_bit.bit_length() - 1
                mask_b ^= b_bit
                nab = A[b] & A[a] & ~((1 << (b + 1)) - 1)
                mask_c = nab
                while mask_c:
                    c_bit = mask_c & -mask_c
                    c = c_bit.bit_length() - 1
                    mask_c ^= c_bit
                    nc = A[c] & A[a] & A[b] & ~((1 << (c + 1)) - 1)
                    if nc:
                        d_bit = nc & -nc
                        d = d_bit.bit_length() - 1
                        return (a, b, c, d)
        return None

    # Helper: Max Independent Set Size via Bron-Kerbosch (Pivot)
    def max_independent_set_size(n, A):
        # Complement graph
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
        
        best = [0]
        
        def bk(R, P, X):
            if not P:
                if len(R) > best[0]:
                    best[0] = len(R)
                return
            if len(R) + bin(P).count('1') <= best[0]:
                return
            
            # Pivot selection: choose u in X (or P if X empty) with max degree in P
            if X:
                # Simple pivot: first bit in X
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                # Pick vertex in P with max degree in P
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
                candidates = P
                
            c = candidates
            while c:
                low = c & -c
                v = low.bit_length() - 1
                bk(R + [v], P & comp[v], X & comp[v])
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], all_mask, 0)
        return best[0]

    # Helper: Find an independent set of size 6 (if exists)
    def find_ind6(n, A):
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
        
        found = [None]
        
        def bk(R, P, X):
            if found[0] is not None:
                return
            if len(R) == 6:
                found[0] = tuple(R)
                return
            if not P:
                return
            if len(R) + bin(P).count('1') < 6:
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
                bk(R + [v], P & comp[v], X & comp[v])
                R.pop() if R else None # R is list, handled in recursion stack? 
                # Wait, R is passed by value in recursive call R+[v], so no pop needed here.
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], all_mask, 0)
        return found[0]

    # Helper: Cost function for optimization
    def get_cost(n, A):
        k4s = count_k4(n, A)
        if k4s > 0:
            return k4s * 1000 # Heavy penalty for K4s
        alpha = max_independent_set_size(n, A)
        if alpha >= 6:
            return (alpha - 5) * 100 # Penalty for large independent sets
        return 0

    # Helper: Local Repair
    def local_repair(n, A, rng, deadline):
        # Copy A
        A = list(A)
        max_iter = 500
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Check K4
            k4 = find_k4(n, A)
            if k4:
                a, b, c, d = k4
                # Remove a random edge from this K4
                edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                u, v = rng.choice(edges)
                A[u] &= ~(1 << v)
                A[v] &= ~(1 << u)
                continue
            
            # Check Ind6
            ind6 = find_ind6(n, A)
            if ind6:
                # Add a random edge within this independent set
                verts = list(ind6)
                u, v = rng.sample(verts, 2)
                A[u] |= (1 << v)
                A[v] |= (1 << u)
                continue
            
            # No violations
            break
        
        return A

    # Helper: Try to grow graph by 1 vertex
    def try_grow(cur_n, cur_A, rng, deadline):
        new_n = cur_n + 1
        if new_n > 40:
            return None
        
        # Try to find a valid adjacency for the new vertex
        # We want to avoid creating K4s (so don't connect to triangles) 
        # and avoid creating Ind6 (so connect to enough of the large independent sets)
        
        # Strategy: Try several random adjacencies biased by constraints
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            cand_A = list(cur_A)
            cand_A.append(0)
            
            # Determine allowed connections to avoid immediate K4s
            # If i, j, k form a triangle in cur_A, new vertex cannot connect to all three.
            # It's safer to just start random and repair.
            
            # Bias: connect to roughly half, but maybe slightly less to avoid K4s?
            # Or slightly more to avoid Ind6?
            # Let's try a few densities
            density = 0.4 + rng.random() * 0.2
            
            for i in range(cur_n):
                if rng.random() < density:
                    cand_A[i] |= (1 << (new_n - 1))
                    cand_A[new_n - 1] |= (1 << i)
            
            # Quick check: does it have K4?
            if count_k4(new_n, cand_A) > 0:
                # Try to repair it
                repaired = local_repair(new_n, cand_A, rng, deadline)
                if count_k4(new_n, repaired) == 0 and max_independent_set_size(new_n, repaired) < 6:
                    return new_n, repaired
            else:
                # Check Ind6
                if max_independent_set_size(new_n, cand_A) < 6:
                    return new_n, cand_A
                else:
                    # Try to repair Ind6
                    repaired = local_repair(new_n, cand_A, rng, deadline)
                    if count_k4(new_n, repaired) == 0 and max_independent_set_size(new_n, repaired) < 6:
                        return new_n, repaired
        
        return None

    # Main Logic
    best_n = 17
    best_A = _paley_17()
    
    # Verify start
    assert count_k4(best_n, best_A) == 0
    assert max_independent_set_size(best_n, best_A) < 6

    # Search Loop
    while time.time() < deadline - 10:
        # Try to grow
        if best_n < 40:
            result = try_grow(best_n, best_A, rng, deadline)
            if result:
                best_n, best_A = result
                continue
        
        # If we can't grow, try to repair/perturb current best to find a better starting point for growth?
        # Or just restart growth with different random seeds?
        # The try_grow function already does multiple attempts.
        # If we fail to grow for a long time, we might be stuck.
        # Let's try a "perturbation and repair" on the current best_n to see if we can find a different valid graph
        # that might be easier to grow? Or just accept we are stuck at this n.
        
        # Actually, if try_grow fails, it's likely because the specific structure of best_A is hard to extend.
        # We could try to "randomize" best_A slightly (while keeping it valid) and retry growth.
        
        # Let's do a few iterations of "perturb valid graph and repair"
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            
            # Perturb
            perturbed = list(best_A)
            num_flips = rng.randint(1, 5)
            for _ in range(num_flips):
                i = rng.randrange(best_n)
                j = rng.randrange(best_n)
                if i == j:
                    continue
                if (perturbed[j] >> i) & 1:
                    perturbed[i] &= ~(1 << j)
                    perturbed[j] &= ~(1 << i)
                else:
                    perturbed[i] |= (1 << j)
                    perturbed[j] |= (1 << i)
            
            # Repair
            repaired = local_repair(best_n, perturbed, rng, deadline)
            
            # Check if valid
            if count_k4(best_n, repaired) == 0 and max_independent_set_size(best_n, repaired) < 6:
                # Try to grow from this new valid graph
                result = try_grow(best_n, repaired, rng, deadline)
                if result:
                    best_n, best_A = result
                    break # Restart the while loop to try growing further
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_A[j] >> i) & 1:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}

if __name__ == "__main__":
    # Simple test to ensure it runs
    res = construct()
    print(f"Found graph with n={res['n']}")
    # Verify
    A = [0] * res['n']
    for u, v in res['edges']:
        A[u] |= (1 << v)
        A[v] |= (1 << u)
    print(f"K4 count: {count_k4(res['n'], A)}")
    print(f"Max Ind Set: {max_independent_set_size(res['n'], A)}")
