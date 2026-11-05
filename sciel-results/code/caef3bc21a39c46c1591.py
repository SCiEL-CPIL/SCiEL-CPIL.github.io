import random
import time
import math

def construct():
    deadline = time.time() + 110
    rng = random.Random(42)
    
    def to_edges(n, A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (A[j] >> i) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(n, A):
        # Fast check
        for a in range(n):
            for b in range(a + 1, n):
                if not (A[b] >> a) & 1: continue
                for c in range(b + 1, n):
                    if not ((A[c] >> a) & 1 and (A[c] >> b) & 1): continue
                    for d in range(c + 1, n):
                        if (A[d] >> a) & 1 and (A[d] >> b) & 1 and (A[d] >> c) & 1:
                            return True
        return False

    def count_k4(n, A):
        count = 0
        for a in range(n):
            for b in range(a + 1, n):
                if not (A[b] >> a) & 1: continue
                for c in range(b + 1, n):
                    if not ((A[c] >> a) & 1 and (A[c] >> b) & 1): continue
                    # Count common neighbors of a,b,c greater than c
                    # Actually, just check d
                    for d in range(c + 1, n):
                        if (A[d] >> a) & 1 and (A[d] >> b) & 1 and (A[d] >> c) & 1:
                            count += 1
        return count

    def find_ind6(n, A):
        # Bron-Kerbosch on complement
        comp = [0] * n
        for i in range(n):
            comp[i] = ((1 << n) - 1) & ~(A[i] | (1 << i))
        
        found = [None]
        
        def bk(R, P, X):
            if found[0] is not None: return
            if len(R) == 6:
                found[0] = tuple(R)
                return
            if not P: return
            if len(R) + bin(P).count("1") < 6: return
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
        
        bk([], (1 << n) - 1, 0)
        return found[0]

    def valid(n, A):
        if count_k4(n, A) > 0: return False
        if find_ind6(n, A) is not None: return False
        return True

    # 1. Try Cyclic Graph Search for n=35 down to 20
    # A cyclic graph is defined by a set of differences D in Z_n
    # Edge (i,j) exists if (j-i) % n in D or (i-j) % n in D
    # We represent D as a bitmask of size n//2 (since d and n-d are equivalent)
    # Actually, for simplicity, we just pick a subset of {1, ..., n-1} and symmetrize.
    
    def build_cyclic_graph(n, diff_mask):
        A = [0] * n
        for i in range(n):
            for j in range(n):
                if i == j: continue
                d = (j - i) % n
                if d == 0: continue
                # Check if d or n-d is in the mask
                # Let's define the mask to contain representatives 1..n//2
                # If d <= n//2, check bit d. Else check bit n-d.
                rep = d if d <= n // 2 else n - d
                if rep == 0: continue
                if (diff_mask >> (rep - 1)) & 1:
                    A[i] |= (1 << j)
        return A

    def cyclic_search(n, time_limit):
        # Hill climbing on the difference set
        # Start with random masks
        best_cost = float('inf')
        best_mask = 0
        
        def cost_of_mask(mask):
            A = build_cyclic_graph(n, mask)
            k4_count = count_k4(n, A)
            if k4_count > 0:
                return 1000 * k4_count
            # Check independent set
            if find_ind6(n, A) is not None:
                return 1
            return 0

        # Initialize with a few random masks
        candidates = []
        for _ in range(50):
            mask = 0
            for d in range(1, n // 2 + 1):
                if rng.random() < 0.5:
                    mask |= (1 << (d - 1))
            candidates.append(mask)
            
        # Also try sparse and dense
        candidates.append(0)
        candidates.append((1 << (n // 2)) - 1)
        
        for mask in candidates:
            if time.time() > time_limit: break
            c = cost_of_mask(mask)
            if c == 0:
                return build_cyclic_graph(n, mask)
            if c < best_cost:
                best_cost = c
                best_mask = mask
        
        # Hill climbing from best
        current_mask = best_mask
        current_cost = best_cost
        start_time = time.time()
        
        while time.time() < time_limit:
            if current_cost == 0:
                return build_cyclic_graph(n, current_mask)
            
            # Pick a random difference to flip
            d = rng.randint(1, n // 2)
            new_mask = current_mask ^ (1 << (d - 1))
            new_cost = cost_of_mask(new_mask)
            
            if new_cost < current_cost:
                current_mask = new_mask
                current_cost = new_cost
            elif rng.random() < 0.1: # Random restart within climb
                current_mask = new_mask
                current_cost = new_cost
                
        return None

    # Try cyclic search for high n
    for n in range(35, 16, -1):
        if time.time() > deadline - 20: break
        # Allocate time proportional to n
        t_limit = deadline - 20
        A = cyclic_search(n, t_limit)
        if A is not None:
            return {'n': n, 'edges': to_edges(n, A)}

    # 2. Fallback: Simulated Annealing on General Graph
    # We will try to find a valid graph for n=35
    n = 35
    A = [0] * n
    # Initialize with random graph
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < 0.5:
                A[i] |= (1 << j)
                A[j] |= (1 << i)
                
    def sa_search(n, A, deadline):
        # Precompute neighbors for faster K4 check?
        # We will use incremental K4 count.
        # This is complex. Let's use a simpler cost:
        # Cost = K4_count * 100 + Ind6_count
        # But counting Ind6 is slow.
        # We will only count Ind6 if K4_count == 0.
        
        def get_cost(A):
            k4 = count_k4(n, A)
            if k4 > 0:
                return k4
            # If no K4, check Ind6
            if find_ind6(n, A) is not None:
                return 1
            return 0

        current_A = list(A)
        current_cost = get_cost(current_A)
        best_A = list(current_A)
        best_cost = current_cost
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        steps = 0
        while time.time() < deadline and T > T_min:
            # Pick random edge
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j: continue
            if i > j: i, j = j, i
            
            # Flip edge
            if (current_A[j] >> i) & 1:
                current_A[i] &= ~(1 << j)
                current_A[j] &= ~(1 << i)
            else:
                current_A[i] |= (1 << j)
                current_A[j] |= (1 << i)
                
            new_cost = get_cost(current_A)
            delta = new_cost - current_cost
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_A = list(current_A)
                    if best_cost == 0:
                        return best_A
            else:
                # Revert
                if (current_A[j] >> i) & 1:
                    current_A[i] &= ~(1 << j)
                    current_A[j] &= ~(1 << i)
                else:
                    current_A[i] |= (1 << j)
                    current_A[j] |= (1 << i)
                    
            T *= cooling
            steps += 1
            if steps % 1000 == 0:
                # Occasionally restart from best if stuck
                if current_cost > 0 and best_cost == 0:
                    return best_A
                if current_cost > best_cost + 10:
                    current_A = list(best_A)
                    current_cost = best_cost
                    T = 0.5
                    
        return best_A if best_cost == 0 else None

    A_sol = sa_search(n, A, deadline)
    if A_sol is not None:
        return {'n': n, 'edges': to_edges(n, A_sol)}
        
    # If all fails, return best found so far (likely none valid for 35)
    # Return a valid smaller graph, e.g., Paley(17)
    q = 17
    squares = {x * x % q for x in range(1, q)}
    n0 = q
    A0 = [0] * n0
    for i in range(n0):
        for j in range(n0):
            if i != j and (j - i) % q in squares:
                A0[i] |= (1 << j)
    return {'n': n0, 'edges': to_edges(n0, A0)}
