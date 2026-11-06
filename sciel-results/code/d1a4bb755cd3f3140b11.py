import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    
    N = 512
    
    # Precompute compatible masks
    # compatible[i] is an integer where bit j is set if lcm(i, j) <= N
    compatible = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i == j:
                # A number is always compatible with itself
                mask |= (1 << j)
                continue
            g = gcd(i, j)
            l = (i // g) * j
            if l <= N:
                mask |= (1 << j)
        compatible[i] = mask
        
    # Initial candidate mask: all numbers 1..N
    # We use bits 1..N. Bit 0 is unused.
    initial_mask = (1 << (N + 1)) - 2  # Bits 1 to N set
    
    best_set = []
    best_size = 0
    
    # To improve robustness, we can try a few different orderings or restarts
    # but the "Most Constrained First" greedy is usually very strong for this N.
    # Let's implement a single high-quality greedy pass first.
    
    def greedy_solve():
        candidates = initial_mask
        solution = []
        
        while candidates:
            # Find the candidate with the minimum number of compatible options
            # among the remaining candidates.
            # This is equivalent to finding the vertex with the minimum degree
            # in the subgraph induced by 'candidates'.
            
            min_count = N + 1
            chosen = -1
            
            # Iterate through all set bits in candidates
            c = candidates
            while c:
                # Get the lowest set bit
                lsb = c & -c
                v = lsb.bit_length() - 1
                
                # Count compatible neighbors in current candidates
                # compatible[v] contains v itself and its compatible neighbors
                # We intersect with candidates to see how many options remain
                compat_in_candidates = compatible[v] & candidates
                count = bin(compat_in_candidates).count('1')
                
                if count < min_count:
                    min_count = count
                    chosen = v
                    # Optimization: If count is 1, it's the best possible (isolated in comp graph? No, min degree)
                    # Actually, if count is 1, it means it's only compatible with itself in the remaining set.
                    # We can't do better than 1 (since it's compatible with itself).
                    if count == 1:
                        break
                
                # Clear the lowest set bit to continue iteration
                c ^= lsb
            
            if chosen == -1:
                break
                
            solution.append(chosen)
            
            # Remove 'chosen' and all its INCOMPATIBLE neighbors from candidates
            # Incompatible neighbors are those in candidates NOT in compatible[chosen]
            # So we keep only the intersection of candidates and compatible[chosen]
            # Wait, we need to remove incompatible ones.
            # New candidates = candidates & compatible[chosen]
            # This works because compatible[chosen] contains chosen and all numbers compatible with it.
            # Any number not in compatible[chosen] is incompatible and must be removed.
            candidates = candidates & compatible[chosen]
            
        return solution
    
    # Run the greedy solver
    sol = greedy_solve()
    best_set = sol
    best_size = len(sol)
    
    # Optional: Try a second strategy if time permits?
    # Strategy 2: Greedy by smallest value (often works well for lcm problems)
    if time.time() < deadline - 10:
        def greedy_solve_small_first():
            candidates = initial_mask
            solution = []
            
            while candidates:
                # Find the smallest value in candidates
                # This is just the index of the lowest set bit
                lsb = candidates & -candidates
                v = lsb.bit_length() - 1
                
                solution.append(v)
                candidates = candidates & compatible[v]
                
            return solution
            
        sol2 = greedy_solve_small_first()
        if len(sol2) > best_size:
            best_set = sol2
            best_size = len(sol2)
            
    # Optional: Strategy 3: Greedy by largest value?
    if time.time() < deadline - 10:
        def greedy_solve_large_first():
            candidates = initial_mask
            solution = []
            
            while candidates:
                # Find the largest value in candidates
                # This is the index of the highest set bit
                v = candidates.bit_length() - 1
                lsb = 1 << v
                
                solution.append(v)
                candidates = candidates & compatible[v]
                
            return solution
            
        sol3 = greedy_solve_large_first()
        if len(sol3) > best_size:
            best_set = sol3
            best_size = len(sol3)

    # Fallback
    if not best_set:
        return [1]
        
    return best_set

if __name__ == "__main__":
    res = construct()
    print(len(res))
