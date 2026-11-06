import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin before 90s kill
    
    n = 512
    k = 4
    N = n + 1  # 1-indexed to 512
    
    # 1. Precompute conflicts
    # conflict[x] is a list of lists. conflict[x][y] is the set of z such that 
    # {x, y, z} is a subset of a 4-AP.
    # If we add x and y, we cannot add any z in conflict[x][y].
    # This allows O(1) check: adding x to a set S is valid iff 
    # for all y in S, conflict[x][y] is disjoint from S.
    # Wait, that's not O(1). 
    # Let's use a different vectorization.
    
    # Precompute a 2D array conflict_count[x][y] = number of z in [1, n] 
    # such that {x, y, z} is part of a 4-AP.
    # Actually, we need to know if a *specific* z is in the set.
    
    # Better: Precompute for each x, a bitset of all y such that x and y 
    # "conflict" in the sense that there exists a z making a 4-AP with x and y?
    # No, a 4-AP has 4 terms. {x, y} is a pair. 
    # If we have a set S, and we want to add x, we need to check if there 
    # exists y1, y2 in S such that {x, y1, y2} can be extended to a 4-AP.
    # Let's define C[x] as the set of pairs (y1, y2) such that {x, y1, y2} 
    # is a subset of a 4-AP.
    # This is too complex for O(1).
    
    # Alternative: Precompute a 3D boolean array or a list of sets.
    # For each x, let neighbors[x] be the set of y such that x and y are in 
    # the same 4-AP.
    # If we add x, we must ensure that for no pair (y1, y2) in S do we have 
    # y1, y2 in neighbors[x] AND {x, y1, y2} forms a 4-AP.
    
    # Let's stick to the precomputed "forbidden triples" approach but optimized.
    # For each x, precompute a list of (y1, y2) pairs. 
    # When adding x, check if any (y1, y2) in list[x] has both y1, y2 in S.
    
    # Even better: Precompute a matrix M where M[x][y] is a bitset of z's 
    # such that {x, y, z} is a 4-AP.
    # If we add x to S, we need to check if for any y in S, 
    # (M[x][y] & S) is non-empty.
    # Using Python ints as bitsets, this is fast.
    
    # Let's build the bitsets.
    # bits[x][y] is an integer where bit z is 1 if {x, y, z} is a 4-AP.
    # Note: A 4-AP is {a, a+d, a+2d, a+3d}.
    # Pairs in it: (a, a+d), (a, a+2d), (a, a+3d), (a+d, a+2d), (a+d, a+3d), (a+2d, a+3d).
    # For a pair (u, v), the other two terms are determined if they exist.
    # Actually, for a pair (u, v), there can be multiple 4-APs containing them.
    # Example: 1, 2, 3, 4. Pair (1, 2) is in 1,2,3,4. 
    # Is (1, 2) in any other 4-AP? 
    # If d=1, 1,2,3,4. 
    # If d is different? 
    # 1, 1+d, 1+2d, 1+3d. If 2=1+d -> d=1.
    # If 2=1+2d -> d=0.5 (not integer).
    # If 2=1+3d -> d=1/3.
    # So for a specific pair (u, v), the possible d's are limited.
    
    # Let's precompute forbidden_pairs[x] = list of (y1, y2) such that 
    # {x, y1, y2} is a subset of a 4-AP.
    # Then adding x is valid iff no (y1, y2) in forbidden_pairs[x] has both 
    # y1 and y2 in S.
    
    # To make this faster, we can precompute for each x, a list of masks?
    # No, let's just use the bitset M[x][y].
    # M is a 513x513 array of Python ints.
    
    M = [[0] * N for _ in range(N)]
    
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            t1 = a
            t2 = a + d
            t3 = a + 2 * d
            t4 = a + 3 * d
            if t4 > n:
                break
            # The 4-AP is {t1, t2, t3, t4}
            # Pairs:
            pairs = [(t1, t2), (t1, t3), (t1, t4), (t2, t3), (t2, t4), (t3, t4)]
            for u, v in pairs:
                # The other two terms are the ones in {t1,t2,t3,t4} \ {u,v}
                others = [t for t in [t1, t2, t3, t4] if t != u and t != v]
                # If we have u and v in set, we cannot have any of 'others' in set.
                # So M[u][v] should have bits set for all 'others'.
                # And M[v][u] should have bits set for all 'others'.
                mask = 0
                for o in others:
                    mask |= (1 << o)
                M[u][v] |= mask
                M[v][u] |= mask
                
    # 2. Greedy Construction using Bitsets
    # We maintain a bitset `S_bits` representing the current set.
    # To check if x can be added:
    # For each y in S, check if (M[x][y] & S_bits) != 0.
    # If any is non-zero, x cannot be added.
    
    def can_add(x, S_bits, S_list):
        # S_list is a list of elements in S for faster iteration?
        # Iterating over bits in S_bits might be faster if sparse.
        # But S is dense-ish. Let's iterate over S_list.
        for y in S_list:
            if M[x][y] & S_bits:
                return False
        return True
        
    # Initial Greedy
    S_bits = 0
    S_list = []
    # Randomize order for diversity? Or just 1..n.
    # Let's try multiple greedy seeds with different random orders.
    
    best_S_bits = 0
    best_S_list = []
    
    def run_greedy(order):
        s_bits = 0
        s_list = []
        for x in order:
            if can_add(x, s_bits, s_list):
                s_list.append(x)
                s_bits |= (1 << x)
        return s_bits, s_list
        
    # Try a few random orders
    random.seed(42)
    for i in range(5):
        order = list(range(1, n + 1))
        random.shuffle(order)
        s_bits, s_list = run_greedy(order)
        if len(s_list) > len(best_S_list):
            best_S_list = s_list
            best_S_bits = s_bits
            
    # 3. Simulated Annealing
    # State: S_bits, S_list
    # Move: Remove x, Add y (or Add y, Remove x, etc.)
    # We want to maximize size.
    # Since adding is harder than removing, we focus on swaps or additions.
    
    current_bits = best_S_bits
    current_list = best_S_list[:]
    current_size = len(current_list)
    
    best_bits = current_bits
    best_list = current_list[:]
    best_size = current_size
    
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    
    # Precompute list of unselected elements
    # This changes as we swap.
    
    # To speed up "can_add", we can use the bitset M.
    # The check `if M[x][y] & S_bits` is very fast in Python.
    
    # Let's implement a simple local search:
    # 1. Try to add a random unselected element.
    # 2. If fails, try to swap: remove a random selected, add a random unselected.
    # 3. If that fails, try to remove a random selected (to allow future additions? No, that reduces size).
    #    Actually, we should only accept moves that maintain or increase size.
    #    Standard SA accepts worse moves (size - 1) to escape local optima.
    
    # Let's track size explicitly.
    
    def get_size(bits):
        return bin(bits).count('1')
        
    # But we maintain S_list, so len(S_list) is O(1).
    
    # We need to be careful with S_list updates.
    # Removing x from S_list: O(N). 
    # To make it O(1), we can use a set for membership and a list for iteration?
    # Or just use the bitset for membership and iterate over bits?
    # Iterating over bits of a 512-bit int is fast.
    
    def bits_to_list(bits):
        return [i for i in range(1, n + 1) if (bits >> i) & 1]
        
    # Let's switch to purely bitset-based state to avoid list sync issues.
    # State: bits
    # Size: bin(bits).count('1') -> slow. 
    # Let's maintain size as a variable.
    
    curr_bits = best_bits
    curr_size = best_size
    
    best_bits = curr_bits
    best_size = curr_size
    
    # For "unselected", we can generate a mask of all bits 1..n and XOR with curr_bits.
    ALL_BITS = 0
    for i in range(1, n + 1):
        ALL_BITS |= (1 << i)
        
    # Helper to get a random bit from a mask
    def random_bit(mask):
        # Find set bits
        bits = []
        m = mask
        while m:
            lsb = m & (-m)
            bits.append(lsb)
            m ^= lsb
        if not bits:
            return None
        return random.choice(bits)
        
    # This is O(N) per call. Might be slow if called many times.
    # Alternative: Precompute list of all elements, filter by mask.
    # Or use numpy?
    
    # Let's use a simpler approach: 
    # Since N=512, we can just iterate.
    
    def try_add(bits, size, x):
        # Check if x can be added to bits
        # x is not in bits
        for y in range(1, n + 1):
            if (bits >> y) & 1:
                if M[x][y] & bits:
                    return False
        return True
        
    # This is O(N) per check.
    # In SA, we do many checks. 
    # 90 seconds / 0.001s per check = 90,000 checks.
    # N=512. 90,000 * 512 = 45M operations. Python can do ~10-50M simple ops/sec.
    # So this is borderline. We need to optimize the inner loop.
    
    # Optimization: Precompute for each x, the list of y such that M[x][y] != 0.
    # Let's call this conf_neighbors[x].
    conf_neighbors = [[] for _ in range(N)]
    for x in range(1, n + 1):
        for y in range(1, n + 1):
            if M[x][y] != 0:
                conf_neighbors[x].append(y)
                
    def try_add_fast(bits, x):
        # Only check y in conf_neighbors[x] that are in bits
        for y in conf_neighbors[x]:
            if (bits >> y) & 1:
                if M[x][y] & bits:
                    return False
        return True
        
    # This reduces the inner loop significantly if conf_neighbors is sparse.
    # How sparse is it?
    # For a given x, how many y conflict with it?
    # x is in ~N/4 4-APs as first term, ~N/4 as second, etc.
    # Total 4-APs containing x is O(N).
    # Each 4-AP has 3 other terms.
    # So conf_neighbors[x] size is O(N).
    # So it's not much faster than O(N).
    
    # Let's stick with the bitset check but optimize the Python loop.
    # Using `bits & mask` is fast.
    
    # Let's try a different move generation:
    # Instead of random x, try all x? No, too slow.
    
    # Let's use the current program's local search but with the fast `try_add_fast`.
    
    # Re-initialize SA state
    curr_bits = best_bits
    curr_size = best_size
    
    # We need a way to quickly find unselected elements.
    # We can maintain a list of unselected elements.
    # When we add x, remove from unselected.
    # When we remove x, add to unselected.
    
    curr_unselected = [i for i in range(1, n + 1) if not ((curr_bits >> i) & 1)]
    
    T = 5.0
    T_min = 0.01
    alpha = 0.9995
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate a move
        # 1. Try Add
        if curr_unselected:
            x = random.choice(curr_unselected)
            if try_add_fast(curr_bits, x):
                # Accept
                new_bits = curr_bits | (1 << x)
                new_size = curr_size + 1
                # Update state
                curr_bits = new_bits
                curr_size = new_size
                curr_unselected.remove(x)
                if curr_size > best_size:
                    best_size = curr_size
                    best_bits = curr_bits
                T *= alpha
                continue
                
        # 2. Try Swap (Remove y, Add x)
        # Pick random y from selected, random x from unselected
        if curr_unselected and curr_size > 0:
            y = random_bit(curr_bits)
            if y:
                y_val = y.bit_length() - 1
                x = random.choice(curr_unselected)
                # Remove y
                temp_bits = curr_bits & ~(1 << y_val)
                # Try add x to temp_bits
                if try_add_fast(temp_bits, x):
                    new_bits = temp_bits | (1 << x)
                    new_size = curr_size # Size unchanged
                    # Accept if size same? 
                    # In SA, we might accept same size to change configuration.
                    # But we want to maximize size. 
                    # If we accept same size, we might escape local optima.
                    # Let's accept with prob 1 for now, or use SA.
                    # Since size is same, delta E = 0. Accept with prob 1.
                    curr_bits = new_bits
                    curr_unselected.remove(x)
                    curr_unselected.append(y_val)
                    T *= alpha
                    continue
                    
        # 3. Try Remove (to allow future adds? Risky)
        # Or Try Double Swap (Remove y1, y2, Add x1, x2) -> Size +0
        # Or Remove y, Add x1, x2 -> Size +1
        if curr_unselected and curr_size > 0 and len(curr_unselected) >= 2:
            y = random_bit(curr_bits)
            if y:
                y_val = y.bit_length() - 1
                temp_bits = curr_bits & ~(1 << y_val)
                # Pick two from unselected
                # Use random.sample
                candidates = random.sample(curr_unselected, 2)
                x1, x2 = candidates
                # Check if x1 and x2 can be added to temp_bits
                if try_add_fast(temp_bits, x1) and try_add_fast(temp_bits | (1 << x1), x2):
                    new_bits = temp_bits | (1 << x1) | (1 << x2)
                    new_size = curr_size + 1
                    curr_bits = new_bits
                    curr_size = new_size
                    curr_unselected.remove(x1)
                    curr_unselected.remove(x2)
                    curr_unselected.append(y_val)
                    if curr_size > best_size:
                        best_size = curr_size
                        best_bits = curr_bits
                    T *= alpha
                    continue
                    
        # If no move accepted, cool down
        T *= alpha
        
    # Return the best set found
    result = [i for i in range(1, n + 1) if (best_bits >> i) & 1]
    return sorted(result)
