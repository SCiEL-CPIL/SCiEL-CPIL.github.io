import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin before 90s kill
    
    n = 512
    N = n + 1  # 1-indexed to 512
    
    # 1. Precompute conflict bitsets
    # conflict[x] is a Python int where bit y is 1 if x and y are in the same 4-AP.
    # Actually, we need: if x is in S, and y is in S, and there exists z in S 
    # such that {x,y,z} is part of a 4-AP, it's invalid.
    # More precisely: A set S is valid iff for all distinct x, y, z, w in S, 
    # they do not form a 4-AP.
    # 
    # Efficient check for adding x to S:
    # For every y in S, if there exists z in S such that {x, y, z} is a subset of a 4-AP, 
    # then adding x is invalid.
    # Let's precompute for each pair (x, y) a bitset `forbidden[x][y]` containing all z 
    # such that {x, y, z} is a subset of some 4-AP.
    # Then x can be added to S iff for all y in S, (forbidden[x][y] & S_bits) == 0.
    
    # To make this faster, we can precompute `conflict_with_x[x]` as a list of (y, mask) pairs?
    # No, let's stick to the matrix `M[x][y]` = bitset of z's.
    # M is a 513x513 array of Python ints.
    
    M = [[0] * N for _ in range(N)]
    
    # Generate all 4-APs
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            t1 = a
            t2 = a + d
            t3 = a + 2 * d
            t4 = a + 3 * d
            
            # The 4-AP is {t1, t2, t3, t4}
            # Any pair of these elements conflicts with the other two.
            # If we have u, v in S, we cannot have w in S if {u, v, w} is a subset of a 4-AP.
            # Specifically, if u, v are in the AP, the other two terms are forbidden.
            
            terms = [t1, t2, t3, t4]
            # There are 6 pairs. For each pair, the other 2 terms are forbidden.
            for i in range(4):
                for j in range(i+1, 4):
                    u = terms[i]
                    v = terms[j]
                    # Forbidden terms are the other two
                    others = [t for t in terms if t != u and t != v]
                    mask = 0
                    for o in others:
                        mask |= (1 << o)
                    
                    # If u and v are in S, then none of 'others' can be in S.
                    # So M[u][v] |= mask and M[v][u] |= mask
                    M[u][v] |= mask
                    M[v][u] |= mask

    # 2. Initial Greedy Construction
    # We use a bitset `S_bits` to represent the set.
    # We try to add elements in a random order.
    
    def try_add(x, S_bits):
        """Check if x can be added to S_bits."""
        # For each y in S_bits, check if M[x][y] & S_bits is non-zero.
        # To speed this up, we iterate over set bits in S_bits.
        # But S_bits is a Python int. Iterating over bits is O(popcount).
        # Alternatively, we can precompute a list of neighbors for x.
        
        # Let's use a precomputed list of neighbors for x to avoid checking all 512 y's.
        # neighbors[x] = list of y such that M[x][y] != 0
        # We can compute this once.
        
        # Actually, let's just iterate over the bits of S_bits.
        # In Python, `while m: lsb = m & -m; m ^= lsb; y = lsb.bit_length()-1` is fast enough for 512 bits.
        
        m = S_bits
        while m:
            lsb = m & (-m)
            y = lsb.bit_length() - 1
            if M[x][y] & S_bits:
                return False
            m ^= lsb
        return True

    # Precompute neighbors for faster iteration? 
    # M[x][y] is non-zero for O(N) y's. 
    # So iterating over S_bits (size ~N/4) and checking M[x][y] is O(N).
    # This is acceptable.
    
    # Greedy seed
    random.seed(42)
    S_bits = 0
    S_list = []
    
    # Try a few random greedy orders to get a good start
    best_start_bits = 0
    best_start_size = 0
    
    for _ in range(10):
        order = list(range(1, n + 1))
        random.shuffle(order)
        s_bits = 0
        s_size = 0
        for x in order:
            if try_add(x, s_bits):
                s_bits |= (1 << x)
                s_size += 1
        if s_size > best_start_size:
            best_start_size = s_size
            best_start_bits = s_bits
    
    curr_bits = best_start_bits
    curr_size = best_start_size
    best_bits = curr_bits
    best_size = curr_size
    
    # 3. Local Search
    # We perform a series of moves:
    # - Add: Try to add a random element not in S.
    # - Swap: Remove a random element from S, try to add a random element not in S.
    # - Double Swap: Remove two, add two.
    #
    # To speed up "Add", we can precompute for each x, the list of y in S that block x.
    # If we remove y, x might become addable.
    
    # Let's maintain a list of elements in S for easy removal.
    curr_set = set([i for i in range(1, n+1) if (curr_bits >> i) & 1])
    
    # We will use a simple hill-climbing with random restarts/swaps.
    # Since adding is hard, we focus on swaps that maintain size but change configuration,
    # or "remove 1, add 2" moves that increase size.
    
    # Optimization: Precompute `blockers[x]` for current S?
    # blockers[x] = set of y in S such that M[x][y] & S_bits != 0.
    # If blockers[x] is empty, x can be added.
    # If we remove y from S, we only need to re-check x if y was in blockers[x].
    
    # Let's implement a move generator that uses this insight.
    
    def get_blockers(x, S_bits, S_set):
        """Return set of y in S such that adding x would create a 4-AP."""
        blockers = set()
        m = S_bits
        while m:
            lsb = m & (-m)
            y = lsb.bit_length() - 1
            if M[x][y] & S_bits:
                blockers.add(y)
            m ^= lsb
        return blockers

    # To make this fast, we'll just re-evaluate candidates.
    # We'll keep a list of "promising" candidates.
    
    # List of all elements
    all_elements = list(range(1, n + 1))
    
    # We'll do a loop of random swaps/adds.
    # To avoid O(N) check for every random x, we'll sample x and check.
    
    # We'll track time and break if deadline is close.
    
    while time.time() < deadline:
        # 1. Try to add a random element
        # Sample a few random elements not in S
        unselected = [x for x in all_elements if x not in curr_set]
        if unselected:
            # Try a few random additions
            for _ in range(10):
                if not unselected: break
                x = random.choice(unselected)
                if try_add(x, curr_bits):
                    curr_bits |= (1 << x)
                    curr_set.add(x)
                    curr_size += 1
                    unselected.remove(x)
                    if curr_size > best_size:
                        best_size = curr_size
                        best_bits = curr_bits
                        best_set = set(curr_set)
                    break
        
        # 2. Try swaps: Remove y, Add x
        # Pick random y in S, random x not in S.
        if curr_set and unselected:
            # Try a few random swaps
            for _ in range(50):
                y = random.choice(list(curr_set))
                x = random.choice(unselected)
                
                # Temporarily remove y
                temp_bits = curr_bits & ~(1 << y)
                
                # Check if x can be added to temp_bits
                if try_add(x, temp_bits):
                    # Accept swap
                    curr_bits = temp_bits | (1 << x)
                    curr_set.remove(y)
                    curr_set.add(x)
                    unselected.remove(x)
                    unselected.append(y)
                    # Size unchanged, but configuration changed
                    # This helps escape local optima
                    break
        
        # 3. Try "Remove 1, Add 2" moves
        # This can increase size by 1.
        if curr_set and len(unselected) >= 2:
            # Pick random y in S
            y = random.choice(list(curr_set))
            temp_bits = curr_bits & ~(1 << y)
            
            # Try to find two x1, x2 from unselected that can be added to temp_bits
            # We can just try random pairs
            found = False
            for _ in range(20):
                if len(unselected) < 2: break
                x1, x2 = random.sample(unselected, 2)
                
                # Check if x1 can be added to temp_bits
                if try_add(x1, temp_bits):
                    temp_bits_1 = temp_bits | (1 << x1)
                    # Check if x2 can be added to temp_bits_1
                    if try_add(x2, temp_bits_1):
                        # Accept
                        curr_bits = temp_bits_1 | (1 << x2)
                        curr_set.remove(y)
                        curr_set.add(x1)
                        curr_set.add(x2)
                        
                        # Update unselected
                        unselected.remove(x1)
                        unselected.remove(x2)
                        unselected.append(y)
                        
                        curr_size += 1
                        if curr_size > best_size:
                            best_size = curr_size
                            best_bits = curr_bits
                            best_set = set(curr_set)
                        found = True
                        break
            
            if found:
                continue # Restart loop to re-evaluate additions
        
        # If we get stuck, we might need to accept a size decrease? 
        # No, the task is to maximize cardinality. We should only accept moves that 
        # maintain or increase size. If we can't increase, we rely on swaps to 
        # find a better configuration for future additions.
        
        # To avoid infinite loops of useless swaps, we can add a cooling factor 
        # or just rely on the randomness.
        
        # If we haven't improved in a long time, maybe restart with a new random seed?
        # For now, the random swaps should suffice.

    # Return the best set found
    result = [i for i in range(1, n + 1) if (best_bits >> i) & 1]
    return sorted(result)
