import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256
    
    # Precompute conflict pairs as bitmasks
    # conflict_pairs[a] is a list of bitmasks M such that if S contains both bits in M, 
    # and a is in S, then there is a conflict.
    # Specifically, M has bits b and c set.
    conflict_pairs = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            k_min = (2 * b + a - 1) // a
            k_max = (n + b) // a
            for k in range(k_min, k_max + 1):
                c = k * a - b
                if b <= c <= n:
                    # Create bitmask for pair (b, c)
                    mask = (1 << b) | (1 << c)
                    conflict_pairs[a].append(mask)

    def is_valid(mask):
        """
        Check if the set represented by mask is valid.
        """
        m = mask
        while m:
            lsb = m & -m
            a = lsb.bit_length() - 1
            # Check all conflict pairs for a
            for pair_mask in conflict_pairs[a]:
                if (mask & pair_mask) == pair_mask:
                    return False
            m ^= lsb
        return True

    def get_conflict(mask, x):
        """
        If mask | (1<<x) is invalid, find one conflicting triple.
        Returns (a, b, c) or None.
        """
        new_mask = mask | (1 << x)
        
        # We need to find a, b, c in new_mask such that a < b <= c and a | b+c
        # We can iterate through a in new_mask
        m = new_mask
        while m:
            lsb_a = m & -m
            a = lsb_a.bit_length() - 1
            
            if a == 1:
                # 1 divides everything.
                # If 1 is in set, we can have at most one other element?
                # No, 1 divides b+c for any b,c.
                # So if 1 is in S, we cannot have any pair (b,c) with 1 < b <= c.
                # This implies if 1 is in S, S can have size at most 2?
                # Wait, if S = {1, 2}, b=2, c must be >=2. If c=2, 1 | 4. Conflict.
                # So if 1 is in S, no other pair can exist.
                # Actually, if 1 is in S, then for any b,c in S with 1 < b <= c, 1 divides b+c.
                # So S can contain 1 and at most one other element.
                if (new_mask & ~1) != 0:
                    # Count bits in new_mask excluding 1
                    rest = new_mask & ~1
                    if rest != 0 and (rest & (rest - 1)) != 0:
                         # More than 1 element in rest
                         # Find two elements b, c
                         b_mask = rest & -rest
                         b = b_mask.bit_length() - 1
                         rest2 = rest ^ b_mask
                         c_mask = rest2 & -rest2
                         c = c_mask.bit_length() - 1
                         return (1, b, c)
                    elif rest != 0:
                         # Exactly one element in rest?
                         # Then no pair b,c with b < c.
                         # But b=c is allowed? "a < b <= c".
                         # If S={1, 5}, b=5, c=5. 1 | 10. Conflict.
                         # So even one other element causes conflict if we can pick b=c.
                         # Yes, b=c is included.
                         b_mask = rest & -rest
                         b = b_mask.bit_length() - 1
                         return (1, b, b)
            
            # Check conflicts for a
            for pair_mask in conflict_pairs[a]:
                if (new_mask & pair_mask) == pair_mask:
                    # Extract b and c from pair_mask
                    # pair_mask has exactly 2 bits set
                    b_mask = pair_mask & -pair_mask
                    b = b_mask.bit_length() - 1
                    c_mask = pair_mask ^ b_mask
                    c = c_mask.bit_length() - 1
                    return (a, b, c)
            
            m ^= lsb_a
        
        return None

    # Step 1: Find best starting interval
    best_start_mask = 0
    best_start_size = 0
    
    for L in range(1, n + 1, 2):
        if time.time() > deadline - 15:
            break
        # Create mask for [L, n]
        S_test_mask = 0
        for x in range(L, n + 1):
            S_test_mask |= (1 << x)
        
        if is_valid(S_test_mask):
            S_greedy_mask = S_test_mask
            # Try to add small elements
            for x in range(1, L):
                if time.time() > deadline - 15:
                    break
                if S_greedy_mask & (1 << x):
                    continue
                test_mask = S_greedy_mask | (1 << x)
                if is_valid(test_mask):
                    S_greedy_mask = test_mask
            
            size = bin(S_greedy_mask).count('1')
            if size > best_start_size:
                best_start_size = size
                best_start_mask = S_greedy_mask

    if best_start_mask == 0:
        best_start_mask = 1 << n
        best_start_size = 1

    best_mask = best_start_mask
    best_size = best_start_size

    # Step 2: Local Search with Swaps
    # We will iterate through all pairs (x not in S, y in S) and try swap
    # To speed up, we can randomize the order of x and y
    
    def local_search(mask, time_limit):
        current_mask = mask
        current_size = bin(current_mask).count('1')
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Get lists of elements in and out
            in_set = []
            out_set = []
            for i in range(1, n + 1):
                if current_mask & (1 << i):
                    in_set.append(i)
                else:
                    out_set.append(i)
            
            random.shuffle(in_set)
            random.shuffle(out_set)
            
            for x in out_set:
                if time.time() > time_limit:
                    break
                for y in in_set:
                    if time.time() > time_limit:
                        break
                    # Try swap y -> x
                    new_mask = current_mask ^ (1 << y) | (1 << x)
                    if is_valid(new_mask):
                        current_mask = new_mask
                        current_size += 0 # Size doesn't change in swap
                        improved = True
                        # Restart loop to find more improvements
                        break
                if improved:
                    break
        
        # Try adding elements
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            out_set = [i for i in range(1, n + 1) if not (current_mask & (1 << i))]
            random.shuffle(out_set)
            for x in out_set:
                if time.time() > time_limit:
                    break
                new_mask = current_mask | (1 << x)
                if is_valid(new_mask):
                    current_mask = new_mask
                    current_size += 1
                    improved = True
                    break
        
        return current_mask, current_size

    # Run multiple local searches from different starts
    # Start from best_start_mask
    # Also try random starts
    
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        if attempt == 0:
            start_mask = best_start_mask
        else:
            # Random start
            start_mask = 0
            # Pick a random subset of size ~100
            # Or just random elements
            for i in range(1, n + 1):
                if random.random() < 0.5:
                    start_mask |= (1 << i)
            # Ensure it's valid? No, local search will handle it?
            # No, local search assumes valid start.
            # So we must start with a valid set.
            # Let's use a greedy valid set.
            start_mask = 0
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
            for x in candidates:
                if time.time() > deadline - 5:
                    break
                test_mask = start_mask | (1 << x)
                if is_valid(test_mask):
                    start_mask = test_mask
        
        mask, size = local_search(start_mask, deadline - 5)
        
        if size > best_size:
            best_size = size
            best_mask = mask

    # Convert mask to list
    result = []
    for i in range(1, n + 1):
        if best_mask & (1 << i):
            result.append(i)
    
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
