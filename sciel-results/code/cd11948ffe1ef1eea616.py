import time
import random

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s kill
    
    # Precompute all 4-term APs
    # For each x, precompute which other elements conflict with it
    # An element x conflicts with y if there exists a 4-AP containing both x and y
    
    # Actually, let's think differently:
    # We want to check if adding x to S creates a 4-AP.
    # A 4-AP is (a, a+d, a+2d, a+3d).
    # When adding x, it could be any of the 4 positions.
    # For x to be position i, we need the other 3 elements in S.
    
    # Let's precompute for each x, the list of "forbidden triples" 
    # - If x is 1st term: need (x+d, x+2d, x+3d) all in S
    # - If x is 2nd term: need (x-d, x+d, x+2d) all in S  
    # - If x is 3rd term: need (x-2d, x-d, x+d) all in S
    # - If x is 4th term: need (x-3d, x-2d, x-d) all in S
    
    # For efficiency, use a bitset for S
    # S_bits: integer where bit i is set if (i+1) is in S
    
    def can_add_bitset(S_bits, x):
        # x is 1-indexed
        # x is 1st term: x, x+d, x+2d, x+3d
        # x + 3d <= N => d <= (N-x)//3
        for d in range(1, (N - x) // 3 + 1):
            a, b, c = x + d, x + 2*d, x + 3*d
            if b > N or c > N:
                break
            # Check if a, b, c all in S
            if (S_bits >> a) & 1 and (S_bits >> b) & 1 and (S_bits >> c) & 1:
                return False
        
        # x is 2nd term: x-d, x, x+d, x+2d
        # x-d >= 1 => d <= x-1
        # x+2d <= N => d <= (N-x)//2
        max_d2 = min(x - 1, (N - x) // 2)
        for d in range(1, max_d2 + 1):
            a, b, c = x - d, x + d, x + 2*d
            if a < 1 or c > N:
                break
            if (S_bits >> a) & 1 and (S_bits >> b) & 1 and (S_bits >> c) & 1:
                return False
        
        # x is 3rd term: x-2d, x-d, x, x+d
        # x-2d >= 1 => d <= (x-1)//2
        # x+d <= N => d <= N-x
        max_d3 = min((x - 1) // 2, N - x)
        for d in range(1, max_d3 + 1):
            a, b, c = x - 2*d, x - d, x + d
            if a < 1 or c > N:
                break
            if (S_bits >> a) & 1 and (S_bits >> b) & 1 and (S_bits >> c) & 1:
                return False
        
        # x is 4th term: x-3d, x-2d, x-d, x
        # x-3d >= 1 => d <= (x-1)//3
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            a, b, c = x - 3*d, x - 2*d, x - d
            if a < 1:
                break
            if (S_bits >> a) & 1 and (S_bits >> b) & 1 and (S_bits >> c) & 1:
                return False
        
        return True

    best_bits = 0
    best_size = 0
    
    # Start with a good base: base-3 construction
    # Numbers with only 0s and 1s in base-3 representation
    # For N=512, max base-3 is 3^6 = 729, so 6 digits
    base3_set = set()
    for digits in range(1 << 6):  # 6 binary digits
        num = 0
        for i in range(6):
            if digits & (1 << i):
                num += 3 ** i
        if 1 <= num <= N:
            base3_set.add(num)
    
    base_bits = 0
    for x in base3_set:
        base_bits |= (1 << x)
    
    best_bits = base_bits
    best_size = len(base3_set)
    
    # Local search: try to improve
    def local_search(bits, size, start_time, time_limit):
        S = set()
        for x in range(1, N + 1):
            if (bits >> x) & 1:
                S.add(x)
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            S_list = list(S)
            random.shuffle(S_list)
            
            for remove_x in S_list:
                if time.time() - start_time >= time_limit:
                    break
                
                # Try removing remove_x
                S.remove(remove_x)
                new_size = len(S)
                
                # Try to add back elements
                added_any = False
                candidates = list(range(1, N + 1))
                random.shuffle(candidates)
                
                temp_bits = 0
                for x in S:
                    temp_bits |= (1 << x)
                
                for y in candidates:
                    if y in S:
                        continue
                    if can_add_bitset(temp_bits, y):
                        S.add(y)
                        temp_bits |= (1 << y)
                        new_size += 1
                        added_any = True
                
                if new_size > size:
                    size = new_size
                    improved = True
                    # Update bits
                    bits = 0
                    for x in S:
                        bits |= (1 << x)
                else:
                    # Restore
                    S.add(remove_x)
                    if added_any:
                        # Need to undo additions
                        # This is getting complex, let's simplify
                        pass
        
        return bits, size

    # Multiple random greedy restarts
    for trial in range(200):
        if time.time() - start_time >= time_limit:
            break
        
        # Random order greedy
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        S_bits = 0
        S_size = 0
        S = set()
        
        for x in candidates:
            if can_add_bitset(S_bits, x):
                S.add(x)
                S_bits |= (1 << x)
                S_size += 1
        
        if S_size > best_size:
            best_size = S_size
            best_bits = S_bits
        
        # Try local search on this result
        if S_size > best_size:
            # We already updated best, now try to improve further
            pass
        
        # Also try local search on the best so far occasionally
        if trial % 10 == 0 and best_size > 0:
            # Create set from best_bits
            S = set()
            for x in range(1, N + 1):
                if (best_bits >> x) & 1:
                    S.add(x)
            
            # Try a few local search moves
            S_list = list(S)
            random.shuffle(S_list)
            
            for remove_x in S_list[:5]:  # Try first 5
                if time.time() - start_time >= time_limit:
                    break
                
                S.remove(remove_x)
                temp_bits = 0
                for x in S:
                    temp_bits |= (1 << x)
                
                new_size = len(S)
                candidates = list(range(1, N + 1))
                random.shuffle(candidates)
                
                for y in candidates:
                    if y in S:
                        continue
                    if can_add_bitset(temp_bits, y):
                        S.add(y)
                        temp_bits |= (1 << y)
                        new_size += 1
                
                if new_size > best_size:
                    best_size = new_size
                    best_bits = temp_bits
                else:
                    S.add(remove_x)
    
    # Convert best_bits to sorted list
    result = []
    for x in range(1, N + 1):
        if (best_bits >> x) & 1:
            result.append(x)
    
    return sorted(result)
