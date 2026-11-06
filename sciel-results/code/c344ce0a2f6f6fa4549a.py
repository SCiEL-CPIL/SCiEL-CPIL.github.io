"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    N = 512
    
    def is_4ap_free(selected_bits):
        """Check if the bitset represents a 4-AP-free set."""
        # For each x in the set, check all possible 4-APs containing x
        x = 0
        while x < N:
            if (selected_bits >> x) & 1:
                # x corresponds to number x+1
                val = x + 1
                # Check if x is the first term: need x+1+d, x+1+2d, x+1+3d all in set
                d = 1
                while val + 3*d <= N:
                    b1 = (selected_bits >> (val + d - 1)) & 1
                    b2 = (selected_bits >> (val + 2*d - 1)) & 1
                    b3 = (selected_bits >> (val + 3*d - 1)) & 1
                    if b1 and b2 and b3:
                        return False
                    d += 1
                # Check if x is the second term: need x-d, x+d, x+2d
                d = 1
                while val - d >= 1 and val + 2*d <= N:
                    b1 = (selected_bits >> (val - d - 1)) & 1
                    b2 = (selected_bits >> (val + d - 1)) & 1
                    b3 = (selected_bits >> (val + 2*d - 1)) & 1
                    if b1 and b2 and b3:
                        return False
                    d += 1
                # Check if x is the third term: need x-2d, x-d, x+d
                d = 1
                while val - 2*d >= 1 and val + d <= N:
                    b1 = (selected_bits >> (val - 2*d - 1)) & 1
                    b2 = (selected_bits >> (val - d - 1)) & 1
                    b3 = (selected_bits >> (val + d - 1)) & 1
                    if b1 and b2 and b3:
                        return False
                    d += 1
                # Check if x is the fourth term: need x-3d, x-2d, x-d
                d = 1
                while val - 3*d >= 1:
                    b1 = (selected_bits >> (val - 3*d - 1)) & 1
                    b2 = (selected_bits >> (val - 2*d - 1)) & 1
                    b3 = (selected_bits >> (val - d - 1)) & 1
                    if b1 and b2 and b3:
                        return False
                    d += 1
            x += 1
        return True
    
    def can_add(selected_bits, x):
        """Check if adding element x (1-indexed) to selected_bits keeps it 4-AP-free."""
        val = x
        # Check if x is the first term
        d = 1
        while val + 3*d <= N:
            b1 = (selected_bits >> (val + d - 1)) & 1
            b2 = (selected_bits >> (val + 2*d - 1)) & 1
            b3 = (selected_bits >> (val + 3*d - 1)) & 1
            if b1 and b2 and b3:
                return False
            d += 1
        # Check if x is the second term
        d = 1
        while val - d >= 1 and val + 2*d <= N:
            b1 = (selected_bits >> (val - d - 1)) & 1
            b2 = (selected_bits >> (val + d - 1)) & 1
            b3 = (selected_bits >> (val + 2*d - 1)) & 1
            if b1 and b2 and b3:
                return False
            d += 1
        # Check if x is the third term
        d = 1
        while val - 2*d >= 1 and val + d <= N:
            b1 = (selected_bits >> (val - 2*d - 1)) & 1
            b2 = (selected_bits >> (val - d - 1)) & 1
            b3 = (selected_bits >> (val + d - 1)) & 1
            if b1 and b2 and b3:
                return False
            d += 1
        # Check if x is the fourth term
        d = 1
        while val - 3*d >= 1:
            b1 = (selected_bits >> (val - 3*d - 1)) & 1
            b2 = (selected_bits >> (val - 2*d - 1)) & 1
            b3 = (selected_bits >> (val - d - 1)) & 1
            if b1 and b2 and b3:
                return False
            d += 1
        return True
    
    def greedy_forward():
        """Greedy forward scan."""
        selected_bits = 0
        for x in range(1, N+1):
            if can_add(selected_bits, x):
                selected_bits |= (1 << (x-1))
        return selected_bits
    
    def greedy_random_order():
        """Greedy with random order."""
        selected_bits = 0
        order = list(range(1, N+1))
        random.shuffle(order)
        for x in order:
            if can_add(selected_bits, x):
                selected_bits |= (1 << (x-1))
        return selected_bits
    
    def popcount(bits):
        """Count bits set."""
        count = 0
        while bits:
            count += 1
            bits &= bits - 1
        return count
    
    def local_search(selected_bits, max_iter=5000):
        """Local search: try adding elements, and if stuck, try removing and re-adding."""
        best = selected_bits
        best_size = popcount(best)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try to add any element that can be added
            added = False
            # Try random order for diversity
            order = list(range(1, N+1))
            random.shuffle(order)
            for x in order:
                if not ((best >> (x-1)) & 1):
                    if can_add(best, x):
                        new_bits = best | (1 << (x-1))
                        new_size = best_size + 1
                        if new_size > best_size:
                            best = new_bits
                            best_size = new_size
                            added = True
                            break
            
            if not added:
                # Try a swap: remove one element, then try to add two
                # Or just try removing a random element and see if we can add more
                if best_size > 0:
                    # Pick a random element to remove
                    remove_candidates = []
                    x = 0
                    while x < N:
                        if (best >> x) & 1:
                            remove_candidates.append(x+1)
                        x += 1
                    
                    if remove_candidates:
                        rm = random.choice(remove_candidates)
                        new_bits = best & ~(1 << (rm-1))
                        new_size = best_size - 1
                        
                        # Try to add elements to the reduced set
                        for x in range(1, N+1):
                            if not ((new_bits >> (x-1)) & 1):
                                if can_add(new_bits, x):
                                    new_bits |= (1 << (x-1))
                                    new_size += 1
                        
                        if new_size > best_size:
                            best = new_bits
                            best_size = new_size
                            added = True
        
        return best, best_size
    
    # Start with greedy forward
    best_bits = greedy_forward()
    best_size = popcount(best_bits)
    
    # Try multiple random restarts with local search
    while time.time() < deadline:
        # Random restart
        bits = greedy_random_order()
        size = popcount(bits)
        
        if size >= best_size:
            bits, size = local_search(bits, max_iter=2000)
            if size > best_size:
                best_bits = bits
                best_size = size
        
        # Also try local search on the current best
        if time.time() < deadline - 5:
            bits2, size2 = local_search(best_bits, max_iter=2000)
            if size2 > best_size:
                best_bits = bits2
                best_size = size2
    
    # Convert bitset to sorted list
    result = []
    for x in range(1, N+1):
        if (best_bits >> (x-1)) & 1:
            result.append(x)
    
    return sorted(result)
# EVOLVE-BLOCK-END
