"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    def is_valid_addition(x, selected_list):
        """
        Check if x can be added to selected_list (list of ints)
        without creating a triple {a, b, x} with gcd(a,b)==gcd(b,x)==gcd(c,x).
        """
        n = len(selected_list)
        for i in range(n):
            a = selected_list[i]
            g_ax = gcd_table[a][x]
            # Optimization: if g_ax is 1, it can never equal gcd(b,x) unless gcd(b,x) is 1,
            # but if gcd(a,b) is 1, then gcd(a,x)=1 and gcd(b,x)=1 is allowed?
            # Wait. Condition: gcd(a,b) == gcd(a,x) == gcd(b,x).
            # If gcd(a,b) = 1, then we need gcd(a,x)=1 and gcd(b,x)=1.
            # This is allowed. The problem is when they are all EQUAL.
            # So if g_ax != gcd_table[a][b], we can skip this pair?
            # No, we need gcd(a,x) == gcd(b,x) == gcd(a,b).
            
            for j in range(i + 1, n):
                b = selected_list[j]
                g_ab = gcd_table[a][b]
                if g_ab == g_ax:
                    g_bx = gcd_table[b][x]
                    if g_bx == g_ab:
                        return False
        return True

    def mask_to_list(mask):
        return [i + 1 for i in range(N) if (mask >> i) & 1]

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << (x - 1))
        return m

    # Greedy Construction
    def greedy_construct(order):
        selected = []
        for x in order:
            if is_valid_addition(x, selected):
                selected.append(x)
        return selected

    best_size = 0
    best_mask = 0
    best_list = []

    # Run multiple restarts with different strategies
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # 1. Generate initial set
        if r % 3 == 0:
            order = elements[:]
            random.shuffle(order)
        elif r % 3 == 1:
            # Primes first
            primes = [x for x in range(2, N + 1) if all(x % p != 0 for p in range(2, int(x**0.5) + 1))]
            composites = [x for x in range(1, N + 1) if x not in primes]
            order = primes + composites
            random.shuffle(order)
        else:
            # Reverse order (larger numbers first)
            order = elements[::-1]
            random.shuffle(order)
            
        current_list = greedy_construct(order)
        current_mask = list_to_mask(current_list)
        current_size = len(current_list)
        
        # 2. Simulated Annealing / Local Search
        # We will use a simpler "Random Walk" with acceptance of neutral/improving moves
        # and occasional backtracking.
        
        improved = True
        local_deadline = min(start_time + 0.5 * (r+1), deadline) # Dynamic budget
        
        while time.time() < local_deadline:
            # Try to add
            added = False
            candidates = [x for x in range(1, N+1) if not (current_mask >> (x-1)) & 1]
            if candidates:
                random.shuffle(candidates)
                for x in candidates:
                    if is_valid_addition(x, current_list):
                        current_list.append(x)
                        current_mask |= (1 << (x - 1))
                        current_size += 1
                        added = True
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
                            best_list = current_list[:]
                        break
                        
            if not added:
                # Try 1-1 Swap
                if current_list:
                    # Pick a random element to remove and a random one to add
                    # This is O(N) to generate candidates, O(|S|^2) to check
                    # We limit swaps per restart to avoid infinite loops
                    for _ in range(10):
                        if time.time() > local_deadline: break
                        x_out = random.choice(current_list)
                        x_in_candidates = [x for x in range(1, N+1) if x != x_out and not (current_mask >> (x-1)) & 1]
                        if not x_in_candidates: continue
                        x_in = random.choice(x_in_candidates)
                        
                        # Check if swap is valid
                        new_list = [z for z in current_list if z != x_out]
                        if is_valid_addition(x_in, new_list):
                            # Accept swap (size remains same)
                            # This helps escape local minima where no single add is possible
                            current_list = new_list
                            current_list.append(x_in)
                            current_mask = list_to_mask(current_list)
                            # size unchanged
                            break
        
        # Final check for this restart
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask
            best_list = current_list[:]

    # Final polish: try to add any remaining
    for x in range(1, N + 1):
        if not (best_mask >> (x - 1)) & 1:
            if is_valid_addition(x, best_list):
                best_list.append(x)
                best_mask |= (1 << (x - 1))
                best_size += 1

    return best_list

# EVOLVE-BLOCK-END
