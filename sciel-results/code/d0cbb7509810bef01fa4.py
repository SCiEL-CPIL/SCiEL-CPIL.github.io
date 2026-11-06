import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 128
    FULL_MASK = (1 << N) - 1
    
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # For each element x, precompute:
    # conflict_pairs[x] = dict mapping (a, b) -> True if (x, a, b) is a bad triple
    # But more efficiently: for each x, and for each a, store the bitmask of b's
    # such that (x, a, b) is a bad triple.
    # Actually, let's precompute: for each x, a bitmask `bad_with[x]` where bit i is set
    # if there exists some j such that (x, i, j) forms a bad triple... no that's not right.
    
    # Better approach: For each pair (a, b), compute the set of x such that (x, a, b) is bad.
    # Then when we have a current set S, the number of bad triples involving a new element x
    # equals the number of pairs (a, b) in S such that x is in bad_set[a][b].
    
    # Precompute: for each unordered pair (a, b), a bitmask of elements x that form a bad triple with them
    bad_pair_mask = {}  # (a, b) -> bitmask of x where (x,a,b) is bad
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            g_ab = gcd_table[a][b]
            mask = 0
            for x in range(1, N+1):
                if x == a or x == b:
                    continue
                if gcd_table[x][a] == g_ab and gcd_table[x][b] == g_ab:
                    mask |= (1 << x)
            if mask:
                bad_pair_mask[(a, b)] = mask
    
    # For fast conflict counting: for each element x, precompute a list of (a, b, mask)
    # where x is in the bad triple with pair (a,b). But that's just iterating over all pairs.
    
    # Alternative: for each x, precompute conflict_count_with[S] efficiently.
    # When adding x to S: count pairs (a,b) in S such that x is in bad_pair_mask[(a,b)].
    # We can precompute for each x: a list of all pairs (a,b) where x is bad with them.
    x_bad_pairs = [[] for _ in range(N+1)]
    for (a, b), mask in bad_pair_mask.items():
        for x in range(1, N+1):
            if mask & (1 << x):
                x_bad_pairs[x].append((a, b))
    
    def count_conflicts(S_mask, x):
        """Count number of bad triples formed by adding x to set S_mask."""
        count = 0
        for (a, b) in x_bad_pairs[x]:
            if (S_mask & (1 << a)) and (S_mask & (1 << b)):
                count += 1
        return count
    
    def total_conflicts(S_mask):
        """Count total number of bad triples in set S_mask."""
        count = 0
        # Iterate over all bad triples and check if all three are in S
        # This is O(|bad_triples|) which could be large.
        # Alternative: iterate over pairs in S and check
        # For efficiency, let's just count by iterating over x_bad_pairs
        # But that double counts. Let's just iterate over all triples.
        # Actually, let's precompute all bad triples as a list
        pass
    
    # Precompute all bad triples
    bad_triples = []
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            g_ab = gcd_table[a][b]
            for c in range(b+1, N+1):
                if gcd_table[b][c] == g_ab and gcd_table[c][a] == g_ab:
                    bad_triples.append((a, b, c))
    
    def total_conflicts(S_mask):
        count = 0
        for (a, b, c) in bad_triples:
            if (S_mask & (1 << a)) and (S_mask & (1 << b)) and (S_mask & (1 << c)):
                count += 1
        return count
    
    # Simulated annealing
    # Start with a greedy solution
    def greedy_start():
        S_mask = 0
        # Try adding elements in random order
        order = list(range(1, N+1))
        random.shuffle(order)
        for x in order:
            if count_conflicts(S_mask, x) == 0:
                S_mask |= (1 << x)
        return S_mask
    
    def popcount(mask):
        return bin(mask).count('1')
    
    best_S = 0
    best_size = 0
    
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize
        S = greedy_start()
        current_conf = total_conflicts(S)
        current_size = popcount(S)
        
        if current_size > best_size:
            best_size = current_size
            best_S = S
        
        # Simulated annealing
        T = 50.0
        T_min = 0.1
        cooling = 0.995
        max_steps = 5000
        
        for step in range(max_steps):
            if time.time() > deadline - 3.0:
                break
            if T < T_min:
                break
            
            # Choose a random move
            move_type = random.random()
            active_bits = S
            inactive_bits = FULL_MASK ^ S
            
            if move_type < 0.4:
                # Add a random inactive element
                if inactive_bits == 0:
                    continue
                # Pick a random set bit from inactive_bits
                # Convert to list and pick
                # For speed, just pick a random position and check
                x = random.randint(1, N)
                if S & (1 << x):
                    continue
                new_S = S | (1 << x)
                delta_conf = count_conflicts(S, x)
                delta_size = 1
            elif move_type < 0.8:
                # Remove a random active element
                if active_bits == 0:
                    continue
                x = random.randint(1, N)
                if not (S & (1 << x)):
                    continue
                new_S = S & ~(1 << x)
                # delta_conf: number of bad triples that were using x and are now broken
                # = count of pairs (a,b) in S\{x} such that (x,a,b) is bad
                delta_conf = -count_conflicts(S & ~(1 << x), x)
                delta_size = -1
            else:
                # Swap: remove one active, add one inactive
                if active_bits == 0 or inactive_bits == 0:
                    continue
                x_remove = random.randint(1, N)
                x_add = random.randint(1, N)
                if (S & (1 << x_remove)) == 0 or (S & (1 << x_add)):
                    continue
                if x_remove == x_add:
                    continue
                new_S = S & ~(1 << x_remove) | (1 << x_add)
                # delta_conf = conflicts added by x_add - conflicts removed by x_remove
                # conflicts added by x_add: pairs in S\{x_remove} that are bad with x_add
                S_without_remove = S & ~(1 << x_remove)
                delta_conf = count_conflicts(S_without_remove, x_add) - count_conflicts(S_without_remove, x_remove)
                delta_size = 0
            
            new_conf = current_conf + delta_conf
            new_size = current_size + delta_size
            
            # Accept if better or with probability exp(-delta/T)
            if delta_conf <= 0 and delta_size >= 0:
                # Strictly better or equal in conflicts, better or equal in size
                S = new_S
                current_conf = new_conf
                current_size = new_size
            else:
                # Compute acceptance probability
                # We want to minimize conflicts, maximize size
                # Use a combined objective: minimize (conflicts * 1000 - size)
                old_obj = current_conf * 1000 - current_size
                new_obj = new_conf * 1000 - new_size
                delta_obj = new_obj - old_obj
                if delta_obj < 0:
                    S = new_S
                    current_conf = new_conf
                    current_size = new_size
                else:
                    # Accept with probability exp(-delta_obj / T)
                    import math
                    p = math.exp(-delta_obj / T)
                    if random.random() < p:
                        S = new_S
                        current_conf = new_conf
                        current_size = new_size
            
            T *= cooling
            
            # Track best
            if current_conf == 0 and current_size > best_size:
                best_size = current_size
                best_S = S
            elif current_conf < best_size * 1000:  # heuristic: prefer low conflicts
                if current_size > best_size:
                    best_size = current_size
                    best_S = S
        
        # Local search: try to add any element that doesn't create conflicts
        for x in range(1, N+1):
            if time.time() > deadline - 2.0:
                break
            if not (S & (1 << x)):
                if count_conflicts(S, x) == 0:
                    S |= (1 << x)
                    current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_S = S
    
    # Convert bitmask to list
    result = [i for i in range(1, N+1) if best_S & (1 << i)]
    if not result:
        result = [1]
    return result
