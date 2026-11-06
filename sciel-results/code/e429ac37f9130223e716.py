import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
            
    # Precompute conflict masks
    # conflict_mask[a][b] is a bitmask of c such that (a,b,c) is a bad triple
    # We only need this for a != b.
    conflict_mask = [[0] * (N + 1) for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g = gcd_table[a][b]
            mask = 0
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if gcd_table[a][c] == g and gcd_table[b][c] == g:
                    mask |= (1 << c)
            conflict_mask[a][b] = mask
            conflict_mask[b][a] = mask
            
    # State
    in_set = [False] * (N + 1)
    current_set = set()
    # conflict_count[x] = number of pairs (a,b) in current_set such that x is in conflict_mask[a][b]
    conflict_count = [0] * (N + 1)
    
    def add_element(x):
        in_set[x] = True
        current_set.add(x)
        # Update conflict counts for all y not in set
        # New pairs are (x, a) for all a in current_set (excluding x)
        for a in list(current_set):
            if a == x:
                continue
            mask = conflict_mask[x][a]
            # Iterate set bits in mask
            m = mask
            while m:
                lsb = m & -m
                y = lsb.bit_length() - 1
                if not in_set[y]:
                    conflict_count[y] += 1
                m ^= lsb
                
    def remove_element(x):
        in_set[x] = False
        current_set.remove(x)
        # Decrement conflict counts
        for a in list(current_set):
            if a == x:
                continue
            mask = conflict_mask[x][a]
            m = mask
            while m:
                lsb = m & -m
                y = lsb.bit_length() - 1
                if not in_set[y]:
                    conflict_count[y] -= 1
                m ^= lsb

    best_set = set()
    best_size = 0
    
    def update_best():
        nonlocal best_set, best_size
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    # Greedy Initialization
    # We can try multiple random orderings or just one greedy with random tie-breaking
    for _ in range(20):
        # Reset state
        in_set = [False] * (N + 1)
        current_set = set()
        conflict_count = [0] * (N + 1)
        
        # Greedy add
        # Randomize the order of candidates?
        # Or just pick any valid one.
        # To escape local optima, randomize which valid element to pick.
        remaining = list(range(1, N + 1))
        random.shuffle(remaining)
        
        # We need to check validity dynamically.
        # Instead of shuffling and iterating, we can just pick a random valid element from all remaining.
        # But "valid" changes as we add.
        
        # Simple greedy:
        while True:
            valid_candidates = [x for x in range(1, N + 1) if not in_set[x] and conflict_count[x] == 0]
            if not valid_candidates:
                break
            # Pick randomly to diversify
            x = random.choice(valid_candidates)
            add_element(x)
            
        update_best()
        if time.time() > deadline:
            break
            
    # Local Search: 1-Remove, Re-Greedy
    # Start from best_set found
    if best_set:
        # Load best_set into state
        in_set = [False] * (N + 1)
        current_set = set()
        conflict_count = [0] * (N + 1)
        for x in best_set:
            # We can't just call add_element because it updates counts based on current_set
            # We need to build the state from scratch or carefully.
            # Easier: Re-run greedy but force the elements of best_set to be added first?
            # Or just rebuild state by adding elements of best_set one by one.
            # Since best_set is valid, adding them one by one in any order will keep conflict_count correct for the rest.
            # Wait, if I add x, then y, the conflict_count for z is updated by pairs (x,z) and (y,z).
            # This is correct.
            add_element(x)
            
        # Now current_set is best_set.
        # Perform local search
        for step in range(5000):
            if time.time() > deadline:
                break
                
            # Try to add
            valid_candidates = [x for x in range(1, N + 1) if not in_set[x] and conflict_count[x] == 0]
            if valid_candidates:
                # Try to add one. If it increases size, keep it.
                # But maybe adding one blocks many others?
                # For now, just add a random one.
                x = random.choice(valid_candidates)
                add_element(x)
                update_best()
                # Optionally, if we want to be more careful, we could check if this was a "bad" move
                # by seeing if it blocked many others, but for now, maximize size immediately.
            else:
                # Stuck. Remove a random element and try again.
                if not current_set:
                    break
                x = random.choice(list(current_set))
                remove_element(x)
                # After removal, try to add again in the next iteration.
                
    # Final Polish: Greedy add from best_set
    # Load best_set
    in_set = [False] * (N + 1)
    current_set = set()
    conflict_count = [0] * (N + 1)
    for x in best_set:
        add_element(x)
        
    # Try to add any remaining valid elements
    for x in range(1, N + 1):
        if not in_set[x] and conflict_count[x] == 0:
            add_element(x)
            
    return list(current_set)
