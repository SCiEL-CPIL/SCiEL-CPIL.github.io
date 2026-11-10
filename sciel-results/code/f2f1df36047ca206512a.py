"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 78  # safety margin
    
    N = 256
    
    def has_4ap_with_x(A_set, x):
        """Check if adding x to A_set creates a 4-AP."""
        for d in range(1, (N - x) // 3 + 1):
            if (x + d) in A_set and (x + 2*d) in A_set and (x + 3*d) in A_set:
                return True
        for d in range(1, min(x-1, (N-x)//2) + 1):
            if (x - d) in A_set and (x + d) in A_set and (x + 2*d) in A_set:
                return True
        for d in range(1, (x-1)//2 + 1):
            if (x - 2*d) in A_set and (x - d) in A_set and (x + d) in A_set:
                return True
        for d in range(1, (x-1)//3 + 1):
            if (x - 3*d) in A_set and (x - 2*d) in A_set and (x - d) in A_set:
                return True
        return False
    
    def gain_3aps_add(A_set, x):
        """Number of new 3-APs created by adding x to A_set."""
        gain = 0
        for d in range(1, (N - x) // 2 + 1):
            if (x + d) in A_set and (x + 2*d) in A_set:
                gain += 1
        for d in range(1, min(x-1, N-x) + 1):
            if (x - d) in A_set and (x + d) in A_set:
                gain += 1
        for d in range(1, (x-1)//2 + 1):
            if (x - 2*d) in A_set and (x - d) in A_set:
                gain += 1
        return gain
    
    def loss_3aps_remove(A_set, x):
        """Number of 3-APs lost by removing x from A_set (x is in A_set)."""
        loss = 0
        for d in range(1, (N - x) // 2 + 1):
            if (x + d) in A_set and (x + 2*d) in A_set:
                loss += 1
        for d in range(1, min(x-1, N-x) + 1):
            if (x - d) in A_set and (x + d) in A_set:
                loss += 1
        for d in range(1, (x-1)//2 + 1):
            if (x - 2*d) in A_set and (x - d) in A_set:
                loss += 1
        return loss
    
    def count_3aps(A_set):
        count = 0
        for a in A_set:
            for d in range(1, (N - a) // 2 + 1):
                if (a + d) in A_set and (a + 2*d) in A_set:
                    count += 1
        return count
    
    # Phase 1: Greedy construction
    A_set = set()
    for x in range(1, N + 1):
        if not has_4ap_with_x(A_set, x):
            A_set.add(x)
    
    current_3aps = count_3aps(A_set)
    best_3aps = current_3aps
    best_A = set(A_set)
    
    # Phase 2: Greedy additions
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if x not in A_set and not has_4ap_with_x(A_set, x):
                g = gain_3aps_add(A_set, x)
                if g > 0:
                    A_set.add(x)
                    current_3aps += g
                    changed = True
                    if current_3aps > best_3aps:
                        best_3aps = current_3aps
                        best_A = set(A_set)
    
    # Phase 3: Simulated annealing-style local search
    random.seed(123)
    temp = 10.0
    cooling = 0.99
    
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try adding a random element
            x = random.randint(1, N)
            if x not in A_set and not has_4ap_with_x(A_set, x):
                g = gain_3aps_add(A_set, x)
                A_set.add(x)
                current_3aps += g
                if current_3aps > best_3aps:
                    best_3aps = current_3aps
                    best_A = set(A_set)
            # If can't add, do nothing
        elif move_type < 0.7:
            # Try removing a random element
            if A_set:
                x = random.choice(list(A_set))
                l = loss_3aps_remove(A_set, x)
                A_set.remove(x)
                current_3aps -= l
                # Keep it removed for now (will try to add others)
        else:
            # Try a swap
            if A_set:
                x = random.choice(list(A_set))
                l = loss_3aps_remove(A_set, x)
                A_set.remove(x)
                current_3aps -= l
                
                # Try to find a good y to add
                best_y = None
                best_net = -l  # need net > 0 to improve
                for _ in range(30):
                    y = random.randint(1, N)
                    if y not in A_set and y != x and not has_4ap_with_x(A_set, y):
                        g = gain_3aps_add(A_set, y)
                        if g > best_net:
                            best_net = g
                            best_y = y
                
                if best_y is not None:
                    A_set.add(best_y)
                    current_3aps += best_net
                    if current_3aps > best_3aps:
                        best_3aps = current_3aps
                        best_A = set(A_set)
                else:
                    # Put x back
                    A_set.add(x)
                    current_3aps += l
        
        temp *= cooling
    
    # Phase 4: Final greedy pass from best
    A_set = set(best_A)
    current_3aps = best_3aps
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if x not in A_set and not has_4ap_with_x(A_set, x):
                g = gain_3aps_add(A_set, x)
                if g > 0:
                    A_set.add(x)
                    current_3aps += g
                    changed = True
                    if current_3aps > best_3aps:
                        best_3aps = current_3aps
                        best_A = set(A_set)
    
    return sorted(best_A)
# EVOLVE-BLOCK-END
