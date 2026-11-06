"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 256

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 25.0
    best_size = N + 1
    best_set = None

    def is_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_addable(S):
        if not S:
            return list(range(1, N + 1))
        S_set = set(S)
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        
        addable = []
        for x in range(1, N + 1):
            if x in S_set:
                continue
            # Check if x can be added
            if 2 * x in sums:
                continue
            can_add = True
            for a in S:
                if (x + a) in sums:
                    can_add = False
                    break
            if can_add:
                addable.append(x)
        return addable

    def compute_energy(S):
        if not S:
            return 0, 0, 0 # size, invalidity, non_maximality
        
        S_set = set(S)
        sums = set()
        invalidity = 0
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    invalidity += 1
                else:
                    sums.add(s)
        
        # If invalid, we can't easily compute addable, but we return high penalty
        if invalidity > 0:
            return len(S), invalidity, 0

        addable_count = len(get_addable(S))
        return len(S), 0, addable_count

    # 1. Generate an initial valid, maximal Sidon set using Greedy
    def greedy_initial():
        S = []
        S_set = set()
        sums = set()
        remaining = list(range(1, N + 1))
        
        while remaining:
            best_x = None
            best_score = -1
            # Score: number of elements blocked
            for x in remaining:
                # Check if x can be added
                if 2 * x in sums:
                    continue
                can_add = True
                for a in S:
                    if (x + a) in sums:
                        can_add = False
                        break
                if not can_add:
                    continue
                
                # Calculate blocking power
                new_sums = set(sums)
                new_sums.add(2 * x)
                for a in S:
                    new_sums.add(x + a)
                
                blocked = 0
                for y in remaining:
                    if y == x:
                        continue
                    # Check if y is blocked by new_sums
                    if 2 * y in new_sums:
                        blocked += 1
                        continue
                    y_blocked = False
                    for a in S:
                        if (y + a) in new_sums:
                            y_blocked = True
                            break
                    if y_blocked:
                        blocked += 1
                    # Also check if x itself blocks y via x+y? 
                    # The condition for y to be addable is 2y not in sums AND y+a not in sums for all a in S.
                    # If we add x, the new sums are old_sums U {2x, x+a for a in S}.
                    # y is blocked if 2y is in new_sums OR (y+a) is in new_sums for some a in S U {x}.
                    # Wait, if we add x, the set becomes S U {x}.
                    # So we check against S U {x}.
                    # The sums of S U {x} are sums(S) U {2x, x+a for a in S}.
                    # So the check above is correct.
                
                if blocked > best_score:
                    best_score = blocked
                    best_x = x
            
            if best_x is None:
                break
            
            S.append(best_x)
            S_set.add(best_x)
            sums.add(2 * best_x)
            for a in S:
                sums.add(best_x + a)
            remaining.remove(best_x)
        
        # Ensure maximality
        # The greedy loop above stops when no more can be added?
        # No, it stops if no *valid* x is found. But it might not be maximal if we just picked the best blocking one.
        # Actually, if no valid x is found in 'remaining', then no x can be added, so it IS maximal.
        return S

    initial_set = greedy_initial()
    if initial_set:
        best_size = len(initial_set)
        best_set = list(initial_set)

    # 2. Simulated Annealing
    current_S = list(initial_set)
    current_size, current_inv, current_nonmax = compute_energy(current_S)
    current_energy = 10 * current_size + 100 * current_inv + 10 * current_nonmax
    
    T = 100.0
    cooling_rate = 0.95
    
    steps = 0
    while time.time() < deadline - 2.0:
        steps += 1
        
        # Generate a neighbor
        # Move 1: Remove a random element
        # Move 2: Add a random element (if valid)
        # Move 3: Swap a random element with a random other
        
        move_type = random.choice([0, 1, 2])
        new_S = list(current_S)
        
        if move_type == 0 and new_S:
            idx = random.randint(0, len(new_S) - 1)
            new_S.pop(idx)
        elif move_type == 1:
            # Add a random element from 1..N not in S
            candidates = [x for x in range(1, N + 1) if x not in new_S]
            if candidates:
                x = random.choice(candidates)
                # Check if valid Sidon addition
                S_set = set(new_S)
                sums = set()
                for i in range(len(new_S)):
                    for j in range(i, len(new_S)):
                        sums.add(new_S[i] + new_S[j])
                
                if 2 * x not in sums:
                    valid = True
                    for a in new_S:
                        if (x + a) in sums:
                            valid = False
                            break
                    if valid:
                        new_S.append(x)
        elif move_type == 2 and new_S:
            # Swap
            idx = random.randint(0, len(new_S) - 1)
            old_x = new_S[idx]
            candidates = [x for x in range(1, N + 1) if x not in new_S]
            if candidates:
                new_x = random.choice(candidates)
                # Check if swapping keeps it Sidon
                # Remove old_x, add new_x
                new_S[idx] = new_x
                # We need to verify if new_S is Sidon.
                # For efficiency, we could do a partial check, but N=256 is small.
                if not is_sidon(new_S):
                    # Revert if invalid? Or just let energy handle it?
                    # Let energy handle it, but is_sidon check is O(N^2).
                    # To save time, only accept if valid? No, SA needs to explore invalid states.
                    pass

        new_size, new_inv, new_nonmax = compute_energy(new_S)
        new_energy = 10 * new_size + 100 * new_inv + 10 * new_nonmax
        
        delta = new_energy - current_energy
        
        if delta < 0:
            accept = True
        else:
            if T > 0:
                accept = random.random() < math.exp(-delta / T)
            else:
                accept = False
        
        if accept:
            current_S = new_S
            current_size, current_inv, current_nonmax = new_size, new_inv, new_nonmax
            current_energy = new_energy
            
            # Update best if valid and maximal
            if current_inv == 0 and current_nonmax == 0:
                if current_size < best_size:
                    best_size = current_size
                    best_set = list(current_S)
        
        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    if best_set is not None:
        return best_set
    else:
        return initial_set
# EVOLVE-BLOCK-END
