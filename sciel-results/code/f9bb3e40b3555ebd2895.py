"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    deadline = time.time() + 85.0
    n = 512
    
    # Precompute 4-AP conflict masks
    x_conflicts = [[] for _ in range(n+1)]
    
    for d in range(1, 171):
        for a in range(1, 512 - 3*d + 1):
            terms = [a, a+d, a+2*d, a+3*d]
            for pos in range(4):
                x = terms[pos]
                others = [terms[j] for j in range(4) if j != pos]
                mask = 0
                for y in others:
                    mask |= (1 << (y-1))
                x_conflicts[x].append(mask)
    
    def is_valid(S, x):
        """Check if adding x to S creates a 4-AP."""
        for mask in x_conflicts[x]:
            if (S & mask) == mask:
                return False
        return True
    
    def greedy_random_order(rng):
        """Greedy with random element ordering."""
        S = 0
        order = list(range(1, n+1))
        rng.shuffle(order)
        for x in order:
            if is_valid(S, x):
                S |= (1 << (x-1))
        return S
    
    # Multiple greedy restarts with different random seeds
    best_S = 0
    best_count = 0
    rng = random.Random(42)
    
    # Do several greedy restarts to find a good initial solution
    for _ in range(50):
        if time.time() > deadline - 10.0:
            break
        S = greedy_random_order(rng)
        count = bin(S).count('1')
        if count > best_count:
            best_count = count
            best_S = S
    
    # Local search: try to add elements that can be added without conflict
    # Then try swaps to improve
    S = best_S
    count = best_count
    
    # Phase 1: Try to add any element that can be added
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for x in range(1, n+1):
            bit = 1 << (x-1)
            if not (S & bit) and is_valid(S, x):
                S |= bit
                count += 1
                improved = True
    
    if count > best_count:
        best_count = count
        best_S = S
    
    # Phase 2: Simulated annealing with swaps
    if time.time() < deadline - 3.0:
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        rng = random.Random(123)
        
        while time.time() < deadline - 2.0 and T > T_min:
            # Random move: add, remove, or swap
            move_type = rng.random()
            x = rng.randint(1, n)
            bit = 1 << (x-1)
            
            if S & bit:
                # Remove x
                S &= ~bit
                count -= 1
            else:
                if is_valid(S, x):
                    # Add x
                    S |= bit
                    count += 1
                else:
                    # Try a swap: remove a random element and add x
                    if move_type < 0.5:
                        # Find a random element to remove
                        y = rng.randint(1, n)
                        ybit = 1 << (y-1)
                        if S & ybit:
                            new_S = (S & ~ybit)
                            if is_valid(new_S, x):
                                S = new_S | bit
                                # count stays the same (removed one, added one)
                                if count > best_count:
                                    best_count = count
                                    best_S = S
            T *= cooling
        
        # Update best
        if count > best_count:
            best_count = count
            best_S = S
    
    # Final phase: try to add any remaining valid elements
    S = best_S
    count = best_count
    for x in range(1, n+1):
        bit = 1 << (x-1)
        if not (S & bit) and is_valid(S, x):
            S |= bit
            count += 1
    
    if count > best_count:
        best_count = count
        best_S = S
    
    # Convert bitmask to sorted list
    result = [x for x in range(1, n+1) if best_S & (1 << (x-1))]
    return result

# EVOLVE-BLOCK-END
