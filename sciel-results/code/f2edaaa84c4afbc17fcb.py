"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    n = 512
    
    # Precompute forbidden triples for each x
    forbidden = [[] for _ in range(n + 1)]
    
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            forbidden[a].append((b, c, e))
            forbidden[b].append((a, c, e))
            forbidden[c].append((a, b, e))
            forbidden[e].append((a, b, c))
    
    def can_add(bits, x):
        for a, b, c in forbidden[x]:
            if (bits & (1 << a)) and (bits & (1 << b)) and (bits & (1 << c)):
                return False
        return True

    best_size = 0
    best_bits = 0
    start_time = time.time()
    time_limit = 55.0

    # Initial random greedy start
    current_bits = 0
    current_size = 0
    order = list(range(1, n + 1))
    random.shuffle(order)
    for x in order:
        if can_add(current_bits, x):
            current_bits |= (1 << x)
            current_size += 1
    
    if current_size > best_size:
        best_size = current_size
        best_bits = current_bits

    while time.time() - start_time < time_limit:
        
        # Local Search Phase
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            
            # 1. Try to add any element
            for x in range(1, n + 1):
                if not (current_bits & (1 << x)):
                    if can_add(current_bits, x):
                        current_bits |= (1 << x)
                        current_size += 1
                        improved = True
            
            # 2. If no additions, try swaps
            if not improved:
                # Try random swaps
                # We try a batch of random swaps
                found_swap = False
                for _ in range(200): # Limit swaps per iteration to avoid getting stuck
                    u = random.randint(1, n)
                    if not (current_bits & (1 << u)):
                        continue
                    
                    v = random.randint(1, n)
                    if (current_bits & (1 << v)):
                        continue
                    
                    # Check swap validity
                    temp_bits = current_bits & ~(1 << u)
                    if can_add(temp_bits, v):
                        current_bits = temp_bits | (1 << v)
                        # Size unchanged
                        found_swap = True
                        improved = True # Restart local search to try additions
                        break
                
                if not found_swap:
                    break
        
        # Update Best
        if current_size > best_size:
            best_size = current_size
            best_bits = current_bits
        
        # Perturbation Phase
        # Remove k random elements
        k = random.randint(3, 10)
        removed_count = 0
        while removed_count < k and (time.time() - start_time < time_limit):
            x = random.randint(1, n)
            if current_bits & (1 << x):
                current_bits &= ~(1 << x)
                current_size -= 1
                removed_count += 1
        
        # Optionally, try to immediately add some back to stabilize?
        # The next Local Search phase will do this.
        
        # If we are stuck at the same best_size for a long time, 
        # we might want to reset to a new random start?
        # For now, ILS should be enough.

    # Convert best_bits to list
    result = []
    for i in range(1, n + 1):
        if best_bits & (1 << i):
            result.append(i)
            
    return result

# EVOLVE-BLOCK-END
