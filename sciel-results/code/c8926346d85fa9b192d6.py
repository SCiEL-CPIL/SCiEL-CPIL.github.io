"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s limit
    p = 1009
    
    # Precompute modular inverses
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # Self-inverse elements: x^2 ≡ 1 mod p => x = 1 or x = p-1
    self_inv = {1, p - 1}
    
    best_start = 1
    best_length = 0
    
    # Iterate over all possible starting positions
    for s in range(1, p):
        if time.time() > deadline:
            break
        
        if s in self_inv:
            continue
        
        current_set = set()
        current_length = 0
        
        # Expand interval to the right
        for e in range(s, p):
            if time.time() > deadline:
                break
            
            # Check self-inverse
            if e in self_inv:
                break
            
            # Check if inverse is already in the set
            if inv[e] in current_set:
                break
            
            # Valid to add
            current_set.add(e)
            current_length += 1
        
        if current_length > best_length:
            best_length = current_length
            best_start = s
            
    return [best_start, best_length]
# EVOLVE-BLOCK-END
