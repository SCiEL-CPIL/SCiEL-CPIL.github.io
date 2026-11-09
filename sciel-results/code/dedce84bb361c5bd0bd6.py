"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s kill
    
    # Use bitmask representation for fast operations
    # Bit i represents whether number i is in the set (1-indexed, so bit 0 = number 1)
    N = 128
    
    def get_bits(mask):
        """Return list of numbers present in mask."""
        result = []
        for i in range(1, N + 1):
            if mask & (1 << (i - 1)):
                result.append(i)
        return result
    
    def is_valid_set(mask):
        """Check if the set represented by mask is valid."""
        items = get_bits(mask)
        n = len(items)
        item_set = set(items)
        
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = items[i], items[j], items[k]
                    if (a + b) in item_set and (a + c) in item_set and (b + c) in item_set:
                        return False
        return True
    
    def count_violations(mask):
        """Count the number of violating triples."""
        items = get_bits(mask)
        n = len(items)
        item_set = set(items)
        violations = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = items[i], items[j], items[k]
                    if (a + b) in item_set and (a + c) in item_set and (b + c) in item_set:
                        violations += 1
        return violations
    
    # Simulated Annealing on the full 128-element set
    # Start with a good initial solution: all numbers 64-128 plus some low numbers
    
    def initial_solution():
        mask = 0
        # Start with high numbers
        for i in range(64, 129):
            mask |= (1 << (i - 1))
        # Add some low numbers greedily
        for i in range(1, 64):
            new_mask = mask | (1 << (i - 1))
            if count_violations(new_mask) == 0:
                mask = new_mask
        return mask
    
    def random_flip(mask, num_flips=1):
        """Randomly flip num_flips bits."""
        for _ in range(num_flips):
            pos = random.randint(0, N - 1)
            mask ^= (1 << pos)
        return mask
    
    # Run simulated annealing
    best_mask = initial_solution()
    best_violations = count_violations(best_mask)
    best_size = bin(best_mask).count('1')
    
    current_mask = best_mask
    current_violations = best_violations
    current_size = best_size
    
    temperature = 100.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Propose a move: flip 1-3 random bits
        num_flips = random.choice([1, 1, 1, 2, 2, 3])
        new_mask = random_flip(current_mask, num_flips)
        
        new_violations = count_violations(new_mask)
        new_size = bin(new_mask).count('1')
        
        # Accept if better or with probability based on temperature
        if new_violations < current_violations or (new_violations == current_violations and new_size > current_size):
            current_mask = new_mask
            current_violations = new_violations
            current_size = new_size
            
            if current_violations == 0 and current_size > best_size:
                best_mask = current_mask
                best_size = current_size
        else:
            # Accept worse with probability exp(-delta/T)
            delta = (new_violations - current_violations) - (new_size - current_size) * 0.1
            if delta < 0:
                current_mask = new_mask
                current_violations = new_violations
                current_size = new_size
            else:
                prob = min(1.0, max(0.0, 1.0 - delta / (temperature + 1e-10)))
                if random.random() < prob:
                    current_mask = new_mask
                    current_violations = new_violations
                    current_size = new_size
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01
    
    # If we found a valid solution, use it
    if best_violations == 0:
        result = get_bits(best_mask)
    else:
        # Fallback: use the high set plus a greedy low set
        mask = 0
        for i in range(64, 129):
            mask |= (1 << (i - 1))
        for i in range(1, 64):
            new_mask = mask | (1 << (i - 1))
            if count_violations(new_mask) == 0:
                mask = new_mask
        result = get_bits(mask)
    
    return result

# EVOLVE-BLOCK-END
