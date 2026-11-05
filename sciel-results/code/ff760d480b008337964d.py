"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    # Safety margin: 90s limit, use 80s for search
    deadline = start_time + 80.0

    # 1. Run Greedy to get a strong initial lower bound
    greedy_values, greedy_products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(greedy_values):
            prod *= a
            tails.append(prod)
        if not greedy_products.intersection(tails):
            greedy_values.append(x)
            greedy_products.update(tails)

    best_sequence = greedy_values
    best_length = len(greedy_values)

    # 2. DFS to try to improve
    current_seq = []
    current_products = set()
    
    # Pre-compute valid moves? No, because validity depends on current products.
    # But we can optimize the intersection check.
    
    def dfs(start_val):
        nonlocal best_sequence, best_length, deadline
        
        # Time check
        if time.time() > deadline:
            return False
        
        # Update best
        if len(current_seq) > best_length:
            best_length = len(current_seq)
            best_sequence = current_seq[:]
            
        # Pruning: Can we beat best_length with remaining candidates?
        # Max possible additional elements is (128 - start_val) + 1 = 129 - start_val
        # If current_len + max_possible <= best_length, we can't improve.
        if len(current_seq) + (129 - start_val) <= best_length:
            return True
            
        # Iterate through possible next values
        for x in range(start_val, 129):
            # Check if x can be added
            # Calculate new products
            new_products = []
            prod = x
            new_products.append(prod)
            for a in reversed(current_seq):
                prod *= a
                new_products.append(prod)
                
            # Check for collisions with existing products
            # Optimization: Check intersection
            if current_products.intersection(new_products):
                continue
                
            # Valid move
            current_seq.append(x)
            current_products.update(new_products)
            
            # Recurse
            # The next value must be > x
            if not dfs(x + 1):
                # Timeout signal propagated
                current_seq.pop()
                current_products.difference_update(new_products)
                return False
            
            # Backtrack
            current_seq.pop()
            current_products.difference_update(new_products)
            
            # Time check after backtrack
            if time.time() > deadline:
                return False
                
        return True

    # Start DFS from the smallest possible value
    dfs(2)
    
    # Return the best sequence found
    return best_sequence
# EVOLVE-BLOCK-END
