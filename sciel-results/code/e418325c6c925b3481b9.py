"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill
    
    best_sequence = []
    best_length = 0
    
    # Global state for DFS
    current_seq = []
    existing_products = set()
    
    # Memoization: (last_val, frozenset of remaining) -> best additional length achievable
    # This is too expensive for large sets, so we use a lighter approach:
    # Track which (last_val, index) pairs have been fully explored and their max depth
    # Actually, let's use a different strategy: greedy initialization + improved DFS
    
    # Precompute suffix products for efficiency
    # For a sequence [a1, a2, ..., ak], the new products when appending x are:
    # x, x*ak, x*ak*a_{k-1}, ..., x*ak*...*a1
    # We can maintain these incrementally.
    
    # Greedy initialization: build a good initial solution
    def greedy_build():
        """Build a greedy solution by trying to add each number if possible."""
        seq = []
        products = set()
        suffix_prod = 1  # product of all elements in seq
        
        for x in range(2, 129):
            # Check if x can be added
            # New products: x, x*last, x*last*second_last, ...
            # We need to check against existing products
            
            # Calculate new products
            new_prods = []
            prod = x
            for a in reversed(seq):
                new_prods.append(prod)
                prod *= a
            new_prods.append(prod)
            
            # Check for collisions
            if any(p in products for p in new_prods):
                continue
            
            # Also check if new_prods have internal collisions (they shouldn't for positive increasing)
            # But let's be safe
            if len(set(new_prods)) != len(new_prods):
                continue
            
            # Add it
            seq.append(x)
            products.update(new_prods)
            suffix_prod *= x
        
        return seq
    
    # Get initial greedy solution
    greedy_seq = greedy_build()
    best_sequence = greedy_seq[:]
    best_length = len(greedy_seq)
    
    # Precompute all products for the greedy solution
    def compute_all_products(seq):
        """Compute all consecutive block products for a sequence."""
        prods = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                prods.add(prod)
        return prods
    
    # Now do DFS to improve, starting from the greedy solution's structure
    # We'll use a beam-like approach: try different starting points and extensions
    
    # Better approach: Use the greedy as a lower bound, then do DFS with strong pruning
    # Key insight: if we have a solution of length L, we only need to find solutions of length > L
    
    # Let's use a recursive DFS with the following improvements:
    # 1. Start from empty, but use greedy bound for pruning
    # 2. Try adding elements in a smart order (prefer those that don't conflict)
    # 3. Use memoization on (last_val, remaining_set) for small remaining sets
    
    # For efficiency, we'll represent remaining as a bitmask or sorted tuple
    # Since we're dealing with values 2-128, we can use a bitmask of 127 bits
    
    # Actually, let's stick with the list approach but add better pruning:
    # - If current length + (number of remaining elements that could possibly be added) <= best_length, prune
    # - We can estimate the max addable by checking which remaining elements don't conflict with current
    
    def can_add(x, seq, products):
        """Check if x can be added to seq with given products set."""
        new_prods = []
        prod = x
        for a in reversed(seq):
            new_prods.append(prod)
            prod *= a
        new_prods.append(prod)
        
        if any(p in products for p in new_prods):
            return None
        if len(set(new_prods)) != len(new_prods):
            return None
        return new_prods
    
    # DFS with improved pruning
    # Use a global counter for time checks
    time_check_counter = [0]
    
    def dfs(seq, products, start_idx):
        """
        DFS where:
        - seq: current sequence (list)
        - products: set of all consecutive block products
        - start_idx: index in candidates[2:129] where we can start picking from
        """
        nonlocal best_sequence, best_length, deadline
        
        # Time check every 1000 calls
        time_check_counter[0] += 1
        if time_check_counter[0] % 1000 == 0:
            if time.time() > deadline:
                return False
        
        # Update best
        if len(seq) > best_length:
            best_length = len(seq)
            best_sequence = seq[:]
        
        # Pruning: even if we add all remaining, can we beat best?
        candidates = list(range(2, 129))
        remaining = candidates[start_idx:]
        if not remaining:
            return True
        if len(seq) + len(remaining) <= best_length:
            return True
        
        # Additional pruning: estimate how many of remaining can actually be added
        # This is expensive, so only do it when close to the bound
        if len(seq) + len(remaining) <= best_length + 2:
            # Count how many remaining elements can be added to current seq
            addable = 0
            for x in remaining:
                if can_add(x, seq, products) is not None:
                    addable += 1
            if len(seq) + addable <= best_length:
                return True
        
        # Try each candidate from start_idx onwards
        for i in range(start_idx, len(candidates)):
            x = candidates[i]
            
            new_prods = can_add(x, seq, products)
            if new_prods is None:
                continue
            
            # Add x
            seq.append(x)
            products.update(new_prods)
            
            # Recurse
            dfs(seq, products, i + 1)
            
            # Backtrack
            seq.pop()
            products.difference_update(new_prods)
            
            # Time check
            if time.time() > deadline:
                return False
        
        return True
    
    # Start DFS from empty sequence
    # But first, let's try a few different strategies:
    
    # Strategy 1: Standard DFS from empty
    # We'll use the greedy solution as the initial best
    
    # To make DFS more effective, let's try starting with different first elements
    # and see which branches are most promising
    
    # Actually, let's just run the DFS with the improved pruning
    # The greedy gives us a good lower bound to prune with
    
    # One more optimization: sort candidates in a way that prefers "safe" numbers
    # Numbers with few prime factors tend to create fewer collisions
    # But for simplicity, let's stick with natural order
    
    # Run DFS
    dfs([], set(), 0)
    
    # Final time check
    if time.time() > deadline:
        return best_sequence
    
    return best_sequence
# EVOLVE-BLOCK-END
