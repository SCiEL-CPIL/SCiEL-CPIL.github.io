import time
import random

def is_valid(seq):
    """
    Check if a strictly increasing sequence has distinct products for all non-empty consecutive blocks.
    """
    n = len(seq)
    if n == 0:
        return True
    
    products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
            
    return True

def construct():
    deadline = time.time() + 85.0
    best_seq = []
    best_len = 0
    
    # Precompute conflict graph: for each pair (a,b) with a < b, 
    # check if a*b can be expressed as a product of a consecutive block
    # Actually, the key conflict is: if a*b equals some element in the set,
    # OR if a*b equals a product of a consecutive block.
    
    # Better approach: Use a fast incremental check when adding elements
    # Key insight: when we add a new element x to a valid sequence, we only need to
    # check products that involve x. These are:
    # 1. x itself (singleton)
    # 2. x * (suffixes of the current sequence)
    # 3. (prefixes of the current sequence) * x
    # 4. (prefixes) * x * (suffixes) - but these would already be checked if we're careful
    
    # Actually, for a strictly increasing sequence, consecutive blocks are:
    # [i..j] for i <= j
    # When we append x to the end, new blocks are [i..n] where n is the new index
    # These are: x, x*last, x*last*second_last, etc.
    
    # When we insert x in the middle, it's more complex. So let's use a different strategy.
    
    # Strategy: Use a greedy approach with a fast validity check
    # Precompute which numbers can form product collisions
    
    # For two numbers a, b in the set, a conflict occurs if a*b equals
    # a product of some consecutive block. The simplest case is if a*b is in the set.
    
    # Let's use a randomized local search approach:
    # 1. Start with a good initial solution
    # 2. Try to add/remove/swap elements to improve
    
    def fast_greedy_asc():
        """Ascending greedy with incremental product tracking"""
        values = []
        products = set()
        for x in range(2, 129):
            # Compute all new products involving x as the rightmost element
            new_products = set()
            prod = x
            new_products.add(prod)
            for a in reversed(values):
                prod *= a
                new_products.add(prod)
            if not (new_products & products):
                values.append(x)
                products |= new_products
        return values
    
    def fast_greedy_desc():
        """Descending greedy - try to include large numbers first"""
        values = []
        products = set()
        for x in range(128, 1, -1):
            # If we insert x, we need to check all blocks involving x
            # This is complex, so let's just check full validity
            new_seq = sorted(values + [x])
            if is_valid(new_seq):
                values.append(x)
                products.clear()
                # Rebuild products
                n = len(values)
                for i in range(n):
                    prod = 1
                    for j in range(i, n):
                        prod *= values[j]
                        products.add(prod)
        return values
    
    # Strategy 1: Ascending greedy
    seq = fast_greedy_asc()
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq
    
    # Strategy 2: Descending greedy
    seq = fast_greedy_desc()
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq
    
    # Strategy 3: Randomized local search
    # Start from best_seq and try to improve by adding elements
    while time.time() < deadline:
        # Try different random orderings
        candidates = list(range(2, 129))
        random.shuffle(candidates)
        
        current = list(best_seq)
        current_set = set(current)
        
        # Try to add each candidate
        for x in candidates:
            if x in current_set:
                continue
            new_seq = sorted(current + [x])
            if is_valid(new_seq):
                current = new_seq
                current_set.add(x)
        
        if len(current) > best_len:
            best_len = len(current)
            best_seq = current
            if best_len > 100:
                break
        
        # Also try removing elements to free up space
        if random.random() < 0.3:
            # Try a fresh start with random order
            candidates2 = list(range(2, 129))
            random.shuffle(candidates2)
            current2 = []
            for x in candidates2:
                new_seq = sorted(current2 + [x])
                if is_valid(new_seq):
                    current2 = new_seq
            if len(current2) > best_len:
                best_len = len(current2)
                best_seq = current2
                if best_len > 100:
                    break
    
    return best_seq
