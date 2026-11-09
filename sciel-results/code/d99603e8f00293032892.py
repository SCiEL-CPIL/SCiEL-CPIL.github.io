import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # 1. Initial Greedy Solution
    def greedy_construct():
        values = []
        products = set()
        for x in range(2, 129):
            # Check if x can be appended
            # Since we want strictly increasing, we only append at the end in greedy?
            # The seed appends at the end.
            # Let's stick to the seed's greedy for initialization.
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    seq = greedy_construct()
    used = set(seq)
    best_seq = seq[:]
    best_len = len(seq)
    
    def is_valid(seq):
        """Check if all subarray products are distinct."""
        if not seq:
            return True
        prods = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in prods:
                    return False
                prods.add(prod)
        return True

    # Local Search
    # Moves:
    # 1. Insert x at pos i
    # 2. Swap a (at pos i) with x
    
    def try_improve(seq, used):
        """
        Attempts to improve the sequence by insertion or swap.
        Returns (new_seq, new_used, improved)
        improved is True if length increased.
        """
        n = len(seq)
        
        # 1. Try Insertions
        # We want to find any valid insertion.
        # To be smart, we can try smaller numbers first or larger?
        # Usually, smaller numbers create smaller products, which are more likely to collide?
        # Or larger numbers create larger products, which are less likely to collide with existing small ones?
        # Let's just try all.
        
        candidates = list(range(2, 129))
        
        # Optimization: Pre-check order constraints
        # For a candidate x to be inserted at pos i:
        # if i > 0: x > seq[i-1]
        # if i < n: x < seq[i]
        
        # We will iterate through possible insertions.
        # If we find one, we return it immediately? 
        # Or do we want the "best" insertion? 
        # Any insertion increases length, so it's a strict improvement.
        
        for x in candidates:
            if x in used:
                continue
            
            # Find valid positions for x
            # Lower bound index: first index where seq[i] > x
            # Upper bound index: last index where seq[i] < x
            # Since seq is sorted, we can binary search or just loop.
            
            left = 0
            while left < n and seq[left] < x:
                left += 1
            # left is the first index where seq[left] >= x
            # Since x not in used, seq[left] != x, so seq[left] > x (if left < n)
            # So x can be inserted at positions:
            # 0 to left (if seq[left] > x or left==n)
            # Actually, if seq is [2, 5, 8] and x=4:
            # left=1 (seq[1]=5 > 4).
            # Valid positions: 1 (between 2 and 5).
            # If x=10, left=3 (end). Valid pos: 3.
            # If x=1, left=0. Valid pos: 0.
            
            # So valid range is [left, left]? No.
            # If x < seq[0], left=0. Pos 0 is valid.
            # If seq[0] < x < seq[1], left=1. Pos 1 is valid.
            # If x > seq[-1], left=n. Pos n is valid.
            
            # Wait, is it unique?
            # Yes, for a sorted unique array, there is exactly one position where a new number fits to maintain order.
            
            pos = left
            
            # Check validity
            new_seq = seq[:pos] + [x] + seq[pos:]
            if is_valid(new_seq):
                new_used = used | {x}
                return new_seq, new_used, True
        
        # 2. Try Swaps
        # If no insertion, try to swap an element a with x.
        # This keeps length same, but changes the set of products.
        # This is a "neutral" move to escape local optima.
        
        # To avoid cycling, we can keep a set of visited states (tuples of seq).
        # But for simplicity, let's just do one swap if found, and hope it leads to an insertion.
        # Or, we can try to find a swap that allows an immediate insertion.
        
        # Let's try all swaps.
        for i, a in enumerate(seq):
            for x in candidates:
                if x in used or x == a:
                    continue
                
                # Check order constraint for swap at i
                # Neighbors: left = seq[i-1] if i>0 else None
                #           right = seq[i+1] if i<n-1 else None
                # x must be > left and < right
                
                valid_order = True
                if i > 0 and x <= seq[i-1]:
                    valid_order = False
                if i < n - 1 and x >= seq[i+1]:
                    valid_order = False
                
                if not valid_order:
                    continue
                
                new_seq = seq[:]
                new_seq[i] = x
                if is_valid(new_seq):
                    # We found a valid swap.
                    # Do we accept it?
                    # If we accept it, we might cycle.
                    # Strategy: Accept swap, then immediately try to insert.
                    # If insertion fails, we might revert?
                    # For now, let's just return the swapped seq and let the main loop handle it.
                    # The main loop will then try to insert into this new seq.
                    new_used = (used - {a}) | {x}
                    return new_seq, new_used, False # Not an improvement in length
        
        # No moves found
        return seq, used, False

    # Main Loop
    no_improvement_count = 0
    max_no_improvement = 50 # Limit on neutral moves before trying something else or stopping
    
    while time.time() < deadline:
        new_seq, new_used, improved = try_improve(seq, used)
        
        if improved:
            seq = new_seq
            used = new_used
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
            no_improvement_count = 0
        else:
            if new_seq != seq: # A swap happened
                seq = new_seq
                used = new_used
                no_improvement_count += 1
                if no_improvement_count > max_no_improvement:
                    # Stuck in a local optimum with only swaps available
                    # Break and try a different strategy? 
                    # Or just break if we can't insert.
                    # Since we can't insert, and swaps don't increase length, 
                    # we might be stuck. 
                    # Let's try a "double swap" or just break.
                    # For now, break to save time, as best_seq is saved.
                    break
            else:
                # No moves at all
                break

    return best_seq

# EVOLVE-BLOCK-START
# The function construct is defined above. 
# We need to ensure the entry point is exactly construct().
# The code above defines construct.
# EVOLVE-BLOCK-END
