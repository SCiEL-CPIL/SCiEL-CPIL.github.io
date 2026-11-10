from fractions import Fraction
import time
import random

def check_addition(current_sums, x):
    """
    Check if adding 1/x to the set of sums current_sums causes a collision.
    Returns (True, new_sums) if valid, (False, None) otherwise.
    """
    inv = Fraction(1, x)
    new_sums = set()
    for s in current_sums:
        new_sums.add(s + inv)
    
    # Check intersection
    if new_sums & current_sums:
        return False, None
    
    # Union
    # new_sums is disjoint from current_sums
    # So new total sums is current_sums U new_sums
    return True, current_sums.union(new_sums)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    candidates = list(range(1, 23))
    
    # Initial best set
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    # We will try to find a better set using randomized greedy + local improvement.
    
    def try_greedy(order):
        """
        Run a greedy construction using the given order of candidates.
        Returns the constructed set and its sums.
        """
        current_set = []
        current_sums = {Fraction(0)}
        
        for x in order:
            valid, new_sums = check_addition(current_sums, x)
            if valid:
                current_set.append(x)
                current_sums = new_sums
                
        return current_set, current_sums

    def try_local_improvement(S, sums):
        """
        Try to add any element not in S to the set.
        Returns improved set and sums if successful.
        """
        S_set = set(S)
        improved = False
        for x in range(1, 23):
            if x in S_set:
                continue
            valid, new_sums = check_addition(sums, x)
            if valid:
                S.append(x)
                S_set.add(x)
                sums = new_sums
                improved = True
                # We can break or continue. Continuing might find more?
                # No, adding one element changes the sums.
                # We should restart the loop to check again?
                # Actually, if we add x, we should check if we can add another.
                # So we return and let the outer loop handle it?
                # Or just continue in the same pass?
                # Let's just return the first improvement found to keep it simple.
                # Or better, loop until no improvement.
                # But check_addition is expensive.
                # Let's just try to add ONE more element.
                break
                
        return S, sums

    # Run multiple randomized greedy passes
    # Also try sorted orders
    
    orders_to_try = []
    # Ascending
    orders_to_try.append(list(range(1, 23)))
    # Descending
    orders_to_try.append(list(range(22, 0, -1)))
    
    # Random shuffles
    # We'll generate shuffles on the fly to save memory
    
    pass_count = 0
    while time.time() < deadline:
        pass_count += 1
        
        # Determine order
        if pass_count <= 2:
            order = orders_to_try[pass_count - 1]
        else:
            order = list(range(1, 23))
            random.shuffle(order)
            
        S, sums = try_greedy(order)
        
        # Local improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add any x
            for x in range(1, 23):
                if x in S:
                    continue
                valid, new_sums = check_addition(sums, x)
                if valid:
                    S.append(x)
                    sums = new_sums
                    improved = True
                    break # Restart the for loop to check with new sums
            
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            if best_size == 22:
                break
                
    return best_set
