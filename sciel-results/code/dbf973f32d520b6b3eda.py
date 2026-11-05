import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def is_valid(s):
        n = len(s)
        if n == 0:
            return True
        prods = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= s[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    # 1. Initialize with primes (guaranteed valid base)
    primes = [x for x in range(2, 129) if is_prime(x)]
    seq = primes[:]
    
    # 2. List of candidates to add
    candidates = [x for x in range(2, 129) if x not in primes]
    candidates.sort()
    
    best_seq = seq[:]
    best_len = len(seq)
    
    # 3. Greedy Insertion Phase
    # Try to insert each candidate. If it fits, insert it.
    # To maximize length, we might want to try inserting in different orders,
    # but simple ascending order is a good start.
    i = 0
    while i < len(candidates):
        c = candidates[i]
        inserted = False
        # Try inserting c at all valid positions in seq
        # Since seq is sorted, c can only be inserted in one specific sorted position
        # BUT, the problem requires the OUTPUT to be strictly increasing.
        # So the position is fixed by the value.
        # Find position
        pos = -1
        for k in range(len(seq)):
            if seq[k] > c:
                pos = k
                break
        if pos == -1:
            pos = len(seq)
            
        # Check if c is already in seq
        if pos < len(seq) and seq[pos] == c:
            i += 1
            continue
        if pos > 0 and seq[pos-1] == c:
            i += 1
            continue

        new_seq = seq[:pos] + [c] + seq[pos:]
        
        if is_valid(new_seq):
            seq = new_seq
            best_seq = seq[:]
            best_len = len(seq)
            # If we inserted, we might need to re-check previous candidates?
            # No, greedy forward is standard. But maybe a later candidate blocks an earlier one?
            # Actually, if we process in ascending order, we don't go back.
            # However, sometimes inserting a small number blocks a larger one, or vice versa.
            # A simple greedy pass might not be optimal.
            # Let's do a second pass later.
            i += 1
        else:
            i += 1
            
    # 4. Local Search Phase (Hill Climbing)
    # Try to improve by swapping or moving elements.
    # Since the sequence must be sorted, "swapping" means removing one and reinserting another.
    # Or simply: Remove one element, try to insert a different one.
    
    # We will try to "fix" the sequence by removing elements that cause conflicts
    # and then re-inserting others.
    
    # A more effective local search:
    # Pick a random element to remove.
    # Then try to insert all remaining candidates (including the removed one) 
    # to see if we can get a longer sequence.
    
    # To keep it fast, we limit the number of iterations.
    iterations = 0
    max_iterations = 5000 # Adjust based on time
    
    # We need a pool of "available" numbers
    in_seq = set(seq)
    available = [x for x in range(2, 129) if x not in in_seq]
    
    current_seq = seq[:]
    
    import random
    random.seed(42)
    
    while time.time() < deadline - 5:
        iterations += 1
        
        # Move: Remove a random element from current_seq
        if len(current_seq) <= 2:
            break
            
        idx = random.randint(0, len(current_seq) - 1)
        removed = current_seq[idx]
        new_seq = current_seq[:idx] + current_seq[idx+1:]
        
        # Now try to insert 'removed' and other available numbers
        # We'll just try to insert 'removed' back at a different position?
        # No, position is fixed by value.
        # So removing an element and re-inserting it is useless unless we change the *other* elements.
        
        # Better Move: Swap two elements?
        # Since order is fixed by value, we can't swap values without breaking sort order.
        # The only "move" is changing the *set* of elements.
        
        # So: Remove one element, add a different one.
        # If we remove 'removed', we have len-1.
        # We try to add a number from 'available' + [removed].
        
        # To be efficient, we just try to add ONE number.
        # If we can add a number that makes the sequence valid AND longer than before?
        # No, we removed one, so length is same. We want to replace 'removed' with 'new_c'
        # such that the new sequence is valid.
        
        # Actually, the goal is to maximize length.
        # If we are stuck at length L, we try to find a configuration of length L+1.
        
        # Let's try a "kick": Remove 2 elements, try to insert 2 different ones.
        # This is complex.
        
        # Simpler: Just try to insert any available number into current_seq.
        # If we can insert, we increase length.
        # If we can't, we remove a random element and try again.
        
        # Step 4a: Try to insert any available number
        inserted = False
        for c in available:
            if time.time() > deadline - 5:
                break
            pos = -1
            for k in range(len(current_seq)):
                if current_seq[k] > c:
                    pos = k
                    break
            if pos == -1:
                pos = len(current_seq)
                
            # Check bounds
            if pos < len(current_seq) and current_seq[pos] == c:
                continue
            if pos > 0 and current_seq[pos-1] == c:
                continue

            new_seq = current_seq[:pos] + [c] + current_seq[pos:]
            if is_valid(new_seq):
                current_seq = new_seq
                available.remove(c)
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                inserted = True
                break # Restart search with new state
                
        if not inserted:
            # Try to remove a random element to explore different states
            # This allows us to potentially add a different set of numbers later
            if len(current_seq) > 2:
                idx = random.randint(0, len(current_seq) - 1)
                removed = current_seq[idx]
                current_seq = current_seq[:idx] + current_seq[idx+1:]
                if removed not in available:
                    available.append(removed)
                    available.sort()
            else:
                break # Can't remove more
                
    return best_seq
