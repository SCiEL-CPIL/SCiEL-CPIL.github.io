import random
import time

def check_valid(seq):
    """
    Check if all non-empty consecutive block products in the strictly increasing
    sequence 'seq' are distinct.
    """
    prods = set()
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            if p in prods:
                return False
            prods.add(p)
    return True

def greedy_construct(order):
    """
    Construct a sequence by iterating through 'order' and adding elements
    if they don't create duplicate block products.
    """
    values = []
    products = set()
    for x in order:
        tails = []
        prod = x
        for a in reversed(values):
            tails.append(prod)
            prod *= a
        
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values

def local_search(seq, time_limit):
    """
    Local search using Delete + Greedy Refill strategy.
    """
    if time_limit <= 0:
        return list(seq)
        
    current = list(seq)
    best = list(current)
    best_len = len(current)
    
    all_nums = set(range(2, 129))
    start_time = time.time()
    
    # Helper to sort and check
    # We can optimize by caching the sorted list if the set doesn't change much,
    # but sets are cheap to sort for N~40.
    
    while time.time() - start_time < time_limit:
        if not current:
            break
            
        # 1. Pick a random element to remove
        idx = random.randint(0, len(current) - 1)
        removed = current[idx]
        
        # 2. Create a candidate set without this element
        candidate_set = set(current)
        candidate_set.remove(removed)
        
        # 3. Refill: Try to add elements from the unused pool (which includes 'removed')
        #    in a random order to escape local optima.
        unused_pool = list(all_nums - candidate_set)
        random.shuffle(unused_pool)
        
        added_any = False
        for x in unused_pool:
            if x in candidate_set:
                continue
            
            # Check if adding x is valid
            new_set = candidate_set | {x}
            new_seq = sorted(new_set)
            
            if check_valid(new_seq):
                candidate_set = new_set
                added_any = True
                
        # 4. Acceptance Criteria
        cand_len = len(candidate_set)
        
        if cand_len > best_len:
            best = sorted(candidate_set)
            best_len = cand_len
            current = best
        elif cand_len == len(current) and random.random() < 0.1:
            # Accept same length to explore
            current = sorted(candidate_set)
        elif cand_len > len(current) and random.random() < 0.5:
            # This case is covered by cand_len > best_len usually, 
            # but if best_len is same as current_len, and we improved current but not best?
            # No, if cand_len > current_len, it must be > best_len or == best_len.
            # If cand_len > current_len, we should accept it as current at least.
            pass # Handled above if it beats best. If it doesn't beat best, but beats current?
            # Wait, if best_len is 10, current_len is 9, cand_len is 9.
            # We don't update current.
            # If cand_len is 10, it beats best (or ties).
            
            # Let's add a clause: if cand_len > len(current), accept as current even if not best?
            # Yes, to climb.
            if cand_len > len(current):
                 current = sorted(candidate_set)

    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s safety margin
    
    # 1. Generate initial candidates
    candidates = []
    
    # Standard Greedy
    candidates.append(greedy_construct(range(2, 129)))
    
    # Reverse Greedy
    candidates.append(greedy_construct(range(128, 1, -1)))
    
    # Randomized Greedy
    for _ in range(20):
        order = list(range(2, 129))
        random.shuffle(order)
        candidates.append(greedy_construct(order))
        
    # Select best initial
    best_init = max(candidates, key=len)
    
    # 2. Local Search
    # Use remaining time
    remaining = deadline - time.time()
    if remaining > 0:
        best_seq = local_search(best_init, remaining)
    else:
        best_seq = best_init
        
    return best_seq
