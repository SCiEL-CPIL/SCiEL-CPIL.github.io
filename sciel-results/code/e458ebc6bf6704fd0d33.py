import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    def get_primes(n):
        sieve = [True] * (n+1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5)+1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]
    
    def is_valid(seq):
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True

    primes = get_primes(128)
    all_nums = list(range(2, 129))
    composites = [x for x in all_nums if x not in primes]
    
    best_seq = primes[:]
    best_len = len(primes)
    
    # Greedy Add
    def greedy_add(seq):
        improved = True
        while improved:
            improved = False
            # Try adding composites
            # Sort composites by value? Or random?
            # Let's try smallest first to keep products small
            for c in sorted(composites):
                if c in seq:
                    continue
                new_seq = sorted(seq + [c])
                if is_valid(new_seq):
                    seq = new_seq
                    improved = True
                    break
        return seq

    # Local Search: Swap + Add
    def local_search(seq):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try all swaps
            seq_set = set(seq)
            candidates = [c for c in composites if c not in seq_set]
            for a in seq:
                for c in candidates:
                    if a == c: continue
                    new_seq = sorted((seq_set - {a}) | {c})
                    if not is_valid(new_seq):
                        continue
                    
                    # Lookahead: Can we add anything to new_seq?
                    new_seq_set = set(new_seq)
                    add_found = False
                    for d in candidates:
                        if d in new_seq_set: continue
                        # Also check composites not in candidates? No, candidates is all composites not in seq.
                        # But d might be a? No, d is composite.
                        # We need to check all numbers not in new_seq
                        # This is expensive.
                        # Optimization: Just check a few?
                        # Let's check if ANY number can be added.
                        # To speed up, we can check a subset.
                        pass
                    
                    # This lookahead is too expensive.
                    # Alternative: Just accept the swap if it's valid, and rely on the next iteration of the outer loop to try adding.
                    # But if we accept a swap that doesn't help, we might cycle.
                    # We need to track visited states or use a time limit.
                    
                    # Let's try a different approach:
                    # 1. Try to add. If stuck, 
                    # 2. Try to swap. If swap is valid, do it.
                    # 3. Try to add again.
                    # 4. If still stuck, try another swap.
                    # To avoid cycles, we can keep a history of states.
                    
                    # Let's implement a simple hill climbing with random restarts.
                    break
                if improved: break
        return seq

    # Main Loop
    current_seq = primes[:]
    current_seq = greedy_add(current_seq)
    if len(current_seq) > best_len:
        best_seq = current_seq
        best_len = len(current_seq)
        
    while time.time() < deadline:
        # Try to improve current_seq
        improved = False
        
        # 1. Try Greedy Add
        new_seq = greedy_add(current_seq)
        if len(new_seq) > len(current_seq):
            current_seq = new_seq
            if len(current_seq) > best_len:
                best_seq = current_seq
                best_len = len(current_seq)
            improved = True
        
        if not improved:
            # 2. Try Random Swap
            seq_set = set(current_seq)
            candidates = [c for c in composites if c not in seq_set]
            if candidates:
                # Pick random a and c
                a = random.choice(current_seq)
                c = random.choice(candidates)
                if a != c:
                    new_seq = sorted((seq_set - {a}) | {c})
                    if is_valid(new_seq):
                        # Accept swap
                        current_seq = new_seq
                        improved = True
        
        if not improved:
            # 3. Try Random Remove (Kick)
            if len(current_seq) > 2:
                a = random.choice(current_seq)
                new_seq = sorted(set(current_seq) - {a})
                if is_valid(new_seq):
                    current_seq = new_seq
                    improved = True
        
        if not improved:
            # 4. Restart with a different random greedy
            # Shuffle composites
            random.shuffle(composites)
            current_seq = primes[:]
            # Run greedy with shuffled composites
            improved_greedy = True
            while improved_greedy:
                improved_greedy = False
                for c in composites:
                    if c in current_seq: continue
                    new_seq = sorted(current_seq + [c])
                    if is_valid(new_seq):
                        current_seq = new_seq
                        improved_greedy = True
                        break
            
            if len(current_seq) > best_len:
                best_seq = current_seq
                best_len = len(current_seq)
                
    return best_seq
