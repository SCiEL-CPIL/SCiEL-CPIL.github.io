import time
import random
import numpy as np
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    def greedy_construct():
        """Original greedy approach, used as a baseline or starting point."""
        values = []
        products = set()
        for x in range(2, 129):
            tails = []
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            tails.append(x)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    def evaluate(seq):
        """Returns (length, num_duplicates)."""
        if not seq:
            return 0, 0
        prods = []
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.append(p)
        unique = set(prods)
        duplicates = len(prods) - len(unique)
        return n, duplicates

    def get_score(seq):
        l, d = evaluate(seq)
        if d == 0:
            return l
        else:
            return -d

    def local_search(seq, time_limit=2.0):
        """Perform local search starting from seq."""
        start = time.time()
        current_seq = seq[:]
        current_score = get_score(current_seq)
        current_dups = evaluate(current_seq)[1]
        
        # Pre-compute missing list for efficiency? It changes as we add/remove.
        # Let's just compute on the fly or maintain a set.
        current_set = set(current_seq)
        
        while time.time() < start + time_limit:
            # Generate neighbor: flip a random bit
            # Options: 
            # 1. Remove a random element from current_seq.
            # 2. Add a random element from [2,128] not in current_seq.
            
            # Bias towards adding if valid, removing if invalid?
            # If invalid, we MUST remove to become valid (or swap).
            # If valid, we try to add.
            
            if current_dups > 0:
                # Try to remove an element involved in a duplicate? 
                # Or just random removal?
                # Random removal is safer to avoid getting stuck.
                if current_seq:
                    idx = random.randint(0, len(current_seq) - 1)
                    x = current_seq[idx]
                    new_seq = current_seq[:idx] + current_seq[idx+1:]
                    new_set = set(new_seq)
                else:
                    break
            else:
                # Try to add an element
                # Efficiently get missing elements
                # Since range is small (2..128), we can just iterate
                missing = [x for x in range(2, 129) if x not in current_set]
                if not missing:
                    break
                x = random.choice(missing)
                # Insert in sorted position
                pos = bisect.bisect_left(current_seq, x)
                new_seq = current_seq[:pos] + [x] + current_seq[pos:]
                new_set = current_set | {x}
            
            new_len, new_dups = evaluate(new_seq)
            new_score = get_score(new_seq)
            
            if new_score > current_score:
                current_seq = new_seq
                current_set = new_set
                current_dups = new_dups
                current_score = new_score
            elif new_score == current_score and random.random() < 0.5:
                current_seq = new_seq
                current_set = new_set
                current_dups = new_dups
                current_score = new_score

        return current_seq

    # Initial baseline
    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    if time.time() > deadline - 5:
        return best_seq

    # Main Loop
    # We use a mix of random restarts and perturbations of the best solution.
    # The key improvement from the "inspiration" is the robust local search loop
    # and the specific scoring mechanism that prioritizes validity (0 duplicates)
    # over length, but allows hill climbing on length when valid.
    
    # To ensure we don't just repeat the same local search, we vary the start.
    
    # 1. Start with best
    # 2. Start with random
    # 3. Start with best + random perturbation
    
    while time.time() < deadline - 5:
        # Choose a starting point
        r = random.random()
        
        if r < 0.4:
            # Start from best, perturb slightly
            start_seq = best_seq[:]
            if start_seq:
                # Remove 1-2 random elements to escape local optimum
                num_removals = random.randint(1, 2)
                for _ in range(num_removals):
                    if start_seq:
                        idx = random.randint(0, len(start_seq) - 1)
                        start_seq.pop(idx)
        elif r < 0.7:
            # Start from greedy but with some random skips?
            # Actually, let's just use random init
            start_seq = []
            for x in range(2, 129):
                if random.random() < 0.5:
                    start_seq.append(x)
        else:
            # Start from empty and build up? Or just random
            start_seq = []
            for x in range(2, 129):
                if random.random() < 0.3:
                    start_seq.append(x)
        
        # Run local search
        # Allocate time based on remaining time
        time_left = deadline - time.time()
        if time_left < 3:
            break
            
        # Run for a chunk of time, e.g., 2 seconds or 10% of remaining
        chunk_time = min(2.0, time_left * 0.1)
        if chunk_time < 0.1:
            chunk_time = 0.1
            
        improved_seq = local_search(start_seq, time_limit=chunk_time)
        
        improved_len, improved_dups = evaluate(improved_seq)
        
        if improved_dups == 0 and improved_len > best_len:
            best_len = improved_len
            best_seq = improved_seq
            # If we found a new best, we can try to extend it immediately
            # by running another local search starting from it?
            # The loop will do that in the next iteration if we choose to start from best.

    return best_seq
