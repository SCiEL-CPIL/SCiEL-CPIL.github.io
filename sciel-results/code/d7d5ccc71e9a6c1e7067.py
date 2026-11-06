import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    def get_products(seq):
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def is_valid(seq):
        # Check if all consecutive block products are distinct
        # Since we only insert into valid sequences, we can optimize, 
        # but for simplicity and correctness with small L, recompute.
        prods = get_products(seq)
        # If there are duplicates, len(prods) < number of blocks
        n = len(seq)
        num_blocks = n * (n + 1) // 2
        return len(prods) == num_blocks

    def greedy_sorted():
        values, products = [], set()
        for x in range(2, 129):
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    def polish(seq):
        # Try to insert any missing number at any valid position
        changed = True
        while changed:
            changed = False
            for x in range(2, 129):
                if x in seq:
                    continue
                # Find possible insertion positions
                # seq is sorted
                # x can be at index k if (k==0 or seq[k-1] < x) and (k==len(seq) or x < seq[k])
                # Since seq is sorted, there is only ONE valid position for x to maintain sorted order.
                # Wait, the problem says "strictly increasing integers". So the order is fixed by values.
                # So we only need to check the single position where x fits in the sorted order.
                # If we insert x, the sequence remains sorted.
                
                # Find insertion index
                lo, hi = 0, len(seq)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if seq[mid] < x:
                        lo = mid + 1
                    else:
                        hi = mid
                k = lo
                
                # Check if inserting x at k is valid
                new_seq = seq[:k] + [x] + seq[k:]
                if is_valid(new_seq):
                    seq = new_seq
                    changed = True
                    break # Restart the loop with the new seq
        return seq

    # Initial best from deterministic greedy
    best_seq = greedy_sorted()
    best_len = len(best_seq)
    
    # Randomized search
    while time.time() < deadline:
        # 1. Randomized Greedy Construction
        candidates = list(range(2, 129))
        random.shuffle(candidates)
        
        current = []
        for x in candidates:
            # Find insertion index for x in current (sorted)
            lo, hi = 0, len(current)
            while lo < hi:
                mid = (lo + hi) // 2
                if current[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            k = lo
            new_seq = current[:k] + [x] + current[k:]
            if is_valid(new_seq):
                current = new_seq
        
        # 2. Polish
        current = polish(current)
        
        if len(current) > best_len:
            best_len = len(current)
            best_seq = current
            
            # Optional: if we hit a high length, we can tighten the search or just keep going
            # The task is to maximize length.
            
    return best_seq
