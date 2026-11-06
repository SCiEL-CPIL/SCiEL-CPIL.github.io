import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Greedy Construction
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    
    best_seq = values[:]
    best_len = len(best_seq)
    
    if best_len == 127:
        return best_seq

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
        return len(get_products(seq)) == len(seq) * (len(seq) + 1) // 2

    # 2. Local Search to reach 127
    # We know max length is 127.
    # Current length is 126.
    # Unused set has 1 element.
    
    while time.time() < deadline:
        current_seq = best_seq[:]
        unused = set(range(2, 129)) - set(current_seq)
        
        # If we have more than 1 unused, try direct insertion first
        if len(unused) > 1:
            improved = False
            for u in list(unused):
                for i in range(len(current_seq) + 1):
                    new_seq = current_seq[:i] + [u] + current_seq[i:]
                    if is_valid(new_seq):
                        best_seq = new_seq
                        best_len += 1
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
        
        # If stuck with 1 unused (or 0, but 0 means done), try 2-for-1 exchange
        # This increases length by 1 by removing 1 and adding 2.
        # This is only possible if we have at least 1 unused.
        if len(unused) < 1:
            break # Should not happen if best_len < 127
        
        # Pick a random element to remove
        # To be efficient, we can try removing elements that are "heavy" or random
        # Let's try random indices
        removed_elem = None
        removed_idx = None
        base_seq = None
        new_unused = None
        
        # Try a few random removals
        for _ in range(10): # Try 10 random removals per outer loop
            idx = random.randint(0, len(best_seq) - 1)
            elem = best_seq[idx]
            base_seq = best_seq[:idx] + best_seq[idx+1:]
            new_unused = set(unused)
            new_unused.add(elem)
            
            if len(new_unused) != 2:
                continue
                
            # Try to insert the 2 elements from new_unused into base_seq
            a, b = list(new_unused)
            # Orders: (a,b) and (b,a)
            # Positions: 0 to len(base_seq)
            # We need to insert two items.
            # This is equivalent to choosing 2 positions in the final sequence of length len(base_seq)+2
            # Or choosing position for a, then position for b in the updated sequence.
            
            found = False
            # Randomly sample insertion strategies
            for _ in range(500): # 500 random attempts per removal
                # Pick position for first insert
                p1 = random.randint(0, len(base_seq))
                # Pick position for second insert
                # If we insert a at p1, the sequence length increases by 1.
                # Then we insert b at p2 in the new sequence.
                # p2 can be 0 to len(base_seq)
                # Note: if p2 > p1, it's effectively p2 in original + 1?
                # Let's just construct it.
                
                # Try order (a, b)
                seq_a = base_seq[:]
                seq_a.insert(p1, a)
                p2 = random.randint(0, len(seq_a))
                seq_ab = seq_a[:]
                seq_ab.insert(p2, b)
                
                if is_valid(seq_ab):
                    best_seq = seq_ab
                    best_len += 1
                    found = True
                    break
                
                # Try order (b, a)
                seq_b = base_seq[:]
                seq_b.insert(p1, b)
                p2 = random.randint(0, len(seq_b))
                seq_ba = seq_b[:]
                seq_ba.insert(p2, a)
                
                if is_valid(seq_ba):
                    best_seq = seq_ba
                    best_len += 1
                    found = True
                    break
            
            if found:
                break
        
        if best_len == 127:
            break

    return best_seq
