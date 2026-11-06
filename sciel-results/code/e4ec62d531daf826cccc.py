import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    # State: (sequence_tuple, frozenset_of_products, tuple_of_suffix_products)
    # Using tuples for sequence and suffixes for immutability and faster hashing/copying
    # frozenset for products
    
    beam = [( (), frozenset(), (1,) )]
    best_seq = ()
    best_len = 0
    
    BEAM_WIDTH = 50  # Reduced from 500 to ensure speed
    
    # Precompute: for each number, which other numbers share a common factor > 1
    # This helps with fast collision detection
    common_factors = {}
    for i in range(2, 129):
        common_factors[i] = set()
    
    for i in range(2, 129):
        for j in range(i+1, 129):
            if gcd(i, j) > 1:
                common_factors[i].add(j)
                common_factors[j].add(i)
    
    while True:
        if time.time() > deadline:
            break
            
        candidates = []
        
        for seq, prods, suffixes in beam:
            last = seq[-1] if seq else 1
            seq_len = len(seq)
            
            # Pruning: if even adding all possible numbers can't beat best, skip
            if seq_len + (128 - last) <= best_len:
                continue
                
            # Generate extensions
            for x in range(last + 1, 129):
                # Quick check: if x shares a factor with any suffix product, likely collision
                # New products: { x * s for s in suffixes }
                valid = True
                new_prods_set = set(prods)
                new_suffixes_list = []
                
                for s in suffixes:
                    p = x * s
                    if p in new_prods_set:
                        valid = False
                        break
                    new_prods_set.add(p)
                    new_suffixes_list.append(p)
                
                if not valid:
                    continue
                
                new_seq = seq + (x,)
                new_prods = frozenset(new_prods_set)
                new_suffixes = (1,) + tuple(new_suffixes_list)
                
                candidates.append((new_seq, new_prods, new_suffixes))
        
        if not candidates:
            break
            
        # Update best
        for c in candidates:
            if len(c[0]) > best_len:
                best_seq = c[0]
                best_len = len(c[0])
                
        # Sort candidates
        # Primary: length descending
        # Secondary: last element ascending (smaller last element allows more future choices)
        # Tertiary: fewer suffix products (less collision risk)
        candidates.sort(key=lambda item: (-len(item[0]), item[0][-1], len(item[2])))
        
        # Dynamic beam width: use wider beam early, narrower later
        if best_len < 20:
            current_beam_width = BEAM_WIDTH
        elif best_len < 30:
            current_beam_width = BEAM_WIDTH // 2
        else:
            current_beam_width = max(10, BEAM_WIDTH // 4)
        
        beam = candidates[:current_beam_width]
        
    return list(best_seq)
