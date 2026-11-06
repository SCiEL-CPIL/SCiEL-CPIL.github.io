import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin
    
    # State: (last_value, products_set, sequence_list)
    beam = [(1, set(), [])]
    width = 2000
    best_seq = []
    
    while beam and time.time() < deadline:
        next_beam = []
        
        for last_val, products, seq in beam:
            # Optimization: If seq is already longer than best, we can update best here?
            # No, we update when we create new_seq.
            
            for x in range(last_val + 1, 129):
                # Calculate new suffix products
                new_prods = [x]
                p = x
                for a in reversed(seq):
                    p *= a
                    new_prods.append(p)
                
                # Check for conflicts with existing products
                if not products.intersection(new_prods):
                    new_products = products.union(new_prods)
                    new_seq = seq + [x]
                    
                    # Update global best
                    if len(new_seq) > len(best_seq):
                        best_seq = new_seq
                        
                    next_beam.append((x, new_products, new_seq))
        
        if not next_beam:
            break
            
        # Sort by last_value to keep the "tightest" sequences
        # This favors sequences that leave more room for future numbers
        next_beam.sort(key=lambda s: s[0])
        
        # Trim to width
        beam = next_beam[:width]
        
    return best_seq
