import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # State: (last_val, sums_frozenset, suffix_sums_frozenset, sequence_tuple)
    # R contains 0 to generate the single-element block sum x
    init_R = frozenset([0, 1])
    init_sums = frozenset([1])
    beam = [(1, init_sums, init_R, (1,))]
    best_seq = (1,)
    
    # Beam width
    K = 1000
    
    while beam:
        if time.time() > deadline:
            break
            
        next_beam = []
        
        for last, sums, R, seq in beam:
            # Try next values
            # To find longest, we try all valid x
            for x in range(last + 1, 257):
                # Check collisions
                valid = True
                new_sums_set = set()
                # R is small (size n+1)
                for r in R:
                    s = x + r
                    if s in sums:
                        valid = False
                        break
                    new_sums_set.add(s)
                
                if not valid:
                    continue
                
                new_sums = sums | frozenset(new_sums_set)
                new_R = frozenset(x + r for r in R)
                new_seq = seq + (x,)
                
                next_beam.append((x, new_sums, new_R, new_seq))
                
                # Time check
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
        
        if not next_beam:
            break
            
        # Update best
        for _, _, _, seq in next_beam:
            if len(seq) > len(best_seq):
                best_seq = seq
        
        # Select best states
        # Heuristic: Smaller last_val allows more future steps
        next_beam.sort(key=lambda s: s[0])
        beam = next_beam[:K]
        
    return list(best_seq)
