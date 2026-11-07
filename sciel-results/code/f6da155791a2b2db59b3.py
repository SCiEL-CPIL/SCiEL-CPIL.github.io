<<<<<<< SEARCH
    # Phase 3: Local Search
    # Try swaps
    last_improvement = time.time()
    
    while time.time() - start_time < timeout - 2.0:
        if time.time() - last_improvement > 5.0:
            # Stagnation? Maybe restart or just continue
            # Let's just continue, maybe pick better candidates
            pass
            
        # Pick random x not in S
        x = random.randint(1, N)
        if x in S:
            continue
            
        if is_valid_addition(S, x):
            S.add(x)
            last_improvement = time.time()
        else:
            # Try swap
            # Pick random y in S
            y = random.choice(list(S))
            S.remove(y)
            if is_valid_addition(S, x):
                S.add(x)
                last_improvement = time.time()
            else:
                S.add(y)
                
    # Ensure types are correct
    return sorted(list(S))
=======
    # Phase 3: Local Search
    # Try swaps
    last_improvement = time.time()
    
    while time.time() - start_time < timeout - 3.0:
        if time.time() - last_improvement > 5.0:
            # Stagnation? Try more aggressive moves
            pass
            
        # Pick random x not in S
        x = random.randint(1, N)
        if x in S:
            continue
            
        if is_valid_addition(S, x):
            S.add(x)
            last_improvement = time.time()
        else:
            # Try swap: remove 1, add 1
            y = random.choice(list(S))
            S.remove(y)
            if is_valid_addition(S, x):
                S.add(x)
                last_improvement = time.time()
            else:
                S.add(y)
                
                # Try multi-swap: remove 2, add 1
                if len(S) >= 3:
                    y1, y2 = random.sample(list(S), 2)
                    S.remove(y1)
                    S.remove(y2)
                    if is_valid_addition(S, x):
                        S.add(x)
                        last_improvement = time.time()
                    else:
                        S.add(y1)
                        S.add(y2)
                        
    # Phase 4: Final exhaustive polish
    # Try adding every element from the complement
    complement = [x for x in range(1, N + 1) if x not in S]
    for x in complement:
        if time.time() - start_time > timeout - 1.0:
            break
        if is_valid_addition(S, x):
            S.add(x)
            
    # Ensure types are correct
    return sorted(list(S))
>>>>>>> REPLACE
