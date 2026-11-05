<<<<<<< SEARCH
    # Main Loop
    no_improvement_count = 0
    max_no_improvement = 20
    random.seed(42)
    
    while time.time() < deadline - 2:  # Safety margin
        # 1. Exhaustive Removal - remove all redundant codewords
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for c in list(code):
                if is_redundant(c):
                    apply_code_change([c], [])
                    changed = True
                    if time.time() > deadline:
                        break
        
        # 2. Kicks - try multiple attempts per iteration for better mixing
        kick_success = False
        for _ in range(8):  # Try up to 8 kicks/swaps per main loop iteration
            if time.time() > deadline:
                break
            
            # Try 1-removal first (most likely to succeed)
            if try_kick(1):
                kick_success = True
                break
            # Try 2-removal
            elif try_kick(2):
                kick_success = True
                break
            # Try 3-removal
            elif try_kick(3):
                kick_success = True
                break
            # Try swap
            elif try_swap():
                kick_success = True
                break
        
        if kick_success:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Perturb the code slightly to escape local minimum
                # Try a random kick with larger k
                if len(code) > 5:
                    k = random.randint(4, min(6, len(code)))
                    if try_kick(k):
                        no_improvement_count = 0
                else:
                    # If we're stuck, try a more aggressive restart
                    if len(code) > 10:
                        if try_kick(3):
                            no_improvement_count = 0
=======
    # Main Loop
    no_improvement_count = 0
    max_no_improvement = 25
    random.seed(42)
    
    while time.time() < deadline - 2:  # Safety margin
        # 1. Exhaustive Removal - remove all redundant codewords
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for c in list(code):
                if is_redundant(c):
                    apply_code_change([c], [])
                    changed = True
                    if time.time() > deadline:
                        break
        
        # 2. Kicks - try multiple attempts per iteration for better mixing
        kick_success = False
        for _ in range(10):  # Try up to 10 kicks/swaps per main loop iteration
            if time.time() > deadline:
                break
            
            # Try 1-removal first (most likely to succeed)
            if try_kick(1):
                kick_success = True
                break
            # Try 2-removal
            elif try_kick(2):
                kick_success = True
                break
            # Try 3-removal
            elif try_kick(3):
                kick_success = True
                break
            # Try swap
            elif try_swap():
                kick_success = True
                break
        
        if kick_success:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Perturb the code slightly to escape local minimum
                # Try a random kick with larger k
                if len(code) > 5:
                    k = random.randint(4, min(7, len(code)))
                    if try_kick(k):
                        no_improvement_count = 0
                else:
                    # If we're stuck, try a more aggressive restart
                    if len(code) > 10:
                        if try_kick(3):
                            no_improvement_count = 0
>>>>>>> REPLACE

