def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    all_nums = list(range(2, 129))
    primes = get_primes_up_to(128)
    composites = [x for x in all_nums if x not in primes]
    
    best_seq = []
    best_len = 0
    
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        current_seq = list(primes)
        current_prods = compute_products(current_seq)
        available = list(composites)
        random.shuffle(available)
        
        for x in available:
            if time.time() > deadline:
                break
            inserted = False
            positions = list(range(len(current_seq) + 1))
            random.shuffle(positions)
            
            for k in positions:
                if can_insert(current_seq, x, k, current_prods):
                    current_seq.insert(k, x)
                    current_prods = compute_products(current_seq)
                    inserted = True
                    break
            
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = list(current_seq)
            if best_len == len(all_nums):
                break

    # Final polish
    if time.time() < deadline:
        current_seq = list(best_seq)
        current_prods = compute_products(current_seq)
        # Try adding remaining numbers in ascending order
        for x in all_nums:
            if x in current_seq:
                continue
            positions = list(range(len(current_seq) + 1))
            for k in positions:
                if can_insert(current_seq, x, k, current_prods):
                    current_seq.insert(k, x)
                    current_prods = compute_products(current_seq)
                    break
        if len(current_seq) > best_len:
            best_seq = current_seq

    return best_seq
