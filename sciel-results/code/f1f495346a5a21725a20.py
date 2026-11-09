<<<<<<< SEARCH
def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    W, H, D = container
    n = len(boxes)
    
    vols = [w * h * d for w, h, d in boxes]
    
    # Generate diverse orders
    orders = []
    
    # By volume descending
    order1 = list(range(n))
    order1.sort(key=lambda i: vols[i], reverse=True)
    orders.append(order1)
    
    # By min dimension descending
    order2 = list(range(n))
    order2.sort(key=lambda i: min(boxes[i]), reverse=True)
    orders.append(order2)
    
    # By max dimension descending
    order3 = list(range(n))
    order3.sort(key=lambda i: max(boxes[i]), reverse=True)
    orders.append(order3)
    
    # By aspect ratio (flat boxes first)
    order4 = list(range(n))
    order4.sort(key=lambda i: max(boxes[i]) / max(min(boxes[i]), 1e-9))
    orders.append(order4)
    
    # Random orders
    rng = random.Random(42)
    for r in range(5):
        ro = list(range(n))
        rng.shuffle(ro)
        orders.append(ro)
    
    best_placements = None
    best_vol = 0.0
    
    strategies = [('extreme', 'compact'), ('extreme', 'low'), ('faces', 'compact'), ('faces', 'low')]
    
    for order in orders:
        for pos_s, score_s in strategies:
            if time.time() > deadline:
                break
            placements, vol = _greedy(boxes, container, order, deadline, pos_s, score_s)
            if vol > best_vol:
                best_vol = vol
                best_placements = placements
    
    # Local search on best
    if best_placements is not None and time.time() < deadline:
        rng2 = random.Random(123)
        best_placements, best_vol = _local_search(best_placements, boxes, container, deadline, rng2)
    
    # Additional random restarts if time permits
    if time.time() < deadline - 20:
        rng3 = random.Random(777)
        base_order = list(range(n))
        base_order.sort(key=lambda i: vols[i], reverse=True)
        
        num_restarts = 50
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            order = base_order[:]
            num_swaps = rng3.randint(1, min(5, n))
            for _ in range(num_swaps):
                a, b = rng3.randrange(n), rng3.randrange(n)
                if a != b:
                    order[a], order[b] = order[b], order[a]
            
            pos_s = rng3.choice(['extreme', 'faces'])
            score_s = rng3.choice(['compact', 'low'])
            placements, vol = _greedy(boxes, container, order, deadline, pos_s, score_s)
            if vol > best_vol:
                best_vol = vol
                best_placements = placements
        
        # Final local search
        if best_placements is not None and time.time() < deadline - 2:
            rng4 = random.Random(999)
            best_placements, best_vol = _local_search(best_placements, boxes, container, deadline, rng4)
    
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
=======
def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    W, H, D = container
    n = len(boxes)
    
    vols = [w * h * d for w, h, d in boxes]
    
    # Generate diverse orders
    orders = []
    
    # By volume descending
    order1 = list(range(n))
    order1.sort(key=lambda i: vols[i], reverse=True)
    orders.append(order1)
    
    # By min dimension descending
    order2 = list(range(n))
    order2.sort(key=lambda i: min(boxes[i]), reverse=True)
    orders.append(order2)
    
    # By max dimension descending
    order3 = list(range(n))
    order3.sort(key=lambda i: max(boxes[i]), reverse=True)
    orders.append(order3)
    
    # By aspect ratio (flat boxes first)
    order4 = list(range(n))
    order4.sort(key=lambda i: max(boxes[i]) / max(min(boxes[i]), 1e-9))
    orders.append(order4)
    
    # By volume ascending (small boxes first to fill gaps)
    order5 = list(range(n))
    order5.sort(key=lambda i: vols[i])
    orders.append(order5)
    
    # Random orders
    rng = random.Random(42)
    for r in range(5):
        ro = list(range(n))
        rng.shuffle(ro)
        orders.append(ro)
    
    best_placements = None
    best_vol = 0.0
    
    strategies = [('extreme', 'compact'), ('extreme', 'low'), ('faces', 'compact'), ('faces', 'low'),
                  ('extreme', 'balanced'), ('faces', 'balanced')]
    
    for order in orders:
        for pos_s, score_s in strategies:
            if time.time() > deadline:
                break
            placements, vol = _greedy(boxes, container, order, deadline, pos_s, score_s)
            if vol > best_vol:
                best_vol = vol
                best_placements = placements
    
    # Local search on best
    if best_placements is not None and time.time() < deadline:
        rng2 = random.Random(123)
        best_placements, best_vol = _local_search(best_placements, boxes, container, deadline, rng2, max_iters=80)
    
    # Additional random restarts if time permits
    if time.time() < deadline - 20:
        rng3 = random.Random(777)
        base_order = list(range(n))
        base_order.sort(key=lambda i: vols[i], reverse=True)
        
        num_restarts = 50
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            order = base_order[:]
            num_swaps = rng3.randint(1, min(5, n))
            for _ in range(num_swaps):
                a, b = rng3.randrange(n), rng3.randrange(n)
                if a != b:
                    order[a], order[b] = order[b], order[a]
            
            pos_s = rng3.choice(['extreme', 'faces'])
            score_s = rng3.choice(['compact', 'low', 'balanced'])
            placements, vol = _greedy(boxes, container, order, deadline, pos_s, score_s)
            if vol > best_vol:
                best_vol = vol
                best_placements = placements
        
        # Final local search
        if best_placements is not None and time.time() < deadline - 2:
            rng4 = random.Random(999)
            best_placements, best_vol = _local_search(best_placements, boxes, container, deadline, rng4, max_iters=100)
    
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
>>>>>>> REPLACE
