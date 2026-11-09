<<<<<<< SEARCH
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115

    if n == 0:
        return []

    orders = _orders(boxes, n)
    best_result = None
    best_vol = -1

    # Greedy with different orderings
    for order in orders:
        if time.time() > deadline - 25:
            break
        res = _greedy(order, boxes, container, deadline)
        v = _vol(res)
        if v > best_vol:
            best_vol = v
            best_result = res

    # A few random restarts
    if time.time() < deadline - 12 and best_result is not None:
        rng_r = random.Random(99)
        for _ in range(5):
            if time.time() > deadline - 8:
                break
            ro = list(range(n))
            rng_r.shuffle(ro)
            res = _greedy(ro, boxes, container, deadline)
            v = _vol(res)
            if v > best_vol:
                best_vol = v
                best_result = res

    # Local search
    remaining = deadline - time.time()
    if remaining > 4 and best_result is not None:
        mi = max(200, int(remaining * 50))
        mi = min(mi, 3000)
        best_result = _local_search(best_result, boxes, container, deadline, max_iters=mi)

    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return [tuple(r) for r in best_result]
=======
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115

    if n == 0:
        return []

    orders = _orders(boxes, n)
    best_result = None
    best_vol = -1

    # Greedy with different orderings
    for order in orders:
        if time.time() > deadline - 30:
            break
        res = _greedy(order, boxes, container, deadline)
        v = _vol(res)
        if v > best_vol:
            best_vol = v
            best_result = res

    # A few random restarts
    if time.time() < deadline - 15 and best_result is not None:
        rng_r = random.Random(99)
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            ro = list(range(n))
            rng_r.shuffle(ro)
            res = _greedy(ro, boxes, container, deadline)
            v = _vol(res)
            if v > best_vol:
                best_vol = v
                best_result = res

    # Local search
    remaining = deadline - time.time()
    if remaining > 5 and best_result is not None:
        mi = max(200, int(remaining * 60))
        mi = min(mi, 4000)
        best_result = _local_search(best_result, boxes, container, deadline, max_iters=mi)

    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return [tuple(r) for r in best_result]
>>>>>>> REPLACE
