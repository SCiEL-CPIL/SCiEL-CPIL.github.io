    base_order = np.arange(SIZE)
    while time.time() < deadline - 2:
        order = base_order.copy()
        rng.shuffle(order)
        code = expand_code(base_code, order)
        if len(code) > len(best_code):
            best_code = code
