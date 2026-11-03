    seed = 0
    while time.time() < deadline:
        seed_start = time.time()
        result = do_search(seed)
        # do_search checks deadline, so it will return None if time is up.
        # But if it returns a result, it was fast.
        # If it returns None, it might have used up time or failed.
        if time.time() - seed_start > 10.0:
             # If a seed takes too long, we might want to reduce candidate_window?
             # Or just accept it.
             pass
        seed += 1
