def _greedy_prune(comparators, start_time, deadline):
    current = list(comparators)
    i = 0
    while i < len(current):
        if time.time() > deadline:
            break
        candidate = current[:i] + current[i+1:]
        if _verify_network_numpy(candidate):
            current = candidate
        else:
            i += 1
    return current
