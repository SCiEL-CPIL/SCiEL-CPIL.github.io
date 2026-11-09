import time

def construct():
    start_time = time.time()
    LIMIT = 1009
    MAX_VAL = 1008

    # Precompute modular inverses
    inv = [0] * (LIMIT + 1)
    for i in range(1, LIMIT):
        inv[i] = pow(i, -1, LIMIT)

    best_len = 0
    best_start = 2

    # The interval must be within [2, 1007] because 1 and 1008 are self-inverse
    # 1*1 = 1, 1008*1008 = 1.

    for s in range(2, 1008):
        current_set = set()
        conflict_at = -1
        for e in range(s, 1008):
            e_inv = inv[e]
            if e_inv in current_set:
                conflict_at = e
                break
            current_set.add(e)

        if conflict_at != -1:
            length = (conflict_at - 1) - s + 1
        else:
            length = 1007 - s + 1

        if length > best_len:
            best_len = length
            best_start = s

        # Time check
        if time.time() - start_time > 80:
            break

    return [best_start, best_len]
