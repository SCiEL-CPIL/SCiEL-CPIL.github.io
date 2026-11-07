import time

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    n = 128
    S = set(range(65, 129))
    
    def can_add(x, S):
        if x in S:
            return False
        for y in S:
            xy = x + y
            if xy not in S:
                continue
            for z in S:
                if z == y:
                    continue
                xz = x + z
                if xz not in S:
                    continue
                yz = y + z
                if yz in S or yz == x:
                    return False
        return True
    
    # Multiple greedy passes with different orders
    orders = [
        list(range(64, 0, -1)),
        list(range(1, 65)),
        list(range(32, 65)) + list(range(1, 33)),
        list(range(48, 65)) + list(range(1, 49)),
    ]
    
    best_S = set(S)
    best_size = len(S)
    
    for order in orders:
        if time.time() > deadline - 30:
            break
        trial = set(S)
        for x in order:
            if time.time() > deadline - 30:
                break
            if x not in trial and can_add(x, trial):
                trial.add(x)
        if len(trial) > best_size:
            best_size = len(trial)
            best_S = set(trial)
    
    S = best_S
    
    # Local search: remove-one-add-two
    def try_add_two(S):
        added = []
        for y in range(1, 129):
            if time.time() > deadline - 10:
                break
            if y in S or y in added:
                continue
            if can_add(y, S | set(added)):
                added.append(y)
                if len(added) == 2:
                    break
        return added
    
    for iteration in range(30):
        if time.time() > deadline - 15:
            break
        improved = False
        S_list = sorted(S)
        for x in S_list:
            if time.time() > deadline - 15:
                break
            S.remove(x)
            added = try_add_two(S)
            if len(added) == 2:
                S.update(added)
                improved = True
            else:
                S.add(x)
                for a in added:
                    S.discard(a)
        if not improved:
            break
    
    # Final greedy pass
    for x in range(64, 0, -1):
        if time.time() > deadline - 2:
            break
        if x not in S and can_add(x, S):
            S.add(x)
    
    result = [int(x) for x in S]
    return result
