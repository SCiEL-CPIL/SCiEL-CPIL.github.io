import time
import random

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    best_size = 0
    best_set = []
    
    N = 1000
    max_sum = 2 * N
    
    def try_construction(ordering):
        candidates = list(range(1, N + 1))
        if ordering == 'asc':
            pass
        elif ordering == 'desc':
            candidates.reverse()
        else:
            random.shuffle(candidates)
            
        chosen = []
        counts = [0] * (max_sum + 1)
        
        for x in candidates:
            valid = True
            if counts[2 * x] >= 2:
                valid = False
            else:
                for y in chosen:
                    if counts[x + y] >= 2:
                        valid = False
                        break
            
            if valid:
                chosen.append(x)
                counts[2 * x] += 1
                for y in chosen[:-1]:
                    counts[x + y] += 1
        
        return chosen, counts

    def local_search(chosen, counts):
        chosen_set = set(chosen)
        missing = [x for x in range(1, N + 1) if x not in chosen_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Randomize order of missing to avoid bias
            random.shuffle(missing)
            
            new_missing = []
            for x in missing:
                valid = True
                if counts[2 * x] >= 2:
                    valid = False
                else:
                    for y in chosen:
                        if counts[x + y] >= 2:
                            valid = False
                            break
                
                if valid:
                    chosen.append(x)
                    chosen_set.add(x)
                    counts[2 * x] += 1
                    for y in chosen[:-1]:
                        counts[x + y] += 1
                    improved = True
                else:
                    new_missing.append(x)
            
            missing = new_missing
        
        return chosen

    # Run multiple restarts
    while time.time() < deadline:
        # Choose ordering strategy
        r = random.random()
        if r < 0.2:
            order = 'asc'
        elif r < 0.4:
            order = 'desc'
        else:
            order = 'rand'
            
        chosen, counts = try_construction(order)
        
        # Perform local search to fill in gaps
        chosen = local_search(chosen, counts)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen[:]
            
        # If we are close to the deadline, do a quick check
        if time.time() > deadline - 5.0:
            break

    return best_set
