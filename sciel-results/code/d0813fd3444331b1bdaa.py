import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Phase 1: Greedy initial construction
    chosen = []
    counts = {}
    for x in range(1, 1001):
        new_sums = [x + y for y in chosen] + [2 * x]
        valid = True
        for s in new_sums:
            if counts.get(s, 0) + 1 > 2:
                valid = False
                break
        if valid:
            chosen.append(x)
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1
    
    chosen_set = set(chosen)
    
    # Phase 2: Local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add elements
        for y in range(1, 1001):
            if y in chosen_set:
                continue
            # Check if y can be added
            if counts.get(2*y, 0) + 1 > 2:
                continue
            valid = True
            for z in chosen:
                if counts.get(y+z, 0) + 1 > 2:
                    valid = False
                    break
            if valid:
                chosen.append(y)
                chosen_set.add(y)
                counts[2*y] = counts.get(2*y, 0) + 1
                for z in chosen:
                    if z == y:
                        continue
                    s = y + z
                    counts[s] = counts.get(s, 0) + 1
                improved = True
                break
        
        if time.time() >= deadline:
            break
        
        # If no addition, try swaps
        if not improved:
            # Try a few random swaps
            for _ in range(100):
                if time.time() >= deadline:
                    break
                if not chosen:
                    break
                x = random.choice(chosen)
                y = random.randint(1, 1000)
                if y in chosen_set or y == x:
                    continue
                
                # Simulate swap
                # Remove x
                temp_counts = dict(counts)
                for z in chosen:
                    if z == x:
                        continue
                    s = x + z
                    temp_counts[s] -= 1
                    if temp_counts[s] == 0:
                        del temp_counts[s]
                temp_counts[2*x] -= 1
                if temp_counts[2*x] == 0:
                    del temp_counts[2*x]
                
                # Check if y can be added
                if temp_counts.get(2*y, 0) + 1 > 2:
                    continue
                valid = True
                for z in chosen:
                    if z == x:
                        continue
                    if temp_counts.get(y+z, 0) + 1 > 2:
                        valid = False
                        break
                if valid:
                    # Accept swap
                    # Remove x
                    for z in chosen:
                        if z == x:
                            continue
                        s = x + z
                        counts[s] -= 1
                        if counts[s] == 0:
                            del counts[s]
                    counts[2*x] -= 1
                    if counts[2*x] == 0:
                        del counts[2*x]
                    chosen.remove(x)
                    chosen_set.remove(x)
                    
                    # Add y
                    chosen.append(y)
                    chosen_set.add(y)
                    counts[2*y] = counts.get(2*y, 0) + 1
                    for z in chosen:
                        if z == y:
                            continue
                        s = y + z
                        counts[s] = counts.get(s, 0) + 1
                    improved = True
                    break
                    
    return chosen
