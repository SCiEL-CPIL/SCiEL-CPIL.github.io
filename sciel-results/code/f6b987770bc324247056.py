import time
import random

def construct():
    deadline = time.time() + 85
    
    best_size = 0
    best_set = []
    
    candidates = list(range(1, 257))
    
    orderings = []
    orderings.append(list(range(1, 257)))
    orderings.append(list(range(256, 0, -1)))
    orderings.append([x for x in range(1, 257, 2)] + [x for x in range(2, 257, 2)])
    orderings.append([x for x in range(2, 257, 2)] + [x for x in range(1, 257, 2)])
    alt = []
    for i in range(128):
        alt.append(1 + i)
        alt.append(129 + i)
    orderings.append(alt)
    
    for _ in range(50):
        o = candidates[:]
        random.shuffle(o)
        orderings.append(o)
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        current_set = []
        max_possible_sum = 60 * 256  # 15360
        # sum_to_card[s] = cardinality, or -1 if not achieved
        sum_to_card = [-1] * (max_possible_sum + 1)
        sum_to_card[0] = 0
        n = 0
        current_max_sum = 0
        
        for x in ordering:
            if n >= 60:
                break
            # Compute new sums and check for conflicts
            valid = True
            new_entries = []
            for s in range(current_max_sum + 1):
                if sum_to_card[s] >= 0:
                    ns = s + x
                    nk = sum_to_card[s] + 1
                    if sum_to_card[ns] >= 0 and sum_to_card[ns] != nk:
                        valid = False
                        break
                    new_entries.append((ns, nk))
            if not valid:
                continue
            # Also check new_entries against each other (shouldn't conflict)
            # Apply
            for ns, nk in new_entries:
                if sum_to_card[ns] < 0 or sum_to_card[ns] == nk:
                    sum_to_card[ns] = nk
                # else: already checked, it's equal
            current_set.append(x)
            n += 1
            current_max_sum += x
        
        if n > best_size:
            best_size = n
            best_set = current_set[:]
    
    return best_set
