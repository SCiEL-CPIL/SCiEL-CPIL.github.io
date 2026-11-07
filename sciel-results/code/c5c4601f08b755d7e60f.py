import random
import itertools

def get_score(values):
    sums = set()
    prods = set()
    for i in range(len(values)):
        for j in range(i+1, len(values)):
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def construct():
    n = 14
    best_set = [2**i for i in range(n)]
    best_score = get_score(best_set)
    
    # Try to improve
    for restart in range(10):
        if restart == 0:
            current = best_set[:]
        else:
            current = sorted(random.sample(range(1, 10001), n))
        
        curr_score = get_score(current)
        if curr_score < best_score:
            best_score = curr_score
            best_set = current[:]
            
        for step in range(2000):
            idx = random.randint(0, n-1)
            new_val = random.randint(1, 10000)
            if new_val in current:
                continue
            old_val = current[idx]
            current[idx] = new_val
            new_score = get_score(current)
            if new_score < curr_score:
                curr_score = new_score
                if curr_score < best_score:
                    best_score = curr_score
                    best_set = current[:]
            else:
                current[idx] = old_val
                
        if best_score == 25:
            break

    edges = [list(e) for e in itertools.combinations(range(n), 2)]
    return {'values': best_set, 'edges': edges}
