import time
import random

P = {'n': 12, 'm': 40}

def construct():
    deadline = time.time() + 85
    n = P['n']
    m = P['m']
    N = 1 << n
    
    def is_subset(a, b):
        return (a & b) == a and a != b
    
    def count_comparable(family):
        count = 0
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                if is_subset(family[i], family[j]) or is_subset(family[j], family[i]):
                    count += 1
        return count
    
    def local_search(family, max_iters, deadline):
        current_score = count_comparable(family)
        best_score = current_score
        best_family = family[:]
        
        for iteration in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            i = random.randint(0, m-1)
            new_mask = random.randint(0, N-1)
            
            if new_mask == family[i]:
                continue
            if new_mask in family[:i] + family[i+1:]:
                continue
            
            old_mask = family[i]
            
            lost = 0
            gained = 0
            for j in range(m):
                if j == i:
                    continue
                other = family[j]
                if is_subset(old_mask, other) or is_subset(other, old_mask):
                    lost += 1
                if is_subset(new_mask, other) or is_subset(other, new_mask):
                    gained += 1
            
            delta = gained - lost
            
            if delta > 0:
                family[i] = new_mask
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
        
        return best_family, best_score
    
    best_overall = None
    best_score_overall = 0
    
    for restart in range(10):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            chain = [0]
            cur = 0
            for b in range(n):
                cur |= (1 << b)
                chain.append(cur)
            remaining = [x for x in range(N) if x not in chain]
            random.seed(42 + restart)
            extra = random.sample(remaining, m - len(chain))
            family = chain + extra
        else:
            random.seed(100 + restart)
            family = random.sample(range(N), m)
        
        fam, score = local_search(family, 200000, deadline)
        
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam
    
    if best_overall is None:
        best_overall = list(range(m))
    
    result = []
    for mask in best_overall:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
# EVOLVE-BLOCK-END
