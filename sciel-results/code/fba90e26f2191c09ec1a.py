import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    n = 12
    m = 40
    
    all_subsets = list(range(1 << n))
    
    def count_comparable(family):
        """Count unordered pairs where one is strict subset of other."""
        count = 0
        for i in range(len(family)):
            a = family[i]
            for j in range(i+1, len(family)):
                b = family[j]
                if (a & b) == a and a != b:
                    count += 1
                elif (a & b) == b and a != b:
                    count += 1
        return count

    def greedy_initial():
        family = [0] 
        remaining = set(all_subsets) - {0}
        for _ in range(m - 1):
            best_set = None
            best_gain = -1
            for s in remaining:
                gain = 0
                for e in family:
                    if (s & e) == s and s != e:
                        gain += 1
                    elif (s & e) == e and s != e:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_set = s
            family.append(best_set)
            remaining.discard(best_set)
        return family

    def local_search(family, time_limit=5.0):
        family = list(family)
        current_score = count_comparable(family)
        best_score = current_score
        best_family = list(family)
        
        start = time.time()
        no_improve_count = 0
        
        while time.time() - start < time_limit:
            if no_improve_count > 200:
                break
                
            i = random.randint(0, m - 1)
            old_i = family[i]
            
            r = random.random()
            if r < 0.3:
                # Try a subset of old_i
                new_set = old_i
                if new_set != 0:
                    bit_to_clear = 1 << random.randint(0, n-1)
                    if new_set & bit_to_clear:
                        new_set &= ~bit_to_clear
                if new_set in family or new_set == old_i:
                    continue
            elif r < 0.6:
                # Try a superset of old_i
                new_set = old_i
                if new_set != (1 << n) - 1:
                    bit_to_set = 1 << random.randint(0, n-1)
                    if not (new_set & bit_to_set):
                        new_set |= bit_to_set
                if new_set in family or new_set == old_i:
                    continue
            else:
                new_set = random.choice(all_subsets)
                if new_set in family or new_set == old_i:
                    continue

            family[i] = new_set
            new_score = count_comparable(family)
            
            if new_score > best_score:
                best_score = new_score
                best_family = list(family)
                no_improve_count = 0
            else:
                family[i] = old_i
                no_improve_count += 1
        
        return best_family, best_score

    best_family = None
    best_score = -1
    
    # Start 1: Greedy
    fam1 = greedy_initial()
    score1 = count_comparable(fam1)
    best_family = fam1
    best_score = score1
    
    # Start 2: Structured (Small sets)
    small_sets = []
    for r in range(0, 3):
        for combo in combinations(range(n), r):
            mask = 0
            for x in combo:
                mask |= (1 << x)
            small_sets.append(mask)
            
    small_sets.sort(key=lambda x: (bin(x).count('1'), x))
    fam2 = small_sets[:m]
    score2 = count_comparable(fam2)
    
    if score2 > best_score:
        best_score = score2
        best_family = fam2
    
    # Local search on best start
    if time.time() < deadline - 5:
        improved_fam, improved_score = local_search(best_family, time_limit=10.0)
        if improved_score > best_score:
            best_score = improved_score
            best_family = improved_fam
            
    # Try a few random restarts if time permits
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        random.shuffle(all_subsets)
        fam = all_subsets[:m]
        score = count_comparable(fam)
        
        if score > best_score:
            best_score = score
            best_family = fam
        
        improved_fam, improved_score = local_search(fam, time_limit=2.0)
        if improved_score > best_score:
            best_score = improved_score
            best_family = improved_fam
            
        if time.time() > deadline - 10:
            break
            
    # Final polish
    if time.time() < deadline - 2:
        final_fam, final_score = local_search(best_family, time_limit=5.0)
        if final_score > best_score:
            best_score = final_score
            best_family = final_fam

    result = []
    for mask in best_family:
        lst = [b for b in range(n) if (mask >> b) & 1]
        result.append(lst)
    
    return result
