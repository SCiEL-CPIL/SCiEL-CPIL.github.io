"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def _check_addition(triple, current_set):
    """
    Check if adding 'triple' to 'current_set' creates a K(2,2,2).
    triple = (a, b, c)
    A K(2,2,2) involving (a,b,c) requires points a', b', c' such that
    {a,a'}, {b,b'}, {c,c'} are the 3 pairs.
    The 8 transversals are:
    (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
    We already have (a,b,c). We need the other 7 to be in current_set.
    Points a,b,c,a',b',c' must be distinct.
    """
    a, b, c = triple
    
    for a_prime in range(16):
        if a_prime == a or a_prime == b or a_prime == c:
            continue
        
        t4 = tuple(sorted((a_prime, b, c)))
        if t4 not in current_set:
            continue
            
        for b_prime in range(16):
            if b_prime == a or b_prime == b or b_prime == c or b_prime == a_prime:
                continue
            
            t2 = tuple(sorted((a, b_prime, c)))
            if t2 not in current_set:
                continue
                
            t6 = tuple(sorted((a_prime, b_prime, c)))
            if t6 not in current_set:
                continue
            
            for c_prime in range(16):
                if c_prime == a or c_prime == b or c_prime == c or c_prime == a_prime or c_prime == b_prime:
                    continue
                
                t1 = tuple(sorted((a, b, c_prime)))
                if t1 not in current_set:
                    continue
                
                t3 = tuple(sorted((a, b_prime, c_prime)))
                if t3 not in current_set:
                    continue
                
                t5 = tuple(sorted((a_prime, b, c_prime)))
                if t5 not in current_set:
                    continue
                
                t7 = tuple(sorted((a_prime, b_prime, c_prime)))
                if t7 in current_set:
                    return False
            
    return True

def _greedy_initial():
    """Generate an initial valid triple set using greedy."""
    all_triples = list(itertools.combinations(range(16), 3))
    random.shuffle(all_triples)
    
    current_set = set()
    
    for t in all_triples:
        if _check_addition(t, current_set):
            current_set.add(t)
            
    return current_set

def _local_search(initial_set, time_limit=10.0):
    """Improve the triple set using local search."""
    start_time = time.time()
    current_set = set(initial_set)
    current_size = len(current_set)
    all_triples = list(itertools.combinations(range(16), 3))
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try to add any missing triple
        missing = [t for t in all_triples if t not in current_set]
        random.shuffle(missing)
        
        for t in missing:
            if _check_addition(t, current_set):
                current_set.add(t)
                current_size += 1
                improved = True
                break
        
        if not improved:
            # 2. Try a 1-remove, 2-add move (net +1)
            if time.time() - start_time > time_limit * 0.9:
                break
                
            remove_candidates = list(current_set)
            random.shuffle(remove_candidates)
            
            for t_rem in remove_candidates[:20]:
                current_set.remove(t_rem)
                
                added_count = 0
                added_triples = []
                
                missing = [t for t in all_triples if t not in current_set and t != t_rem]
                random.shuffle(missing)
                
                for t_add in missing:
                    if _check_addition(t_add, current_set):
                        current_set.add(t_add)
                        added_triples.append(t_add)
                        added_count += 1
                        
                if added_count > 1:
                    current_size += (added_count - 1)
                    improved = True
                    break
                else:
                    for t_add in added_triples:
                        current_set.remove(t_add)
                    current_set.add(t_rem)
                    
            if not improved:
                break

    return current_set

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_set = set()
    best_size = 0
    
    while time.time() < deadline:
        g_set = _greedy_initial()
        ls_set = _local_search(g_set, time_limit=5.0)
        
        if len(ls_set) > best_size:
            best_size = len(ls_set)
            best_set = ls_set
            
    return [list(t) for t in best_set]
# EVOLVE-BLOCK-END
