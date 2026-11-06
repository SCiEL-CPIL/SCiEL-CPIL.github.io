"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(S_set, x):
        # Check if x can be added to S_set without creating a sum-free violation
        if 2 * x in S_set:
            return False
        
        for a in S_set:
            if a + x in S_set:
                return False
            if (x - a) in S_set:
                return False
        return True

    def get_conflicts(S_set, x):
        conflicts = set()
        if 2 * x in S_set:
            conflicts.add(2 * x)
            
        for a in S_set:
            c = a + x
            if c in S_set:
                conflicts.add(a)
                conflicts.add(c)
                
            b = x - a
            if b in S_set:
                conflicts.add(a)
                conflicts.add(b)
                
        return conflicts

    def greedy_build(S, order_key):
        S_sorted = sorted(S, key=order_key)
        S_set = set()
        for x in S_sorted:
            if can_add(S_set, x):
                S_set.add(x)
        return S_set

    def local_search(S_set, U, max_iters=50):
        S_set = set(S_set)
        removed = U - S_set
        rng = random.Random(42)
        
        for _ in range(max_iters):
            improved = False
            
            # Phase 1: Try to add any removed element
            removed_list = list(removed)
            rng.shuffle(removed_list)
            
            for x in removed_list:
                if can_add(S_set, x):
                    S_set.add(x)
                    removed.discard(x)
                    improved = True
            
            if not removed:
                break
                
            if not improved:
                # Phase 2: Try swaps
                swap_found = False
                if removed:
                    x = rng.choice(list(removed))
                    conflicts = get_conflicts(S_set, x)
                    if conflicts:
                        conflict_list = list(conflicts)
                        rng.shuffle(conflict_list)
                        for y in conflict_list[:5]:
                            temp_S_set = S_set - {y}
                            if can_add(temp_S_set, x):
                                S_set = temp_S_set
                                S_set.add(x)
                                removed.discard(x)
                                removed.add(y)
                                improved = True
                                break
                                
            if not improved:
                break
                
        return S_set

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        U = set(S)
        
        candidates = []
        
        cand1 = greedy_build(S, lambda x: x)
        candidates.append(cand1)
        
        cand2 = greedy_build(S, lambda x: -x)
        candidates.append(cand2)
        
        cand3 = greedy_build(S, lambda x: abs(x))
        candidates.append(cand3)
        
        cand4 = greedy_build(S, lambda x: -abs(x))
        candidates.append(cand4)
        
        best_greedy = max(candidates, key=len)
        S_final = local_search(best_greedy, U)
        
        if not is_sum_free(S_final):
             S_final = best_greedy
             
        results.append(list(S_final))
    
    return results
# EVOLVE-BLOCK-END
