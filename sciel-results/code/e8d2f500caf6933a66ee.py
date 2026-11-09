"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _independence_number(triples):
    """Compute the independence number: largest subset with no complete triple."""
    n = 16
    if not triples:
        return n
    
    # Precompute masks
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
        
    # Iterate all possible subsets (bitmasks)
    # We want the max size independent set.
    # We can iterate size from 16 down to 1.
    # To avoid generating combinations, we can just iterate all 2^16 masks
    # and check popcount.
    # 65536 iterations is fast.
    
    # Optimization: Sort triples? No.
    
    # We'll iterate size descending.
    # For a given size, we need to check if ANY mask of that size is independent.
    # Instead of generating masks of specific size, we can just check all masks
    # and track the max size found? No, that requires checking all 65536 masks
    # and for each, checking all triples.
    # 65536 * len(triples). If len(triples)=100, 6.5M ops. ~0.1s in Py.
    
    max_indep = 0
    for mask in range(1 << n):
        # Optional: skip if popcount <= max_indep
        if bin(mask).count('1') <= max_indep:
            continue
            
        is_independent = True
        for tm in triple_masks:
            if (mask & tm) == tm:
                is_independent = False
                break
        
        if is_independent:
            max_indep = bin(mask).count('1')
            # We can't break immediately because there might be a larger one?
            # No, we are iterating 0 to 2^n.
            # To find MAX, we should iterate in an order that allows early termination?
            # Iterating 0 to 2^n doesn't guarantee max is found first.
            # So we must check all, or iterate by size.
            # Let's stick to the simple all-mask check and track max.
            # It's O(2^n * T).
            
    return max_indep

def _is_valid(triples):
    """Check if any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i+1, len(triples)):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_triples = None
    best_score = 16  # Worst case
    
    def is_valid_triple_set(triples):
        """Check if any two triples share at most one point."""
        n = len(triples)
        for i in range(n):
            for j in range(i+1, n):
                common = set(triples[i]) & set(triples[j])
                if len(common) > 1:
                    return False
        return True

    def random_valid_system():
        """Generate a random valid set of triples."""
        triples = []
        all_triples = list(combinations(range(16), 3))
        random.shuffle(all_triples)
        for t in all_triples:
            t_set = set(t)
            valid = True
            for existing in triples:
                if len(t_set & set(existing)) > 1:
                    valid = False
                    break
            if valid:
                triples.append(list(t))
        return triples

    # Simulated Annealing / Hill Climbing
    def optimize(initial_triples):
        triples = [list(t) for t in initial_triples]
        current_score = _independence_number(triples)
        
        # Temperature for Simulated Annealing
        temp = 1.0
        cooling = 0.995
        
        while time.time() < deadline:
            # Choose a random move
            move_type = random.choice(['add', 'remove', 'swap'])
            
            new_triples = None
            
            if move_type == 'add':
                # Try to add a random triple
                t = list(random.sample(range(16), 3))
                t_set = set(t)
                valid = True
                for existing in triples:
                    if len(t_set & set(existing)) > 1:
                        valid = False
                        break
                if valid:
                    new_triples = triples + [t]
                    
            elif move_type == 'remove':
                if triples:
                    idx = random.randint(0, len(triples)-1)
                    new_triples = triples[:idx] + triples[idx+1:]
                    
            elif move_type == 'swap':
                if triples:
                    idx = random.randint(0, len(triples)-1)
                    # Generate a replacement triple
                    attempts = 0
                    while attempts < 20:
                        t = list(random.sample(range(16), 3))
                        t_set = set(t)
                        valid = True
                        for j, existing in enumerate(triples):
                            if j == idx:
                                continue
                            if len(t_set & set(existing)) > 1:
                                valid = False
                                break
                        if valid:
                            new_triples = triples[:idx] + [t] + triples[idx+1:]
                            break
                        attempts += 1
            
            if new_triples is None:
                continue
                
            new_score = _independence_number(new_triples)
            
            # Acceptance criterion
            if new_score < current_score:
                triples = new_triples
                current_score = new_score
            else:
                # Simulated Annealing: accept worse with probability exp(-delta/T)
                delta = new_score - current_score
                if delta > 0:
                    prob = math.exp(-delta / temp)
                    if random.random() < prob:
                        triples = new_triples
                        current_score = new_score
                        
            temp *= cooling
            
        return triples, current_score

    # Run optimization with multiple restarts
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        initial = random_valid_system()
        triples, score = optimize(initial)
        
        if score < best_score:
            best_score = score
            best_triples = [list(t) for t in triples]
            if best_score <= 5:
                break
        
        # If we are running low on time, do a final check
        if time.time() > deadline - 5.0:
            break

    if best_triples is None:
        best_triples = random_valid_system()
        if not is_valid_triple_set(best_triples):
             # Fallback
             best_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    return best_triples
# EVOLVE-BLOCK-END
