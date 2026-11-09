import random
import time

def construct():
    n = 180
    elements = list(range(1, n + 1))
    
    # 1. Precompute forbidden triples and conflict structures
    conflicts = {i: set() for i in range(1, n + 1)}
    
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            if b == a:
                continue
            den = b - a
            if den == 0:
                continue
            num = a * b
            if num % den != 0:
                continue
            c = num // den
            if c < 1 or c > n:
                continue
            if c == a or c == b:
                continue
            
            conflicts[a].add(frozenset((b, c)))
            conflicts[b].add(frozenset((a, c)))
            conflicts[c].add(frozenset((a, b)))

    # 2. Greedy Initialization
    def greedy_construct(order):
        S = set()
        for e in order:
            has_conflict = False
            for pair in conflicts[e]:
                u, v = tuple(pair)
                if u in S and v in S:
                    has_conflict = True
                    break
            if not has_conflict:
                S.add(e)
        return S

    best_init = set()
    for _ in range(20):
        random.shuffle(elements)
        S = greedy_construct(elements)
        if len(S) > len(best_init):
            best_init = S
            
    # 3. Simulated Annealing
    LAMBDA = 10000
    current_S = best_init
    
    def count_conflicts(S):
        count = 0
        for e in S:
            for pair in conflicts[e]:
                u, v = tuple(pair)
                if u in S and v in S:
                    count += 1
        return count

    current_conflicts_sum = count_conflicts(current_S)
    
    def get_energy(conf_sum, size):
        return -size + LAMBDA * conf_sum

    current_energy = get_energy(current_conflicts_sum, len(current_S))
    best_S = set(current_S)
    best_energy = current_energy
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    max_iters = 200000
    start_time = time.time()
    
    # Helper to pick random element from S or complement using rejection sampling
    def pick_from_S():
        while True:
            e = random.randint(1, n)
            if e in current_S:
                return e
                
    def pick_from_complement():
        while True:
            e = random.randint(1, n)
            if e not in current_S:
                return e

    for it in range(max_iters):
        if time.time() - start_time > 55:
            break
            
        r = random.random()
        if r < 0.3:
            # Add move
            e = pick_from_complement()
            
            delta_conf = 0
            for pair in conflicts[e]:
                u, v = tuple(pair)
                if u in current_S and v in current_S:
                    delta_conf += 1
            
            delta_E = -1 + LAMBDA * delta_conf
            
            if delta_E < 0 or random.random() < (2.718281828 ** (-delta_E / T)):
                current_S.add(e)
                current_conflicts_sum += delta_conf
                current_energy += delta_E
                if current_conflicts_sum == 0 and len(current_S) > len(best_S):
                    best_S = set(current_S)
                    best_energy = current_energy
                    
        elif r < 0.6:
            # Remove move
            e = pick_from_S()
            
            delta_conf = 0
            for pair in conflicts[e]:
                u, v = tuple(pair)
                if u in current_S and v in current_S:
                    delta_conf += 1
            
            delta_E = 1 - LAMBDA * delta_conf
            
            if delta_E < 0 or random.random() < (2.718281828 ** (-delta_E / T)):
                current_S.discard(e)
                current_conflicts_sum -= delta_conf
                current_energy += delta_E
                if current_conflicts_sum == 0 and len(current_S) > len(best_S):
                    best_S = set(current_S)
                    best_energy = current_energy
                    
        else:
            # Swap move
            e_out = pick_from_S()
            e_in = pick_from_complement()
            
            # Remove e_out
            rem_conf = 0
            for pair in conflicts[e_out]:
                u, v = tuple(pair)
                if u in current_S and v in current_S:
                    rem_conf += 1
            
            # Add e_in
            add_conf = 0
            for pair in conflicts[e_in]:
                u, v = tuple(pair)
                if u in current_S and v in current_S:
                    if u != e_out and v != e_out:
                        add_conf += 1
            
            delta_conf_sum = add_conf - rem_conf
            delta_E = LAMBDA * delta_conf_sum
            
            if delta_E < 0 or random.random() < (2.718281828 ** (-delta_E / T)):
                current_S.discard(e_out)
                current_S.add(e_in)
                current_conflicts_sum += delta_conf_sum
                current_energy += delta_E
                if current_conflicts_sum == 0 and len(current_S) > len(best_S):
                    best_S = set(current_S)
                    best_energy = current_energy
        
        T *= cooling
        if T < T_min:
            T = T_min

    if not best_S:
        best_S = {180}
        
    return list(best_S)
