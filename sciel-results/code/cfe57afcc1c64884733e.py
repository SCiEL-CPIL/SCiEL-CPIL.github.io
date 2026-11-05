import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # 1. Identify all bad triples
    # A triple (a, b, c) is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm(a, b)
            for c in range(b + 1, n + 1):
                lac = lcm(a, c)
                if lac != lab:
                    continue
                lbc = lcm(b, c)
                if lbc == lab:
                    bad_triples.append((a, b, c))
                    
    if time.time() > deadline:
        return numbers # Fallback
        
    # 2. Precompute conflict structures
    # conflicts[z] = list of pairs (u, v) such that (u, v, z) is a bad triple
    # This allows fast checking: can we add z to set S?
    # Yes, if no pair (u, v) in conflicts[z] is fully contained in S.
    
    conflicts = {x: [] for x in numbers}
    for a, b, c in bad_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))
        
    # Helper function to check if x can be added to current_set
    def can_add(x, current_set):
        for u, v in conflicts[x]:
            if u in current_set and v in current_set:
                return False
        return True

    # 3. Greedy initialization with random tie-breaking
    def greedy_init():
        S = set(numbers)
        counts = {x: 0 for x in numbers}
        for t in bad_triples:
            for x in t:
                counts[x] += 1
        
        active_triples = [True] * len(bad_triples)
        
        while True:
            max_count = 0
            candidates = []
            
            for x in S:
                if counts[x] > max_count:
                    max_count = counts[x]
                    candidates = [x]
                elif counts[x] == max_count and max_count > 0:
                    candidates.append(x)
            
            if not candidates or max_count == 0:
                break
                
            target = random.choice(candidates)
            S.remove(target)
            
            for i, t in enumerate(bad_triples):
                if not active_triples[i]:
                    continue
                if target in t:
                    active_triples[i] = False
                    for x in t:
                        if x != target:
                            counts[x] -= 1
        
        return S

    # 4. Simulated Annealing
    def simulate_annealing(S, iterations=2000, initial_temp=10.0, cooling_rate=0.995):
        S = set(S)
        current_size = len(S)
        best_S = set(S)
        best_size = current_size
        temp = initial_temp
        
        for it in range(iterations):
            if time.time() > deadline - 2.0:
                break
            
            # Choose a random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add a random element not in S
                not_in_S = [x for x in numbers if x not in S]
                if not not_in_S:
                    continue
                x = random.choice(not_in_S)
                if can_add(x, S):
                    S.add(x)
                    new_size = len(S)
                    delta = new_size - current_size
                    if delta > 0 or random.random() < pow(2.718281828, delta / temp):
                        current_size = new_size
                        if current_size > best_size:
                            best_size = current_size
                            best_S = set(S)
                    else:
                        S.remove(x)
            elif move_type < 0.6:
                # Try to remove a random element from S
                if not S:
                    continue
                x = random.choice(list(S))
                S.remove(x)
                new_size = len(S)
                delta = new_size - current_size
                if delta > 0 or random.random() < pow(2.718281828, delta / temp):
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_S = set(S)
                else:
                    S.add(x)
            else:
                # Swap: remove one from S, add one not in S
                if not S:
                    continue
                not_in_S = [x for x in numbers if x not in S]
                if not not_in_S:
                    continue
                x = random.choice(list(S))
                y = random.choice(not_in_S)
                
                S.remove(x)
                if can_add(y, S):
                    S.add(y)
                    new_size = len(S)
                    delta = new_size - current_size
                    if delta > 0 or random.random() < pow(2.718281828, delta / temp):
                        current_size = new_size
                        if current_size > best_size:
                            best_size = current_size
                            best_S = set(S)
                    else:
                        S.remove(y)
                        S.add(x)
                else:
                    S.add(x)
            
            temp *= cooling_rate
            if temp < 0.01:
                break
        
        return best_S, best_size

    # 5. Greedy local search (keep as fallback)
    def greedy_local_search(S):
        S = set(S)
        improved = True
        max_iters = 50
        
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            if time.time() > deadline - 5.0:
                break
                
            s_list = list(S)
            random.shuffle(s_list)
            
            for y in s_list:
                if time.time() > deadline - 5.0:
                    break
                    
                S_tentative = S - {y}
                
                candidates_to_add = [x for x in numbers if x not in S]
                random.shuffle(candidates_to_add)
                
                for z in candidates_to_add:
                    if can_add(z, S_tentative):
                        S_tentative.add(z)
                        
                if len(S_tentative) > len(S):
                    S = S_tentative
                    improved = True
                    break
        
        return S

    # Main search loop
    best_set = set()
    best_size = -1
    
    num_restarts = 30  # More restarts with SA
    
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
            
        # Get initial solution via greedy
        S = greedy_init()
        
        # Apply simulated annealing
        S_sa, size_sa = simulate_annealing(S, iterations=1500, initial_temp=8.0, cooling_rate=0.997)
        
        # Apply greedy local search on the SA result
        S_final = greedy_local_search(S_sa)
        
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_set = S_final
            
        # Also try just the SA result
        if len(S_sa) > best_size:
            best_size = len(S_sa)
            best_set = S_sa

    if best_size == -1:
        return [1]
        
    return list(best_set)
