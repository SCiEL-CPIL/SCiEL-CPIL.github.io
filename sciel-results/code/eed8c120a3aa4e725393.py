import random
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 115
    N = 284
    PRESCRIBED = [1, 2, 5, 11]
    
    def get_sums(S):
        sums = []
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.append(S[i] + S[j])
        return sums

    def collisions(S):
        sums = get_sums(S)
        return len(sums) - len(set(sums))

    def score(S):
        c = collisions(S)
        if c > 0:
            return len(S) - 100 * c
        return len(S)

    # Initial construction: Greedy
    S = list(PRESCRIBED)
    S_set = set(S)
    # Try to add elements greedily in random order
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    for x in candidates:
        if x in S_set:
            continue
        S_new = S + [x]
        if collisions(S_new) == 0:
            S = S_new
            S_set.add(x)
    
    best_S = list(S)
    best_score = score(best_S)
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    alpha = 0.9999
    
    current_S = list(S)
    current_score = score(current_S)
    
    while time.time() - start_time < DEADLINE:
        # Generate neighbor
        move_type = random.random()
        new_S = list(current_S)
        new_S_set = set(new_S)
        
        if move_type < 0.4:
            # Add
            # Pick random x not in S
            while True:
                x = random.randint(1, N)
                if x not in new_S_set:
                    break
            new_S.append(x)
        elif move_type < 0.8:
            # Swap
            # Pick random y in S (non-prescribed)
            non_prescribed = [y for y in new_S if y not in PRESCRIBED]
            if not non_prescribed:
                continue
            y = random.choice(non_prescribed)
            new_S.remove(y)
            new_S_set.discard(y)
            
            while True:
                x = random.randint(1, N)
                if x not in new_S_set:
                    break
            new_S.append(x)
            new_S_set.add(x)
        else:
            # Remove
            non_prescribed = [y for y in new_S if y not in PRESCRIBED]
            if not non_prescribed:
                continue
            y = random.choice(non_prescribed)
            new_S.remove(y)
            new_S_set.discard(y)
            
        new_score = score(new_S)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_S = new_S
            current_score = new_score
            
            # Update best
            if current_score > best_score:
                best_score = current_score
                best_S = list(current_S)
                
        T *= alpha
        if T < T_min:
            T = T_min
            # Restart temperature occasionally? Or just keep going
            # For SA, cooling is usually monotonic.
            # But if we hit a plateau, we might want to reheat.
            # Let's just let it cool to min and stay there, effectively doing greedy/local search.
            
    return best_S
