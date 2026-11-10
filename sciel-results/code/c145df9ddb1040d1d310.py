import time
import random
import math

N = 256

def _calc_score(S):
    return sum(1.0 / x for x in S)

def _is_valid_add(S_set, y):
    """Check if y can be added to S_set without forming a 3-AP."""
    for a in S_set:
        b = 2 * y - a
        if 1 <= b <= N and b in S_set and b != a and b != y:
            return False
        if (a + y) % 2 == 0:
            b = (a + y) // 2
            if 1 <= b <= N and b in S_set and b != a and b != y:
                return False
        b = 2 * a - y
        if 1 <= b <= N and b in S_set and b != a and b != y:
            return False
    return True

def _greedy_init():
    S = set()
    for x in range(1, N + 1):
        if _is_valid_add(S, x):
            S.add(x)
    return S

def _perturb(S, k=5):
    S_new = set(S)
    to_remove = random.sample(list(S_new), min(k, len(S_new)))
    for x in to_remove:
        S_new.remove(x)
    candidates = [x for x in range(1, N + 1) if x not in S_new]
    random.shuffle(candidates)
    for y in candidates[:k]:
        if _is_valid_add(S_new, y):
            S_new.add(y)
    return S_new

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_S = _greedy_init()
    best_score = _calc_score(best_S)
    
    T = 5.0
    cooling_rate = 0.9995
    min_T = 0.01
    
    current_S = set(best_S)
    current_score = best_score
    
    recips = [0.0] + [1.0 / i for i in range(1, N + 1)]
    
    # Maintain list of available elements for faster sampling
    # But updating it on every add/remove is tricky with sets.
    # Given N=256, filtering is fast enough.
    
    while time.time() < deadline:
        if T < min_T:
            T = 5.0
            current_S = _perturb(best_S, k=5)
            current_score = _calc_score(current_S)
            continue
            
        move_type = random.random()
        
        if move_type < 0.4:
            candidates = [x for x in range(1, N + 1) if x not in current_S]
            if candidates:
                y = random.choice(candidates)
                if _is_valid_add(current_S, y):
                    delta = recips[y]
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_S.add(y)
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_S = set(current_S)
        elif move_type < 0.7:
            if current_S:
                x = random.choice(list(current_S))
                delta = -recips[x]
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_S.remove(x)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(current_S)
        else:
            if current_S:
                x = random.choice(list(current_S))
                candidates = [y for y in range(1, N + 1) if y not in current_S]
                if candidates:
                    y = random.choice(candidates)
                    S_temp = current_S - {x}
                    if _is_valid_add(S_temp, y):
                        delta = recips[y] - recips[x]
                        if delta > 0 or random.random() < math.exp(delta / T):
                            current_S.remove(x)
                            current_S.add(y)
                            current_score += delta
                            if current_score > best_score:
                                best_score = current_score
                                best_S = set(current_S)
                                
        T *= cooling_rate
        
    return list(best_S)
