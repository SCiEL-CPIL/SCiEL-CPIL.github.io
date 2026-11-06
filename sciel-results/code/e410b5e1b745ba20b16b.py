def construct():
    import time
    import random
    import math
    
    start_time = time.time()
    deadline = start_time + 110.0
    N = 511

    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    pindex = {p: i for i, p in enumerate(primes)}

    mult_mask = {}
    for p in primes:
        m = 0
        for b in range(p, N + 1, p):
            m |= (1 << b)
        mult_mask[p] = m

    pmask = [0] * (N + 1)
    conflict = [0] * (N + 1)
    for a in range(2, N + 1):
        x = a
        pm = 0
        cm = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                pm |= (1 << pindex[p])
                cm |= mult_mask[p]
                while x % p == 0:
                    x //= p
        if x > 1:
            pm |= (1 << pindex[x])
            cm |= mult_mask[x]
        pmask[a] = pm
        conflict[a] = cm

    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (512 - a)

    def greedy_init():
        """Greedy initialization: pick highest weight non-conflicting elements"""
        S = set()
        S_mask = 0
        for a in range(N, 0, -1):
            if (conflict[a] & S_mask) == 0:
                S_mask |= (1 << a)
                S.add(a)
        return S, S_mask

    def local_search(S, S_mask, current_score, T_init, T_min, cooling, max_iter, seed):
        """Local search with simulated annealing"""
        random.seed(seed)
        T = T_init
        best_score = current_score
        best_S = S.copy()
        best_S_mask = S_mask
        
        for iteration in range(max_iter):
            if time.time() >= deadline - 5.0:
                break
            if T < T_min:
                break
                
            # Random move: add, remove, or swap
            move_type = random.random()
            new_S = S.copy()
            new_S_mask = S_mask
            delta = 0.0
            
            if move_type < 0.3:
                # Try to add a random non-conflicting element
                candidates = [a for a in range(1, N + 1) if not (S_mask >> a) & 1 and (conflict[a] & S_mask) == 0]
                if candidates:
                    # Weighted random selection favoring higher weights
                    weights = [w[a] for a in candidates]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cum = 0
                    for i, cw in enumerate(weights):
                        cum += cw
                        if r <= cum:
                            a = candidates[i]
                            break
                    else:
                        a = candidates[-1]
                    new_S.add(a)
                    new_S_mask |= (1 << a)
                    delta = w[a]
            elif move_type < 0.5:
                # Try to remove a random element
                if S:
                    x = random.choice(list(S))
                    new_S.discard(x)
                    new_S_mask ^= (1 << x)
                    delta = -w[x]
            else:
                # Try to swap: remove one, add another
                if S:
                    x = random.choice(list(S))
                    rest = S_mask ^ (1 << x)
                    candidates = [a for a in range(1, N + 1) if not (S_mask >> a) & 1 and (conflict[a] & rest) == 0]
                    if candidates:
                        # Weighted random selection favoring higher weights
                        weights = [w[a] for a in candidates]
                        total_w = sum(weights)
                        r = random.random() * total_w
                        cum = 0
                        for i, cw in enumerate(weights):
                            cum += cw
                            if r <= cum:
                                a = candidates[i]
                                break
                        else:
                            a = candidates[-1]
                        new_S.discard(x)
                        new_S.add(a)
                        new_S_mask = rest | (1 << a)
                        delta = w[a] - w[x]
            
            # Acceptance criterion
            if delta > 0 or (delta < 0 and random.random() < math.exp(delta / T)):
                S = new_S
                S_mask = new_S_mask
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_S = S.copy()
                    best_S_mask = S_mask
            
            T *= cooling
        
        return best_S, best_S_mask, best_score

    def final_polish(S, S_mask):
        """Final deterministic polish: try to add any remaining non-conflicting elements in order of weight"""
        final_S = S.copy()
        final_S_mask = S_mask
        # Try adding elements in descending order of weight
        for a in range(N, 0, -1):
            if not (final_S_mask >> a) & 1 and (conflict[a] & final_S_mask) == 0:
                final_S_mask |= (1 << a)
                final_S.add(a)
        return final_S, final_S_mask

    # Initial greedy solution
    S, S_mask = greedy_init()
    current_score = sum(w[a] for a in S)
    
    best_score = current_score
    best_S = S.copy()
    best_S_mask = S_mask
    
    # Multiple restarts with different seeds and temperature schedules
    random.seed(42)
    seeds = [42, 123, 456, 789, 101, 202, 303]
    temp_schedules = [
        (0.1, 1e-6, 0.9995),
        (0.05, 1e-6, 0.9997),
        (0.2, 1e-5, 0.9993),
    ]
    
    for seed in seeds:
        if time.time() >= deadline - 10.0:
            break
        
        for T_init, T_min, cooling in temp_schedules:
            if time.time() >= deadline - 10.0:
                break
            
            # Reset to best solution found so far for this restart
            S_restart = best_S.copy()
            S_mask_restart = best_S_mask
            score_restart = sum(w[a] for a in S_restart)
            
            S_new, S_mask_new, score_new = local_search(
                S_restart, S_mask_restart, score_restart,
                T_init, T_min, cooling, 50000, seed
            )
            
            if score_new > best_score:
                best_score = score_new
                best_S = S_new
                best_S_mask = S_mask_new
    
    # Final polish
    best_S, best_S_mask = final_polish(best_S, best_S_mask)
    final_score = sum(w[a] for a in best_S)
    
    if final_score > best_score:
        best_score = final_score
    
    return list(best_S)
