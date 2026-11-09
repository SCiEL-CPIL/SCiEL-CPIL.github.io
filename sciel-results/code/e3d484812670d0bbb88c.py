import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    n = 180
    
    # Precompute triples (a,b,c) such that 1/a = 1/b + 1/c
    # a < b < c
    # (b-a)(c-a) = a^2
    triples = []
    for a in range(1, n + 1):
        a2 = a * a
        d = 1
        while d * d <= a2:
            if a2 % d == 0:
                e = a2 // d
                if d < e:
                    b = a + d
                    c = a + e
                    if b <= n and c <= n:
                        triples.append((a, b, c))
            d += 1
            
    # Build conflict lists
    # conflicts[i] contains list of (j, k) such that (i+1, j+1, k+1) is a triple
    # Note: i, j, k are 0-indexed
    conflicts = [[] for _ in range(n)]
    for (a, b, c) in triples:
        a_idx = a - 1
        b_idx = b - 1
        c_idx = c - 1
        conflicts[a_idx].append((b_idx, c_idx))
        conflicts[b_idx].append((a_idx, c_idx))
        conflicts[c_idx].append((a_idx, b_idx))
        
    # Helper to calculate penalty (number of violated triples)
    def get_penalty(S):
        p = 0
        for (a, b, c) in triples:
            if S[a-1] and S[b-1] and S[c-1]:
                p += 1
        return p

    # Helper to calculate size
    def get_size(S):
        return sum(S)

    # Simulated Annealing
    def simulated_annealing(S_init, T_start, T_end, steps):
        S = S_init[:]
        cur_penalty = get_penalty(S)
        cur_size = get_size(S)
        best_S = S[:]
        best_penalty = cur_penalty
        best_size = cur_size
        
        # We want to maximize size, so if penalty is 0, we track max size.
        # If penalty > 0, we prefer lower penalty, then higher size.
        # Objective function for SA: Minimize (penalty * 1000 - size)
        # This ensures we drive penalty to 0 first, then maximize size.
        
        def objective(S):
            return get_penalty(S) * 1000 - get_size(S)
            
        cur_obj = objective(S)
        best_obj = cur_obj
        best_S = S[:]
        
        for step in range(steps):
            if time.time() > DEADLINE - 5:
                break
                
            # Choose random element to flip
            i = random.randint(0, n - 1)
            
            # Calculate delta objective
            # If S[i] is 0 -> 1:
            #   size increases by 1
            #   penalty increases by count of (j,k) in conflicts[i] where S[j] and S[k] are 1
            # If S[i] is 1 -> 0:
            #   size decreases by 1
            #   penalty decreases by count of (j,k) in conflicts[i] where S[j] and S[k] are 1
            
            if S[i]:
                # Flip 1 -> 0
                delta_size = -1
                delta_penalty = 0
                for (j, k) in conflicts[i]:
                    if S[j] and S[k]:
                        delta_penalty -= 1
                # Wait, if S[i] is 1, the triples involving i are currently counted in penalty.
                # If we remove i, those triples are no longer violated.
                # So penalty decreases by the number of active triples involving i.
                # delta_penalty = - (number of active triples involving i)
                # Objective change: delta_obj = 1000 * delta_penalty - delta_size
                # delta_obj = 1000 * (-count) - (-1) = 1 - 1000*count
            else:
                # Flip 0 -> 1
                delta_size = 1
                delta_penalty = 0
                for (j, k) in conflicts[i]:
                    if S[j] and S[k]:
                        delta_penalty += 1
                # Objective change: delta_obj = 1000 * delta_penalty - delta_size
                # delta_obj = 1000 * count - 1
            
            # Recalculate delta_obj properly
            count = 0
            for (j, k) in conflicts[i]:
                if S[j] and S[k]:
                    count += 1
            
            if S[i]:
                # Removing i
                # Penalty drops by 'count'
                # Size drops by 1
                delta_obj = 1000 * (-count) - (-1)
            else:
                # Adding i
                # Penalty rises by 'count'
                # Size rises by 1
                delta_obj = 1000 * count - 1
                
            new_obj = cur_obj + delta_obj
            
            # Acceptance probability
            if new_obj < cur_obj:
                S[i] = not S[i]
                cur_obj = new_obj
            else:
                # Calculate temperature
                T = T_start * (T_end / T_start) ** (step / steps)
                if random.random() < math.exp(-delta_obj / T):
                    S[i] = not S[i]
                    cur_obj = new_obj
            
            # Update best
            if cur_obj < best_obj:
                best_obj = cur_obj
                best_S = S[:]
                
        return best_S

    def local_search_polish(S):
        """Greedy addition of elements that don't violate constraints."""
        S = S[:]
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if not S[i]:
                    # Check if i can be added
                    can_add = True
                    for (j, k) in conflicts[i]:
                        if S[j] and S[k]:
                            can_add = False
                            break
                    if can_add:
                        S[i] = True
                        improved = True
        return S

    best_S = None
    best_size = 0
    
    # Strategy 1: Random restarts with SA
    num_restarts = 20
    steps_per_sa = 5000
    
    for r in range(num_restarts):
        if time.time() > DEADLINE - 10:
            break
            
        # Initialize S randomly
        S_init = [False] * n
        for i in range(n):
            if random.random() < 0.5:
                S_init[i] = True
                
        S_sa = simulated_annealing(S_init, 10.0, 0.01, steps_per_sa)
        
        # Polish
        S_polished = local_search_polish(S_sa)
        
        # Check validity
        if get_penalty(S_polished) == 0:
            size = get_size(S_polished)
            if size > best_size:
                best_size = size
                best_S = S_polished[:]
                
    # Strategy 2: Greedy initialization + SA
    # Start with high numbers (often better for this type of problem)
    S_greedy = [False] * n
    for i in range(n // 2, n):
        S_greedy[i] = True
    # Polish
    S_greedy = local_search_polish(S_greedy)
    
    if get_penalty(S_greedy) == 0:
        size = get_size(S_greedy)
        if size > best_size:
            best_size = size
            best_S = S_greedy[:]
            
    # Run SA from greedy start
    if time.time() < DEADLINE - 10:
        S_sa2 = simulated_annealing(S_greedy, 5.0, 0.01, steps_per_sa)
        S_polished2 = local_search_polish(S_sa2)
        if get_penalty(S_polished2) == 0:
            size = get_size(S_polished2)
            if size > best_size:
                best_size = size
                best_S = S_polished2[:]

    if best_S is None:
        # Fallback: just pick 180
        best_S = [False] * n
        best_S[-1] = True
        best_size = 1

    out = [i + 1 for i in range(n) if best_S[i]]
    return out
