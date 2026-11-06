import random
import math
import time

def construct():
    N = 128
    gcd = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            gcd[i][j] = math.gcd(i, j)
            gcd[j][i] = gcd[i][j]
            
    bad_pairs = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            for b in range(a + 1, N + 1):
                if gcd[x][a] == gcd[x][b] == gcd[a][b]:
                    bad_pairs[x].append((a, b))
                    
    # Use a set for S for O(1) membership
    # Start with a random set of size 80 from [2, 128]
    candidates = list(range(2, N + 1))
    random.shuffle(candidates)
    S = set(candidates[:80])
    
    # Calculate initial violations
    def calc_violations(S):
        total = 0
        for c in S:
            for a, b in bad_pairs[c]:
                if a in S and b in S:
                    total += 1
        return total // 3

    current_violations = calc_violations(S)
    
    best_S = set(S)
    if current_violations == 0:
        best_size = len(S)
    else:
        best_size = 0
        
    start_time = time.time()
    timeout = 55
    
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    
    while time.time() - start_time < timeout:
        if T < T_min:
            T = 1.0 # Restart temperature? Or just break?
            # Let's break and do a few restarts if time permits.
            break
            
        S_list = list(S)
        not_in_S = [x for x in range(2, N + 1) if x not in S]
        
        r = random.random()
        accepted = False
        
        if r < 0.4 and not_in_S:
            x = random.choice(not_in_S)
            delta = 0
            for a, b in bad_pairs[x]:
                if a in S and b in S:
                    delta += 1
            if delta < 0 or random.random() < math.exp(-delta / T):
                S.add(x)
                current_violations += delta
                accepted = True
        elif r < 0.8 and S_list:
            x = random.choice(S_list)
            delta = 0
            for a, b in bad_pairs[x]:
                if a in S and b in S:
                    delta += 1
            # Removing x reduces violations by delta
            # New cost = current - delta
            # Change = -delta
            if -delta < 0 or random.random() < math.exp(delta / T):
                S.remove(x)
                current_violations -= delta
                accepted = True
        else:
            if S_list and not_in_S:
                x = random.choice(S_list)
                y = random.choice(not_in_S)
                # Calculate violations removed by x
                rem = 0
                for a, b in bad_pairs[x]:
                    if a in S and b in S:
                        rem += 1
                # Calculate violations added by y
                # Must exclude pairs involving x
                add = 0
                for a, b in bad_pairs[y]:
                    if a in S and b in S and a != x and b != x:
                        add += 1
                delta_cost = add - rem
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    S.remove(x)
                    S.add(y)
                    current_violations += delta_cost
                    accepted = True

        if accepted and current_violations == 0:
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
                
        T *= cooling
        
    return list(best_S)
