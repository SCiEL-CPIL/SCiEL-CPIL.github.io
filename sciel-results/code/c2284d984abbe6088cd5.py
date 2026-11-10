import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 12
    m = 40
    N_masks = 1 << n
    
    # Precompute popcounts
    pop = [0] * N_masks
    for i in range(1, N_masks):
        pop[i] = pop[i >> 1] + (i & 1)
        
    # Greedy Init
    C = [0] * N_masks
    S = []
    InS = [False] * N_masks
    
    for k in range(m):
        best_m = -1
        best_score = -1
        for cand in range(N_masks):
            if InS[cand]:
                continue
            score = C[cand]
            # Tie break: prefer smaller popcount
            if score > best_score or (score == best_score and (best_m == -1 or pop[cand] < pop[best_m])):
                best_score = score
                best_m = cand
                
        if best_m == -1:
            break
            
        S.append(best_m)
        InS[best_m] = True
        
        # Update C
        bm = best_m
        for i in range(N_masks):
            if i != bm:
                if (i & bm) == i or (i & bm) == bm:
                    C[i] += 1

    best_score = sum(C[s] for s in S) // 2
    
    # Local Search
    while time.time() < deadline:
        improved = False
        indices = list(range(m))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            s_i = S[i]
            best_delta = 0
            best_m_new = -1
            
            for m_new in range(N_masks):
                if InS[m_new]:
                    continue
                
                comp_si = 0
                if (m_new & s_i) == m_new or (m_new & s_i) == s_i:
                    comp_si = 1
                
                delta = C[m_new] - C[s_i] - comp_si
                
                if delta > best_delta:
                    best_delta = delta
                    best_m_new = m_new
            
            if best_delta > 0:
                S[i] = best_m_new
                InS[s_i] = False
                InS[best_m_new] = True
                
                for m in range(N_masks):
                    dec = 0
                    inc = 0
                    if m != s_i:
                        if (m & s_i) == m or (m & s_i) == s_i:
                            dec = 1
                    if m != best_m_new:
                        if (m & best_m_new) == m or (m & best_m_new) == best_m_new:
                            inc = 1
                    C[m] += inc - dec
                    
                best_score += best_delta
                improved = True
                break
        
        if not improved:
            i = random.randint(0, m-1)
            s_i = S[i]
            
            while True:
                m_new = random.randint(0, N_masks-1)
                if not InS[m_new]:
                    break
            
            comp_si = 0
            if (m_new & s_i) == m_new or (m_new & s_i) == s_i:
                comp_si = 1
            
            delta = C[m_new] - C[s_i] - comp_si
            
            S[i] = m_new
            InS[s_i] = False
            InS[m_new] = True
            
            for m in range(N_masks):
                dec = 0
                inc = 0
                if m != s_i:
                    if (m & s_i) == m or (m & s_i) == s_i:
                        dec = 1
                if m != m_new:
                    if (m & m_new) == m or (m & m_new) == m_new:
                        inc = 1
                C[m] += inc - dec
                
            best_score += delta

    result = []
    for mask in S:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result
