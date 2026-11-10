import time

def construct():
    S = 123457
    # Sieve for smallest prime factor is not strictly needed if we do the LPF sieve directly
    P = [1] * (S + 1)
    for i in range(2, S + 1):
        if P[i] == 1: # i is prime
            for j in range(i, S + 1, i):
                P[j] = i
                
    best_score = float('inf')
    best_a = 1
    best_b = S - 1
    
    # We can add a time check, but this should take < 0.1s
    for a in range(1, S // 2 + 1):
        b = S - a
        score = P[a] if P[a] > P[b] else P[b]
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            
    return [best_a, best_b]
