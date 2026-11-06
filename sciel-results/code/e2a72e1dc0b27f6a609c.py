import numpy as np

def construct():
    limit = 10000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = [i for i, prime in enumerate(is_prime) if prime]
    N = len(primes)
    G = limit
    
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # M[i, g] = max length of valid sequence ending at primes[i] with last gap <= g
    # We use int16 to save memory, max length is small (< 100)
    M = np.ones((N, G + 1), dtype=np.int16)
    
    # M[i, 0] = 1 is already set by ones
    
    for j in range(1, N):
        pj = primes[j]
        # M[j, 0] = 1
        current_max = 1
        # Iterate g from 1 to pj - 2
        # If g > pj - 2, pj - g < 2, not prime.
        # We only need to go up to G, but for g > pj, M[j, g] = M[j, g-1].
        # We can stop early if we reach pj? No, M is defined up to G.
        # But for backtracking, we might need M[j, g] for large g.
        # However, M[j, g] for g >= pj is constant (equal to M[j, pj-2]).
        # So we can fill up to pj-2, then fill the rest with the final value.
        
        max_g_to_check = pj - 2
        if max_g_to_check > G:
            max_g_to_check = G
            
        for g in range(1, max_g_to_check + 1):
            # Default is carry over
            M[j, g] = current_max
            val = pj - g
            if is_prime[val]:
                i = prime_to_idx[val]
                cand = 1 + M[i, g]
                if cand > current_max:
                    current_max = cand
                    M[j, g] = cand
        
        # Fill the rest with current_max
        if max_g_to_check < G:
            M[j, max_g_to_check + 1:] = current_max

    # Find max
    max_len = np.max(M)
    flat_idx = np.argmax(M)
    j_end = flat_idx // (G + 1)
    g_end = flat_idx % (G + 1)
    
    # Backtrack
    seq = []
    j = j_end
    g = g_end
    while True:
        seq.append(primes[j])
        if M[j, g] == 1:
            break
        
        # Find the actual gap used
        # M[j, g] > M[j, g-1] implies the transition happened at g?
        # Not necessarily, M[j, g] is max up to g.
        # We need to find the smallest g_min such that M[j, g_min] == M[j, g].
        # The transition that achieved this max must have used a gap g' <= g_min.
        # Actually, if M[j, g] > M[j, g-1], the max was achieved by a transition with gap exactly g?
        # Yes, because M[j, g-1] is the max for gaps < g.
        # If M[j, g] > M[j, g-1], the new value comes from 1 + M[i_g, g].
        # So the gap used is g.
        
        while g > 0 and M[j, g] == M[j, g - 1]:
            g -= 1
            
        if g == 0:
            break # Should not happen
            
        val = primes[j] - g
        if not is_prime[val]:
            # This should not happen if logic is correct
            break
            
        i = prime_to_idx[val]
        
        # Find g_prev for i
        # We used M[i, g]. This is max for gap <= g.
        # We need the specific gap that achieved this max.
        # Find smallest g_prev <= g such that M[i, g_prev] == M[i, g].
        g_prev = g
        while g_prev > 0 and M[i, g_prev - 1] == M[i, g_prev]:
            g_prev -= 1
            
        j = i
        g = g_prev
        
    seq.reverse()
    return seq
