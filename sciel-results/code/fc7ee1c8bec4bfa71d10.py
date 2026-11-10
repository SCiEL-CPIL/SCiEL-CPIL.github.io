import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # H is fixed: 65..128
    # Sums of any two distinct elements in H are > 128, so no bad triples in H
    H = set(range(65, 129))
    
    # Candidates for S' are 1..64
    candidates = list(range(1, 65))
    
    best_S_prime = set()
    best_size = 0
    
    # Helper to check if sum is in S = S_prime U H
    def in_S(x, S_prime):
        if x in S_prime:
            return True
        if 65 <= x <= 128:
            return True
        return False

    def can_add(g, S_prime):
        # Check if adding g creates a bad triple with any a, b in S_prime
        # Triple: {g, a, b}
        # Sums: g+a, g+b, a+b must be in S
        S_list = list(S_prime)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            sum_ga = g + a
            if not in_S(sum_ga, S_prime):
                continue
            for j in range(i + 1, n):
                b = S_list[j]
                sum_gb = g + b
                if not in_S(sum_gb, S_prime):
                    continue
                sum_ab = a + b
                if in_S(sum_ab, S_prime):
                    return False
        return True

    def greedy(order):
        S_prime = set()
        for g in order:
            if can_add(g, S_prime):
                S_prime.add(g)
        return S_prime

    # Run multiple randomized greedy attempts
    while time.time() < deadline:
        order = candidates[:]
        random.shuffle(order)
        
        S_prime = greedy(order)
        
        if len(S_prime) > best_size:
            best_size = len(S_prime)
            best_S_prime = S_prime

    result = list(best_S_prime) + list(H)
    result = [int(x) for x in result]
    
    return result
