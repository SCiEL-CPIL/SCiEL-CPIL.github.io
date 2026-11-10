import random
import time
from math import gcd

N = 128
gcd_matrix = [[1] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(i, N + 1):
        g = gcd(i, j)
        gcd_matrix[i][j] = g
        gcd_matrix[j][i] = g

def is_valid_with_x(S, x):
    # S is a set of integers
    # Check if adding x creates a bad triple
    # Bad triple: a, b in S such that gcd(x,a) == gcd(x,b) == gcd(a,b)
    
    # Optimization: Group elements of S by gcd(x, a)
    # But simple double loop is fine for n < 128
    
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        d_a = gcd_matrix[x][a]
        for j in range(i + 1, n):
            b = S_list[j]
            if gcd_matrix[x][b] == d_a:
                if gcd_matrix[a][b] == d_a:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Initialize with a greedy set
    S = set()
    # Sort by number of divisors? Or just random?
    # Let's try to add numbers that are "safe" first.
    # Numbers with many small prime factors are risky.
    # Numbers that are prime powers are safer?
    # Let's just use random order for initialization to avoid bias.
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    for x in candidates:
        if is_valid_with_x(S, x):
            S.add(x)
    
    best_S = set(S)
    
    # Local search
    # We will try to add elements, and if stuck, swap.
    no_improvement_count = 0
    
    while time.time() < deadline:
        # Try to add
        added = False
        if len(S) < N:
            # Pick a random candidate not in S
            # To be efficient, we can maintain a list of candidates
            # But for simplicity, pick random 1..N
            for _ in range(10): # Try a few times to find a candidate not in S
                x = random.randint(1, N)
                if x not in S:
                    if is_valid_with_x(S, x):
                        S.add(x)
                        added = True
                        if len(S) > len(best_S):
                            best_S = set(S)
                            no_improvement_count = 0
                        break
        
        if not added:
            # Try to swap
            if S:
                y = random.choice(list(S))
                # Pick x not in S
                x_found = False
                for _ in range(10):
                    x = random.randint(1, N)
                    if x not in S:
                        x_found = True
                        break
                
                if x_found:
                    S_temp = set(S)
                    S_temp.remove(y)
                    if is_valid_with_x(S_temp, x):
                        S = S_temp
                        S.add(x)
                        # Swapping doesn't increase size, so no_improvement_count increases?
                        # Or we can consider it a "move"
                        no_improvement_count += 1
        
        # If we are stuck for too long, restart with a new random seed?
        # Or just keep swapping.
        
    return list(best_S)
