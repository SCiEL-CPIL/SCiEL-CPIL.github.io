import time
from math import gcd
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    N = 128
    UNIVERSE = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute forbidden triples
    # forbidden[a][b] = set of c such that (a, b, c) is forbidden
    forbidden = [[set() for _ in range(N + 1)] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_mat[a][b]
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if gcd_mat[a][c] == gab and gcd_mat[b][c] == gab:
                    forbidden[a][b].add(c)
                    forbidden[b][a].add(c)
    
    # Precompute degree in conflict graph for heuristic ordering
    degree = [0] * (N + 1)
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            if forbidden[u][v]:
                degree[u] += 1
                degree[v] += 1

    def can_add(S, x):
        # Check if x can be added to S without forming a forbidden triple
        # A forbidden triple (a, b, x) exists if x in forbidden[a][b]
        for i in range(len(S)):
            a = S[i]
            for j in range(i + 1, len(S)):
                b = S[j]
                if x in forbidden[a][b]:
                    return False
        return True

    def greedy_initial(ordering_key):
        S = []
        P = set(UNIVERSE)
        # Sort P by the ordering key
        P_sorted = sorted(P, key=ordering_key)
        for v in P_sorted:
            if time.time() > DEADLINE:
                break
            if can_add(S, v):
                S.append(v)
        return S

    # Try multiple different orderings to find a good initial solution
    # Ordering 1: By degree (high degree first) - "most constrained"
    S1 = greedy_initial(lambda x: -degree[x])
    
    # Ordering 2: By value (small first)
    S2 = greedy_initial(lambda x: x)
    
    # Ordering 3: By number of divisors (approximated by degree or just value for speed)
    # Let's try random shuffles with a few restarts
    import random
    random.seed(42)
    
    best_S = max([S1, S2], key=len)
    
    # Run a few random greedy restarts
    for _ in range(20):
        if time.time() > DEADLINE - 10:
            break
        P_shuffled = UNIVERSE[:]
        random.shuffle(P_shuffled)
        S_rand = []
        for v in P_shuffled:
            if can_add(S_rand, v):
                S_rand.append(v)
        if len(S_rand) > len(best_S):
            best_S = S_rand[:]

    # Local Search: 1-swap and 1-add
    S = best_S[:]
    S_set = set(S)
    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        
        # 1-add
        for x in UNIVERSE:
            if x not in S_set:
                if can_add(S, x):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    break
        
        if improved:
            continue
            
        # 1-swap
        found_swap = False
        for i, a in enumerate(S):
            if time.time() > DEADLINE - 5:
                break
            for b in UNIVERSE:
                if b in S_set:
                    continue
                # Try swapping a with b
                # Remove a, check if b can be added
                S_temp = S[:i] + S[i+1:]
                if can_add(S_temp, b):
                    S = S_temp + [b]
                    S_set.remove(a)
                    S_set.add(b)
                    found_swap = True
                    break
            if found_swap:
                break
        
        if found_swap:
            improved = True

    # If we have time left, try a more aggressive local search or simulated annealing
    if time.time() < DEADLINE - 2:
        # Simulated Annealing
        current_S = S[:]
        current_S_set = set(current_S)
        best_SA_S = current_S[:]
        best_SA_len = len(current_S)
        
        T = 100.0
        cooling = 0.99
        while time.time() < DEADLINE - 1:
            if T < 1.0:
                break
            
            # Pick a random move: add, remove, or swap
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add' and len(current_S) < N:
                candidates = [x for x in UNIVERSE if x not in current_S_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                if can_add(current_S, x):
                    new_S = current_S + [x]
                    delta = 1
                    if delta > 0 or random.random() < pow(2.718, delta / T):
                        current_S = new_S
                        current_S_set.add(x)
                        if len(current_S) > best_SA_len:
                            best_SA_len = len(current_S)
                            best_SA_S = current_S[:]
                else:
                    # If can't add, skip or try a swap instead
                    pass
                    
            elif move_type == 'remove' and len(current_S) > 0:
                i = random.randint(0, len(current_S) - 1)
                x = current_S[i]
                new_S = current_S[:i] + current_S[i+1:]
                delta = -1
                if delta > 0 or random.random() < pow(2.718, delta / T):
                    current_S = new_S
                    current_S_set.remove(x)
                    
            elif move_type == 'swap' and len(current_S) > 0:
                i = random.randint(0, len(current_S) - 1)
                a = current_S[i]
                candidates = [x for x in UNIVERSE if x not in current_S_set]
                if not candidates:
                    continue
                b = random.choice(candidates)
                S_temp = current_S[:i] + current_S[i+1:]
                if can_add(S_temp, b):
                    new_S = S_temp + [b]
                    delta = 0 # Size doesn't change, but structure might improve for future adds
                    # We accept with high probability to explore
                    if random.random() < pow(2.718, delta / T) or random.random() < 0.1:
                        current_S = new_S
                        current_S_set.remove(a)
                        current_S_set.add(b)
                        if len(current_S) > best_SA_len:
                            best_SA_len = len(current_S)
                            best_SA_S = current_S[:]
            
            T *= cooling

        if len(best_SA_S) > len(S):
            S = best_SA_S[:]

    return S if S else [1]

# Wrap to ensure nonempty
_orig_construct = construct
def construct():
    result = _orig_construct()
    if not result:
        return [1]
    return result
