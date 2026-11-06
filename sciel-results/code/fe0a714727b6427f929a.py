import time
import random
import sys
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 90s kill, 5s safety margin
    N = 256
    
    # Precompute weights
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
    
    conflict_pairs = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for d in range(1, (N - x) // 2 + 1):
            a = x + d
            b = x + 2 * d
            if a <= N and b <= N:
                conflict_pairs[x].append((a, b))
        for d in range(1, min(x - 1, N - x) + 1):
            a = x - d
            b = x + d
            if a >= 1 and b <= N:
                conflict_pairs[x].append((a, b))
        for d in range(1, (x - 1) // 2 + 1):
            a = x - 2 * d
            b = x - d
            if a >= 1 and b >= 1:
                conflict_pairs[x].append((a, b))
    
    def is_valid(x, S_bitset):
        for (a, b) in conflict_pairs[x]:
            if ((S_bitset >> (a - 1)) & 1) and ((S_bitset >> (b - 1)) & 1):
                return False
        return True
    
    def greedy_solve(order, S_init_bitset=0, S_init_score=0.0):
        S = S_init_bitset
        score = S_init_score
        for x in order:
            if (S >> (x - 1)) & 1:
                continue
            if is_valid(x, S):
                S |= (1 << (x - 1))
                score += weights[x]
        return S, score
    
    def simulated_annealing(S, score, max_iterations=10000, T0=1.0, T_end=0.01):
        best_S = S
        best_score = score
        current_S = S
        current_score = score
        
        all_elements = list(range(1, N + 1))
        
        for i in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            T = T0 * (T_end / T0) ** (i / max_iterations)
            
            move_type = random.random()
            best_gain = -1e9
            best_move = None
            
            if move_type < 0.3:
                # Add
                # Pick random x not in S
                while True:
                    x = random.choice(all_elements)
                    if not ((current_S >> (x - 1)) & 1):
                        break
                if is_valid(x, current_S):
                    gain = weights[x]
                    best_gain = gain
                    best_move = ('add', x)
            elif move_type < 0.6:
                # Remove
                # Pick random x in S
                in_set_count = bin(current_S).count('1')
                if in_set_count > 0:
                    # Find a random set bit
                    # Efficient way: pick random index, check if set, else retry
                    while True:
                        x = random.choice(all_elements)
                        if (current_S >> (x - 1)) & 1:
                            break
                    gain = -weights[x]
                    best_gain = gain
                    best_move = ('remove', x)
            else:
                # Swap
                in_set_count = bin(current_S).count('1')
                if in_set_count > 0:
                    while True:
                        u = random.choice(all_elements)
                        if (current_S >> (u - 1)) & 1:
                            break
                    while True:
                        v = random.choice(all_elements)
                        if not ((current_S >> (v - 1)) & 1):
                            break
                    S_temp = current_S & ~(1 << (u - 1))
                    if is_valid(v, S_temp):
                        gain = weights[v] - weights[u]
                        best_gain = gain
                        best_move = ('swap', u, v)
            
            if best_move:
                delta = best_gain
                if delta > 0 or random.random() < math.exp(delta / T):
                    if best_move[0] == 'add':
                        x = best_move[1]
                        current_S |= (1 << (x - 1))
                        current_score += weights[x]
                    elif best_move[0] == 'remove':
                        x = best_move[1]
                        current_S &= ~(1 << (x - 1))
                        current_score -= weights[x]
                    else:
                        u, v = best_move[1], best_move[2]
                        current_S &= ~(1 << (u - 1))
                        current_S |= (1 << (v - 1))
                        current_score += weights[v] - weights[u]
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S = current_S
        
        return best_S, best_score

    best_S = 0
    best_score = 0.0

    rng = random.Random(42)
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S, score = greedy_solve(order)
        S, score = simulated_annealing(S, score, max_iterations=5000)
        if score > best_score:
            best_score = score
            best_S = S

    result = []
    for x in range(1, N + 1):
        if (best_S >> (x - 1)) & 1:
            result.append(x)
    
    if not result:
        result = [1]
        
    return result
