import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
    
    def is_3ap_free(S_set):
        """Check if S_set has no 3-term AP."""
        for a in S_set:
            for b in S_set:
                if a >= b:
                    continue
                c = 2 * b - a
                if c > b and c in S_set:
                    return False
        return True
    
    def can_add(x, S_set):
        """Check if x can be added to S_set without creating a 3-AP."""
        for a in S_set:
            # x is middle: a, x, c => c = 2x - a
            c = 2 * x - a
            if 1 <= c <= N and c != x and c != a and c in S_set:
                return False
            # a is middle: x, a, c => c = 2a - x
            c = 2 * a - x
            if 1 <= c <= N and c != x and c != a and c in S_set:
                return False
            # x, a are endpoints: x, m, a => m = (x+a)/2
            s = x + a
            if s % 2 == 0:
                m = s // 2
                if 1 <= m <= N and m != x and m != a and m in S_set:
                    return False
        return True
    
    def greedy_init(order):
        """Greedy add elements in given order."""
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S
    
    def local_search(S, deadline):
        """Improve S via local search."""
        S = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_gain = 1e-9
            best_move = None
            
            # Try single additions
            for x in range(1, N + 1):
                if time.time() > deadline:
                    break
                if x not in S and can_add(x, S):
                    gain = weights[x]
                    if gain > best_gain:
                        best_gain = gain
                        best_move = ('add', x, None)
            
            # Try single swaps: remove u, add v
            for v in range(1, N + 1):
                if time.time() > deadline:
                    break
                if v in S:
                    continue
                # Find conflicts
                for u in S:
                    S_temp = S - {u}
                    if can_add(v, S_temp):
                        gain = weights[v] - weights[u]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = ('swap', v, u)
            
            # Try 2-for-1: remove u, v, add w
            for w in range(1, N + 1):
                if time.time() > deadline:
                    break
                if w in S:
                    continue
                for u in S:
                    if time.time() > deadline:
                        break
                    S_temp = S - {u}
                    if not can_add(w, S_temp):
                        continue
                    for v in S_temp:
                        if v >= u:
                            continue
                        S_temp2 = S_temp - {v}
                        if can_add(w, S_temp2):
                            gain = weights[w] - weights[u] - weights[v]
                            if gain > best_gain:
                                best_gain = gain
                                best_move = ('swap2', w, u, v)
            
            if best_move:
                if best_move[0] == 'add':
                    x = best_move[1]
                    S.add(x)
                elif best_move[0] == 'swap':
                    v, u = best_move[1], best_move[2]
                    S.remove(u)
                    S.add(v)
                elif best_move[0] == 'swap2':
                    w, u, v = best_move[1], best_move[2], best_move[3]
                    S.remove(u)
                    S.remove(v)
                    S.add(w)
                improved = True
        
        return S
    
    # Try multiple initializations
    best_S = set()
    best_score = 0.0
    
    # Standard greedy
    S1 = greedy_init(range(1, N + 1))
    S1 = local_search(S1, deadline)
    score1 = sum(weights[x] for x in S1)
    if score1 > best_score:
        best_score = score1
        best_S = S1
    
    # Reverse greedy (add larger first)
    S2 = greedy_init(range(N, 0, -1))
    S2 = local_search(S2, deadline)
    score2 = sum(weights[x] for x in S2)
    if score2 > best_score:
        best_score = score2
        best_S = S2
    
    # Odd-only then add evens
    S3 = set()
    for x in range(1, N + 1, 2):
        S3.add(x)
    for x in range(2, N + 1, 2):
        if can_add(x, S3):
            S3.add(x)
    S3 = local_search(S3, deadline)
    score3 = sum(weights[x] for x in S3)
    if score3 > best_score:
        best_score = score3
        best_S = S3
    
    # Powers of 2 based
    S4 = set()
    x = 1
    while x <= N:
        S4.add(x)
        x *= 2
    for x in range(1, N + 1):
        if x not in S4 and can_add(x, S4):
            S4.add(x)
    S4 = local_search(S4, deadline)
    score4 = sum(weights[x] for x in S4)
    if score4 > best_score:
        best_score = score4
        best_S = S4
    
    # Final polish on best_S
    if time.time() < deadline:
        best_S = local_search(best_S, deadline)
    
    return list(best_S)

