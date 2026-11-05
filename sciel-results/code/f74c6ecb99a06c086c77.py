import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    for n in [20, 19, 18]:
        if time.time() > deadline - 25:
            break
        result = _solve_n(n, deadline)
        if result is not None:
            return result
    
    return {'n': 5, 'edges': []}

def _solve_n(n, deadline):
    deadline_global = deadline - 10
    
    # Precompute 6-subsets
    six_subs = list(combinations(range(n), 6))
    
    # Random restart local search
    for restart in range(200):
        if time.time() > deadline_global:
            break
        
        # Random initial graph
        adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = adj[j][i] = 1
        
        # Local search
        for iteration in range(20000):
            if time.time() > deadline_global:
                break
            
            # Find violations
            violations = []
            for s in six_subs:
                d0 = 0
                for w in s:
                    if w != s[0] and adj[s[0]][w]:
                        d0 += 1
                is_reg = True
                for v in s[1:]:
                    d = 0
                    for w in s:
                        if w != v and adj[v][w]:
                            d += 1
                    if d != d0:
                        is_reg = False
                        break
                if is_reg:
                    violations.append(s)
            
            if not violations:
                # Check larger subsets (limited)
                if _check_larger(adj, n, deadline_global):
                    edge_list = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i][j]:
                                edge_list.append([i, j])
                    return {'n': n, 'edges': edge_list}
            
            # Fix random violation
            s = random.choice(violations)
            i, j = random.choice(list(combinations(s, 2)))
            adj[i][j] = 1 - adj[i][j]
            adj[j][i] = adj[i][j]
    
    return None

def _check_larger(adj, n, deadline):
    for k in range(7, min(n+1, 10)):  # Only check up to size 9 for speed
        for s in combinations(range(n), k):
            if time.time() > deadline:
                return True
            d0 = 0
            for w in s:
                if w != s[0] and adj[s[0]][w]:
                    d0 += 1
            for v in s[1:]:
                d = 0
                for w in s:
                    if w != v and adj[v][w]:
                        d += 1
                if d != d0:
                    break
            else:
                return False
    return True
