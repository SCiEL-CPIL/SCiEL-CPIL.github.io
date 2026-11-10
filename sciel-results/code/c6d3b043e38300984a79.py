import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    for n in [22, 21, 20]:
        if time.time() > deadline - 5:
            break
        
        max_attempts = 10 if n == 22 else (20 if n == 21 else 40)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            A = np.random.randint(0, 2, (n, n))
            np.fill_diagonal(A, 0)
            A = (A + A.T) // 2
            
            valid = False
            repair_rounds = 0
            max_repair_rounds = 300
            consecutive_clean = 0
            clean_threshold = 20
            
            while repair_rounds < max_repair_rounds and not valid:
                if time.time() > deadline - 5:
                    break
                
                found_violation = False
                violation_subset = None
                
                # Sample 300 subsets
                for _ in range(300):
                    k = random.randint(6, n)
                    S = random.sample(range(n), k)
                    
                    sub_A = A[np.ix_(S, S)]
                    degs = sub_A.sum(axis=1)
                    
                    if len(set(degs)) == 1:
                        found_violation = True
                        violation_subset = S
                        break
                
                if not found_violation:
                    consecutive_clean += 1
                    if consecutive_clean >= clean_threshold:
                        # High confidence, do full check if n <= 20
                        if n <= 20:
                            full_valid = True
                            for k in range(6, n + 1):
                                for S in combinations(range(n), k):
                                    S = list(S)
                                    sub_A = A[np.ix_(S, S)]
                                    degs = sub_A.sum(axis=1)
                                    if len(set(degs)) == 1:
                                        full_valid = False
                                        found_violation = True
                                        violation_subset = S
                                        break
                                if not full_valid:
                                    break
                            if full_valid:
                                valid = True
                        else:
                            valid = True
                else:
                    consecutive_clean = 0
                    if violation_subset is not None:
                        u, v = random.sample(violation_subset, 2)
                        A[u, v] = 1 - A[u, v]
                        A[v, u] = A[u, v]
                
                repair_rounds += 1
            
            if valid:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if A[i, j] == 1:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}

    return {'n': 5, 'edges': []}
