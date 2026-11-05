import random
import time

def construct():
    n = 20
    max_val = 10000
    start_time = time.time()
    deadline = start_time + 80  # 80s to be safe
    
    def set_sizes(A):
        sums = set()
        prods = set()
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return len(sums), len(prods)
    
    def objective(A):
        s, p = set_sizes(A)
        return max(s, p)
    
    best_A = None
    best_score = float('inf')
    
    def try_candidate(A):
        nonlocal best_A, best_score
        if len(set(A)) != n:
            return
        if any(x < 1 or x > max_val for x in A):
            return
        score = objective(A)
        if score < best_score:
            best_score = score
            best_A = list(A)
    
    # Multiple initializations
    random.seed(12345)
    
    # AP
    try_candidate(list(range(1, 21)))
    try_candidate(list(range(1, 21, 2)))  # 1,3,5,...,39
    try_candidate(list(range(10, 200, 10)))  # 10,20,...,190
    
    # GP
    for base in [2, 3, 4]:
        A = [base**i for i in range(n)]
        if max(A) <= max_val:
            try_candidate(A)
    
    # Random
    for _ in range(200):
        A = random.sample(range(1, max_val + 1), n)
        try_candidate(A)
    
    # Local search
    current_A = list(best_A)
    current_score = best_score
    temperature = 500.0
    
    while time.time() < deadline:
        A = list(current_A)
        move = random.random()
        
        if move < 0.4:
            i = random.randrange(n)
            A[i] = max(1, min(max_val, A[i] + random.randint(-30, 30)))
        elif move < 0.7:
            i = random.randrange(n)
            A[i] = random.randint(1, max_val)
        else:
            i, j = random.sample(range(n), 2)
            A[i] = max(1, min(max_val, A[i] + random.randint(-30, 30)))
            A[j] = max(1, min(max_val, A[j] + random.randint(-30, 30)))
        
        if len(set(A)) < n:
            continue
        
        new_score = objective(A)
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temperature, 0.01)):
            current_A = A
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = list(current_A)
        
        temperature *= 0.9995
        if temperature < 1:
            temperature = 500.0
    
    return best_A
