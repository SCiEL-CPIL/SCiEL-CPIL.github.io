import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    def get_cost(A):
        sums = set()
        prods = set()
        n = len(A)
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return max(len(sums), len(prods))
        
    def local_search(A, time_limit):
        A = list(A)
        current_cost = get_cost(A)
        best_cost = current_cost
        best_A = list(A)
        T = 1000.0
        cooling = 0.9995
        n = 20
        
        while time.time() < time_limit:
            i = random.randint(0, n - 1)
            new_val = random.randint(1, 10000)
            if new_val in A:
                continue
            old_val = A[i]
            A[i] = new_val
            new_cost = get_cost(A)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_A = list(A)
                current_cost = new_cost
            else:
                if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                else:
                    A[i] = old_val
            
            T *= cooling
            if T < 1.0:
                T = 1.0
                
        return best_A, best_cost

    candidates = [
        list(range(1, 21)),
        [int(10 ** (4 * i / 19)) for i in range(20)],
    ]
    
    # Add a random candidate
    random_set = set()
    while len(random_set) < 20:
        random_set.add(random.randint(1, 10000))
    candidates.append(list(random_set))
    
    # Find best initial
    init_costs = [(get_cost(c), c) for c in candidates]
    init_costs.sort(key=lambda x: x[0])
    
    best_init_cost, best_init_set = init_costs[0]
    
    # Run local search
    final_set, final_cost = local_search(best_init_set, deadline)
    
    return final_set
