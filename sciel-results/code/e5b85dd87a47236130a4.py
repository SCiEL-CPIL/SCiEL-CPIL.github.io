import random
import time
import math

def construct():
    N = 512
    
    # Precompute all 4-term arithmetic progressions
    all_aps = []
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map each number to the indices of APs it belongs to
    ap_indices_by_num = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(all_aps):
        for x in ap:
            ap_indices_by_num[x].append(idx)
            
    num_aps = len(all_aps)
    
    # State
    S = set()
    ap_count = [0] * num_aps
    
    def add_element(x):
        for idx in ap_indices_by_num[x]:
            if ap_count[idx] >= 3:
                return False
        for idx in ap_indices_by_num[x]:
            ap_count[idx] += 1
        S.add(x)
        return True
        
    def remove_element(x):
        for idx in ap_indices_by_num[x]:
            ap_count[idx] -= 1
        S.remove(x)
        
    def greedy_fill_recorded():
        added = []
        for x in range(1, N + 1):
            if x not in S:
                if add_element(x):
                    added.append(x)
        return added
        
    def greedy_build(order):
        nonlocal S, ap_count
        S = set()
        ap_count = [0] * num_aps
        for x in order:
            add_element(x)
        greedy_fill_recorded()
        return len(S)
        
    # 1. Greedy Initialization
    best_size = 0
    best_set = set()
    
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
    ]
    for _ in range(10):
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        orders.append(random_order)
        
    for order in orders:
        size = greedy_build(order)
        if size > best_size:
            best_size = size
            best_set = set(S)
            
    # 2. Simulated Annealing
    S = set(best_set)
    ap_count = [0] * num_aps
    for x in S:
        for idx in ap_indices_by_num[x]:
            ap_count[idx] += 1
            
    current_size = len(S)
    best_size_sa = current_size
    best_set_sa = set(S)
    
    start_time = time.time()
    deadline = 55.0
    
    T = 1.0
    cooling_rate = 0.9995
    
    while time.time() - start_time < deadline:
        if not S:
            break
            
        x = random.choice(list(S))
        saved_size = current_size
        
        # Remove x
        remove_element(x)
        
        # Greedy fill with recording
        added = greedy_fill_recorded()
        
        new_size = len(S)
        delta = new_size - saved_size
        
        if delta >= 0:
            # Accept
            current_size = new_size
            if current_size > best_size_sa:
                best_size_sa = current_size
                best_set_sa = set(S)
        else:
            # delta is negative
            prob = math.exp(delta / T)
            if random.random() < prob:
                # Accept
                current_size = new_size
            else:
                # Revert
                for y in added:
                    remove_element(y)
                add_element(x)
                current_size = saved_size
                
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
    return sorted(best_set_sa)
