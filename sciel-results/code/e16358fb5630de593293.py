# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    time_limit = 58.0  # Safety margin for 60s limit
    
    best_A = []
    best_B = []
    best_size = 0

    def can_add(x, target_set, target_sums, target_diffs, other_diffs):
        # Check Sidon condition for target_set
        if 2 * x in target_sums:
            return False
        for a in target_set:
            if x + a in target_sums:
                return False
            d = abs(x - a)
            if d in target_diffs or d in other_diffs:
                return False
        return True

    def add_element(x, target_set, target_sums, target_diffs):
        target_set.append(x)
        target_sums.add(2 * x)
        for a in target_set[:-1]:
            target_sums.add(x + a)
            d = abs(x - a)
            target_diffs.add(d)

    def remove_element(x, target_set, target_sums, target_diffs):
        # This is expensive, so we only use it in local search if necessary
        # For now, we just rebuild if we remove, or we track carefully.
        # Given time constraints, we will stick to adding in local search or 
        # simple swaps that don't require complex removal logic if we can avoid it.
        # Actually, for local search, we can just try to add. 
        # To improve, we might need to remove. 
        # Let's implement a simple remove by rebuilding the sets.
        pass

    def greedy_construct(order, strategy):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in order:
            if time.time() - start_time > time_limit:
                break
            
            added = False
            # Strategy 0: Balanced (try the smaller set first)
            # Strategy 1: A first
            # Strategy 2: B first
            # Strategy 3: Random
            
            try_A = False
            try_B = False
            
            if strategy == 0:
                if len(A) < len(B):
                    try_A = True
                    try_B = True
                else:
                    try_B = True
                    try_A = True
            elif strategy == 1:
                try_A = True
                try_B = True
            elif strategy == 2:
                try_B = True
                try_A = True
            else:
                if random.random() < 0.5:
                    try_A = True
                    try_B = True
                else:
                    try_B = True
                    try_A = True
            
            # The order of trying matters.
            # If strategy 0, we want to prioritize the smaller one.
            # Let's refine the logic inside the loop based on strategy.
            
            if strategy == 0:
                if len(A) <= len(B):
                    if can_add(x, A, sums_A, diffs_A, diffs_B):
                        add_element(x, A, sums_A, diffs_A)
                        added = True
                    elif can_add(x, B, sums_B, diffs_B, diffs_A):
                        add_element(x, B, sums_B, diffs_B)
                        added = True
                else:
                    if can_add(x, B, sums_B, diffs_B, diffs_A):
                        add_element(x, B, sums_B, diffs_B)
                        added = True
                    elif can_add(x, A, sums_A, diffs_A, diffs_B):
                        add_element(x, A, sums_A, diffs_A)
                        added = True
            elif strategy == 1:
                if can_add(x, A, sums_A, diffs_A, diffs_B):
                    add_element(x, A, sums_A, diffs_A)
                    added = True
                elif can_add(x, B, sums_B, diffs_B, diffs_A):
                    add_element(x, B, sums_B, diffs_B)
                    added = True
            elif strategy == 2:
                if can_add(x, B, sums_B, diffs_B, diffs_A):
                    add_element(x, B, sums_B, diffs_B)
                    added = True
                elif can_add(x, A, sums_A, diffs_A, diffs_B):
                    add_element(x, A, sums_A, diffs_A)
                    added = True
            else:
                if random.random() < 0.5:
                    if can_add(x, A, sums_A, diffs_A, diffs_B):
                        add_element(x, A, sums_A, diffs_A)
                        added = True
                    elif can_add(x, B, sums_B, diffs_B, diffs_A):
                        add_element(x, B, sums_B, diffs_B)
                        added = True
                else:
                    if can_add(x, B, sums_B, diffs_B, diffs_A):
                        add_element(x, B, sums_B, diffs_B)
                        added = True
                    elif can_add(x, A, sums_A, diffs_A, diffs_B):
                        add_element(x, A, sums_A, diffs_A)
                        added = True

        return A, B

    def local_search(A, B):
        # Try to add any remaining elements
        # This is a simple pass. We can iterate this a few times.
        # We need to be careful with time.
        A_set = set(A)
        B_set = set(B)
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Rebuild state
        for a in A:
            sums_A.add(2*a)
            for b in A:
                if a <= b:
                    sums_A.add(a+b)
            for b in A:
                if a != b:
                    diffs_A.add(abs(a-b))
        
        for b in B:
            sums_B.add(2*b)
            for a in B:
                if b <= a:
                    sums_B.add(b+a)
            for a in B:
                if b != a:
                    diffs_B.add(abs(b-a))
            
        # Actually, rebuilding is O(N^2). N is small (<64). 64^2 = 4096.
        # This is fast enough.
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for x in range(1, 257):
                if time.time() - start_time > time_limit:
                    break
                if x in A_set or x in B_set:
                    continue
                if can_add(x, A, sums_A, diffs_A, diffs_B):
                    add_element(x, A, sums_A, diffs_A)
                    A_set.add(x)
                    improved = True
                elif can_add(x, B, sums_B, diffs_B, diffs_A):
                    add_element(x, B, sums_B, diffs_B)
                    B_set.add(x)
                    improved = True
        return A, B

    # 1. Run deterministic strategies
    orders = [
        list(range(1, 257)),
        list(range(256, 0, -1)),
    ]
    strategies = [0, 1, 2]
    
    for order in orders:
        for strat in strategies:
            if time.time() - start_time > time_limit:
                break
            A, B = greedy_construct(order, strat)
            A, B = local_search(A, B)
            size = len(A) + len(B)
            if size > best_size:
                best_size = size
                best_A, best_B = A, B

    # 2. Random restarts
    while time.time() - start_time < time_limit:
        random.seed()
        order = list(range(1, 257))
        random.shuffle(order)
        strat = random.randint(0, 3)
        A, B = greedy_construct(order, strat)
        A, B = local_search(A, B)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A, best_B = A, B

    return [best_A, best_B]
# EVOLVE-BLOCK-END
