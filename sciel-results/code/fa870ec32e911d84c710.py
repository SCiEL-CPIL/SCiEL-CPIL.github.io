import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def is_addable(A, counts, x):
        if counts[2*x] >= 2:
            return False
        for a in A:
            if counts[x+a] >= 2:
                return False
        return True

    def add_element(A, counts, x):
        A.append(x)
        counts[2*x] += 1
        for a in A[:-1]:
            counts[x+a] += 1

    def remove_element(A, counts, a):
        # Find index
        idx = A.index(a)
        A.pop(idx)
        counts[2*a] -= 1
        for a2 in A:
            counts[a+a2] -= 1

    def greedy_run(order):
        A = []
        counts = [0] * 2001
        for x in order:
            if is_addable(A, counts, x):
                add_element(A, counts, x)
        return A, counts

    best_A = []
    best_size = 0
    best_counts = [0] * 2001

    # Phase 1: Randomized Greedy
    # 500 restarts is fast. Each run ~1ms. 0.5s total.
    for i in range(1000):
        if time.time() > deadline:
            break
        order = list(range(1, 1001))
        random.shuffle(order)
        A, counts = greedy_run(order)
        if len(A) > best_size:
            best_size = len(A)
            best_A = A
            best_counts = counts

    # Phase 2: Local Search
    A = best_A
    counts = best_counts
    set_A = set(A)
    
    # To speed up "not in A", we can just use the set.
    # But we need to pick random x not in A.
    # Rejection sampling.
    
    while time.time() < deadline:
        # Try to add
        added = False
        # Try a few random additions
        for _ in range(10):
            x = random.randint(1, 1000)
            if x in set_A:
                continue
            if is_addable(A, counts, x):
                add_element(A, counts, x)
                set_A.add(x)
                added = True
                break
        
        if not added:
            # Try a swap
            if len(A) == 0:
                break
            a = random.choice(A)
            # Find b not in A
            b = random.randint(1, 1000)
            while b in set_A:
                b = random.randint(1, 1000)
            
            # Try swap
            # Remove a
            remove_element(A, counts, a)
            set_A.remove(a)
            
            # Check b
            if is_addable(A, counts, b):
                # Accept
                add_element(A, counts, b)
                set_A.add(b)
            else:
                # Revert
                add_element(A, counts, a)
                set_A.add(a)
        
        # Update best
        if len(A) > best_size:
            best_size = len(A)
            best_A = A.copy()
            best_counts = counts.copy()

    return best_A
