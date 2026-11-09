import time
import random
from collections import deque

N = 12

def compute_threshold(trans_a, trans_b):
    """
    Computes the exact reset threshold using BFS on the subset automaton.
    Returns the length of the shortest reset word.
    """
    full_set = (1 << N) - 1
    visited = {full_set: 0}
    queue = deque([(full_set, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        if mask.bit_count() == 1:
            return dist
        
        new_mask_a = 0
        for i in range(N):
            if mask & (1 << i):
                new_mask_a |= (1 << trans_a[i])
        
        if new_mask_a not in visited:
            visited[new_mask_a] = dist + 1
            queue.append((new_mask_a, dist + 1))
            
        new_mask_b = 0
        for i in range(N):
            if mask & (1 << i):
                new_mask_b |= (1 << trans_b[i])
        
        if new_mask_b not in visited:
            visited[new_mask_b] = dist + 1
            queue.append((new_mask_b, dist + 1))
            
    return -1

def construct():
    """
    Uses fast BFS and simulated annealing to maximize the reset threshold.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill
    
    def fast_threshold(trans_a, trans_b):
        """Fast BFS with on-the-fly mask transition computation."""
        full_set = (1 << N) - 1
        
        # Precompute per-state destination bitmasks
        dest_a = [1 << trans_a[i] for i in range(N)]
        dest_b = [1 << trans_b[i] for i in range(N)]
        
        visited = {full_set: 0}
        queue = deque([(full_set, 0)])
        
        while queue:
            mask, dist = queue.popleft()
            if mask.bit_count() == 1:
                return dist
            
            # Compute next mask for letter a
            new_mask_a = 0
            m = mask
            while m:
                lsb = m & (-m)
                i = lsb.bit_length() - 1
                new_mask_a |= dest_a[i]
                m ^= lsb
            
            if new_mask_a not in visited:
                visited[new_mask_a] = dist + 1
                queue.append((new_mask_a, dist + 1))
                
            # Compute next mask for letter b
            new_mask_b = 0
            m = mask
            while m:
                lsb = m & (-m)
                i = lsb.bit_length() - 1
                new_mask_b |= dest_b[i]
                m ^= lsb
            
            if new_mask_b not in visited:
                visited[new_mask_b] = dist + 1
                queue.append((new_mask_b, dist + 1))
        
        return -1
    
    best_a = None
    best_b = None
    best_score = -1
    
    def update_best(a, b, score):
        nonlocal best_a, best_b, best_score
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
    
    # Family 1: Černý/Pawlik automaton
    a1 = [(i + 1) % N for i in range(N)]
    b1 = [i - 1 if i > 0 else 0 for i in range(N)]
    score1 = fast_threshold(a1, b1)
    update_best(a1, b1, score1)
    
    # Family 2: Cycle + constant reset
    a2 = [(i + 1) % N for i in range(N)]
    b2 = [0] * N
    score2 = fast_threshold(a2, b2)
    update_best(a2, b2, score2)
    
    # Family 3: Cycle + contracting map
    a3 = [(i + 1) % N for i in range(N)]
    b3 = [i // 2 if i % 2 == 0 else (i + 1) // 2 for i in range(N)]
    score3 = fast_threshold(a3, b3)
    update_best(a3, b3, score3)
    
    # Family 4: Parameterized cycle + partial reset
    for k in range(1, N):
        a4 = [(i + 1) % N for i in range(N)]
        b4 = [0 if i < k else i - 1 for i in range(N)]
        score4 = fast_threshold(a4, b4)
        update_best(a4, b4, score4)
        if time.time() > deadline:
            break
    
    # Family 5: Reverse cycle + constant
    a5 = [(i - 1) % N for i in range(N)]
    b5 = [0] * N
    score5 = fast_threshold(a5, b5)
    update_best(a5, b5, score5)
    
    # Family 6: Various permutations + reset
    import itertools
    perms = [
        [(i + 1) % N for i in range(N)],
        [(i - 1) % N for i in range(N)],
        [(-i) % N for i in range(N)],
        [(2 * i) % N for i in range(N)],
        [(3 * i) % N for i in range(N)],
        [(5 * i) % N for i in range(N)],
    ]
    for perm in perms:
        if len(set(perm)) != N:
            continue
        for reset_target in range(N):
            a6 = perm[:]
            b6 = [reset_target] * N
            score6 = fast_threshold(a6, b6)
            update_best(a6, b6, score6)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Family 7: Pawlik-type with parameter
    for k in range(1, N):
        a7 = [(i + 1) % N for i in range(N)]
        b7 = [0 if i == 0 else (i - 1 if i <= k else i) for i in range(N)]
        score7 = fast_threshold(a7, b7)
        update_best(a7, b7, score7)
        if time.time() > deadline:
            break
    
    # Family 8: Bidirectional cycle with asymmetric reset
    for hole_pos in range(N):
        a8 = [(i + 1) % N for i in range(N)]
        b8 = []
        for i in range(N):
            if i == 0:
                b8.append(0)
            elif i == hole_pos:
                b8.append(0)
            else:
                b8.append(i - 1)
        score8 = fast_threshold(a8, b8)
        update_best(a8, b8, score8)
        if time.time() > deadline:
            break
    
    # Family 9: Known high-threshold structure
    a9 = [(i + 1) % N for i in range(N)]
    b9 = []
    for i in range(N):
        if i == 0:
            b9.append(0)
        elif i % 2 == 0:
            b9.append(i - 2)
        else:
            b9.append(i - 1)
    score9 = fast_threshold(a9, b9)
    update_best(a9, b9, score9)
    
    # Family 10: Two-letter structure where a is a long cycle and b has a specific pattern
    # Try: a = cycle, b maps state i to (i - d) % N for various d, with one exception
    for d in range(1, N):
        for exc_state in range(N):
            for exc_dest in range(N):
                a10 = [(i + 1) % N for i in range(N)]
                b10 = []
                for i in range(N):
                    if i == exc_state:
                        b10.append(exc_dest)
                    else:
                        b10.append((i - d) % N)
                score10 = fast_threshold(a10, b10)
                update_best(a10, b10, score10)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # If we have time, do simulated annealing from the best found
    if time.time() < deadline - 10.0 and best_a is not None:
        random.seed(42)
        current_a = best_a[:]
        current_b = best_b[:]
        current_score = best_score
        
        # Simulated annealing parameters
        initial_temp = 5.0
        cooling_rate = 0.999
        min_temp = 0.01
        max_steps = 50000
        
        temp = initial_temp
        steps_done = 0
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            if temp < min_temp:
                break
            
            letter = random.randint(0, 1)
            state = random.randint(0, N - 1)
            old_dest = current_a[state] if letter == 0 else current_b[state]
            
            # Try a random new destination
            new_dest = random.randint(0, N - 1)
            if new_dest == old_dest:
                continue
            
            if letter == 0:
                current_a[state] = new_dest
            else:
                current_b[state] = new_dest
                
            new_score = fast_threshold(current_a, current_b)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_a = current_a[:]
                    best_b = current_b[:]
            else:
                # Accept with probability exp(delta / temp)
                delta = new_score - current_score
                if delta < 0:
                    import math
                    prob = math.exp(delta / temp)
                    if random.random() > prob:
                        # Reject: revert
                        if letter == 0:
                            current_a[state] = old_dest
                        else:
                            current_b[state] = old_dest
                        new_score = current_score
                # If delta >= 0, already accepted above
            
            temp *= cooling_rate
            steps_done += 1
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END
