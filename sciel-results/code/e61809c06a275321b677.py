import time
import random
import sys

N = 12
FULL_MASK = (1 << N) - 1

def compute_threshold(trans_a, trans_b):
    """
    Computes the length of the shortest reset word for the automaton defined by trans_a and trans_b.
    Returns 0 if not synchronizing.
    """
    next_a = [0] * (1 << N)
    next_b = [0] * (1 << N)
    
    for mask in range(1, 1 << N):
        lb = mask & -mask
        i = (lb.bit_length() - 1)
        prev_mask = mask ^ lb
        
        next_a[mask] = next_a[prev_mask] | (1 << trans_a[i])
        next_b[mask] = next_b[prev_mask] | (1 << trans_b[i])
        
    dist = [-1] * (1 << N)
    dist[FULL_MASK] = 0
    queue = [FULL_MASK]
    head = 0
    
    while head < len(queue):
        cur = queue[head]
        head += 1
        
        if cur & (cur - 1) == 0:
            return dist[cur]
        
        nxt_a = next_a[cur]
        nxt_b = next_b[cur]
        
        if dist[nxt_a] == -1:
            dist[nxt_a] = dist[cur] + 1
            queue.append(nxt_a)
            
        if dist[nxt_b] == -1:
            dist[nxt_b] = dist[cur] + 1
            queue.append(nxt_b)
            
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Start with a strong candidate
    # The Černý automaton structure
    a_start = [(i + 1) % N for i in range(N)]
    b_start = [0] * N
    for i in range(1, N):
        b_start[i] = i - 1
    # b_start[0] = 0
    
    cur_a = a_start[:]
    cur_b = b_start[:]
    cur_score = compute_threshold(cur_a, cur_b)
    
    best_score = cur_score
    best_a = cur_a[:]
    best_b = cur_b[:]
    
    # Local Search with Simulated Annealing
    temp = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Mutation
        state = random.randint(0, N-1)
        if random.random() < 0.5:
            old_dest = cur_a[state]
            new_dest = random.randint(0, N-1)
            if new_dest == old_dest: continue
            cur_a[state] = new_dest
            new_score = compute_threshold(cur_a, cur_b)
            if new_score > cur_score:
                cur_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_a = cur_a[:]
                    best_b = cur_b[:]
                    temp = 10.0 # Reset temp on improvement
            else:
                # Accept with probability p
                delta = new_score - cur_score
                if delta < 0:
                    p = pow(2.71828, delta / temp)
                    if random.random() < p:
                        cur_score = new_score
                    else:
                        cur_a[state] = old_dest
                # If delta == 0, accept (random walk)
                else:
                    cur_score = new_score
        else:
            old_dest = cur_b[state]
            new_dest = random.randint(0, N-1)
            if new_dest == old_dest: continue
            cur_b[state] = new_dest
            new_score = compute_threshold(cur_a, cur_b)
            if new_score > cur_score:
                cur_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_a = cur_a[:]
                    best_b = cur_b[:]
                    temp = 10.0
            else:
                delta = new_score - cur_score
                if delta < 0:
                    p = pow(2.71828, delta / temp)
                    if random.random() < p:
                        cur_score = new_score
                    else:
                        cur_b[state] = old_dest
                else:
                    cur_score = new_score
        
        temp *= cooling
        if temp < 0.1:
            temp = 0.1

    if best_score == 0:
        return [[(i + 1) % N for i in range(N)], [0] * N]
        
    return [best_a, best_b]
