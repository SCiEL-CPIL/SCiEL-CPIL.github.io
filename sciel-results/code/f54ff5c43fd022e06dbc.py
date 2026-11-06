import time
import random
import math
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def compute_threshold(trans_a, trans_b):
    """
    Computes the length of the shortest reset word for the automaton defined by trans_a and trans_b.
    Returns 0 if not synchronizing.
    Optimized for speed.
    """
    # Precompute bit masks for single states
    trans_a_bits = [1 << x for x in trans_a]
    trans_b_bits = [1 << x for x in trans_b]
    
    next_a = [0] * (1 << N)
    next_b = [0] * (1 << N)
    
    # Precompute transitions for all masks
    for mask in range(1 << N):
        ma = 0
        mb = 0
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            ma |= trans_a_bits[i]
            mb |= trans_b_bits[i]
            m ^= lsb
        next_a[mask] = ma
        next_b[mask] = mb

    # BFS from FULL_MASK
    dist = [-1] * (1 << N)
    dist[FULL_MASK] = 0
    queue = deque([FULL_MASK])
    
    while queue:
        cur = queue.popleft()
        # Check if singleton
        if cur & (cur - 1) == 0:
            return dist[cur]
        
        nxt_a = next_a[cur]
        if dist[nxt_a] == -1:
            dist[nxt_a] = dist[cur] + 1
            queue.append(nxt_a)
            
        nxt_b = next_b[cur]
        if dist[nxt_b] == -1:
            dist[nxt_b] = dist[cur] + 1
            queue.append(nxt_b)
            
    return 0 # Not synchronizing

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    # Initial seeds
    starts = []
    
    # Seed 1: Cycle a, Staircase b (b[i] = max(0, i-1))
    a1 = [(i + 1) % N for i in range(N)]
    b1 = [max(0, i-1) for i in range(N)]
    starts.append((a1, b1))
    
    # Seed 2: Cycle a, Fold b (b[i] = i % 2)
    a2 = [(i + 1) % N for i in range(N)]
    b2 = [i % 2 for i in range(N)]
    starts.append((a2, b2))
    
    # Seed 3: Cycle a, Random b
    a3 = [(i + 1) % N for i in range(N)]
    b3 = [random.randint(0, N-1) for _ in range(N)]
    starts.append((a3, b3))
    
    # Seed 4: Cycle a, "Ping-Pong" like structure
    # b[0]=0, b[1]=0, b[2]=1, b[3]=2, ..., b[N-1]=N-2
    a4 = [(i + 1) % N for i in range(N)]
    b4 = [0] * N
    for i in range(1, N):
        b4[i] = i - 1
    b4[1] = 0 # Explicitly set
    starts.append((a4, b4))
    
    best_score = 0
    best_a = None
    best_b = None
    
    # Evaluate seeds
    for a_start, b_start in starts:
        score = compute_threshold(a_start, b_start)
        if score > best_score:
            best_score = score
            best_a = a_start[:]
            best_b = b_start[:]
            
    # Simulated Annealing
    # Start from the best seed
    cur_a = best_a[:]
    cur_b = best_b[:]
    cur_score = best_score
    
    T = 10.0 # Initial temperature
    cooling = 0.999
    
    while time.time() < deadline:
        # Mutation
        state = random.randint(0, N-1)
        if random.random() < 0.5:
            old_dest = cur_a[state]
            new_dest = random.randint(0, N-1)
            if new_dest == old_dest:
                continue
            cur_a[state] = new_dest
            new_score = compute_threshold(cur_a, cur_b)
            delta = new_score - cur_score
            if delta > 0 or random.random() < math.exp(delta / T):
                cur_a[state] = new_dest
                cur_score = new_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_a = cur_a[:]
                    best_b = cur_b[:]
            else:
                cur_a[state] = old_dest
        else:
            old_dest = cur_b[state]
            new_dest = random.randint(0, N-1)
            if new_dest == old_dest:
                continue
            cur_b[state] = new_dest
            new_score = compute_threshold(cur_a, cur_b)
            delta = new_score - cur_score
            if delta > 0 or random.random() < math.exp(delta / T):
                cur_b[state] = new_dest
                cur_score = new_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_a = cur_a[:]
                    best_b = cur_b[:]
            else:
                cur_b[state] = old_dest
        
        T *= cooling
        if T < 0.1:
            T = 10.0 # Reheat if stuck too low? Or just keep going.
            # Reheating can help escape local optima.
            # Let's reheat if we haven't improved in a while.
            # Simple reheat: if T < 1.0, set T = 5.0
            if T < 1.0:
                T = 5.0

    # Ensure we return a synchronizing automaton
    if best_score == 0:
        # Fallback to seed
        return [[(i + 1) % N for i in range(N)], [0] * N]
        
    return [best_a, best_b]
