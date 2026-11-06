import time
import random

N = 12
FULL_MASK = (1 << N) - 1

def compute_threshold(trans_a, trans_b):
    """
    Computes the length of the shortest reset word for the automaton defined by trans_a and trans_b.
    Returns 0 if not synchronizing.
    Uses DP to compute subset transitions in O(2^N) time.
    """
    # Precompute transition of subsets using DP
    # next_a[mask] is the resulting mask after applying 'a'
    next_a = [0] * (1 << N)
    next_b = [0] * (1 << N)
    
    # DP: next_trans[mask] = next_trans[mask without LSB] | (1 << trans[LSB_index])
    for mask in range(1, 1 << N):
        lb = mask & -mask
        i = (lb.bit_length() - 1)
        prev_mask = mask ^ lb
        
        next_a[mask] = next_a[prev_mask] | (1 << trans_a[i])
        next_b[mask] = next_b[prev_mask] | (1 << trans_b[i])
        
    # BFS from FULL_MASK
    dist = [-1] * (1 << N)
    dist[FULL_MASK] = 0
    queue = [FULL_MASK]
    head = 0
    
    while head < len(queue):
        cur = queue[head]
        head += 1
        
        # Check if singleton
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
            
    return 0 # Not synchronizing

def construct():
    N = 12
    
    # Generate Černý automaton candidates
    # Standard Černý automaton:
    # a: i -> (i+1) % N
    # b: 0->0, 1->0, i->i+1 (2<=i<=N-2), N-1->N-2
    
    a_1 = [(i + 1) % N for i in range(N)]
    b_1 = [0] * N
    b_1[0] = 0
    b_1[1] = 0
    for i in range(2, N - 1):
        b_1[i] = i + 1
    b_1[N - 1] = N - 2
    
    # Variant 2: Swapped letters
    a_2 = b_1[:]
    b_2 = a_1[:]
    
    # Variant 3: Reverse cycle for a
    a_3 = [(i - 1) % N for i in range(N)]
    b_3 = b_1[:]
    
    # Variant 4: Reverse cycle for a, swapped
    a_4 = b_1[:]
    b_4 = a_3[:]
    
    candidates = [
        (a_1, b_1),
        (a_2, b_2),
        (a_3, b_3),
        (a_4, b_4)
    ]
    
    best_score = 0
    best_a = None
    best_b = None
    
    for a, b in candidates:
        score = compute_threshold(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
            
    # If we haven't reached the theoretical max, try local search
    if best_score < (N - 1) ** 2:
        start_time = time.time()
        deadline = start_time + 50.0
        
        cur_a = best_a[:]
        cur_b = best_b[:]
        cur_score = best_score
        
        while time.time() < deadline:
            # Try to improve via single state mutation
            # Focus on states that might be causing bottlenecks, but random is fine
            if random.random() < 0.5:
                idx = random.randint(0, N - 1)
                old_dest = cur_a[idx]
                new_dest = random.randint(0, N - 1)
                if new_dest == old_dest:
                    continue
                cur_a[idx] = new_dest
                new_score = compute_threshold(cur_a, cur_b)
                if new_score > cur_score:
                    cur_score = new_score
                    if cur_score > best_score:
                        best_score = cur_score
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                        if best_score >= (N - 1) ** 2:
                            break
                else:
                    cur_a[idx] = old_dest
            else:
                idx = random.randint(0, N - 1)
                old_dest = cur_b[idx]
                new_dest = random.randint(0, N - 1)
                if new_dest == old_dest:
                    continue
                cur_b[idx] = new_dest
                new_score = compute_threshold(cur_a, cur_b)
                if new_score > cur_score:
                    cur_score = new_score
                    if cur_score > best_score:
                        best_score = cur_score
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                        if best_score >= (N - 1) ** 2:
                            break
                else:
                    cur_b[idx] = old_dest

    if best_score == 0:
        return [[(i + 1) % N for i in range(N)], [0] * N]
        
    return [best_a, best_b]
