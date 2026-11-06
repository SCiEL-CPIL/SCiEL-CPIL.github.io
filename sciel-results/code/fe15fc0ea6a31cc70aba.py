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
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin for safety
    
    # Generate strong initial candidates
    starts = []
    
    # 1. Černý Automaton Structure
    # a: cycle 0->1->...->N-1->0
    # b: maps i to i-1 for i>0, and 0 to 0 (or some variant that creates long chains)
    # Standard Černý: a is cycle, b maps x to x-1 mod N? No, usually b maps to 0 or folds.
    # Let's use the classic "Shift and Fold":
    # a[i] = (i+1)%N
    # b[i] = 0 if i==0 else i-1  (This is the standard Černý automaton for N states)
    a_cz = [(i + 1) % N for i in range(N)]
    b_cz = [0] * N
    for i in range(1, N):
        b_cz[i] = i - 1
    starts.append((a_cz, b_cz))
    
    # 2. Variant: a cycle, b maps to 0 except last maps to 0
    a2 = [(i + 1) % N for i in range(N)]
    b2 = [0] * N
    starts.append((a2, b2))
    
    # 3. Random but structured: a is cycle, b is random permutation-like
    a3 = [(i + 1) % N for i in range(N)]
    b3 = list(range(N))
    random.shuffle(b3)
    starts.append((a3, b3))
    
    # 4. Another heuristic: a is cycle, b maps i to i//2
    a4 = [(i + 1) % N for i in range(N)]
    b4 = [i // 2 for i in range(N)]
    starts.append((a4, b4))
    
    best_score = 0
    best_a = None
    best_b = None
    
    # Simulated Annealing Parameters
    # We will run SA for each start, or maybe just one long SA from the best start?
    # Better: Run SA from each start for a fraction of the time, or just one main SA loop
    # that can jump between starts if stuck? 
    # Simpler and robust: Run SA from the best initial candidate found in the first pass.
    
    # First, evaluate all starts to find the best one to seed the SA
    for a_start, b_start in starts:
        score = compute_threshold(a_start, b_start)
        if score > best_score:
            best_score = score
            best_a = a_start[:]
            best_b = b_start[:]
            
    # If we found a good one, we use it as the start for SA.
    # If best_score is 0, pick a random one or the first one.
    if best_a is None:
        best_a = starts[0][0][:]
        best_b = starts[0][1][:]
        best_score = compute_threshold(best_a, best_b)
        
    cur_a = best_a[:]
    cur_b = best_b[:]
    cur_score = best_score
    
    # Simulated Annealing Loop
    # Temperature schedule: Start high, decay exponentially
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # We need to ensure we don't timeout.
    # Estimate cost of one iteration: compute_threshold is ~1-2ms.
    # 55s / 2ms = 27,500 iterations possible.
    
    while time.time() < deadline:
        # Mutation
        # 1. Single transition change (most common)
        # 2. Swap transitions
        move_type = random.random()
        
        old_a = cur_a[:]
        old_b = cur_b[:]
        
        if move_type < 0.7:
            # Change single transition
            state = random.randint(0, N-1)
            if random.random() < 0.5:
                old_dest = cur_a[state]
                new_dest = random.randint(0, N-1)
                if new_dest == old_dest:
                    continue
                cur_a[state] = new_dest
            else:
                old_dest = cur_b[state]
                new_dest = random.randint(0, N-1)
                if new_dest == old_dest:
                    continue
                cur_b[state] = new_dest
        else:
            # Swap two transitions in the same letter
            if random.random() < 0.5:
                i, j = random.sample(range(N), 2)
                cur_a[i], cur_a[j] = cur_a[j], cur_a[i]
            else:
                i, j = random.sample(range(N), 2)
                cur_b[i], cur_b[j] = cur_b[j], cur_b[i]
        
        new_score = compute_threshold(cur_a, cur_b)
        
        # Acceptance Criterion
        if new_score > cur_score:
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_a = cur_a[:]
                best_b = cur_b[:]
        else:
            # Accept worse solution with probability exp(-delta / T)
            delta = cur_score - new_score
            if delta < 0:
                # This shouldn't happen in else, but safe
                pass
            else:
                # If new_score == cur_score, delta is 0, exp(0)=1, always accept (or prob 1)
                # If new_score < cur_score, delta > 0
                prob = 1.0
                if delta > 0:
                    prob = 2.71828 ** (-delta / T)
                
                if random.random() < prob:
                    cur_score = new_score
                else:
                    # Reject: revert
                    cur_a = old_a
                    cur_b = old_b
        
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Final Check: Ensure we return a valid synchronizing automaton
    if best_score == 0:
        # Fallback: Return a known synchronizing automaton
        # Cycle + Reset to 0
        return [[(i + 1) % N for i in range(N)], [0] * N]
        
    return [best_a, best_b]
