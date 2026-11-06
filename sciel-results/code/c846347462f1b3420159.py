import time
import random
import itertools

N = 12
FULL_MASK = (1 << N) - 1

def compute_threshold(trans_a, trans_b):
    """
    Computes the length of the shortest reset word for the automaton defined by trans_a and trans_b.
    Returns 0 if not synchronizing.
    """
    # Precompute transition of subsets
    # We can compute next_a and next_b more efficiently.
    # For each mask, next_a[mask] = OR of (1 << trans_a[i]) for i in mask.
    
    # Optimization: Precompute for each state i, the bit mask for a and b.
    # bit_a[i] = 1 << trans_a[i]
    # bit_b[i] = 1 << trans_b[i]
    # Then next_a[mask] = OR of bit_a[i] for i where mask has bit i.
    
    # Even better: Since N=12, 2^12=4096. 
    # We can compute next_a and next_b by iterating masks and using dynamic programming or just simple loops.
    # Simple loop over 4096 masks * 12 states is ~50k ops. 
    # Current code does this. 
    # Let's try to speed up the inner loop.
    
    bit_a = [1 << x for x in trans_a]
    bit_b = [1 << x for x in trans_b]
    
    next_a = [0] * (1 << N)
    next_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        ma = 0
        mb = 0
        m = mask
        i = 0
        while m:
            if m & 1:
                ma |= bit_a[i]
                mb |= bit_b[i]
            i += 1
            m >>= 1
        # The above while loop is faster than range(N) with if mask & (1<<i)
        # because it skips zero bits.
        next_a[mask] = ma
        next_b[mask] = mb

    # BFS from FULL_MASK
    # dist[mask] = distance from FULL_MASK
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
    # 90s limit. Leave 5s margin.
    deadline = start_time + 85.0
    
    # Initial seeds
    # We want high synchronization threshold.
    # Known good structures often involve a cycle and a map that "breaks" the cycle slowly.
    
    candidates = []
    
    # Start 1: Cycle + specific map
    a1 = [(i + 1) % N for i in range(N)]
    b1 = [0] * N
    b1[0] = 1
    b1[1] = 0
    candidates.append((a1, b1))
    
    # Start 2: Cycle + random map
    a2 = [(i + 1) % N for i in range(N)]
    b2 = [random.randint(0, N-1) for _ in range(N)]
    candidates.append((a2, b2))
    
    # Start 3: Another structure
    a3 = [(i + 1) % N for i in range(N)]
    b3 = [i % 2 for i in range(N)]
    candidates.append((a3, b3))
    
    best_score = 0
    best_a = None
    best_b = None
    
    for a_start, b_start in candidates:
        score = compute_threshold(a_start, b_start)
        if score > best_score:
            best_score = score
            best_a = a_start[:]
            best_b = b_start[:]
            
        # Local Search
        cur_a = a_start[:]
        cur_b = b_start[:]
        cur_score = score
        
        # If current is not synchronizing, we might want to force synchronization first?
        # Or just let the search try to fix it.
        # If cur_score is 0, we might get stuck in non-sync region.
        # Let's add a check: if cur_score == 0, try to find a sync word by random mutation?
        # Actually, if it's not synchronizing, compute_threshold returns 0.
        # The search will try to improve score. If score is 0, any synchronizing automaton has score > 0.
        
        iters = 0
        while time.time() < deadline:
            iters += 1
            
            # Mutation strategy:
            # Pick a random state.
            # Pick a random letter (a or b).
            # Try ALL possible destinations for that state/letter? 
            # N=12, so 12 options. 
            # Current code picks 1 random destination.
            # Trying all 12 allows finding a better neighbor if one exists, 
            # at the cost of up to 12 evaluations.
            # 12 evaluations * ~1ms = 12ms. 
            # 85s / 12ms = ~7000 iterations.
            # This is reasonable.
            
            state = random.randint(0, N-1)
            use_a = random.random() < 0.5
            
            if use_a:
                old_dest = cur_a[state]
                # Try all other destinations
                best_new_score = cur_score
                best_new_dest = old_dest
                
                for new_dest in range(N):
                    if new_dest == old_dest:
                        continue
                    cur_a[state] = new_dest
                    new_score = compute_threshold(cur_a, cur_b)
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_dest = new_dest
                    if time.time() > deadline:
                        break
                cur_a[state] = best_new_dest
                cur_score = best_new_score
                
            else:
                old_dest = cur_b[state]
                best_new_score = cur_score
                best_new_dest = old_dest
                
                for new_dest in range(N):
                    if new_dest == old_dest:
                        continue
                    cur_b[state] = new_dest
                    new_score = compute_threshold(cur_a, cur_b)
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_dest = new_dest
                    if time.time() > deadline:
                        break
                cur_b[state] = best_new_dest
                cur_score = best_new_score
            
            if cur_score > best_score:
                best_score = cur_score
                best_a = cur_a[:]
                best_b = cur_b[:]
                
            # Random restart if stuck for a while?
            # If we haven't improved in last 500 iterations, maybe restart from best found so far with a random perturbation?
            # For now, let's just keep searching.
            
    # Ensure we return a synchronizing automaton
    if best_score == 0:
        # Fallback to a known synchronizing automaton
        # a: cycle, b: all to 0
        a_fb = [(i + 1) % N for i in range(N)]
        b_fb = [0] * N
        return [a_fb, b_fb]
        
    return [best_a, best_b]
