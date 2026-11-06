import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 60s limit (90s kill, 5s buffer)
    N = 256
    
    # Precompute 3-APs and 4-APs (0-based indexing)
    threes = []
    for d in range(1, N // 2 + 1):
        for a in range(0, N - 2 * d):
            threes.append((a, a + d, a + 2 * d))
    
    fours = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            fours.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Build index maps for fast delta computation
    # For each element, store which APs it's in and the other elements
    elem_to_3aps = [[] for _ in range(N)]
    for a, b, c in threes:
        elem_to_3aps[a].append((b, c))
        elem_to_3aps[b].append((a, c))
        elem_to_3aps[c].append((a, b))
    
    elem_to_4aps = [[] for _ in range(N)]
    for a, b, c, d in fours:
        elem_to_4aps[a].append((b, c, d))
        elem_to_4aps[b].append((a, c, d))
        elem_to_4aps[c].append((a, b, d))
        elem_to_4aps[d].append((a, b, c))
    
    def greedy_init():
        """Greedy construction: add element if it doesn't create a 4-AP"""
        state = [False] * N
        for x in range(N):
            creates_4ap = False
            for o1, o2, o3 in elem_to_4aps[x]:
                if state[o1] and state[o2] and state[o3]:
                    creates_4ap = True
                    break
            if not creates_4ap:
                state[x] = True
        return state
    
    def count_3aps(state):
        """Count 3-APs fully contained in state"""
        count = 0
        for a, b, c in threes:
            if state[a] and state[b] and state[c]:
                count += 1
        return count
    
    def count_4aps(state):
        """Count 4-APs fully contained in state"""
        count = 0
        for a, b, c, d in fours:
            if state[a] and state[b] and state[c] and state[d]:
                count += 1
        return count
    
    def delta_3aps(state, x, new_val):
        """Compute change in 3-AP count if state[x] becomes new_val"""
        sign = 1 if new_val else -1
        delta = 0
        for o1, o2 in elem_to_3aps[x]:
            if state[o1] and state[o2]:
                delta += sign
        return delta
    
    def delta_4aps(state, x, new_val):
        """Compute change in 4-AP count if state[x] becomes new_val"""
        sign = 1 if new_val else -1
        delta = 0
        for o1, o2, o3 in elem_to_4aps[x]:
            if state[o1] and state[o2] and state[o3]:
                delta += sign
        return delta
    
    # Parallel tempering setup
    num_replicas = 12
    T_min = 1.0
    T_max = 50.0
    temperatures = [T_min * (T_max / T_min) ** (i / (num_replicas - 1)) for i in range(num_replicas)]
    
    # Initialize replicas
    replicas = []
    for i in range(num_replicas):
        # Start with greedy, then perturb based on temperature
        state = greedy_init()
        if i > 0:
            # Higher temperature replicas get more perturbation
            num_flips = int(5 * (i / (num_replicas - 1)))
            for _ in range(num_flips):
                x = random.randint(0, N - 1)
                state[x] = not state[x]
        
        s3 = count_3aps(state)
        s4 = count_4aps(state)
        score = s3 - 100.0 * s4  # Heavy penalty for 4-APs
        
        replicas.append({
            'state': state,
            's3': s3,
            's4': s4,
            'score': score,
            'T': temperatures[i]
        })
    
    # Track best valid solution
    best_valid_state = None
    best_valid_score = -1
    
    for r in replicas:
        if r['s4'] == 0 and r['s3'] > best_valid_score:
            best_valid_score = r['s3']
            best_valid_state = r['state'].copy()
    
    # Parallel tempering loop
    swap_interval = max(1, num_replicas // 2)
    step_count = 0
    
    while time.time() < deadline:
        # Perform one step on each replica
        for i in range(num_replicas):
            if time.time() > deadline:
                break
            
            r = replicas[i]
            state = r['state']
            T = r['T']
            
            # Choose a random element to flip
            x = random.randint(0, N - 1)
            new_val = not state[x]
            
            # Compute deltas
            d3 = delta_3aps(state, x, new_val)
            d4 = delta_4aps(state, x, new_val)
            
            # New scores
            new_s3 = r['s3'] + d3
            new_s4 = r['s4'] + d4
            new_score = new_s3 - 100.0 * new_s4
            
            # Acceptance criterion
            if new_score >= r['score']:
                accept = True
            else:
                delta_score = new_score - r['score']
                if T > 0:
                    accept = random.random() < (1.0 / T) * delta_score
                else:
                    accept = False
            
            if accept:
                state[x] = new_val
                r['state'] = state
                r['s3'] = new_s3
                r['s4'] = new_s4
                r['score'] = new_score
                
                # Update best valid solution
                if new_s4 == 0 and new_s3 > best_valid_score:
                    best_valid_score = new_s3
                    best_valid_state = state.copy()
        
        step_count += 1
        
        # Swap neighboring replicas based on Metropolis criterion
        if step_count % swap_interval == 0:
            for i in range(num_replicas - 1):
                if time.time() > deadline:
                    break
                
                r1 = replicas[i]
                r2 = replicas[i + 1]
                
                # Compute acceptance probability for swap
                # P(accept) = min(1, exp((1/T1 - 1/T2) * (score1 - score2)))
                T1 = r1['T']
                T2 = r2['T']
                s1 = r1['score']
                s2 = r2['score']
                
                if T1 == T2:
                    accept_swap = True
                else:
                    exponent = (1.0 / T1 - 1.0 / T2) * (s1 - s2)
                    if exponent >= 0:
                        accept_swap = True
                    else:
                        accept_swap = random.random() < exponent
                
                if accept_swap:
                    # Swap states and scores
                    replicas[i], replicas[i + 1] = replicas[i + 1], replicas[i]
    
    # Final verification
    if best_valid_state is None:
        # Fallback to greedy
        best_valid_state = greedy_init()
    
    # Convert to 1-based and return
    result = [i + 1 for i in range(N) if best_valid_state[i]]
    
    # Verify no 4-APs
    state_1based = [False] * N
    for x in result:
        state_1based[x - 1] = True
    if count_4aps(state_1based) > 0:
        # Should not happen, but fallback
        result = [i + 1 for i in range(N) if greedy_init()[i]]
    
    return sorted(result)

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
    # Quick count
    state = [False] * 256
    for x in res:
        state[x - 1] = True
    s3 = count_3aps(state)
    s4 = count_4aps(state)
    print(f"3-APs: {s3}, 4-APs: {s4}")
