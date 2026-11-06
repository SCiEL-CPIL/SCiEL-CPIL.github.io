import time
import random
import itertools

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
    
    # Precompute transitions for all subsets
    # Using local variables for speed
    ta = trans_a
    tb = trans_b
    na = next_a
    nb = next_b
    
    for mask in range(1, 1 << N):
        lb = mask & -mask
        i = lb.bit_length() - 1
        prev_mask = mask ^ lb
        
        na[mask] = na[prev_mask] | (1 << ta[i])
        nb[mask] = nb[prev_mask] | (1 << tb[i])
        
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
        
        nxt_a = na[cur]
        nxt_b = nb[cur]
        
        if dist[nxt_a] == -1:
            dist[nxt_a] = dist[cur] + 1
            queue.append(nxt_a)
            
        if dist[nxt_b] == -1:
            dist[nxt_b] = dist[cur] + 1
            queue.append(nxt_b)
            
    return 0 # Not synchronizing

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin for safety
    
    best_score = 0
    best_a = None
    best_b = None
    
    # Strategy 1: Cycle + Defect family
    # a is a cycle: a[i] = (i+1) % N
    # b has structure: b[i] = (i-1) % N for most i, but some states map to a sink
    # The defect positions determine the reset threshold
    
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # Try different sink targets and defect patterns
    # For each possible sink s, try various subsets of states that map to s
    # while others follow the reverse cycle
    
    # First, try the classic Černý construction (all except one map to 0)
    for sink in range(N):
        # Pattern: b[i] = (i-1) % N unless i == sink, then b[sink] = sink
        # This is the standard Černý automaton variant
        b = [(i - 1) % N for i in range(N)]
        b[sink] = sink
        score = compute_threshold(a_cycle, b)
        if score > best_score:
            best_score = score
            best_a = a_cycle[:]
            best_b = b[:]
    
    # Now do constructive search: vary which states are "defective" (map to sink)
    # and where the sink is
    
    # Beam search over defect configurations
    # State: (set of defective states, sink)
    # Start with all single-defect configurations
    
    candidates = []
    for sink in range(N):
        for defect in range(N):
            b = [(i - 1) % N for i in range(N)]
            b[defect] = sink
            score = compute_threshold(a_cycle, b)
            if score > 0:
                candidates.append((score, sink, [defect], b[:]))
    
    # Sort by score descending
    candidates.sort(key=lambda x: -x[0])
    
    # Keep top beam
    beam_size = 20
    beam = candidates[:beam_size]
    
    # Expand: add one more defect to each beam member
    max_defects = 4  # Limit number of defects to keep search tractable
    
    while time.time() < deadline:
        new_beam = []
        for score, sink, defects, b in beam:
            if len(defects) >= max_defects:
                new_beam.append((score, sink, defects, b[:]))
                continue
            
            # Try adding one more defect
            for new_defect in range(N):
                if new_defect in defects:
                    continue
                new_defects = defects + [new_defect]
                new_b = b[:]
                new_b[new_defect] = sink
                new_score = compute_threshold(a_cycle, new_b)
                if new_score > 0:
                    new_beam.append((new_score, sink, new_defects, new_b[:]))
        
        # Also try changing the sink for existing defect sets
        for score, sink, defects, b in beam:
            for new_sink in range(N):
                if new_sink == sink:
                    continue
                new_b = [(i - 1) % N for i in range(N)]
                for d in defects:
                    new_b[d] = new_sink
                new_score = compute_threshold(a_cycle, new_b)
                if new_score > 0:
                    new_beam.append((new_score, new_sink, defects[:], new_b[:]))
        
        # Deduplicate and keep top beam
        seen = set()
        unique_beam = []
        for item in new_beam:
            key = (item[1], tuple(sorted(item[2])))
            if key not in seen:
                seen.add(key)
                unique_beam.append(item)
        
        unique_beam.sort(key=lambda x: -x[0])
        beam = unique_beam[:beam_size]
        
        # Update best
        if beam:
            top_score = beam[0][0]
            if top_score > best_score:
                best_score = top_score
                best_a = a_cycle[:]
                best_b = beam[0][3][:]
        
        # If top score is stable and high, we can also try local mutations
        # on the best found so far
        if best_score > 0 and best_b is not None:
            # Try small mutations on the best automaton
            for _ in range(5):
                if time.time() >= deadline:
                    break
                cur_a = best_a[:]
                cur_b = best_b[:]
                # Mutate one transition in b
                state = random.randint(0, N-1)
                old_val = cur_b[state]
                new_val = random.randint(0, N-1)
                cur_b[state] = new_val
                score = compute_threshold(cur_a, cur_b)
                if score > best_score:
                    best_score = score
                    best_a = cur_a[:]
                    best_b = cur_b[:]
        
        # If beam is empty or all scores are 0, break
        if not beam or beam[0][0] == 0:
            break
    
    # Strategy 2: Also try some random structured automata as backup
    if time.time() < deadline - 5:
        for _ in range(50):
            if time.time() >= deadline:
                break
            # Random b with bias toward low values
            a = a_cycle[:]
            b = [random.randint(0, 3) for _ in range(N)]
            score = compute_threshold(a, b)
            if score > best_score:
                best_score = score
                best_a = a[:]
                best_b = b[:]
    
    # Final check: Ensure we return a valid synchronizing automaton
    if best_score == 0 or best_a is None:
        # Fallback: classic Černý
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [(i - 1) % N for i in range(N)]
        best_b[0] = 0
        best_score = compute_threshold(best_a, best_b)
    
    return [best_a, best_b]
