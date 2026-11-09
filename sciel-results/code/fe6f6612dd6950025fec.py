"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import heapq

def _all_block_sums(seq):
    """Return set of all nonempty consecutive block sums, or None if duplicate found."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return None
            sums.add(s)
    return sums

def _try_append(seq, sums, x):
    """Try to append x to seq. Returns (True, new_sums) if valid, else (False, None)."""
    tails = [x]
    total = x
    for a in reversed(seq):
        total += a
        tails.append(total)
    for t in tails:
        if t in sums:
            return False, None
    new_sums = sums | set(tails)
    return True, new_sums

def greedy_smallest(deadline):
    values, sums = [], set()
    for x in range(1, 257):
        if time.time() > deadline:
            break
        ok, new_sums = _try_append(values, sums, x)
        if ok:
            values.append(x)
            sums = new_sums
    return values

def greedy_largest(deadline):
    values, sums = [], set()
    for x in range(256, 0, -1):
        if time.time() > deadline:
            break
        ok, new_sums = _try_append(values, sums, x)
        if ok:
            values.append(x)
            sums = new_sums
    return sorted(values)

def greedy_with_gap(deadline, min_gap=1):
    values, sums = [], set()
    for x in range(1, 257):
        if time.time() > deadline:
            break
        if values and x - values[-1] < min_gap:
            continue
        ok, new_sums = _try_append(values, sums, x)
        if ok:
            values.append(x)
            sums = new_sums
    return values

def dag_longest_path(deadline, max_states=200000):
    """
    Model sequence construction as a DAG where nodes are (last_val, sums_fingerprint).
    Use priority queue to explore most promising states first.
    A state is (negative_length, last_val, seq_tuple, sums_frozen).
    We prune if we've seen a state with same last_val and sums is a superset.
    """
    start_time = time.time()
    
    # Priority queue: (-length, last_val, seq_tuple, sums_frozenset)
    # We want to explore longer sequences and smaller last values first
    pq = []
    visited = {}  # last_val -> list of (sums_frozenset, length) for dominance checking
    
    # Initialize: single-element sequences
    for x in range(1, 257):
        sums = frozenset([x])
        heapq.heappush(pq, (-1, x, (x,), sums))
    
    best_seq = []
    best_len = 0
    states_explored = 0
    
    while pq:
        if time.time() > deadline or states_explored >= max_states:
            break
        
        neg_len, last_val, seq, sums = heapq.heappop(pq)
        states_explored += 1
        cur_len = -neg_len
        
        if cur_len > best_len:
            best_len = cur_len
            best_seq = list(seq)
        
        # Try to extend
        for x in range(last_val + 1, min(last_val + 40, 257)):
            ok, new_sums_set = _try_append(list(seq), set(sums), x)
            if ok:
                new_sums = frozenset(new_sums_set)
                new_seq = seq + (x,)
                new_len = cur_len + 1
                
                # Pruning: check if this state is dominated
                dominated = False
                if x in visited:
                    for prev_sums, prev_len in visited[x]:
                        if new_sums <= prev_sums and new_len <= prev_len:
                            # New state is dominated by existing
                            dominated = True
                            break
                        elif prev_sums <= new_sums and prev_len <= new_len:
                            # Existing state is dominated, remove it
                            visited[x].remove((prev_sums, prev_len))
                            break
                
                if not dominated:
                    if x not in visited:
                        visited[x] = []
                    visited[x].append((new_sums, new_len))
                    heapq.heappush(pq, (-new_len, x, new_seq, new_sums))
        
        # Also try larger jumps for diversity
        if last_val + 40 < 256:
            for x in range(last_val + 40, min(last_val + 80, 257), 5):
                ok, new_sums_set = _try_append(list(seq), set(sums), x)
                if ok:
                    new_sums = frozenset(new_sums_set)
                    new_seq = seq + (x,)
                    new_len = cur_len + 1
                    
                    dominated = False
                    if x in visited:
                        for prev_sums, prev_len in visited[x]:
                            if new_sums <= prev_sums and new_len <= prev_len:
                                dominated = True
                                break
                            elif prev_sums <= new_sums and prev_len <= new_len:
                                visited[x].remove((prev_sums, prev_len))
                                break
                    
                    if not dominated:
                        if x not in visited:
                            visited[x] = []
                        visited[x].append((new_sums, new_len))
                        heapq.heappush(pq, (-new_len, x, new_seq, new_sums))
    
    return best_seq

def beam_search(deadline, beam_width=40):
    """Beam search over sequences, keeping top candidates by length and room."""
    start_time = time.time()
    
    beams = []
    for x in range(1, 100):
        sums = {x}
        beams.append(([x], sums))
    
    best = []
    
    while beams:
        if time.time() > deadline:
            break
        
        def score(b):
            seq, s = b
            # Prioritize length, then remaining room, then fewer distinct sums
            return (len(seq), 256 - seq[-1] if seq else 0, -len(s))
        
        beams.sort(key=score, reverse=True)
        beams = beams[:beam_width]
        
        new_beams = []
        for seq, sums in beams:
            if len(seq) > len(best):
                best = list(seq)
            if seq[-1] >= 256:
                continue
            
            for x in range(seq[-1]+1, min(seq[-1]+25, 257)):
                ok, new_sums = _try_append(seq, sums, x)
                if ok:
                    new_beams.append((seq + [x], new_sums))
            
            if seq[-1] + 25 < 256:
                for x in range(seq[-1]+25, min(seq[-1]+50, 257), 5):
                    ok, new_sums = _try_append(seq, sums, x)
                    if ok:
                        new_beams.append((seq + [x], new_sums))
        
        if not new_beams:
            break
        
        seen = set()
        deduped = []
        for seq, sums in new_beams:
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                deduped.append((seq, sums))
        
        beams = deduped[:beam_width * 3]
    
    return best if best else []

def local_search(start, deadline):
    """Hill-climbing: try replacing elements with alternatives that allow extension."""
    if not start:
        return start
    
    current = list(start)
    current_sums = _all_block_sums(current)
    if current_sums is None:
        return current
    
    best = list(current)
    
    for iteration in range(80):
        if time.time() > deadline:
            break
        improved = False
        
        for i in range(len(current)):
            orig = current[i]
            found = False
            
            for delta in range(1, 12):
                for new_val in [orig - delta, orig + delta]:
                    if new_val < 1 or new_val > 256:
                        continue
                    if i > 0 and new_val <= current[i-1]:
                        continue
                    if i < len(current)-1 and new_val >= current[i+1]:
                        continue
                    
                    trial = current.copy()
                    trial[i] = new_val
                    sums = _all_block_sums(trial)
                    
                    if sums is not None:
                        extended = list(trial)
                        ext_sums = sums
                        for x in range(extended[-1]+1, 257):
                            if time.time() > deadline:
                                break
                            ok, new_sums = _try_append(extended, ext_sums, x)
                            if ok:
                                extended.append(x)
                                ext_sums = new_sums
                        
                        if len(extended) > len(current):
                            current = extended
                            current_sums = ext_sums
                            improved = True
                            found = True
                            break
                        elif len(extended) == len(current) and extended[-1] < current[-1]:
                            current = extended
                            current_sums = ext_sums
                            improved = True
                            found = True
                            break
                if found:
                    break
            if improved:
                break
        
        if len(current) > len(best):
            best = list(current)
        
        if not improved:
            if current_sums is not None:
                for x in range(current[-1]+1, 257):
                    if time.time() > deadline:
                        break
                    ok, new_sums = _try_append(current, current_sums, x)
                    if ok:
                        current.append(x)
                        current_sums = new_sums
                        improved = True
                        break
        
        if not improved:
            break
    
    return best

def simulated_annealing(start, deadline):
    """SA: perturb elements and greedily extend, accepting worse solutions probabilistically."""
    if not start:
        return start
    
    current = list(start)
    current_sums = _all_block_sums(current)
    if current_sums is None:
        return start
    
    best = list(current)
    best_len = len(current)
    
    temperature = 5.0
    cooling = 0.995
    
    while time.time() < deadline and temperature > 0.05:
        # Perturb: replace one element
        i = random.randint(0, len(current)-1)
        orig = current[i]
        
        attempts = 0
        while attempts < 8:
            new_val = orig + random.randint(-8, 8)
            if new_val < 1 or new_val > 256:
                attempts += 1
                continue
            if i > 0 and new_val <= current[i-1]:
                attempts += 1
                continue
            if i < len(current)-1 and new_val >= current[i+1]:
                attempts += 1
                continue
            
            trial = current.copy()
            trial[i] = new_val
            sums = _all_block_sums(trial)
            
            if sums is not None:
                # Greedily extend
                ext = list(trial)
                ext_sums = sums
                for x in range(ext[-1]+1, 257):
                    if time.time() > deadline:
                        break
                    ok, new_sums = _try_append(ext, ext_sums, x)
                    if ok:
                        ext.append(x)
                        ext_sums = new_sums
                
                new_len = len(ext)
                if new_len > best_len:
                    current = ext
                    current_sums = ext_sums
                    best = list(current)
                    best_len = new_len
                elif new_len >= best_len:
                    # Accept with probability based on temperature
                    if random.random() < 0.15:
                        current = ext
                        current_sums = ext_sums
                break
            attempts += 1
        
        temperature *= cooling
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 88  # 88s safety margin (90s kill)
    
    candidates = []
    
    # Strategy 1: Greedy smallest (fast, good baseline)
    if time.time() < deadline - 5:
        c1 = greedy_smallest(deadline)
        candidates.append(c1)
    
    # Strategy 2: Greedy largest
    if time.time() < deadline - 5:
        c2 = greedy_largest(deadline)
        candidates.append(c2)
    
    # Strategy 3: Greedy with various gaps
    for gap in [1, 2, 3]:
        if time.time() < deadline - 10:
            c = greedy_with_gap(deadline, min_gap=gap)
            candidates.append(c)
    
    # Strategy 4: Multiple greedy variants with different starting points
    for start in range(1, 120, 3):
        if time.time() > deadline - 15:
            break
        values, sums = [], set()
        for x in range(start, 257):
            if time.time() > deadline:
                break
            ok, new_sums = _try_append(values, sums, x)
            if ok:
                values.append(x)
                sums = new_sums
        if values:
            candidates.append(values)
    
    # Strategy 5: DAG longest path (the new approach)
    if time.time() < deadline - 25:
        dag_deadline = min(deadline - 10, time.time() + 25)
        c5 = dag_longest_path(dag_deadline, max_states=250000)
        if c5:
            candidates.append(c5)
    
    # Strategy 6: Beam search
    if time.time() < deadline - 20:
        beam_deadline = min(deadline - 10, time.time() + 20)
        c6 = beam_search(beam_deadline, beam_width=40)
        if c6:
            candidates.append(c6)
    
    # Find best candidate so far
    best = max(candidates, key=len) if candidates else []
    
    # Refine using local search
    if time.time() < deadline - 12 and best:
        ls_deadline = min(deadline - 5, time.time() + 12)
        ls_result = local_search(best, ls_deadline)
        if len(ls_result) > len(best):
            best = ls_result
    
    # Simulated annealing on the best
    if time.time() < deadline - 6 and best:
        sa_deadline = min(deadline - 2, time.time() + 6)
        sa_result = simulated_annealing(best, sa_deadline)
        if len(sa_result) > len(best):
            best = sa_result
    
    # Final greedy extension
    if best and time.time() < deadline - 1:
        sums = _all_block_sums(best)
        if sums is not None:
            for x in range(best[-1]+1, 257):
                if time.time() > deadline - 1:
                    break
                ok, new_sums = _try_append(best, sums, x)
                if ok:
                    best.append(x)
                    sums = new_sums
    
    return best if best else greedy_smallest(deadline)
# EVOLVE-BLOCK-END
