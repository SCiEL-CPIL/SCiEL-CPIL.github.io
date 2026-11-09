"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

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
    """Greedy that enforces minimum gap between consecutive elements."""
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

def backtracking_search(initial, deadline, max_nodes=500000):
    """Depth-first backtracking with greedy pruning."""
    if not initial:
        return []
    
    sums = _all_block_sums(initial)
    if sums is None:
        return list(initial)
    
    best = list(initial)
    nodes = 0
    
    def dfs(seq, sums, candidates):
        nonlocal best, nodes
        if time.time() > deadline or nodes > max_nodes:
            return
        nodes += 1
        if len(seq) > len(best):
            best = list(seq)
        if not candidates:
            return
        
        for i, x in enumerate(candidates):
            ok, new_sums = _try_append(seq, sums, x)
            if ok:
                remaining = candidates[i+1:]
                seq.append(x)
                dfs(seq, new_sums, remaining)
                seq.pop()
                if time.time() > deadline or nodes > max_nodes:
                    return
    
    candidates = [c for c in range(initial[-1]+1, 257)]
    dfs(list(initial), sums, candidates)
    return best

def beam_search(deadline, beam_width=20):
    """Beam search over sequences, keeping top candidates by length and 'room'."""
    start_time = time.time()
    
    # Each beam entry: (seq, sums, score)
    # Score: length * 100 + (256 - last element) to favor shorter last (more room)
    beams = []
    
    # Initialize with first element candidates
    for x in range(1, 50):
        sums = {x}
        beams.append(([x], sums))
    
    best = []
    
    max_len = 0
    while beams:
        if time.time() > deadline:
            break
        
        # Score and sort beams
        def score(b):
            seq, s = b
            return (len(seq), 256 - seq[-1] if seq else 0)
        
        beams.sort(key=score, reverse=True)
        
        # Keep top beam_width
        beams = beams[:beam_width]
        
        # Expand each beam
        new_beams = []
        for seq, sums in beams:
            if len(seq) > len(best):
                best = list(seq)
            if len(seq) > max_len:
                max_len = len(seq)
            
            if seq[-1] >= 256:
                continue
            
            # Try next few candidates
            for x in range(seq[-1]+1, min(seq[-1]+15, 257)):
                ok, new_sums = _try_append(seq, sums, x)
                if ok:
                    new_beams.append((seq + [x], new_sums))
        
        # Also try larger jumps for diversity
        for seq, sums in beams:
            if seq[-1] + 15 < 256:
                for x in range(seq[-1]+15, min(seq[-1]+30, 257), 3):
                    ok, new_sums = _try_append(seq, sums, x)
                    if ok:
                        new_beams.append((seq + [x], new_sums))
        
        if not new_beams:
            break
        
        # Deduplicate by sequence
        seen = set()
        deduped = []
        for seq, sums in new_beams:
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                deduped.append((seq, sums))
        
        beams = deduped[:beam_width * 2]
    
    if len(best) > 0:
        return best
    return []

def local_search(start, deadline):
    """Hill-climbing: try replacing elements with alternatives that allow extension."""
    if not start:
        return start
    
    current = list(start)
    current_sums = _all_block_sums(current)
    if current_sums is None:
        return current
    
    best = list(current)
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        improved = False
        
        for i in range(len(current)):
            orig = current[i]
            found = False
            
            # Try nearby values
            for delta in range(1, 10):
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
                        # Try to extend
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
                        elif len(extended) == len(current):
                            # Accept if it opens more room
                            if extended[-1] < current[-1]:
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
            # Try to extend current
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

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    # Generate diverse starting points
    candidates = []
    
    # Strategy 1: Greedy smallest
    if time.time() < deadline:
        c1 = greedy_smallest(deadline)
        candidates.append(c1)
    
    # Strategy 2: Greedy largest
    if time.time() < deadline:
        c2 = greedy_largest(deadline)
        candidates.append(c2)
    
    # Strategy 3: Greedy with gap
    if time.time() < deadline:
        c3 = greedy_with_gap(deadline, min_gap=1)
        candidates.append(c3)
        c4 = greedy_with_gap(deadline, min_gap=2)
        candidates.append(c4)
    
    # Strategy 4: Multiple greedy variants with different starting points
    for start in range(1, 60, 2):
        if time.time() > deadline - 10:
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
    
    # Strategy 5: Beam search
    if time.time() < deadline - 20:
        c5 = beam_search(deadline, beam_width=20)
        if c5:
            candidates.append(c5)
    
    # Find best candidate so far
    best = max(candidates, key=len) if candidates else []
    
    # Refine best candidates using local search
    if time.time() < deadline - 15 and best:
        ls_result = local_search(best, deadline)
        if len(ls_result) > len(best):
            best = ls_result
    
    # Use backtracking on the best candidate if time permits
    if time.time() < deadline - 10 and best:
        bt_result = backtracking_search(best, deadline, max_nodes=100000)
        if len(bt_result) > len(best):
            best = bt_result
    
    # Final greedy extension
    if best and time.time() < deadline:
        sums = _all_block_sums(best)
        if sums is not None:
            for x in range(best[-1]+1, 257):
                if time.time() > deadline:
                    break
                ok, new_sums = _try_append(best, sums, x)
                if ok:
                    best.append(x)
                    sums = new_sums
    
    return best if best else greedy_smallest(deadline)
# EVOLVE-BLOCK-END
