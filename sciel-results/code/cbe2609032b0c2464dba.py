"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _compute_sums_bitset(seq):
    """Return integer bitset of all nonempty consecutive block sums."""
    sums = 0
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums |= (1 << s)
    return sums

def _try_append_bitset(seq, sums_bitset, x):
    """Try to append x to seq using bitset. Returns (True, new_sums_bitset) if valid, else (False, None)."""
    tails = [x]
    total = x
    for a in reversed(seq):
        total += a
        tails.append(total)
    # Check for conflicts using bitset
    mask = 0
    for t in tails:
        if t > 256 * 15:  # Max possible sum for 15 elements of 256
            return False, None
        mask |= (1 << t)
    if mask & sums_bitset:
        return False, None
    return True, sums_bitset | mask

def _validate_seq_bitset(seq):
    """Validate that all consecutive block sums are distinct using bitset."""
    sums = 0
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if (sums >> s) & 1:
                return False
            sums |= (1 << s)
    return True

def greedy_smallest(deadline):
    values, sums = [], 0
    for x in range(1, 257):
        if time.time() > deadline:
            break
        ok, new_sums = _try_append_bitset(values, sums, x)
        if ok:
            values.append(x)
            sums = new_sums
    return values

def greedy_largest(deadline):
    values, sums = [], 0
    for x in range(256, 0, -1):
        if time.time() > deadline:
            break
        ok, new_sums = _try_append_bitset(values, sums, x)
        if ok:
            values.append(x)
            sums = new_sums
    return sorted(values)

def _backtracking_search(deadline, max_depth=15, start_val=1):
    """
    Backtracking search with bitset optimization.
    Tries to find a sequence of length max_depth starting with values >= start_val.
    """
    best = []
    
    def dfs(seq, sums_bitset, min_next):
        nonlocal best
        if time.time() > deadline:
            return
        if len(seq) > len(best):
            best = list(seq)
        if len(seq) >= max_depth:
            return
        
        # Pruning: if we can't possibly beat best, stop
        # (This is a weak prune, but helps)
        
        for x in range(min_next, 257):
            if time.time() > deadline:
                return
            ok, new_sums = _try_append_bitset(seq, sums_bitset, x)
            if ok:
                seq.append(x)
                dfs(seq, new_sums, x + 1)
                seq.pop()
                if len(best) >= max_depth:
                    return
    
    dfs([], 0, start_val)
    return best

def _beam_search_bitset(deadline, beam_width=50):
    """
    Beam search using bitset representation.
    State: (sequence_tuple, sums_bitset)
    """
    # Initial beam: single-element sequences
    beam = []
    for x in range(1, 257):
        beam.append(((x,), 1 << x))
    
    best_seq = []
    
    while time.time() < deadline:
        next_beam = {}
        for seq_tuple, sums_bitset in beam:
            last_val = seq_tuple[-1]
            seq_list = list(seq_tuple)
            
            for x in range(last_val + 1, 257):
                ok, new_sums = _try_append_bitset(seq_list, sums_bitset, x)
                if ok:
                    new_seq = seq_tuple + (x,)
                    key = new_seq
                    if key not in next_beam:
                        next_beam[key] = (new_seq, new_sums)
        
        if not next_beam:
            break
        
        # Score: prefer longer, then smaller last element
        scored = []
        for seq_tuple, sums_bitset in next_beam.values():
            scored.append((len(seq_tuple), -seq_tuple[-1], seq_tuple, sums_bitset))
        
        scored.sort(reverse=True)
        beam = [(s, sf) for _, _, s, sf in scored[:beam_width]]
        
        best_in_beam = max(beam, key=lambda x: len(x[0]))
        if len(best_in_beam[0]) > len(best_seq):
            best_seq = list(best_in_beam[0])
        
        if not beam:
            break
        
        max_last = max(s[-1] for s, _ in beam)
        if max_last >= 256:
            break
    
    return best_seq

def _greedy_extend_from_bitset(seq, deadline):
    """Greedily extend a sequence with smallest valid values using bitset."""
    if not seq:
        return []
    sums = _compute_sums_bitset(seq)
    current = list(seq)
    for x in range(current[-1] + 1, 257):
        if time.time() > deadline:
            break
        ok, new_sums = _try_append_bitset(current, sums, x)
        if ok:
            current.append(x)
            sums = new_sums
    return current

def _local_search_bitset(seq, deadline, max_iterations=100):
    """
    Local search: try replacing elements with nearby values to enable better extensions.
    """
    if not seq or len(seq) < 2:
        return seq
    
    current = list(seq)
    best = list(current)
    best_len = len(best)
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        improved = False
        
        for i in range(len(current)):
            if time.time() > deadline:
                break
            
            orig = current[i]
            lb = current[i-1] + 1 if i > 0 else 1
            ub = current[i+1] - 1 if i < len(current) - 1 else 256
            
            # Try nearby values
            for new_val in range(max(lb, orig - 5), min(ub, orig + 5) + 1):
                if new_val == orig:
                    continue
                
                trial = current.copy()
                trial[i] = new_val
                
                # Validate using bitset
                if not _validate_seq_bitset(trial):
                    continue
                
                # Try to extend
                ext = _greedy_extend_from_bitset(trial, deadline)
                
                if len(ext) > best_len:
                    best = ext
                    current = ext
                    best_len = len(best)
                    improved = True
                    break
                elif len(ext) == best_len and ext[-1] < best[-1]:
                    best = ext
                    current = ext
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best

def _randomized_greedy_bitset(seq, deadline, num_trials=15):
    """Try multiple greedy extensions with randomized value ordering using bitset."""
    if not seq:
        return []
    best = list(seq)
    
    for _ in range(num_trials):
        if time.time() > deadline:
            break
        current = list(seq)
        sums = _compute_sums_bitset(current)
        
        candidates = list(range(current[-1] + 1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            ok, new_sums = _try_append_bitset(current, sums, x)
            if ok:
                current.append(x)
                sums = new_sums
        
        if len(current) > len(best):
            best = current
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    candidates = []
    
    # Phase 1: Greedy strategies
    if time.time() < deadline:
        candidates.append(greedy_smallest(deadline))
    
    if time.time() < deadline:
        candidates.append(greedy_largest(deadline))
    
    # Multiple greedy starts with different initial values
    for start in range(1, 50, 3):
        if time.time() > deadline - 15:
            break
        values, sums = [], 0
        for x in range(start, 257):
            if time.time() > deadline:
                break
            ok, new_sums = _try_append_bitset(values, sums, x)
            if ok:
                values.append(x)
                sums = new_sums
        if values:
            candidates.append(values)
    
    best = max(candidates, key=len) if candidates else []
    
    # Phase 2: Backtracking search for deeper sequences
    if time.time() < deadline - 30:
        bt_result = _backtracking_search(deadline - 5, max_depth=14, start_val=1)
        if len(bt_result) > len(best):
            best = bt_result
    
    # Phase 3: Beam search
    if time.time() < deadline - 20:
        bs_result = _beam_search_bitset(deadline - 5, beam_width=40)
        if len(bs_result) > len(best):
            best = bs_result
    
    # Phase 4: Local search on best candidate
    if time.time() < deadline - 10 and best:
        ls_result = _local_search_bitset(best, deadline - 5)
        if len(ls_result) > len(best):
            best = ls_result
    
    # Phase 5: Randomized greedy extensions
    if time.time() < deadline - 5 and best:
        rg_result = _randomized_greedy_bitset(best, deadline, num_trials=20)
        if len(rg_result) > len(best):
            best = rg_result
    
    # Final polish: greedy extension
    if best and time.time() < deadline:
        final = _greedy_extend_from_bitset(best, deadline)
        if len(final) > len(best):
            best = final
    
    return best if best else greedy_smallest(deadline)
# EVOLVE-BLOCK-END
