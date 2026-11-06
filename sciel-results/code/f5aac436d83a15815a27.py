import time
import random

def construct():
    deadline = time.time() + 85  # safety margin (limit is 90s)
    
    def compute_tails(x, values):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        return tails
    
    def greedy_deterministic():
        values, sums = [], set()
        for x in range(1, 257):
            tails = compute_tails(x, values)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values, sums
    
    def greedy_biased(rng, p_small=0.4, max_scan=80):
        """At each step, scan up to max_scan candidates > last, pick valid ones, choose based on bias."""
        values, sums = [], set()
        last = 0
        
        while last < 256:
            valid = []
            for x in range(last + 1, min(last + 1 + max_scan, 257)):
                tails = compute_tails(x, values)
                if not sums.intersection(tails):
                    valid.append((x, tails))
            
            if not valid:
                # Expand search if needed
                for x in range(last + 1 + max_scan, 257):
                    tails = compute_tails(x, values)
                    if not sums.intersection(tails):
                        valid.append((x, tails))
                        break
            
            if not valid:
                break
            
            if rng.random() < p_small:
                x, tails = valid[0]
            else:
                x, tails = rng.choice(valid)
            
            values.append(x)
            sums.update(tails)
            last = x
        
        return values, sums
    
    def greedy_shuffled_window(rng, window=4):
        """Shuffle a window of candidates and pick the first valid."""
        values, sums = [], set()
        last = 0
        
        while last < 256:
            # Take a window of candidates starting from last+1
            start = last + 1
            end = min(start + window * 5, 257)
            window_candidates = list(range(start, end))
            rng.shuffle(window_candidates)
            
            chosen = None
            for x in window_candidates:
                tails = compute_tails(x, values)
                if not sums.intersection(tails):
                    chosen = (x, tails)
                    break
            
            if chosen is None:
                # Fallback: scan all
                for x in range(start, 257):
                    tails = compute_tails(x, values)
                    if not sums.intersection(tails):
                        chosen = (x, tails)
                        break
            
            if chosen is None:
                break
            
            x, tails = chosen
            values.append(x)
            sums.update(tails)
            last = x
        
        return values, sums
    
    # Deterministic baseline
    base_seq, base_sums = greedy_deterministic()
    best = base_seq
    best_len = len(base_seq)
    
    rng = random.Random(777)
    
    # Run many randomized greedy searches
    while time.time() < deadline - 18:
        r = rng.random()
        if r < 0.25:
            seq, sums = greedy_biased(rng, p_small=0.2, max_scan=40)
        elif r < 0.5:
            seq, sums = greedy_biased(rng, p_small=0.5, max_scan=60)
        elif r < 0.75:
            seq, sums = greedy_shuffled_window(rng, window=rng.randint(3, 8))
        else:
            seq, sums = greedy_biased(rng, p_small=0.7, max_scan=30)
        
        if len(seq) > best_len:
            best = list(seq)
            best_len = len(best)
    
    # Final: limited backtracking to extend best
    if time.time() < deadline - 5:
        best = _local_extend(best, deadline)
    
    return best


def _local_extend(seq, deadline):
    """Limited-depth backtracking to extend a sequence."""
    best = list(seq)
    best_len = len(best)
    
    def compute_tails(x, values):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        return tails
    
    # Compute current sums
    s_sums = set()
    n = len(seq)
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += seq[j]
            s_sums.add(total)
    
    def backtrack(s, s_sums, depth):
        nonlocal best, best_len
        if time.time() > deadline:
            return
        if depth > 4:
            return
        
        if len(s) > best_len:
            best = list(s)
            best_len = len(best)
        
        last = s[-1] if s else 0
        for x in range(last + 1, 257):
            if time.time() > deadline:
                return
            tails = compute_tails(x, s)
            if any(t in s_sums for t in tails):
                continue
            s.append(x)
            s_sums.update(tails)
            backtrack(s, s_sums, depth + 1)
            s_sums.difference_update(tails)
            s.pop()
    
    backtrack(list(seq), s_sums, 0)
    return best
