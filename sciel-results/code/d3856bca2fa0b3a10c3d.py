"""Improved program: local search with element removal + re-extension."""
import time
import random

DEADLINE = 80.0  # safety margin before 90s kill

def _compute_sums(seq):
    """Return set of all nonempty consecutive block sums for a sequence."""
    n = len(seq)
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _is_valid(seq):
    """Check if sequence is strictly increasing in [1,256] with distinct block sums."""
    if not seq:
        return True
    if seq[0] < 1 or seq[-1] > 256:
        return False
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    return len(_compute_sums(seq)) == len(seq) * (len(seq) + 1) // 2

def _greedy_extend(seq, start_val=1, end_val=257):
    """Greedily extend the sequence by trying to add valid elements."""
    if not seq:
        return seq
    sums = _compute_sums(seq)
    last = seq[-1]
    result = list(seq)
    for x in range(last + 1, end_val):
        # Check if x can be appended
        tails = []
        total = x
        tails.append(total)
        for a in reversed(result):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            result.append(x)
            sums.update(tails)
    return result

def _greedy_smallest(seq, end_val=257):
    """Try to extend from the smallest possible value after the last element."""
    if not seq:
        return seq
    sums = _compute_sums(seq)
    last = seq[-1]
    result = list(seq)
    for x in range(last + 1, end_val):
        tails = []
        total = x
        tails.append(total)
        for a in reversed(result):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            result.append(x)
            sums.update(tails)
    return result

def _greedy_largest(seq, start_val=1):
    """Try to extend using larger values first (greedy from top)."""
    if not seq:
        return seq
    sums = _compute_sums(seq)
    last = seq[-1]
    result = list(seq)
    # Try larger values first to potentially fit more
    for x in range(256, last, -1):
        if x <= last:
            continue
        tails = []
        total = x
        tails.append(total)
        for a in reversed(result):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            result.append(x)
            sums.update(tails)
    return result

def _remove_and_reextend(seq, pos):
    """Remove element at pos and try to re-extend with different choices."""
    if pos < 0 or pos >= len(seq):
        return seq
    new_seq = seq[:pos] + seq[pos+1:]
    return _greedy_extend(new_seq)

def _local_search(seq, max_iters=5000, deadline=DEADLINE):
    """Local search: randomly remove elements and re-extend."""
    best = list(seq)
    best_len = len(best)
    t0 = time.time()
    
    for it in range(max_iters):
        if time.time() - t0 > deadline:
            break
        
        # Randomly choose an element to remove (not the first or last usually)
        if len(best) <= 1:
            break
        
        # Weight towards removing middle elements
        pos = random.randint(1, len(best) - 2) if len(best) > 2 else 0
        
        candidate = _remove_and_reextend(best, pos)
        
        if len(candidate) > best_len:
            best = candidate
            best_len = len(candidate)
    
    return best

def _beam_search(width=10, deadline=DEADLINE):
    """Beam search over prefixes to find longer sequences."""
    t0 = time.time()
    # Start with single elements
    beams = [[i] for i in range(1, 257)]
    
    for _ in range(50):  # max depth
        if time.time() - t0 > deadline:
            break
        if not beams:
            break
        
        # Extend each beam
        new_beams = []
        for b in beams:
            if time.time() - t0 > deadline:
                break
            last = b[-1]
            sums = _compute_sums(b)
            for x in range(last + 1, 257):
                tails = [x]
                total = x
                for a in reversed(b):
                    total += a
                    tails.append(total)
                if not sums.intersection(tails):
                    new_beams.append(b + [x])
        
        if not new_beams:
            break
        
        # Keep top beams by length, then by "potential" (more room to grow)
        new_beams.sort(key=lambda b: (len(b), -b[-1]), reverse=True)
        beams = new_beams[:width]
    
    if not beams:
        return []
    return beams[0]

def _construct_initial():
    """Initial greedy construction."""
    values = []
    sums = set()
    for x in range(1, 257):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def construct():
    t_start = time.time()
    
    # Phase 1: Initial greedy
    seq = _construct_initial()
    best = list(seq)
    best_len = len(best)
    
    # Phase 2: Beam search for better prefixes
    if time.time() - t_start < 10:
        beam_result = _beam_search(width=20, deadline=10)
        if len(beam_result) > best_len:
            best = beam_result
            best_len = len(best)
    
    # Phase 3: Local search with removal + re-extension
    if time.time() - t_start < 70:
        best = _local_search(best, max_iters=10000, deadline=70)
        best_len = len(best)
    
    # Phase 4: Final greedy extension
    if time.time() - t_start < 78:
        extended = _greedy_extend(best)
        if len(extended) > best_len:
            best = extended
            best_len = len(best)
    
    # Phase 5: Another round of local search if time permits
    if time.time() - t_start < 78:
        best = _local_search(best, max_iters=5000, deadline=78)
    
    # Ensure validity
    if not _is_valid(best):
        # Fallback to initial greedy
        best = _construct_initial()
    
    return best
