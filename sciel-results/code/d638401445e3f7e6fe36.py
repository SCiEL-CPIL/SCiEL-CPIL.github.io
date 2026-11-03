"""Ternary covering code K_3(7,1) via multi-restart randomized greedy with bitsets."""
import time
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin for 150s limit

def _time_ok():
    return time.time() < _DEADLINE

def _neighbors_of(idx, q, n):
    """Compute indices of all words within Hamming distance 1 of idx."""
    word = np.unravel_index(idx, (q,) * n)
    out = [idx]
    for pos in range(n):
        for v in range(q):
            if v == word[pos]:
                continue
            w = list(word)
            w[pos] = v
            out.append(int(np.ravel_multi_index(tuple(w), (q,) * n)))
    return out

def construct(q, n, radius):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 115.0

    if radius != 1:
        import itertools
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    total = q ** n
    # Precompute neighbor sets as lists of indices
    # For q=3, n=7, total=2187. Each set has 1 + 7*2 = 15 elements.
    neighbor_sets = []
    for idx in range(total):
        neighbor_sets.append(_neighbors_of(idx, q, n))
    
    # Precompute bitsets for fast intersection/union
    # Python integers are arbitrary precision, so 2187 bits is fine.
    neighbor_bitsets = []
    for ns in neighbor_sets:
        bits = 0
        for idx in ns:
            bits |= (1 << idx)
        neighbor_bitsets.append(bits)
    
    all_bits = (1 << total) - 1
    
    def _remove_redundant(chosen):
        """Remove any chosen word whose coverage is fully covered by others."""
        if not _time_ok():
            return chosen
        m = len(chosen)
        if m == 0:
            return chosen
        # Precompute union of all chosen
        # We will check each word c_i: is union(others) & ball(c_i) == ball(c_i)?
        # i.e., is ball(c_i) subset of union(others)?
        
        # To speed up, we can maintain a count or just recompute.
        # With ~200 words, m^2 * bit_ops is fast.
        changed = True
        while changed and _time_ok():
            changed = False
            for i in range(len(chosen)):
                if not _time_ok():
                    break
                # Check if chosen[i] is redundant
                # Union of all others
                others_union = 0
                for j in range(len(chosen)):
                    if j != i:
                        others_union |= neighbor_bitsets[chosen[j]]
                
                if (neighbor_bitsets[chosen[i]] & ~others_union) == 0:
                    # Redundant
                    chosen.pop(i)
                    changed = True
                    break
        return chosen

    def _randomized_greedy(seed):
        """
        Randomized greedy set cover.
        At each step, pick a candidate from the top K% with highest gain,
        chosen randomly to escape local optima.
        """
        rng = random.Random(seed)
        uncovered = all_bits
        chosen = []
        
        while uncovered and _time_ok():
            # Find best candidates
            # To avoid O(N) full scan every time if N is large, we can just scan.
            # N=2187 is small enough.
            best_gain = 0
            candidates = []
            
            for c in range(total):
                if c in chosen: 
                    # Optimization: we don't track 'in chosen' set, but chosen is small.
                    # Actually, checking `c in chosen` is O(M). M~200.
                    # 2187 * 200 = 400k ops per step. 200 steps -> 80M ops. Might be slow in Python.
                    # Better: maintain a set of chosen indices.
                    pass
            
            # Let's optimize the loop:
            chosen_set = set(chosen)
            for c in range(total):
                if c in chosen_set:
                    continue
                gain = bin(uncovered & neighbor_bitsets[c]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    candidates = [c]
                elif gain == best_gain:
                    candidates.append(c)
            
            if not candidates:
                break
                
            # Randomized selection: pick from top candidates
            # If we pick strictly best, it's deterministic greedy.
            # Picking randomly from ties is standard.
            # To explore more, we could pick from top 5% of gains.
            # But for now, random tie-breaking is a good start.
            
            # Actually, a stronger strategy: "Las Vegas" style or "Greedy with noise".
            # Let's try picking randomly from the set of all candidates with gain == best_gain.
            # This handles ties.
            
            c = rng.choice(candidates)
            chosen.append(c)
            uncovered &= ~neighbor_bitsets[c]
            
            # Early stop if fully covered
            if uncovered == 0:
                break
                
        return chosen

    best_code = None
    best_len = total + 1
    
    # Multi-restart
    # Each restart: Greedy + Remove Redundant
    # How many restarts can we do?
    # Greedy is ~0.05-0.1s per run (estimated).
    # Remove redundant is ~0.01s.
    # We have ~115s.
    # Let's do as many as possible.
    
    restart_count = 0
    while _time_ok():
        code = _randomized_greedy(restart_count)
        if len(code) < best_len:
            # Polish with redundant removal
            code = _remove_redundant(code)
            if len(code) < best_len:
                best_len = len(code)
                best_code = code[:]
                
        # If we hit a very good solution, we can try to speed up subsequent runs
        # or just keep going.
        
        restart_count += 1
        # Safety: if we've done many, check time more strictly
        if restart_count % 10 == 0 and not _time_ok():
            break

    if best_code is None:
        # Fallback: deterministic greedy
        best_code = _randomized_greedy(0)
        best_code = _remove_redundant(best_code)

    # Final validation and conversion
    # Check validity
    covered = 0
    for c in best_code:
        covered |= neighbor_bitsets[c]
    if covered != all_bits:
        # Should not happen if greedy finished, but just in case
        # Fallback to full space (expensive but valid)
        import itertools
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Convert indices to lists of ints
    out = []
    for c in best_code:
        out.append(list(np.unravel_index(c, (q,) * n)))
    return out
