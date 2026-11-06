"""Improved program for dense increasing sequences with unique block products."""
import time
import itertools
import math

# EVOLVE-BLOCK-START

def _block_products(seq):
    """Return set of all nonempty consecutive block products for a sequence."""
    prods = set()
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return prods

def _is_valid(seq):
    """Check if sequence has all distinct block products."""
    return len(_block_products(seq)) == len(seq) * (len(seq) + 1) // 2

def _tail_products(x, seq):
    """Compute tail products when adding x to seq: x, x*seq[-1], x*seq[-1]*seq[-2], ..."""
    tails = []
    prod = x
    tails.append(prod)
    for a in reversed(seq):
        prod *= a
        tails.append(prod)
    return tails

def _prime_factor_count(n):
    """Count total prime factors with multiplicity."""
    count = 0
    d = 2
    while d * d <= n:
        while n % d == 0:
            count += 1
            n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def _greedy_construct(candidates, deadline=None):
    """Greedy: add smallest candidate that keeps all block products distinct."""
    if deadline is None:
        deadline = time.time() + 58
    values = []
    products = set()
    for x in candidates:
        if time.time() > deadline:
            break
        tails = _tail_products(x, values)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values

def _beam_search_dynamic(candidates, deadline=None):
    """
    Beam search with dynamic beam width and improved heuristic using prime factor counts.
    """
    if deadline is None:
        deadline = time.time() + 58
    
    start_time = time.time()
    total_candidates = len(candidates)
    
    # Precompute prime factor counts for all candidates
    pfc = {x: _prime_factor_count(x) for x in candidates}
    
    # Each beam: (sequence, set_of_products, last_element)
    beams = [([], set(), 0)]
    best_seq = []
    
    for step, x in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # Determine beam width based on progress
        progress = step / total_candidates
        if progress < 0.3:
            beam_width = 100
        elif progress < 0.6:
            beam_width = 60
        else:
            beam_width = 30
        
        new_beams = []
        for seq, prods, last_el in beams:
            tails = _tail_products(x, seq)
            if not prods.intersection(tails):
                new_prods = prods | set(tails)
                new_beams.append((seq + [x], new_prods, x))
        
        if not new_beams:
            break
        
        # Score beams: prefer longer sequences, then lower collision potential
        def beam_score(beam):
            seq, prods, last_el = beam
            if not seq:
                return (0, 0, 0, 0)
            
            length = len(seq)
            total_pfc = sum(pfc.get(a, 0) for a in seq)
            max_prod = max(prods) if prods else 1
            log_max = math.log(max_prod + 1) if max_prod > 1 else 0
            remaining = sum(1 for c in candidates if c > last_el)
            
            return (length, -total_pfc, -log_max, remaining)
        
        new_beams.sort(key=beam_score, reverse=True)
        
        if len(new_beams) > beam_width:
            beams = new_beams[:beam_width]
        else:
            beams = new_beams
        
        if beams[0][0] and len(beams[0][0]) > len(best_seq):
            best_seq = beams[0][0]
    
    return best_seq

def _beam_search_restart(candidates, deadline=None, max_restarts=5):
    """
    Beam search with multiple restarts using different candidate orderings.
    """
    if deadline is None:
        deadline = time.time() + 58
    
    best_overall = []
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            cands = list(candidates)
        elif restart == 1:
            cands = [x for x in candidates if x >= 3]
        elif restart == 2:
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(math.sqrt(n)) + 1):
                    if n % i == 0:
                        return False
                return True
            primes = [x for x in candidates if is_prime(x)]
            composites = [x for x in candidates if not is_prime(x)]
            cands = primes + composites
        elif restart == 3:
            mid = len(candidates) // 2
            cands = candidates[mid:] + candidates[:mid]
        else:
            cands = list(reversed(candidates))
        
        seq = _beam_search_dynamic(cands, deadline=deadline - 5)
        if len(seq) > len(best_overall):
            best_overall = seq
    
    return best_overall

def _local_search_extend(seq, candidates, max_iter=50, deadline=None):
    """Try to extend sequence by adding elements at the end."""
    if deadline is None:
        deadline = time.time() + 58
    
    current = list(seq)
    current_prods = _block_products(current)
    best = list(current)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for x in candidates:
            if current and x <= current[-1]:
                continue
            if x in current:
                continue
            tails = _tail_products(x, current)
            if not current_prods.intersection(tails):
                new_prods = current_prods | set(tails)
                new_seq = current + [x]
                if len(new_seq) > len(best):
                    best = new_seq
                current = new_seq
                current_prods = new_prods
                improved = True
                break
        
        if not improved:
            break
    
    return best

def _local_search_swap(seq, candidates, max_iter=30, deadline=None):
    """Try to improve sequence by replacing elements with better candidates."""
    if deadline is None:
        deadline = time.time() + 58
    
    current = list(seq)
    current_prods = _block_products(current)
    best = list(current)
    best_len = len(current)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        
        for i in range(len(current)):
            old_val = current[i]
            prev_val = current[i-1] if i > 0 else 1
            next_val = current[i+1] if i < len(current) - 1 else 129
            
            for x in candidates:
                if x in current:
                    continue
                if x <= prev_val or x >= next_val:
                    continue
                
                test_seq = current[:i] + [x] + current[i+1:]
                test_prods = _block_products(test_seq)
                if len(test_prods) == len(test_seq) * (len(test_seq) + 1) // 2:
                    if len(test_seq) > best_len:
                        best = test_seq
                        best_len = len(test_seq)
                    current = test_seq
                    current_prods = test_prods
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best

def construct():
    deadline = time.time() + 58  # Safety margin
    
    # Phase 1: Greedy construction (fast baseline)
    candidates = list(range(2, 129))
    greedy_seq = _greedy_construct(candidates, deadline)
    
    # Phase 2: Dynamic beam search with restarts
    if time.time() < deadline - 20:
        beam_seq = _beam_search_restart(candidates, deadline=deadline - 15, max_restarts=5)
        if len(beam_seq) > len(greedy_seq):
            greedy_seq = beam_seq
    
    # Phase 3: Local search - extend
    if time.time() < deadline - 8:
        remaining = [x for x in candidates if x not in set(greedy_seq)]
        extended = _local_search_extend(greedy_seq, remaining, max_iter=50, deadline=deadline - 5)
        if len(extended) > len(greedy_seq):
            greedy_seq = extended
    
    # Phase 4: Local search - swap/improve
    if time.time() < deadline - 3:
        swapped = _local_search_swap(greedy_seq, candidates, max_iter=30, deadline=deadline - 2)
        if len(swapped) > len(greedy_seq):
            greedy_seq = swapped
    
    # Final verification
    if _is_valid(greedy_seq):
        return greedy_seq
    else:
        return _greedy_construct(candidates, deadline=deadline)

# EVOLVE-BLOCK-END
