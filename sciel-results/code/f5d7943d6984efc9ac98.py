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

def _greedy_construct(candidates, deadline=None):
    """Greedy: add smallest candidate that keeps all block products distinct."""
    if deadline is None:
        deadline = time.time() + 58
    values = []
    products = set()
    for x in candidates:
        if time.time() > deadline:
            break
        # Compute new tail products
        tails = []
        prod = x
        tails.append(prod)
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values

def _beam_search(candidates, beam_width=50, deadline=None):
    """Beam search to find a longer valid sequence."""
    if deadline is None:
        deadline = time.time() + 58
    
    start_time = time.time()
    
    # Each beam entry: (sequence, set_of_products)
    # Start with empty sequence
    beams = [([], set())]
    best_seq = []
    
    for x in candidates:
        if time.time() > deadline:
            break
        
        new_beams = []
        for seq, prods in beams:
            # Try adding x
            tails = []
            prod = x
            tails.append(prod)
            for a in reversed(seq):
                prod *= a
                tails.append(prod)
            
            if not prods.intersection(tails):
                new_prods = prods | set(tails)
                new_beams.append((seq + [x], new_prods))
        
        if not new_beams:
            break
        
        # Keep only the longest sequences; if tied, keep diverse ones
        max_len = max(len(b[0]) for b in new_beams)
        longest = [b for b in new_beams if len(b[0]) == max_len]
        
        if len(longest) > beam_width:
            # Sample diverse beams: prefer those with different last few elements
            # Simple strategy: take evenly spaced from sorted by last element
            longest.sort(key=lambda b: b[0][-1])
            step = max(1, len(longest) // beam_width)
            beams = longest[::step][:beam_width]
        else:
            beams = longest
        
        # Update best
        if max_len > len(best_seq):
            best_seq = beams[0][0]
    
    return best_seq

def _local_search(seq, candidates, max_iter=200, deadline=None):
    """Try to extend or improve sequence by local modifications."""
    if deadline is None:
        deadline = time.time() + 58
    
    current = list(seq)
    current_prods = _block_products(current)
    best = list(current)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try extending: add a candidate at the end
        for x in candidates:
            if x <= current[-1] if current else False:
                continue
            if x in current:
                continue
            # Check if adding x is valid
            tails = []
            prod = x
            tails.append(prod)
            for a in reversed(current):
                prod *= a
                tails.append(prod)
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

def construct():
    deadline = time.time() + 58  # Safety margin
    
    # Phase 1: Greedy construction
    candidates = list(range(2, 129))
    greedy_seq = _greedy_construct(candidates, deadline)
    
    # Phase 2: Beam search from scratch
    if time.time() < deadline - 10:
        beam_seq = _beam_search(candidates, beam_width=30, deadline=deadline - 5)
        if len(beam_seq) > len(greedy_seq):
            greedy_seq = beam_seq
    
    # Phase 3: Local search / extension
    if time.time() < deadline - 5:
        # Try to extend the best sequence found
        remaining = [x for x in candidates if x not in set(greedy_seq)]
        extended = _local_search(greedy_seq, remaining, max_iter=100, deadline=deadline - 2)
        if len(extended) > len(greedy_seq):
            greedy_seq = extended
    
    # Final verification
    if _is_valid(greedy_seq):
        return greedy_seq
    else:
        # Fallback to greedy (should always be valid)
        return _greedy_construct(candidates, deadline=deadline)

# EVOLVE-BLOCK-END
