# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    denominators = list(range(2, 49))
    n = len(denominators)
    fracs = [Fraction(1, d) for d in denominators]
    
    # Beam Search
    # State: (sum, indices_tuple)
    # We want to maximize sum < 1.
    # We process denominators in order.
    # beam is a list of (sum, indices)
    
    beam_width = 5000
    # Initialize with one state: sum=0, indices=()
    beam = [(Fraction(0), ())]
    
    for i in range(n):
        f = fracs[i]
        new_beam = []
        for s, inds in beam:
            # Option 1: Do not take i
            new_beam.append((s, inds))
            
            # Option 2: Take i
            new_s = s + f
            if new_s < 1:
                new_inds = inds + (i,)
                new_beam.append((new_s, new_inds))
        
        # Keep top beam_width states with highest sum < 1
        # Sort by sum descending
        # We need to be careful: we want the sum closest to 1 from below.
        # So we sort by sum descending and take the first beam_width.
        
        # Optimization: Instead of sorting the entire list which can be 2*beam_width,
        # we can use heapq.nlargest.
        import heapq
        # heapq.nlargest returns the largest elements.
        # We want the largest sums.
        # The key is the sum.
        
        # But we also need to keep the indices.
        # heapq.nlargest(k, iterable, key=func)
        
        # To avoid excessive memory allocation of tuples, we can just sort.
        # 5000 * 2 = 10000 items. Sorting 10000 items is fast.
        
        new_beam.sort(key=lambda x: x[0], reverse=True)
        
        # Truncate to beam_width
        beam = new_beam[:beam_width]
        
        # Early termination? If the best possible sum from remaining items
        # cannot beat the current best in the beam, we could stop?
        # No, because we might need to exclude some current items to fit better?
        # No, beam search keeps the best prefixes.
        # Actually, if we have processed all items, we are done.
        
    # The best solution is the one in the beam with the highest sum < 1
    # Since we kept top beam_width, the first one is the best.
    best_sum, best_inds = beam[0]
    
    return list(best_inds)
# EVOLVE-BLOCK-END
