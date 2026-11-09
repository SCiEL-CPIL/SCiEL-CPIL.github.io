# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import heapq
    
    denominators = list(range(2, 49))
    n = len(denominators)
    fracs = [Fraction(1, d) for d in denominators]
    
    # Beam Search with bitmask for indices
    # State: (sum, mask)
    # We want to maximize sum < 1.
    
    beam_width = 20000
    beam = [(Fraction(0), 0)]
    
    for i in range(n):
        f = fracs[i]
        new_beam = []
        
        for s, m in beam:
            # Option 1: Do not take i
            new_beam.append((s, m))
            
            # Option 2: Take i
            new_s = s + f
            if new_s < 1:
                new_m = m | (1 << i)
                new_beam.append((new_s, new_m))
        
        # Keep top beam_width states with highest sum < 1
        # Use heapq.nlargest for efficiency
        beam = heapq.nlargest(beam_width, new_beam, key=lambda x: x[0])
        
    # The best solution is the one in the beam with the highest sum < 1
    best_sum, best_mask = beam[0]
    
    # Convert mask to list of indices
    indices = [i for i in range(n) if (best_mask >> i) & 1]
    
    return indices
# EVOLVE-BLOCK-END
