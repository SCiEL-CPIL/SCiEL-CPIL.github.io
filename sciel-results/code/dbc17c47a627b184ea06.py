"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    base = 51975
    # All odd divisors of 51975 = 3^3 * 5^2 * 7 * 11
    # Generate all divisors
    import itertools
    prime_powers = [
        [3**i for i in range(4)],   # 1,3,9,27
        [5**i for i in range(3)],   # 1,5,25
        [7**i for i in range(2)],   # 1,7
        [11**i for i in range(2)],  # 1,11
    ]
    divisors = set()
    for combo in itertools.product(*prime_powers):
        d = 1
        for x in combo:
            d *= x
        divisors.add(d)
    # Remove 1 and keep only odd (all are odd since base is odd)
    moduli = sorted([d for d in divisors if d > 1])
    
    # Represent coverage as a numpy boolean array of size base
    covered = np.zeros(base, dtype=np.uint8)
    
    # Precompute for each modulus m and residue r, the indices it covers
    # To speed up, we'll compute the count of newly covered residues efficiently
    
    classes = []
    
    # Greedy set cover: at each step, pick the (m, r) that covers the most uncovered residues
    # We process moduli in a smart order: larger moduli cover fewer residues per class,
    # but smaller moduli cover more. Actually for covering, we want to maximize coverage.
    # 
    # Key insight: for modulus m, each residue class covers base//m residues.
    # So smaller m means more residues per class. We should consider all (m,r) pairs.
    
    # Precompute the coverage mask for each (m, r) as a list of indices
    # But base=51975, and there are ~40 moduli, each with up to m residues.
    # Total (m,r) pairs: sum of m over all divisors m>1.
    
    # Let's compute the total number of candidate classes
    total_candidates = sum(m for m in moduli)
    # This could be large. Let's estimate:
    # moduli are divisors > 1 of 51975
    # sum of divisors of 51975 = (1+3+9+27)(1+5+25)(1+7)(1+11) = 40*31*8*12 = 119040
    # minus 1 for divisor=1, so total_candidates ~ 119039
    # That's manageable.
    
    # For each candidate (m, r), we need to count how many uncovered residues it covers.
    # covered[r::m] gives the residues in that class. We want count of 0s in that slice.
    
    # We'll use numpy for fast counting.
    
    # Sort moduli: we want to try all (m, r) pairs. To be efficient, 
    # we can iterate through moduli and for each, find the best r.
    # But the global greedy picks the single best (m,r) across all moduli.
    
    # Strategy: maintain a list of all candidate (m, r) pairs.
    # At each iteration, find the one that covers the most uncovered residues.
    # Remove it, update covered, repeat.
    
    # To make this efficient, we can:
    # 1. For each modulus m, precompute the list of residues 0..m-1
    # 2. For each (m, r), the covered indices are r, r+m, r+2m, ... < base
    # 3. We want to maximize sum(covered[idx] == 0) for idx in that arithmetic progression
    
    # Let's build a data structure for fast evaluation.
    
    # Alternative approach: since base is only 51975, we can use a bitset or 
    # a numpy array and compute the count via slicing.
    
    # Let's precompute for each m the step size and use numpy's fancy indexing.
    
    # For efficiency, let's process moduli in order of decreasing base//m (i.e., increasing m)
    # because smaller m means each class covers more residues, so they're likely to be chosen first.
    
    # Actually, let's just do a full greedy: at each step, evaluate all remaining (m,r) pairs.
    
    # To speed up, we can note that once a modulus m has been used, we can't use it again.
    # So we pick one (m, r) per modulus at most.
    
    # Revised greedy:
    # - Maintain set of unused moduli
    # - At each step, for each unused modulus m, find the best residue r that covers most uncovered
    # - Pick the (m, r) with the maximum coverage
    # - Add to classes, mark covered, remove m from unused
    
    unused_moduli = set(moduli)
    
    # Precompute for each m, the list of residue indices for each r
    # m_to_residues[m] = list of arrays, where array r is the indices r, r+m, ...
    # But this might use too much memory. Let's compute on the fly.
    
    while unused_moduli:
        if time.time() > deadline - 5.0:
            break
        
        best_m = None
        best_r = None
        best_count = -1
        
        for m in unused_moduli:
            # For this modulus, find the best residue
            # The residues are 0, 1, ..., m-1
            # For each r, the covered indices are r, r+m, r+2m, ...
            # We want to count how many of these are uncovered (covered == 0)
            
            # Using numpy: covered[r::m] gives the slice
            # Count of zeros in that slice
            
            # To speed up, we can compute all residues at once using a 2D approach
            # But m can be up to 51975, so this might be expensive.
            
            # For small m, we can iterate over r. For large m, each class has few elements.
            
            # Let's compute the count for each r in 0..m-1
            # We can vectorize this using numpy.
            
            # Create an array of all indices covered by this modulus
            # indices = np.arange(m) + np.arange(0, base, m)[:, None]  # This is too big
            
            # Better: for each r, count = np.sum(covered[r::m] == 0)
            
            # Let's do it in a loop but with numpy slicing
            best_r_for_m = 0
            best_count_for_m = -1
            
            for r in range(m):
                count = np.sum(covered[r::m] == 0)
                if count > best_count_for_m:
                    best_count_for_m = count
                    best_r_for_m = r
            
            if best_count_for_m > best_count:
                best_count = best_count_for_m
                best_m = m
                best_r = best_r_for_m
        
        if best_m is None or best_count <= 0:
            break
        
        classes.append([best_m, best_r])
        covered[best_r::best_m] = 1
        unused_moduli.discard(best_m)
    
    return classes
# EVOLVE-BLOCK-END
