import time
import random
import numpy as np

def construct():
    base = 51975
    deadline = time.time() + 115  # Safety margin
    
    # 1. Get divisors
    # Divisors must be > 1 and divide base.
    # base = 51975 = 3^3 * 5^2 * 7 * 11
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    # Precompute residue sets for each divisor to speed up gain calculation?
    # For a given m, the residues are r, r+m, ..., r+k*m < base.
    # Using numpy slicing counts[r::m] is efficient.
    
    def get_uncovered(counts):
        return np.sum(counts == 0)

    def greedy_init(order):
        """
        Greedy construction.
        Returns: classes list, final uncovered count, final counts array
        """
        counts = np.zeros(base, dtype=np.int8)
        classes = []
        uncovered = base
        
        for m in order:
            best_r = 0
            best_gain = -1
            # Try all residues
            for r in range(m):
                # Count how many currently uncovered residues are in this class
                gain = np.sum(counts[r::m] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            classes.append([m, best_r])
            counts[best_r::m] += 1
            uncovered -= best_gain
            
            if uncovered == 0:
                break
                
        return classes, uncovered, counts

    def local_search(classes, initial_counts):
        """
        Local search using incremental updates.
        Returns: best classes, best uncovered count
        """
        classes = [list(c) for c in classes]
        counts = initial_counts.copy()
        
        # Calculate initial uncovered
        uncovered = get_uncovered(counts)
        
        improved = True
        passes = 0
        max_passes = 500 # Increased passes
        
        while improved and passes < max_passes:
            if time.time() > deadline - 5:
                break
            improved = False
            passes += 1
            
            for i in range(len(classes)):
                m, r_old = classes[i]
                
                # 1. Remove current class
                counts[r_old::m] -= 1
                
                # Calculate uncovered state without this class
                # We need to find the best residue to cover the most currently uncovered residues
                # Note: 'currently uncovered' means counts[x] == 0 in the state WITHOUT class i
                
                best_r = r_old
                best_gain = -1 # We want to maximize gain. If we keep r_old, gain is #newly covered.
                
                # Optimization: If we keep r_old, the gain is the number of residues in S(m, r_old) 
                # that are currently 0 (i.e., were 1 before removal, now 0).
                # Actually, let's just iterate all r to be safe and simple, 
                # but we can break early if we find a gain equal to the total uncovered? 
                # No, max gain is bounded by len(S(m, r)).
                
                for r_new in range(m):
                    # If we add r_new to the current counts (which lacks class i),
                    # how many new residues are covered?
                    # Those are residues in S(m, r_new) where counts[x] == 0.
                    gain = np.sum(counts[r_new::m] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r_new
                
                # 3. Apply best residue
                counts[best_r::m] += 1
                
                # 4. Update state
                if best_r != r_old:
                    classes[i] = [m, best_r]
                    improved = True
                    # Recalculate uncovered? 
                    # We can update it incrementally:
                    # Let U_old be uncovered before removal.
                    # After removal, some residues became uncovered.
                    # Let L = number of residues in S(m, r_old) that had count 1 before removal.
                    # These became 0. So uncovered increased by L.
                    # Then we add best_r. Let G = number of residues in S(m, best_r) that had count 0 after removal.
                    # These become 1. So uncovered decreases by G.
                    # This is tricky to track perfectly without extra arrays.
                    # Given base=51975, np.sum(counts==0) is ~0.1ms.
                    # 50 moduli * 500 passes * 0.1ms = 2.5s. Acceptable.
                    uncovered = get_uncovered(counts)
                else:
                    # If no change, the state is identical to before the removal+addition.
                    # So uncovered is unchanged.
                    pass
                
        return classes, uncovered

    # --- Main Execution ---
    
    best_classes = None
    best_uncovered = base
    best_counts = None

    # 1. Greedy with increasing order (small moduli first)
    classes, unc, counts = greedy_init(divisors)
    if unc < best_uncovered:
        best_uncovered = unc
        best_classes = classes
        best_counts = counts
    
    # 2. Greedy with decreasing order (large moduli first)
    classes, unc, counts = greedy_init(divisors[::-1])
    if unc < best_uncovered:
        best_uncovered = unc
        best_classes = classes
        best_counts = counts
        
    # 3. Random orders
    random.seed(42)
    start_time = time.time()
    
    # Run random greedy inits until time is low
    # We reserve 40s for local search to allow deeper optimization
    while time.time() < deadline - 40: 
        order = divisors[:]
        random.shuffle(order)
        classes, unc, counts = greedy_init(order)
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = classes
            best_counts = counts
            if best_uncovered == 0:
                break
    
    # 4. Local Search on the best found
    if best_classes is not None and time.time() < deadline - 10:
        best_classes, best_uncovered = local_search(best_classes, best_counts)
        
    return best_classes
