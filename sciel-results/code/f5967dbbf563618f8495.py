import random
import itertools
import time

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit
    
    n = 16
    points = list(range(n))
    
    # Precompute all possible triples and their bitmasks
    all_triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                mask = (1 << a) | (1 << b) | (1 << c)
                # Store the pair masks as well for quick lookup
                p_ab = (a, b)
                p_ac = (a, c)
                p_bc = (b, c)
                all_triples.append((mask, p_ab, p_ac, p_bc, (a,b,c)))
                
    # Total triples: 560
    # Max possible linear packing size: 37
    
    best_alpha = n + 1
    best_packing = []
    
    # Function to calculate independence number
    def calc_alpha(packing_masks):
        # packing_masks is a list of int masks
        # We check all subsets of points
        # A subset S (mask) is independent if no triple mask T is a subset of S
        # i.e., (T & S) == T
        
        # Optimization: Sort triples by mask? No, just check.
        # 2^16 = 65536 subsets.
        # For each subset, check against packing_masks.
        # 65536 * 37 ~ 2.4M ops.
        
        max_ind = 0
        # We can iterate by size to stop early if we find one?
        # No, we need the MAX size.
        # So we check all.
        
        # To speed up, we can precompute the list of triple masks
        # and for each subset, check if any triple is contained.
        
        # Alternative: 
        # For each triple, it "blocks" all supersets.
        # This is hard to do efficiently for max independent set.
        # Direct check is fine for N=16.
        
        # We can prune: if a subset of size k is independent, 
        # any subset of it is independent.
        # But we want the largest.
        
        # Let's just iterate all masks from 1 to 2^16 - 1
        # and check independence.
        
        # To make it faster, we can check sizes in descending order?
        # No, we have to check all to be sure.
        
        # Actually, we can use a recursive search for max independent set.
        # But 2^16 is small enough for brute force if optimized.
        
        # Let's use a simple loop.
        # To speed up the inner check:
        # If we have a list of triple masks, we can check:
        # any((t & s) == t for t in triple_masks)
        
        # We can speed this up by noting that if a subset S contains a triple, 
        # it's not independent.
        
        # Let's try to optimize the check.
        # Precompute nothing, just loop.
        
        max_ind = 0
        for s in range(1, 1 << n):
            # Calculate popcount (size of s)
            # If popcount(s) <= max_ind, skip? 
            # No, we need to know if it's independent first.
            # But if it's independent, we update max_ind.
            
            # Check independence
            is_ind = True
            for t in packing_masks:
                if (t & s) == t:
                    is_ind = False
                    break
            
            if is_ind:
                # Count bits
                cnt = bin(s).count('1')
                if cnt > max_ind:
                    max_ind = cnt
                    # If max_ind is 1, we can't get better? 
                    # No, we want to maximize the size of the independent set.
                    # The score is the size. We want to MINIMIZE this score.
                    # So we want the packing that yields the SMALLEST max_ind.
                    
        return max_ind

    # Greedy packing generator
    def generate_packing():
        random.shuffle(all_triples)
        used_pairs = set()
        packing = []
        packing_masks = []
        
        for mask, p1, p2, p3, _ in all_triples:
            if p1 in used_pairs or p2 in used_pairs or p3 in used_pairs:
                continue
            # Add
            packing.append((p1, p2, p3, mask))
            packing_masks.append(mask)
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            
        return packing_masks, len(packing)

    # We want packings of size 37 to have the best chance of small alpha.
    # If we get smaller, alpha is likely larger.
    
    iterations = 0
    while time.time() - start_time < timeout:
        iterations += 1
        masks, size = generate_packing()
        
        if size < 37:
            # Skip sub-optimal packings? 
            # A packing with 36 triples might still have a small alpha, 
            # but 37 is the theoretical max. 
            # It's safer to focus on 37.
            continue
            
        alpha = calc_alpha(masks)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_packing = [list(m) for m in [(t[0] for t in [0] for _ in [0])] ] # Placeholder, need to return triples
            
            # We need to return the actual triples [a,b,c]
            # Let's reconstruct from masks or store them properly.
            
            # Re-doing storage:
            # In generate_packing, we should return the triples list too.
            
    # Let's refine generate_packing to return triples.
    
    # Reset
    best_alpha = n + 1
    best_triples = []
    
    def generate_packing_v2():
        random.shuffle(all_triples)
        used_pairs = set()
        triples = []
        masks = []
        
        for mask, p1, p2, p3, pts in all_triples:
            if p1 in used_pairs or p2 in used_pairs or p3 in used_pairs:
                continue
            triples.append(pts)
            masks.append(mask)
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            
        return triples, masks, len(triples)

    # Run search
    while time.time() - start_time < timeout:
        triples, masks, size = generate_packing_v2()
        
        if size < 37:
            continue
            
        alpha = calc_alpha(masks)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            
            if best_alpha == 4: # Theoretical lower bound?
                # For v=16, can alpha be 4?
                # If alpha=4, every 5-set contains a triple.
                # This is a covering design C(16,5,3).
                # If we find 4, we can stop early as it's likely optimal.
                break

    # If no packing of size 37 found (unlikely), return the best found or seed
    if not best_triples:
        # Fallback to seed
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    return best_triples

# EVOLVE-BLOCK-START
def construct():
    # Use the logic developed above
    import time
    import random
    import itertools

    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    n = 16
    points = list(range(n))
    
    # Precompute all possible triples
    all_triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                mask = (1 << a) | (1 << b) | (1 << c)
                p_ab = (a, b)
                p_ac = (a, c)
                p_bc = (b, c)
                all_triples.append((mask, p_ab, p_ac, p_bc, (a,b,c)))
                
    best_alpha = n + 1
    best_triples = []
    
    def calc_alpha(masks):
        max_ind = 0
        # Iterate all subsets
        for s in range(1, 1 << n):
            is_ind = True
            for t in masks:
                if (t & s) == t:
                    is_ind = False
                    break
            if is_ind:
                cnt = bin(s).count('1')
                if cnt > max_ind:
                    max_ind = cnt
        return max_ind

    def generate_packing():
        random.shuffle(all_triples)
        used_pairs = set()
        triples = []
        masks = []
        
        for mask, p1, p2, p3, pts in all_triples:
            if p1 in used_pairs or p2 in used_pairs or p3 in used_pairs:
                continue
            triples.append(pts)
            masks.append(mask)
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            
        return triples, masks, len(triples)

    while time.time() - start_time < timeout:
        triples, masks, size = generate_packing()
        
        # We prioritize maximal packings (size 37)
        if size < 37:
            continue
            
        alpha = calc_alpha(masks)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            
            if best_alpha <= 4:
                break

    if not best_triples:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    return best_triples
# EVOLVE-BLOCK-END
