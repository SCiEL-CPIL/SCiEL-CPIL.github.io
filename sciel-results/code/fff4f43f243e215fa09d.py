import time
import random
import math

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 80  # Leave 10s margin for safety

    # Precompute comparability using bit manipulation
    # For each mask, we need to know which other masks are comparable
    # Two masks a, b are comparable if (a & ~b) == 0 or (b & ~a) == 0
    
    # Precompute subset relationships
    # For each mask, list of masks that are subsets of it (including itself)
    subsets_of = [[] for _ in range(total_masks)]
    supersets_of = [[] for _ in range(total_masks)]
    
    # For each mask, find all its subsets
    for mask in range(total_masks):
        # All subsets of mask
        sub = mask
        while True:
            subsets_of[mask].append(sub)
            if sub == 0:
                break
            sub = (sub - 1) & mask
        # All supersets of mask
        # Supersets are masks that contain mask
        # Iterate over all masks and check if they contain mask
        # This is O(2^n) per mask, total O(4^n) which is too slow
        # Instead, we'll compute comparability on the fly but optimize
    
    # Alternative: precompute a bitset for each mask indicating comparable masks
    # But 4096 * 4096 bits = 2MB, which is fine
    # Let's use a list of bytearrays or just compute on the fly
    
    # For efficiency in search, let's precompute for each mask a list of comparable masks
    # comparable[mask] = list of masks s.t. mask is subset or superset of that mask
    comparable = [[] for _ in range(total_masks)]
    for a in range(total_masks):
        for b in range(total_masks):
            if a == b:
                continue
            if (a & ~b) == 0 or (b & ~a) == 0:
                comparable[a].append(b)
    
    # Now comparable[a] contains all masks comparable to a (excluding a itself)
    
    # Dense initialization: Start with a good structure
    # Use a multi-chain approach
    # Chain 1: subsets of {0,1,2,3,4,5} - 64 elements, too many
    # Let's use a different approach
    
    # Start with all subsets of {0,1,2,3,4} (32 elements) plus some supersets
    family = []
    fam_set = set()
    
    # Add all subsets of {0,1,2,3,4}
    for mask in range(1 << 5):
        family.append(mask)
        fam_set.add(mask)
    
    # scores[s] = number of sets in family comparable to s
    scores = [0] * total_masks
    
    # Initialize scores
    for a in family:
        for b in comparable[a]:
            if b in fam_set:
                scores[a] += 1
    
    def update_scores_for_addition(mask, delta):
        """Update scores when adding/removing a mask from family"""
        for s in comparable[mask]:
            scores[s] += delta
        scores[mask] += delta  # mask is comparable to itself in terms of counting, but we exclude self
    
    # Actually, let's be more careful. scores[s] should count how many OTHER family members are comparable to s
    # So when we add mask, for each s comparable to mask, scores[s] += 1
    # And scores[mask] should be the count of family members comparable to mask
    
    # Let me redo this more carefully
    scores = [0] * total_masks
    
    # For each pair in family, increment scores for both
    for i, a in enumerate(family):
        for j in range(i + 1, len(family)):
            b = family[j]
            if (a & ~b) == 0 or (b & ~a) == 0:
                scores[a] += 1
                scores[b] += 1
    
    def update_scores_add(mask):
        """Update scores when adding mask to family"""
        for s in comparable[mask]:
            if s in fam_set:
                scores[s] += 1
        scores[mask] = sum(1 for s in comparable[mask] if s in fam_set)
    
    def update_scores_remove(mask):
        """Update scores when removing mask from family"""
        for s in comparable[mask]:
            if s in fam_set:
                scores[s] -= 1
    
    # Greedily add remaining sets
    while len(family) < m and time.time() < deadline:
        best_score = -1
        best_mask = -1
        
        for s in range(total_masks):
            if s in fam_set:
                continue
            # Compute how many family members are comparable to s
            cnt = sum(1 for t in comparable[s] if t in fam_set)
            if cnt > best_score:
                best_score = cnt
                best_mask = s
                
        if best_mask == -1:
            break
            
        family.append(best_mask)
        fam_set.add(best_mask)
        update_scores_add(best_mask)
                
    def get_total_score():
        return sum(scores) // 2
    
    current_score = get_total_score()
    best_family = family[:]
    best_score = current_score
    
    # Simulated annealing
    T = 5.0
    cooling = 0.95
    
    while time.time() < deadline and T > 0.01:
        improved = False
        
        for i in range(len(family)):
            if time.time() > deadline:
                break
            A = family[i]
            
            best_gain = 0
            best_B = -1
            
            for B in range(total_masks):
                if B in fam_set:
                    continue
                # Gain from replacing A with B
                # Remove A: lose scores[A] comparable pairs
                # Add B: gain (number of family members comparable to B, excluding A)
                # But we need to be careful about double counting
                # 
                # Current contribution of A to total score: scores[A]
                # After removing A and adding B:
                # New scores for other elements change
                # 
                # Simpler: compute the change in total score
                # Total score = sum of scores / 2
                # When we remove A: each s comparable to A loses 1 from its score, A is removed
                # When we add B: each s comparable to B gains 1, B is added
                # 
                # Change in sum of scores:
                # - Remove A: -scores[A] (A's score) - sum over s comparable to A of 1 (each loses 1)
                #   But scores[A] already counts comparable family members, so removing A reduces each comparable s by 1
                #   Total change from removal: -scores[A] - scores[A] = -2*scores[A]? No...
                # 
                # Let me think again. Total score = number of comparable pairs.
                # When we remove A, we lose all pairs (A, s) where s is comparable to A and s is in family.
                # That's scores[A] pairs.
                # When we add B, we gain all pairs (B, s) where s is comparable to B and s is in family (after removal of A).
                # That's the number of family members (excluding A) comparable to B.
                # 
                # So gain = (comparable count of B in family without A) - scores[A]
                
                comp_B = sum(1 for s in comparable[B] if s in fam_set and s != A)
                gain = comp_B - scores[A]
                
                if gain > best_gain:
                    best_gain = gain
                    best_B = B
                    
            if best_B != -1:
                accept = False
                if best_gain > 0:
                    accept = True
                else:
                    if random.random() < math.exp(best_gain / T):
                        accept = True
                
                if accept:
                    fam_set.remove(A)
                    fam_set.add(best_B)
                    family[i] = best_B
                    
                    # Update scores
                    update_scores_remove(A)
                    update_scores_add(best_B)
                    
                    new_score = get_total_score()
                    if new_score > best_score:
                        best_score = new_score
                        best_family = family[:]
                    
                    improved = True
        
        T *= cooling
    
    # Final greedy polish
    for _ in range(100):
        if time.time() > deadline:
            break
        improved_any = False
        for i in range(len(family)):
            if time.time() > deadline:
                break
            A = family[i]
            
            best_gain = 0
            best_B = -1
            
            for B in range(total_masks):
                if B in fam_set:
                    continue
                comp_B = sum(1 for s in comparable[B] if s in fam_set and s != A)
                gain = comp_B - scores[A]
                
                if gain > best_gain:
                    best_gain = gain
                    best_B = B
            
            if best_B != -1 and best_gain > 0:
                fam_set.remove(A)
                fam_set.add(best_B)
                family[i] = best_B
                
                update_scores_remove(A)
                update_scores_add(best_B)
                
                new_score = get_total_score()
                if new_score > best_score:
                    best_score = new_score
                    best_family = family[:]
                
                improved_any = True
        
        if not improved_any:
            break
    
    # Restore best family
    family = best_family
    
    return _convert(family)

def _convert(family_masks):
    result = []
    for mask in family_masks:
        s = []
        for b in range(12):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    return result
