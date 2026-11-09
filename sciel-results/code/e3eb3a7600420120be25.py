import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 16
    k = 4
    
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
    
    num_sets = len(all_masks)
    
    # Precompute conflict graph: two sets conflict if they are part of a sunflower triple
    # Actually, for this problem, we need to check if adding a set creates a sunflower with
    # two existing sets. Let's precompute which pairs of sets "conflict" in the sense that
    # they would form a sunflower with some third set.
    
    # Better approach: Precompute for each pair (i, j), whether they can be in a sunflower
    # with some k. Two sets A, B are in a sunflower with C if A∩B = A∩C = B∩C.
    # This means C must contain A∩B and be disjoint from (A\B) ∪ (B\A) outside A∩B.
    # Actually: A∩B = v, then C must satisfy C∩A = v and C∩B = v.
    # So C must contain v, and C must be disjoint from (A\v) and (B\v).
    # So C ⊆ (complement of (A\B) ∪ (B\A)) and C ⊇ v, and |C| = k.
    
    # For the greedy minimum-degree approach, we need to know conflicts between sets.
    # Two sets i and j "conflict" if there exists a third set k such that i, j, k form a sunflower.
    # But that's expensive to precompute. Instead, let's use a different approach.
    
    # Alternative: Use a greedy algorithm that maintains the family and checks compatibility.
    # The key insight from the directive is to use minimum-degree greedy on the conflict graph.
    
    # Let's precompute the conflict graph where edge (i,j) means i and j cannot both be in
    # the family because they would form a sunflower with some third set.
    # Actually, that's not quite right either. The constraint is that NO three sets form a sunflower.
    # So i and j can both be in the family as long as no third set k in the family forms a sunflower with them.
    
    # Let me think differently. The constraint is: for any three distinct sets A, B, C in the family,
    # it's NOT the case that A∩B = A∩C = B∩C.
    
    # So when adding a new set C to family F, we need to check that for every pair (A, B) in F,
    # C does not form a sunflower with A and B.
    
    # Precompute: for each pair (i, j), what is the set of masks k that would form a sunflower with i and j?
    # Then when adding k, we check if any pair (i, j) in F has k in their forbidden set.
    
    # This is still expensive. Let's just use a direct check but with minimum-degree greedy.
    
    best_family_size = 0
    best_family = []
    
    def is_compatible(mask, F):
        """Check if mask can be added to F without creating a sunflower."""
        for i in range(len(F)):
            a = F[i]
            for j in range(i+1, len(F)):
                b = F[j]
                # Check if mask, a, b form a sunflower
                ab = a & b
                ac = a & mask
                bc = b & mask
                if ab == ac == bc:
                    return False
        return True
    
    def compute_degree(mask, available):
        """Compute the number of available sets that conflict with mask."""
        # Two sets conflict if they would form a sunflower with some third set
        # For simplicity, let's just count how many available sets would be blocked
        # Actually, for minimum-degree greedy, we want to pick the set with the fewest
        # conflicts, meaning it blocks the fewest other sets.
        
        # A simpler heuristic: pick the set that is compatible with the most other available sets
        count = 0
        for other in available:
            if other == mask:
                continue
            # Check if mask and other can coexist (i.e., there's no third set that would
            # form a sunflower with both). This is complex.
            # Let's just use a simpler heuristic: random selection with some bias.
            pass
        return count
    
    # Simplified approach: Use minimum-degree greedy where degree = number of sets
    # that would be incompatible if we picked this one.
    
    while time.time() < deadline:
        random.shuffle(all_masks)
        
        # Use a priority-based greedy: at each step, pick the set with minimum "cost"
        # where cost is the number of remaining sets it would conflict with.
        
        F = []
        remaining = list(range(num_sets))
        
        while remaining and time.time() < deadline:
            # Find the set with minimum degree (fewest conflicts with remaining)
            best_idx = -1
            best_degree = float('inf')
            
            # Sample a subset for speed
            sample_size = min(len(remaining), 200)
            sample = random.sample(remaining, sample_size)
            
            for idx in sample:
                mask = all_masks[idx]
                # Count how many remaining sets are incompatible with this mask
                # Two sets are incompatible if they form a sunflower with some third set
                # For now, use a simpler metric: just check compatibility with F
                if not is_compatible(mask, F):
                    continue
                
                # Count conflicts with other remaining sets
                degree = 0
                for other_idx in remaining:
                    if other_idx == idx:
                        continue
                    other_mask = all_masks[other_idx]
                    # Check if mask and other_mask would form a sunflower with any set in F
                    # This is the key conflict: if there exists c in F such that
                    # mask, other_mask, c form a sunflower
                    for f_mask in F:
                        m_o = mask & other_mask
                        m_f = mask & f_mask
                        o_f = other_mask & f_mask
                        if m_o == m_f == o_f:
                            degree += 1
                            break
                
                if degree < best_degree:
                    best_degree = degree
                    best_idx = idx
            
            if best_idx == -1:
                break
            
            F.append(all_masks[best_idx])
            remaining.remove(best_idx)
        
        if len(F) > best_family_size:
            best_family_size = len(F)
            best_family = F.copy()
    
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i+1)
        result.append(sorted(elems))
    
    return result
