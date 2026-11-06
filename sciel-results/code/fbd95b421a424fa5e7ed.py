import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = 16
    k = 4
    all_sets = []
    for comb in combinations(range(n), k):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        all_sets.append(mask)
    
    total_sets = len(all_sets)  # C(16,4) = 1820
    
    # Precompute pairwise intersections
    # inter[i][j] = all_sets[i] & all_sets[j]
    # This is a 1820x1820 matrix, but we can use a dict or list of lists
    # Actually, for speed, let's compute on the fly but cache
    
    # Better approach: for the genetic algorithm, we need fast conflict checking
    # Two sets A and B conflict with a third set C if they form a sunflower
    # Sunflower: A∩B = A∩C = B∩C
    # For a given pair (A,B), the sets C that conflict are those where:
    #   C contains (A∩B), and C is disjoint from (A\B) and (B\A)
    # Wait, that's not quite right. Let me re-derive.
    # A∩B = A∩C = B∩C = I
    # This means:
    #   I ⊆ A, I ⊆ B, I ⊆ C
    #   A∩C = I means A\C ⊆ A\I and C\A ⊆ C\I, and (A\I)∩C = empty
    #   Similarly (B\I)∩C = empty
    # So C must contain I, and be disjoint from (A\I) ∪ (B\I)
    # Also C must be a k-set.
    
    # For the greedy/evolutionary approach, when we have a family F, 
    # adding a new set s is valid iff for every pair (a,b) in F, 
    # s does not form a sunflower with a and b.
    # But that's O(|F|^2) per insertion, which is too slow.
    
    # Alternative: maintain for each pair in F, the "forbidden" condition.
    # Actually, the constraint is: for any three distinct sets in F, they don't form a sunflower.
    # This is a 3-uniform hypergraph independence problem.
    
    # Let me think about the structure differently.
    # For k=4, n=16, the maximum sunflower-free family...
    # The Erdős-Rado sunflower theorem says for k=4, the sunflower number is bounded.
    # But we want the maximum size of a family with NO three-petal sunflower.
    
    # Let me try a different approach: 
    # 1. Build a conflict graph where each node is a 4-set, and edges represent "can coexist"
    #    Actually, the constraint is 3-wise, not pairwise. So it's not a simple graph.
    
    # Let me use the following approach:
    # - Greedy construction with multiple random orderings
    # - Local search: try to add sets, or swap sets to improve
    # - Key optimization: for a given family F, maintain a data structure to quickly check
    #   if a new set s can be added.
    
    # For a family F, s can be added iff for every pair (a,b) in F, 
    # s does NOT form a sunflower with a and b.
    # s forms a sunflower with a,b iff a∩b = a∩s = b∩s.
    # Let I = a∩b. Then we need a∩s = I and b∩s = I.
    # a∩s = I means: s contains I, and s is disjoint from a\I.
    # b∩s = I means: s contains I, and s is disjoint from b\I.
    # So s contains I and is disjoint from (a\I) ∪ (b\I).
    
    # For each pair (a,b) in F, we can precompute:
    #   I = a & b
    #   forbidden_mask = (a & ~I) | (b & ~I)  # = a^b essentially, but only the parts outside I
    #   Actually a\I = a & ~I, b\I = b & ~I
    #   So the condition is: (s & I) == I and (s & (a&~I | b&~I)) == 0
    
    # To check if s can be added to F, we need to verify this for ALL pairs in F.
    # That's O(|F|^2) per check, which is too slow for large F.
    
    # Optimization: group pairs by their core I.
    # For each possible core I (which is a subset of [16] with size 0,1,2,3,4),
    # maintain the union of forbidden masks for all pairs with that core.
    # When checking s: for each core I that s contains, check if s is disjoint from the union of forbidden masks for pairs with core I.
    
    # Number of possible cores: sum_{i=0}^{4} C(16,i) = 1+16+120+560+1820 = 2517
    # But we only care about cores that actually appear in pairs from F.
    
    # Let me implement this:
    
    def build_family_greedy(order, time_limit):
        """Build a family greedily given an ordering of all_sets."""
        family = []
        # core_to_forbidden: dict from core_mask -> union of forbidden masks
        core_to_forbidden = {}
        
        for idx, s in enumerate(order):
            if time.time() > time_limit:
                break
            
            # Check if s can be added
            valid = True
            # s must contain core I for the pair to be relevant
            # For each core I in core_to_forbidden:
            #   if (s & I) == I, then check (s & core_to_forbidden[I]) == 0
            for I, forb in core_to_forbidden.items():
                if (s & I) == I:
                    if (s & forb) != 0:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Add s to family
            # Update core_to_forbidden for all pairs (s, existing)
            for existing in family:
                I = s & existing
                # forbidden mask for this pair: (s & ~I) | (existing & ~I)
                forb = (s & ~I) | (existing & ~I)
                if I in core_to_forbidden:
                    core_to_forbidden[I] |= forb
                else:
                    core_to_forbidden[I] = forb
            
            family.append(s)
        
        return family
    
    # Run multiple greedy constructions with different random orderings
    best_family = []
    best_size = 0
    
    while time.time() < deadline:
        order = all_sets[:]
        random.shuffle(order)
        family = build_family_greedy(order, deadline)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
        
        if best_size >= 250:
            break
    
    # Local search: try to improve by adding more sets
    # For the best family, try to add any set that wasn't included
    if time.time() < deadline:
        family_set = set(best_family)
        # Build core_to_forbidden for the current family
        core_to_forbidden = {}
        for i in range(len(best_family)):
            for j in range(i+1, len(best_family)):
                s = best_family[i]
                e = best_family[j]
                I = s & e
                forb = (s & ~I) | (e & ~I)
                if I in core_to_forbidden:
                    core_to_forbidden[I] |= forb
                else:
                    core_to_forbidden[I] = forb
        
        # Try to add sets not in the family
        added = True
        while added and time.time() < deadline:
            added = False
            for s in all_sets:
                if s in family_set:
                    continue
                valid = True
                for I, forb in core_to_forbidden.items():
                    if (s & I) == I:
                        if (s & forb) != 0:
                            valid = False
                            break
                if valid:
                    # Add s
                    for existing in best_family:
                        I = s & existing
                        forb = (s & ~I) | (existing & ~I)
                        if I in core_to_forbidden:
                            core_to_forbidden[I] |= forb
                        else:
                            core_to_forbidden[I] = forb
                    best_family.append(s)
                    family_set.add(s)
                    added = True
        
        best_size = len(best_family)
    
    # Convert to list of lists
    result = []
    for mask in best_family:
        elements = []
        for i in range(n):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(elements)
    
    return result
