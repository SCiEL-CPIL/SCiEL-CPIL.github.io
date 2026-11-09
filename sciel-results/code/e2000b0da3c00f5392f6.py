import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline with 5s safety margin
    
    # 255 = 3 * 5 * 17
    # We need a subset of [1,255] containing 255 where gcd(a,b) > 1 for all pairs.
    
    # Baseline: multiples of 3 (size 85)
    baseline = list(range(3, 256, 3))
    best_set = baseline[:]
    best_size = len(best_set)
    
    # Numbers that share a factor with 255: divisible by 3, 5, or 17
    # Any element in our set must have gcd > 1 with 255, so must be divisible by 3, 5, or 17
    candidates = [n for n in range(1, 256) if n % 3 == 0 or n % 5 == 0 or n % 17 == 0]
    
    # Build adjacency: two numbers are "compatible" if gcd > 1
    n_cand = len(candidates)
    cand_set = set(candidates)
    
    # Precompute gcd compatibility
    # compat[i][j] = True if gcd(candidates[i], candidates[j]) > 1
    # Use a bitset approach for efficiency
    # First, let's build the conflict graph (edges where gcd == 1)
    # and find maximum independent set = maximum clique in complement
    
    # Actually, let's directly build the "compatible" graph and find max clique
    # Two vertices are adjacent if gcd > 1
    
    # For Bron-Kerbosch with pivot, we need adjacency as sets
    adj = [set() for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Find maximum clique using Bron-Kerbosch with pivot
    # But the graph is large (candidates can be up to ~170 nodes), so we need to be careful
    
    # Let's first try a greedy approach to get a good lower bound, then try to improve
    
    # Greedy: start with multiples of 3, try to add elements
    # Actually, let's use a different strategy:
    # The set must contain 255. Every element must share a factor with 255.
    # Let's categorize candidates by their prime factors among {3, 5, 17}
    
    # Strategy: try all combinations of "which prime factors to focus on"
    # Option 1: All multiples of 3 (size 85)
    # Option 2: Mix - include some multiples of 5 or 17 that don't conflict
    
    # Let's try a greedy local search
    # Start with multiples of 3
    current_set = set(range(3, 256, 3))
    
    # Try to add elements not in current_set
    for n in candidates:
        if n in current_set:
            continue
        if time.time() > deadline:
            break
        # Check if n is compatible with all elements in current_set
        can_add = True
        for m in current_set:
            if math.gcd(n, m) == 1:
                can_add = False
                break
        if can_add:
            current_set.add(n)
    
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = list(current_set)
    
    # Try different starting points
    # Start with multiples of 5
    if time.time() < deadline:
        current_set = set(range(5, 256, 5))
        # Add 255 if not present (it is, 255/5=51)
        if 255 not in current_set:
            current_set.add(255)
        for n in candidates:
            if n in current_set:
                continue
            if time.time() > deadline:
                break
            can_add = True
            for m in current_set:
                if math.gcd(n, m) == 1:
                    can_add = False
                    break
            if can_add:
                current_set.add(n)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
    
    # Start with multiples of 17
    if time.time() < deadline:
        current_set = set(range(17, 256, 17))
        if 255 not in current_set:
            current_set.add(255)
        for n in candidates:
            if n in current_set:
                continue
            if time.time() > deadline:
                break
            can_add = True
            for m in current_set:
                if math.gcd(n, m) == 1:
                    can_add = False
                    break
            if can_add:
                current_set.add(n)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
    
    # Try mixed starts: multiples of 3 + some multiples of 5 that are compatible
    if time.time() < deadline:
        # For each multiple of 5, check if it's compatible with all multiples of 3
        # A multiple of 5 is compatible with all multiples of 3 iff it shares a factor with 3
        # i.e., it's also a multiple of 3 (i.e., multiple of 15)
        # OR... wait, gcd(5, 3) = 1, so 5 is NOT compatible with 3.
        # So a multiple of 5 (not multiple of 3) has gcd 1 with 3, hence incompatible.
        # This means we can't simply mix multiples of 3 and 5.
        
        # What about multiples of 15? They're already in the multiples of 3 set.
        # So mixing 3 and 5 multiples doesn't help beyond what's already in mult of 3.
        
        # What about adding numbers that are multiples of 17 (not 3 or 5)?
        # gcd(17, 3) = 1, so 17 is incompatible with 3.
        # So we can't add pure multiples of 17 to the multiples of 3 set either.
        
        # This suggests the multiples of 3 set might be optimal.
        # Let's verify with a more systematic approach.
        
        pass
    
    # Let's try a more systematic Bron-Kerbosch on a smaller subgraph
    # Focus on the "interesting" candidates: those not already in multiples of 3
    # and see if we can build a set larger than 85
    
    # Actually, let's try: what if we use a different prime?
    # The key insight: any valid set containing 255 must have every element 
    # sharing a factor with 255 = 3*5*17.
    
    # Let's try to find if there's a set larger than 85 by considering
    # numbers divisible by 5 or 17 but not 3, and checking conflicts.
    
    # Numbers divisible by 5 but not 3: 5, 10, 20, 25, 35, 40, 50, 55, 65, 70, 80, 85, 95, 100, 110, 115, 125, 130, 140, 145, 155, 160, 170, 175, 185, 190, 200, 205, 215, 220, 230, 235, 245, 250
    # These have gcd 1 with 3, so they conflict with ALL multiples of 3.
    # So if we include any of these, we can't include any multiple of 3.
    
    # Similarly for multiples of 17 but not 3 or 5: 17, 34, 68, 85, 119, 153, 170, 204, 238
    # Wait, 85 = 5*17, 170 = 2*5*17, 153 = 9*17 = 3^2*17
    # Let me recheck: multiples of 17 in [1,255]: 17,34,51,68,85,102,119,136,153,170,187,204,221,238,255
    # Those also divisible by 3: 51(3*17), 102(6*17), 153(9*17), 204(12*17), 255(15*17)
    # Those divisible by 5: 85(5*17), 170(10*17), 255(15*17)
    # Pure 17 (not 3 or 5): 17, 34, 68, 119, 136, 187, 221, 238
    
    # So the structure is:
    # Group A: multiples of 3 (85 elements) - all mutually compatible
    # Group B: multiples of 5 not 3 (34 elements) - all conflict with Group A
    # Group C: multiples of 17 not 3 or 5 (8 elements) - all conflict with Group A
    # Group D: multiples of 15 (17 elements) - in both A and B, compatible with A
    # Group E: multiples of 51 (5 elements) - in both A and C, compatible with A
    # Group F: multiples of 85 (3 elements: 85, 170, 255) - in B and C, conflict with A
    
    # Within Group B: two multiples of 5 are compatible iff they share another factor
    # Within Group C: two multiples of 17 are compatible iff they share another factor
    
    # The question: can we build a set from B ∪ C ∪ F that's larger than 85?
    # |B| = 34, |C| = 8, |F| = 3 (but F ⊂ B ∩ C)
    # B ∪ C has at most 34 + 8 = 42 elements, which is less than 85.
    # So we can't beat 85 by using only B and C.
    
    # What about mixing A with some elements?
    # Any element in B (not in A) conflicts with all of A.
    # Any element in C (not in A) conflicts with all of A.
    # So if we include any element from B\A or C\A, we lose all of A.
    
    # Therefore, the maximum set is either:
    # 1. All of A (size 85), or
    # 2. A subset of (B ∪ C) that's a clique (size ≤ 42)
    
    # So 85 is optimal!
    
    # But wait - what about numbers that are in A but we might exclude to gain more from B∪C?
    # If we exclude some from A, we can add some from B∪C, but B∪C elements conflict with ALL of A.
    # So excluding one from A lets us add at most... no, each B or C element conflicts with ALL of A.
    # So we can't add any B or C element unless we remove ALL of A.
    
    # Hence, the answer is 85 (all multiples of 3).
    
    # Let's also verify: are there numbers not divisible by 3, 5, or 17 that could be added?
    # No! Every element must have gcd > 1 with 255 = 3*5*17, so must be divisible by 3, 5, or 17.
    
    # Final verification of the baseline
    result = best_set
    if 255 not in result:
        result.append(255)
    
    # Verify the result
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                # Invalid - fall back to pure multiples of 3
                result = list(range(3, 256, 3))
                break
        else:
            continue
        break
    
    return result
