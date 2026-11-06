import random

def construct():
    N = 180
    # Pre-compute all reciprocal triples {a, b, c} with a < b < c
    triples = []
    for a in range(1, N + 1):
        # b must be > a
        # c = ab / (b - a)
        # c > b => 2a > b
        # c <= N
        max_b = min(N, 2 * a - 1)
        for b in range(a + 1, max_b + 1):
            denom = b - a
            if (a * b) % denom == 0:
                c = (a * b) // denom
                if c > b and c <= N:
                    triples.append((a, b, c))
    
    # If no triples, return all
    if not triples:
        return list(range(1, N + 1))
        
    # Map element to indices of triples it appears in
    # This helps in fast updates
    elem_to_triples = {i: [] for i in range(1, N + 1)}
    for i, (a, b, c) in enumerate(triples):
        elem_to_triples[a].append(i)
        elem_to_triples[b].append(i)
        elem_to_triples[c].append(i)
        
    best_size = 0
    best_set = set()
    
    # Randomized Greedy Hitting Set
    # We want to remove the minimum number of elements to break all triples.
    # Let removed = set()
    # Active triples = triples not intersecting removed
    
    num_restarts = 1000 # Adjust based on speed, 1000 should be fast enough for N=180
    
    for _ in range(num_restarts):
        removed = set()
        # To speed up, we maintain a list of active triple indices
        # But checking "is this triple active" requires checking if any of its elements are in removed.
        # Since N is small, we can just iterate through all triples to find the best candidate.
        
        # Optimization: 
        # We pick an element to remove. 
        # Strategy: Pick element that appears in the most *active* triples.
        # Tie-breaking: Random among top few, or just random from all max-count.
        
        # To avoid O(M) scan for every removal (M is #triples), we can just do it.
        # M is likely small.
        
        while True:
            # Find active triples
            # A triple is active if none of its elements are in removed
            
            # Calculate frequency of each element in active triples
            # We only care about elements NOT in removed
            counts = {}
            has_active = False
            
            # Iterate through all triples
            for t in triples:
                a, b, c = t
                # Check if active
                if a in removed or b in removed or c in removed:
                    continue
                has_active = True
                # Count
                if a not in counts: counts[a] = 0
                if b not in counts: counts[b] = 0
                if c not in counts: counts[c] = 0
                counts[a] += 1
                counts[b] += 1
                counts[c] += 1
                
            if not has_active:
                break
                
            # Find max count
            max_count = 0
            candidates = []
            for x, cnt in counts.items():
                if cnt > max_count:
                    max_count = cnt
                    candidates = [x]
                elif cnt == max_count:
                    candidates.append(x)
            
            # Random selection from candidates
            # To diversify, we can pick randomly from the top K candidates if we wanted, 
            # but picking from max count is standard greedy. 
            # To improve, we can sometimes pick from second-best? 
            # Let's stick to random from max count for now.
            if candidates:
                rem = random.choice(candidates)
                removed.add(rem)
        
        # Construct S
        S = set(range(1, N + 1)) - removed
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    return list(best_set)
