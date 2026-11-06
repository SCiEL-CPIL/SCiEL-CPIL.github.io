import itertools
import random
import time

def generate_sts15():
    # Points 0..14
    # Lines of PG(3,2)
    # We can hardcode or generate
    # Generation:
    # Points are non-zero vectors in GF(2)^4
    # We can just use a known list of STS(15) or generate it.
    
    # Simple generator for STS(15)
    # Actually, let's just use a standard construction.
    # Or generate all triples and pick a Steiner system?
    # Easier: Use the fact that STS(15) is unique up to isomorphism? No, there are many.
    # But any STS(15) works.
    
    # Let's generate one using the affine geometry AG(4,2) minus infinity?
    # No, PG(3,2) is easier.
    
    points = []
    for i in range(1, 16):
        points.append(i-1)
        
    # We need the lines.
    # A line is {u, v, u+v} in GF(2)^4.
    # Let's map 0..14 to non-zero vectors.
    # 1: 0001, 2: 0010, 4: 0100, 8: 1000
    # 3: 0011, 5: 0101, 6: 0110, 7: 0111
    # 9: 1001, 10: 1010, 11: 1011, 12: 1100
    # 13: 1101, 14: 1110, 15: 1111
    # This mapping is not 1-to-1 with indices 0..14 directly if we use 1..15.
    # Let's just use indices 0..14 and define vectors.
    
    vecs = []
    for i in range(15):
        # Use binary representation of i+1
        v = 0
        x = i + 1
        for j in range(4):
            if x & (1 << j):
                v |= (1 << j)
        vecs.append(v)
        
    lines = []
    seen = set()
    for i in range(15):
        for j in range(i+1, 15):
            u = vecs[i]
            v = vecs[j]
            w = u ^ v
            # Find index of w
            # w is non-zero since u!=v
            k = vecs.index(w)
            # The line is {i, j, k}
            # Sort to unique
            line = tuple(sorted([i, j, k]))
            if line not in seen:
                seen.add(line)
                lines.append(line)
                
    # There should be 35 lines
    return [list(l) for l in lines]

def compute_alpha(triples):
    # 16 points
    # Check all subsets
    # 2^16 = 65536
    # A subset is independent if it contains no triple.
    # We want max size.
    
    # Precompute: for each point, which triples contain it?
    # Or simpler: for each subset, check if any triple is subset.
    # Optimization: 
    # If a subset S contains a triple, it's not independent.
    # We can iterate k from 16 down to 1.
    # If we find an independent set of size k, return k.
    
    # Bitmask representation
    # triples_mask: list of ints
    triple_masks = [0] * len(triples)
    for i, t in enumerate(triples):
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks[i] = m
        
    n_triples = len(triples)
    
    # Iterate subset size from 16 down
    # This might be slow if we iterate all subsets for each k.
    # Better: Iterate all subsets once, store max independent size.
    # Or:
    # For mask in range(1<<16):
    #    if mask is independent:
    #       update max
    
    # 65536 iterations is fast.
    max_size = 0
    for mask in range(1 << 16):
        # Check if mask contains any triple
        is_independent = True
        for m in triple_masks:
            if (mask & m) == m:
                is_independent = False
                break
        if is_independent:
            size = bin(mask).count("1")
            if size > max_size:
                max_size = size
    return max_size

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    sts15 = generate_sts15()
    base_triples = [list(t) for t in sts15]
    
    best_alpha = 16 # Init high
    best_triples = None
    
    # Initial guess: STS(15) + isolated 15
    # Alpha is 8
    test_triples = base_triples + [] # 15 is isolated
    # Wait, if 15 is isolated, it's not in any triple.
    # So test_triples is just sts15.
    alpha = compute_alpha(test_triples)
    # Note: compute_alpha checks subsets of 16 points.
    # If 15 is not in any triple, any independent set in 0..14 can include 15.
    # So alpha = alpha(0..14) + 1.
    # For STS(15), alpha is 7. So total 8.
    
    if alpha < best_alpha:
        best_alpha = alpha
        best_triples = test_triples
    
    # Search for better configurations
    # Strategy: Randomly select k disjoint pairs from 0..14.
    # Remove corresponding lines from sts15.
    # Add triples {pair, 15}.
    
    # Map pair to line index in sts15
    pair_to_line = {}
    for i, line in enumerate(sts15):
        for a in range(3):
            for b in range(a+1, 3):
                p1, p2 = line[a], line[b]
                pair_to_line[(min(p1,p2), max(p1,p2))] = i
                
    points_15 = list(range(15))
    
    while time.time() < deadline:
        # Random k between 1 and 7
        k = random.randint(1, 7)
        # Select k disjoint pairs
        available = set(points_15)
        selected_pairs = []
        while len(selected_pairs) < k and available:
            # Pick two random points
            p1, p2 = random.sample(list(available), 2)
            pair = (min(p1,p2), max(p1,p2))
            # Check if this pair's line is still available?
            # We are removing lines. We need to ensure we don't pick the same line twice.
            # Since pairs are disjoint, lines are distinct?
            # In STS, distinct disjoint pairs are in distinct lines.
            # Yes.
            selected_pairs.append(pair)
            available.remove(p1)
            available.remove(p2)
            
        # Construct new triples
        new_triples = []
        removed_indices = set()
        for pair in selected_pairs:
            idx = pair_to_line.get(pair)
            if idx is not None:
                removed_indices.add(idx)
            # Add new triple
            new_triples.append([pair[0], pair[1], 15])
            
        # Add remaining sts15 lines
        for i, line in enumerate(sts15):
            if i not in removed_indices:
                new_triples.append(line)
                
        # Check linearity (should be valid by construction, but let's be safe)
        # Actually, the construction guarantees linearity.
        
        alpha = compute_alpha(new_triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = new_triples
            # If we reach 7, we might stop or continue to see if 6 is possible?
            # Theoretical min is likely 7.
            if best_alpha <= 7:
                # We can try to find 6, but 7 is a huge jump from 8.
                # Let's run a bit longer to be sure 7 is stable or if 6 exists.
                pass

    return best_triples
