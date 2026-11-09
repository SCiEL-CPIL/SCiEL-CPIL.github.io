import time
import random

N = 9
MAX_VERTICES = 1 << N

def construct():
    # Deadline guard: 140 seconds to be safe against 150s limit
    deadline = time.time() + 140.0
    best_path = []
    
    # Precompute neighbors for all vertices
    # neighbors[v] is a list of vertices adjacent to v
    neighbors = [[] for _ in range(MAX_VERTICES)]
    for v in range(MAX_VERTICES):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
            
    def grow_and_refine():
        # Random start
        start = random.randint(0, MAX_VERTICES - 1)
        path = [start]
        used = {start}
        # blocked[v] is True if v is adjacent to any vertex in path
        # We don't need to store it as a set if we just check, but a set is faster for lookup
        blocked = set()
        
        # Initialize blocked with neighbors of start
        for n in neighbors[start]:
            if n not in used:
                blocked.add(n)
                
        # Bit orders for randomization
        # We will randomize the order of bits we try for extension
        bit_order = list(range(N))
        
        # Bidirectional Growth
        # We maintain left_end and right_end indices in path
        # Actually, just use path[0] and path[-1]
        
        while True:
            if time.time() > deadline:
                return path
            
            left_candidates = []
            right_candidates = []
            
            # Check left end
            cur = path[0]
            for b in bit_order:
                v = cur ^ (1 << b)
                if v not in used and v not in blocked:
                    # Calculate Warnsdorff score: number of valid neighbors of v
                    # Valid neighbors of v are those not in used and not in blocked
                    # Note: v itself is not in used/blocked yet
                    score = 0
                    for nb in neighbors[v]:
                        if nb not in used and nb not in blocked:
                            score += 1
                    left_candidates.append((score, v))
                    
            # Check right end
            cur = path[-1]
            for b in bit_order:
                v = cur ^ (1 << b)
                if v not in used and v not in blocked:
                    score = 0
                    for nb in neighbors[v]:
                        if nb not in used and nb not in blocked:
                            score += 1
                    right_candidates.append((score, v))
                    
            if not left_candidates and not right_candidates:
                break
                
            # Choose the move with the lowest score (most constrained)
            # If tie, randomize
            if left_candidates and right_candidates:
                min_score = min(min(s for s, _ in left_candidates), min(s for s, _ in right_candidates))
                # Filter candidates with min score
                left_min = [v for s, v in left_candidates if s == min_score]
                right_min = [v for s, v in right_candidates if s == min_score]
                
                # Randomly pick from the combined set of min-score candidates
                # Or just pick one side? Randomizing side is good.
                if random.random() < 0.5:
                    v = random.choice(left_min)
                    path.insert(0, v)
                else:
                    v = random.choice(right_min)
                    path.append(v)
            elif left_candidates:
                min_score = min(s for s, _ in left_candidates)
                left_min = [v for s, v in left_candidates if s == min_score]
                v = random.choice(left_min)
                path.insert(0, v)
            else:
                min_score = min(s for s, _ in right_candidates)
                right_min = [v for s, v in right_candidates if s == min_score]
                v = random.choice(right_min)
                path.append(v)
                
            # Update state
            used.add(v)
            for nb in neighbors[v]:
                if nb not in used:
                    blocked.add(nb)
                    
            # Randomize bit order occasionally to escape local minima in heuristic
            if random.random() < 0.05:
                random.shuffle(bit_order)

        # Local Refinement: Insertion
        # We try to insert a vertex between u and v if there exists w such that
        # dist(u,w)=1, dist(w,v)=1, w not in used, w not in blocked.
        # Note: If we insert w, w becomes part of the path.
        # Its neighbors might become blocked.
        
        modified = True
        while modified and time.time() < deadline:
            modified = False
            # Iterate backwards to handle insertions correctly? 
            # Or just restart the loop if modification happens.
            # Since N=9, path length is small (<512), O(L^2) is fine.
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i+1]
                # Find common neighbors of u and v
                # Common neighbors w must satisfy: w^u has 1 bit, w^v has 1 bit.
                # This implies w = u ^ (1<<b) = v ^ (1<<c).
                # So u^v must have exactly 2 bits set.
                diff = u ^ v
                if diff.bit_count() != 2:
                    continue
                
                # The two bits are the ones that differ.
                # Let the bits be b1, b2.
                # w1 = u ^ (1<<b1) = v ^ (1<<b2)
                # w2 = u ^ (1<<b2) = v ^ (1<<b1)
                
                # Extract bits
                b1 = (diff & -diff).bit_length() - 1
                b2 = (diff // (1 << b1) & -diff // (1 << b1)).bit_length() - 1 # This is clunky
                # Simpler:
                bits = []
                temp_diff = diff
                while temp_diff:
                    lsb = temp_diff & -temp_diff
                    bits.append(lsb.bit_length() - 1)
                    temp_diff ^= lsb
                
                w1 = u ^ (1 << bits[0])
                w2 = u ^ (1 << bits[1])
                
                # Check w1
                if w1 not in used and w1 not in blocked:
                    # Insert w1
                    path.insert(i+1, w1)
                    used.add(w1)
                    for nb in neighbors[w1]:
                        if nb not in used:
                            blocked.add(nb)
                    modified = True
                    break # Restart scan
                    
                # Check w2
                if w2 not in used and w2 not in blocked:
                    # Insert w2
                    path.insert(i+1, w2)
                    used.add(w2)
                    for nb in neighbors[w2]:
                        if nb not in used:
                            blocked.add(nb)
                    modified = True
                    break # Restart scan
                    
        return path

    # Run multiple trials
    while time.time() < deadline:
        p = grow_and_refine()
        if len(p) > len(best_path):
            best_path = p
            # Optional: print progress for debugging if needed, but stdout might be slow?
            # print(f"New best: {len(best_path)}")
            
    return best_path
