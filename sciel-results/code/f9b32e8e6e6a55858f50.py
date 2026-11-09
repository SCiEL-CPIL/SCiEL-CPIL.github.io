import random
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer for safety
    
    n = 16
    best_alpha = n
    best_triples = []
    
    all_triples = list(itertools.combinations(range(n), 3))
    
    def get_alpha(triples):
        if not triples:
            return n
            
        # Convert triples to bitmasks
        t_masks = set()
        for a, b, c in triples:
            t_masks.add((1 << a) | (1 << b) | (1 << c))
        t_masks = list(t_masks)
        
        max_alpha = 0
        # Iterate all masks and track max popcount of independent sets
        for mask in range((1 << n) - 1, -1, -1):
            pc = mask.bit_count()
            if pc <= max_alpha:
                continue
            # Check if independent
            independent = True
            for t in t_masks:
                if (t & mask) == t:
                    independent = False
                    break
            if independent:
                max_alpha = pc
                if max_alpha == n:
                    break
                
        return max_alpha

    def try_random_greedy(seed_offset=0):
        triples = []
        used_pairs = set()
        
        # Shuffle all triples
        candidates = all_triples[:]
        random.shuffle(candidates)
        
        for t in candidates:
            a, b, c = t
            p1 = (a, b)
            p2 = (a, c)
            p3 = (b, c)
            
            # Check if any pair is used
            if p1 in used_pairs or p2 in used_pairs or p3 in used_pairs:
                continue
                
            # Add triple
            triples.append(t)
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            
        return triples

    def algebraic_base():
        """Generate triples using GF(2)^4 affine line structure.
        
        In GF(2)^4, each 1-dimensional subspace has 2 nonzero elements.
        An affine line through point p in direction v is {p, p+v, p+2v} = {p, p+v, p}
        Wait, in GF(2), 2v = 0, so this gives only 2 points.
        
        Instead, use 2-dimensional subspaces: each has 4 points forming a "square".
        We can form triples from these squares.
        
        Actually, let's use the fact that in GF(2)^4, we can partition into 15 parallel classes
        of 8 lines each (each line being a 1-dim affine subspace = 2 points).
        That doesn't give triples directly.
        
        Better approach: Use the 16 points as GF(2)^4. For each nonzero vector v,
        consider the 8 cosets of the 1-dim subspace <v>. Each coset has 2 points.
        This gives pairs, not triples.
        
        Alternative: Use quadratic forms or other structures.
        
        Let me try a different algebraic construction:
        Points are 0..15. For each pair (i, j) with i < j, consider i XOR j.
        We want triples {a, b, c} such that some algebraic relation holds.
        
        Simplest: Use the 15 nonzero vectors of GF(2)^4. Each nonzero v defines
        a perfect matching on the 16 points (pairs {x, x+v}).
        We can't directly get triples from this.
        
        Let me try: For each 2-dim subspace W of GF(2)^4, W has 4 elements {0, u, v, u+v}.
        The cosets of W are 4 sets of 4 points. From each 4-set, we can form 4 triples
        (choosing 3 of 4 points). But we need to be careful about linearity.
        
        Actually, let's just use random greedy with multiple restarts and local search.
        The algebraic insight might help with initialization.
        """
        # Generate a structured initial set using XOR-based patterns
        triples = []
        used_pairs = set()
        
        # For each nonzero direction v in GF(2)^4, create triples
        # using the structure of the vector space
        for v in range(1, 16):
            # Find points that can form triples with this direction
            for base in range(16):
                a = base
                b = base ^ v
                c = base ^ (v << 1) if (v << 1) < 16 else base ^ (v ^ (v << 1))
                # This is getting complicated. Let's just do random greedy.
                break
            break
        
        # Fall back to random greedy
        return try_random_greedy()

    def local_search(triples, time_limit=5.0):
        """Try to improve the independence number by swapping triples."""
        if time.time() > time_limit:
            return triples
            
        current_alpha = get_alpha(triples)
        current_triples = triples[:]
        
        # Build used pairs
        used_pairs = set()
        for a, b, c in current_triples:
            used_pairs.add((a, b))
            used_pairs.add((a, c))
            used_pairs.add((b, c))
        
        # Try removing and replacing triples
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(len(current_triples)):
                if time.time() > time_limit:
                    break
                # Remove triple i
                a, b, c = current_triples[i]
                removed = current_triples.pop(i)
                used_pairs.discard((a, b))
                used_pairs.discard((a, c))
                used_pairs.discard((b, c))
                
                # Try adding different triples
                for new_t in all_triples:
                    na, nb, nc = new_t
                    p1 = (na, nb)
                    p2 = (na, nc)
                    p3 = (nb, nc)
                    
                    if p1 in used_pairs or p2 in used_pairs or p3 in used_pairs:
                        continue
                    
                    # Add new triple
                    current_triples.append(new_t)
                    used_pairs.add(p1)
                    used_pairs.add(p2)
                    used_pairs.add(p3)
                    
                    new_alpha = get_alpha(current_triples)
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        improved = True
                    else:
                        # Revert
                        current_triples.pop()
                        used_pairs.discard(p1)
                        used_pairs.discard(p2)
                        used_pairs.discard(p3)
                
                # Put back original triple
                current_triples.insert(i, removed)
                used_pairs.add((a, b))
                used_pairs.add((a, c))
                used_pairs.add((b, c))
                
                if improved:
                    break
        
        return current_triples

    trials = 0
    while time.time() < deadline:
        trials += 1
        trs = try_random_greedy()
        
        # Calculate alpha
        alpha = get_alpha(trs)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = trs[:]
            if alpha <= 7:
                break
            # Try local search to improve
            if time.time() < deadline - 10:
                improved = local_search(trs, time_limit=min(5.0, deadline - time.time()))
                improved_alpha = get_alpha(improved)
                if improved_alpha < best_alpha:
                    best_alpha = improved_alpha
                    best_triples = improved
                    if alpha <= 7:
                        break

    return best_triples
# EVOLVE-BLOCK-END
