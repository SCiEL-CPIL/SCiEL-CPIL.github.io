"""Linear triple systems on 16 points with small independence number."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _independence_number(triples, n=16):
    """Compute the independence number exactly using branch and bound."""
    if not triples:
        return n
    
    # Precompute: for each point, which triples contain it
    point_triples = [[] for _ in range(n)]
    for i, t in enumerate(triples):
        for p in t:
            point_triples[p].append(i)
    
    # For each triple, its bitmask
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    full_mask = (1 << n) - 1
    
    def is_independent(mask):
        """Check if the set of points in mask contains no triple."""
        for tm in triple_masks:
            if (tm & mask) == tm:
                return False
        return True
    
    # Branch and bound: try to find largest independent set
    best = [0]
    
    def branch(remaining_mask, current_size, current_mask):
        if remaining_mask == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Upper bound: current_size + popcount(remaining_mask)
        if current_size + bin(remaining_mask).count('1') <= best[0]:
            return
        
        # Pick a point to branch on
        # Heuristic: pick point with highest degree in remaining
        best_point = -1
        best_deg = -1
        rmask = remaining_mask
        while rmask:
            lsb = rmask & (-rmask)
            p = lsb.bit_length() - 1
            deg = len(point_triples[p])
            if deg > best_deg:
                best_deg = deg
                best_point = p
            rmask ^= lsb
        
        if best_point == -1:
            return
        
        p = best_point
        bit = 1 << p
        
        # Option 1: include p
        new_mask = current_mask | bit
        # Check if including p creates a triple
        valid = True
        for tm in triple_masks:
            if (tm & new_mask) == tm:
                valid = False
                break
        if valid:
            # Remove points that conflict with p (points that form a triple with p and another included point)
            # Actually, we just need to ensure no triple is fully in new_mask
            # For remaining, we need to exclude points that would complete a triple
            new_remaining = remaining_mask & ~bit
            # For each triple containing p, the other two points cannot both be in the set
            # This is complex; simpler: just recurse and let is_independent check handle it
            # But that's slow. Let's use a simpler bound.
            branch(new_remaining, current_size + 1, new_mask)
        
        # Option 2: exclude p
        branch(remaining_mask & ~bit, current_size, current_mask)
    
    branch(full_mask, 0, 0)
    return best[0]


def _independence_number_fast(triples, n=16):
    """Faster independence number using greedy + verification."""
    if not triples:
        return n
    
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    # Greedy: try to build large independent set
    # Then verify and try to improve
    best = 0
    
    # Try multiple random orderings
    for attempt in range(50):
        order = list(range(n))
        random.shuffle(order)
        
        included = set()
        for p in order:
            # Check if adding p creates a triple
            can_add = True
            for tm in triple_masks:
                if (tm & (1 << p)) and (tm & (1 << p)) == (1 << p):
                    # p is in this triple; check if other two are in included
                    others = tm & ~(1 << p)
                    if others and all((others >> q) & 1 for q in included):
                        # Need to check properly
                        pass
                    # Check if any two points in included complete the triple with p
                    if len(included) >= 2:
                        for i, a in enumerate(included):
                            for b in included[i+1:]:
                                if (a != p and b != p):
                                    # Check if {p, a, b} is a triple
                                    m = (1 << p) | (1 << a) | (1 << b)
                                    for t2m in triple_masks:
                                        if t2m == m:
                                            can_add = False
                                            break
                            if not can_add:
                                break
                    if not can_add:
                        break
            
            if can_add:
                included.add(p)
        
        if len(included) > best:
            best = len(included)
    
    # Now try to find if we can do better
    # Check from best+1 upward if possible
    for size in range(best + 1, n + 1):
        found = False
        # Try a few random subsets of this size
        for _ in range(200):
            subset = set(random.sample(range(n), size))
            independent = True
            for tm in triple_masks:
                if all((tm >> p) & 1 for p in subset):
                    independent = False
                    break
            if independent:
                found = True
                break
        if found:
            best = size
        else:
            break
    
    return best


def _is_linear(triples):
    """Check if any two triples share at most one point."""
    pair_set = set()
    for t in triples:
        for i in range(3):
            for j in range(i+1, 3):
                pair = (min(t[i], t[j]), max(t[i], t[j]))
                if pair in pair_set:
                    return False
                pair_set.add(pair)
    return True


def _build_initial_system():
    """Build an initial linear triple system using structured construction."""
    triples = []
    pair_used = set()
    
    def try_add(a, b, c):
        t = tuple(sorted((a, b, c)))
        if len(set(t)) < 3:
            return False
        pairs = [(min(t[0],t[1]), max(t[0],t[1])),
                 (min(t[0],t[2]), max(t[0],t[2])),
                 (min(t[1],t[2]), max(t[1],t[2]))]
        for p in pairs:
            if p in pair_used:
                return False
        for p in pairs:
            pair_used.add(p)
        triples.append(list(t))
        return True
    
    # Use structured construction based on 4x4 grid
    # Points: (r, c) -> r*4 + c
    # Type 1: Triples within rows using modular patterns
    # Type 2: Triples within columns
    # Type 3: Diagonal triples
    
    # Let's use a known good construction
    # Start with a set of triples that covers many pairs
    
    # Row-based triples (each row has 4 points, pick 3)
    for r in range(4):
        base = r * 4
        # Use different patterns for different rows
        if r % 2 == 0:
            try_add(base, base+1, base+2)
            try_add(base, base+2, base+3)
        else:
            try_add(base, base+1, base+3)
            try_add(base+1, base+2, base+3)
    
    # Column-based triples
    for c in range(4):
        pts = [c, 4+c, 8+c, 12+c]
        if c % 2 == 0:
            try_add(pts[0], pts[1], pts[2])
            try_add(pts[0], pts[2], pts[3])
        else:
            try_add(pts[0], pts[1], pts[3])
            try_add(pts[1], pts[2], pts[3])
    
    # Diagonal triples
    for d in range(4):
        pts = [(d + i) % 4 + i*4 for i in range(4)]
        try_add(pts[0], pts[1], pts[2])
        try_add(pts[0], pts[2], pts[3])
    
    # Anti-diagonal triples
    for d in range(4):
        pts = [(d + i) % 4 + (3-i)*4 for i in range(4)]
        try_add(pts[0], pts[1], pts[2])
        try_add(pts[0], pts[2], pts[3])
    
    # Add more triples to increase density
    # Cross triples
    for i in range(4):
        for j in range(i+1, 4):
            a = i*4
            b = j*4
            c = (i+j) % 4
            try_add(a, b, c*4)
    
    return triples


def _local_search(triples, n=16, max_iter=50000, time_limit=100):
    """Local search to minimize independence number."""
    start = time.time()
    best_triples = [list(t) for t in triples]
    best_indep = _independence_number_fast(best_triples, n)
    
    if not _is_linear(best_triples):
        return best_triples, best_indep
    
    current_triples = [list(t) for t in triples]
    current_indep = best_indep
    
    # Build pair tracking
    def get_pair_set(triples_list):
        ps = set()
        for t in triples_list:
            for i in range(3):
                for j in range(i+1, 3):
                    ps.add((min(t[i],t[j]), max(t[i],t[j])))
        return ps
    
    def try_swap(triples_list, idx1, new_triple):
        """Replace triple at idx1 with new_triple, maintaining linearity."""
        old = triples_list[idx1]
        # Remove old triple's pairs
        old_pairs = set()
        for i in range(3):
            for j in range(i+1, 3):
                old_pairs.add((min(old[i],old[j]), max(old[i],old[j])))
        
        # Check new triple doesn't conflict
        new_pairs = set()
        for i in range(3):
            for j in range(i+1, 3):
                new_pairs.add((min(new_triple[i],new_triple[j]), max(new_triple[i],new_triple[j])))
        
        # All new pairs must not be in old_pairs of other triples
        # Build set of all pairs except old triple's
        all_pairs = get_pair_set(triples_list) - old_pairs
        for p in new_pairs:
            if p in all_pairs:
                return False
        
        # Do the swap
        triples_list[idx1] = list(new_triple)
        return True
    
    iteration = 0
    no_improve = 0
    temp = 0.5
    
    while iteration < max_iter:
        if time.time() - start > time_limit:
            break
        
        iteration += 1
        no_improve += 1
        
        if no_improve > 5000:
            # Reset temperature and restart from best
            current_triples = [list(t) for t in best_triples]
            current_indep = best_indep
            temp = max(0.1, temp * 0.5)
            no_improve = 0
        
        # Generate a random move: replace a random triple with a random valid triple
        idx = random.randint(0, len(current_triples) - 1)
        old_triple = current_triples[idx]
        
        # Generate candidate triple
        candidates = []
        for _ in range(20):
            a, b, c = random.sample(range(n), 3)
            candidates.append([a, b, c])
        
        improved = False
        for cand in candidates:
            if try_swap(current_triples, idx, cand):
                # Evaluate
                new_indep = _independence_number_fast(current_triples, n)
                
                if new_indep < current_indep:
                    current_indep = new_indep
                    if new_indep < best_indep:
                        best_indep = new_indep
                        best_triples = [list(t) for t in current_triples]
                    improved = True
                    no_improve = 0
                    break
                elif new_indep == current_indep:
                    # Accept with some probability (for diversity)
                    if random.random() < 0.3:
                        no_improve = 0
                        break
                else:
                    # Worse, reject (or accept with small prob)
                    if random.random() < temp * 0.01:
                        break
                    else:
                        # Undo
                        current_triples[idx] = list(old_triple)
        
        if improved:
            continue
        
        # Also try adding a new triple
        if len(current_triples) < 40:
            for _ in range(10):
                a, b, c = random.sample(range(n), 3)
                new_t = [a, b, c]
                # Check if can add
                all_pairs = get_pair_set(current_triples)
                new_pairs = set()
                for i in range(3):
                    for j in range(i+1, 3):
                        new_pairs.add((min(new_t[i],new_t[j]), max(new_t[i],new_t[j])))
                if not (new_pairs & all_pairs):
                    current_triples.append(new_t)
                    new_indep = _independence_number_fast(current_triples, n)
                    if new_indep <= current_indep:
                        if new_indep < best_indep:
                            best_indep = new_indep
                            best_triples = [list(t) for t in current_triples]
                        current_indep = new_indep
                        no_improve = 0
                        break
                    else:
                        current_triples.pop()
    
    return best_triples, best_indep


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    best_triples = None
    best_indep = 16
    
    # Try multiple initial constructions
    initial_systems = []
    
    # Structured construction
    sys1 = _build_initial_system()
    if _is_linear(sys1):
        initial_systems.append(sys1)
    
    # Random constructions with different seeds
    for seed in range(10):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        triples = []
        pair_used = set()
        
        def try_add(a, b, c, pair_used=pair_used, triples=triples):
            t = tuple(sorted((a, b, c)))
            if len(set(t)) < 3:
                return False
            pairs = [(min(t[0],t[1]), max(t[0],t[1])),
                     (min(t[0],t[2]), max(t[0],t[2])),
                     (min(t[1],t[2]), max(t[1],t[2]))]
            for p in pairs:
                if p in pair_used:
                    return False
            for p in pairs:
                pair_used.add(p)
            triples.append(list(t))
            return True
        
        attempts = 0
        while len(triples) < 35 and attempts < 3000:
            attempts += 1
            a, b, c = rng.sample(range(16), 3)
            try_add(a, b, c)
        
        if triples and _is_linear(triples):
            initial_systems.append(triples)
    
    # Also try the algebraic construction
    # STS(15) on points 1-15, then modify to include point 0
    sts_triples = []
    seen = set()
    for a in range(1, 16):
        for b in range(a+1, 16):
            c = a ^ b
            if c == 0 or c == a or c == b:
                continue
            t = tuple(sorted((a, b, c)))
            if t not in seen:
                seen.add(t)
                sts_triples.append(list(t))
    
    if _is_linear(sts_triples):
        initial_systems.append(sts_triples)
    
    # Evaluate all initial systems and run local search on best ones
    evaluated = []
    for sys in initial_systems:
        if time.time() > deadline:
            break
        indep = _independence_number_fast(sys, 16)
        evaluated.append((indep, sys))
    
    evaluated.sort(key=lambda x: x[0])
    
    # Run local search on top candidates
    for indep, sys in evaluated[:5]:
        if time.time() > deadline:
            break
        
        if indep < best_indep:
            best_indep = indep
            best_triples = [list(t) for t in sys]
        
        # Run local search
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            result_triples, result_indep = _local_search(sys, 16, max_iter=30000, time_limit=min(remaining_time - 5, 60))
            if result_indep < best_indep:
                best_indep = result_indep
                best_triples = result_triples
    
    if best_triples is None:
        # Fallback
        best_triples = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
        best_indep = 16
    
    return best_triples
# EVOLVE-BLOCK-END
