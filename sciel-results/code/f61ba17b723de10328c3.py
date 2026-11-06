import random
import time
import itertools
from functools import lru_cache

N = 16
ALL_POINTS = frozenset(range(N))
ALL_PAIRS = frozenset(itertools.combinations(range(N), 2))

def _triple_to_mask(triple):
    """Convert a triple (a,b,c) to a bitmask."""
    mask = 0
    for x in triple:
        mask |= (1 << x)
    return mask

def _compute_alpha_bitmask(triples):
    """
    Compute independence number using bitmask superset marking.
    For each triple, mark all supersets as 'not independent'.
    Then find the largest k with at least one unmarked subset of size k.
    """
    n = N
    # A subset S is independent if no triple is a subset of S.
    # Equivalently, S is NOT independent if there exists a triple T with T ⊆ S.
    # We'll use a different approach: enumerate all subsets from largest to smallest.
    # But that's 2^16 = 65536 subsets, which is fine.
    
    # For each triple, create its bitmask.
    triple_masks = [_triple_to_mask(t) for t in triples]
    
    # A subset mask S is independent iff for all triple_masks T, (S & T) != T
    # i.e., no triple is fully contained in S.
    
    # We want the max popcount(S) such that S is independent.
    # Iterate from k=16 down.
    for k in range(n, 0, -1):
        for subset in itertools.combinations(range(n), k):
            s_mask = 0
            for x in subset:
                s_mask |= (1 << x)
            is_ind = True
            for tm in triple_masks:
                if (s_mask & tm) == tm:
                    is_ind = False
                    break
            if is_ind:
                return k
    return 1

def _check_linearity_mask(triple_mask, used_pairs_mask):
    """
    Check if a triple (given as bitmask) is linear with respect to used pairs.
    used_pairs_mask: a set of frozenset pairs, or we can use a bitmask over pairs.
    Actually, let's use a set of pairs for clarity and speed with small n.
    """
    # Extract points from mask
    points = []
    for i in range(N):
        if triple_mask & (1 << i):
            points.append(i)
    if len(points) != 3:
        return False
    a, b, c = points
    pairs = {frozenset((a,b)), frozenset((a,c)), frozenset((b,c))}
    for p in pairs:
        if p in used_pairs_mask:
            return False
    return True

def _get_pairs_from_mask(mask):
    """Get the three pairs from a triple bitmask."""
    points = []
    for i in range(N):
        if mask & (1 << i):
            points.append(i)
    a, b, c = points
    return {frozenset((a,b)), frozenset((a,c)), frozenset((b,c))}

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_triples = None
    best_alpha = N
    
    def update_best(triples):
        nonlocal best_triples, best_alpha
        alpha = _compute_alpha_bitmask(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [list(t) for t in triples]
            if best_alpha <= 5:
                return True
        return False
    
    # Precompute all possible triples as bitmasks
    all_triple_masks = []
    all_triple_points = []
    for combo in itertools.combinations(range(N), 3):
        mask = _triple_to_mask(combo)
        all_triple_masks.append(mask)
        all_triple_points.append(combo)
    
    # Number of all triples
    num_triples = len(all_triple_masks)
    
    # Strategy: Backtracking with forward checking + heuristic ordering
    # We build a list of triples. At each step, we choose the next triple.
    # Forward checking: maintain used_pairs set.
    
    # Heuristic: prioritize triples that share points with existing triples
    # (to maximize coverage and minimize independence number)
    
    def backtrack(triples, used_pairs, depth, max_depth):
        nonlocal best_alpha, best_triples
        if time.time() > deadline:
            return False
        
        # Evaluate current state
        if triples:
            if update_best(triples):
                return True
        
        if depth >= max_depth:
            return False
        
        # Generate candidate triples: those that are linear with current used_pairs
        candidates = []
        for i, tm in enumerate(all_triple_masks):
            if _check_linearity_mask(tm, used_pairs):
                candidates.append(i)
        
        if not candidates:
            return False
        
        # Heuristic: score each candidate by how many of its points are already
        # "covered" by existing triples (i.e., appear in at least one existing triple)
        if triples:
            covered_points = set()
            for t in triples:
                covered_points.update(t)
            
            def score_candidate(idx):
                tm = all_triple_masks[idx]
                pts = all_triple_points[idx]
                overlap = sum(1 for p in pts if p in covered_points)
                return -overlap  # Prefer more overlap (lower score)
            
            candidates.sort(key=score_candidate)
        
        # Limit branching factor for speed
        max_branch = 50
        if len(candidates) > max_branch:
            # Keep the best ones
            candidates = candidates[:max_branch]
        
        # Also add some random ones for diversity
        if len(candidates) > 10:
            random.shuffle(candidates[10:])
        
        for idx in candidates:
            if time.time() > deadline:
                return False
            
            tm = all_triple_masks[idx]
            new_used_pairs = used_pairs | _get_pairs_from_mask(tm)
            new_triples = triples + [all_triple_points[idx]]
            
            if backtrack(new_triples, new_used_pairs, depth + 1, max_depth):
                return True
        
        return False
    
    # Run multiple backtracking searches with different max_depth and randomization
    random.seed(42)
    
    # First, try a deep search
    initial_triples = []
    initial_used_pairs = set()
    
    # Start with a few random triples to break symmetry
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Random initial seed
        seed_triples = []
        seed_used_pairs = set()
        
        # Add 3-5 random triples as seed
        num_seed = random.randint(3, 5)
        for _ in range(num_seed):
            idx = random.randint(0, num_triples - 1)
            tm = all_triple_masks[idx]
            if _check_linearity_mask(tm, seed_used_pairs):
                seed_triples.append(all_triple_points[idx])
                seed_used_pairs |= _get_pairs_from_mask(tm)
        
        if not seed_triples:
            continue
        
        # Try backtracking from this seed
        result = backtrack(seed_triples, seed_used_pairs, 0, 40)
        if result:
            break
    
    if best_triples is None:
        # Fallback: greedy random construction
        triples = []
        used_pairs = set()
        for _ in range(200):
            idx = random.randint(0, num_triples - 1)
            tm = all_triple_masks[idx]
            if _check_linearity_mask(tm, used_pairs):
                triples.append(list(all_triple_points[idx]))
                used_pairs |= _get_pairs_from_mask(tm)
        best_triples = triples
        best_alpha = _compute_alpha_bitmask(triples)
    
    return best_triples
