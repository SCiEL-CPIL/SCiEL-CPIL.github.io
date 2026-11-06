import random
import time
import itertools
from collections import defaultdict

def _check_validity(triples):
    """Check that any two triples share at most one point."""
    n = len(triples)
    for i in range(n):
        for j in range(i + 1, n):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True

def _independence_number(triples):
    """Compute the independence number (largest set with no triple contained)."""
    n = 16
    # For each subset, check if it contains any triple
    # We want the largest subset that contains NO triple
    # A subset S is independent if no triple is a subset of S
    
    # Use bitmasks: each triple is a 16-bit mask
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    # Find largest independent set
    # Brute force over all 2^16 subsets is 65536, which is fine
    # But we can be smarter: iterate subsets by size descending
    best = 0
    # For each subset, check if any triple is contained
    # Precompute: for each subset mask, does it contain any triple?
    # A subset S contains triple T iff (S & T_mask) == T_mask
    
    # Iterate all subsets, find max size with no triple contained
    for size in range(16, 0, -1):
        found = False
        for subset in itertools.combinations(range(16), size):
            smask = 0
            for p in subset:
                smask |= (1 << p)
            # Check if any triple is contained in this subset
            has_triple = False
            for tm in triple_masks:
                if (smask & tm) == tm:
                    has_triple = True
                    break
            if not has_triple:
                found = True
                break
        if found:
            best = size
            break
    return best

def _greedy_build(rnd):
    """Build a linear triple system greedily with randomization."""
    points = list(range(16))
    triples = []
    
    # Try to build as many triples as possible
    # Each new triple must share at most 1 point with every existing triple
    # We want to cover as many points as possible and create dense interactions
    
    # Strategy: repeatedly pick 3 points that form a valid triple
    # and that "connect" well to existing structure
    remaining_attempts = 500
    while remaining_attempts > 0:
        remaining_attempts -= 1
        # Pick 3 random distinct points
        a, b, c = rnd.sample(points, 3)
        candidate = [a, b, c]
        
        # Check validity against all existing triples
        valid = True
        for t in triples:
            common = set(candidate) & set(t)
            if len(common) > 1:
                valid = False
                break
        
        if valid:
            triples.append(candidate)
    
    return triples

def _local_search(triples, rnd, time_limit=10.0):
    """Improve independence number via local search (add/swap triples)."""
    start_time = time.time()
    triples = [list(t) for t in triples]
    
    def score(t):
        return _independence_number(t)
    
    best_score = score(triples)
    best_triples = [list(t) for t in triples]
    
    while time.time() - start_time < time_limit:
        # Try adding a random triple
        if rnd.random() < 0.7:
            # Add a triple
            a, b, c = rnd.sample(range(16), 3)
            candidate = triples + [[a, b, c]]
            if _check_validity(candidate):
                s = score(candidate)
                if s < best_score:
                    best_score = s
                    best_triples = [list(t) for t in candidate]
                    triples = [list(t) for t in candidate]
        else:
            # Try swapping: remove one triple, add another
            if len(triples) < 1:
                continue
            idx = rnd.randrange(len(triples))
            new_triples = [list(t) for i, t in enumerate(triples) if i != idx]
            a, b, c = rnd.sample(range(16), 3)
            candidate = new_triples + [[a, b, c]]
            if _check_validity(candidate):
                s = score(candidate)
                if s < best_score:
                    best_score = s
                    best_triples = [list(t) for t in candidate]
                    triples = [list(t) for t in candidate]
    
    return best_triples, best_score

def _structured_build(rnd):
    """Build using a more structured approach based on Steiner system ideas."""
    # Start with a known good configuration and perturb
    # Use a near-Steiner triple system approach
    
    # For 16 points, a Steiner triple system doesn't exist (16 mod 6 = 4)
    # But we can get close. Let's use a greedy approach with bias toward
    # covering all points multiple times
    
    triples = []
    point_counts = [0] * 16
    
    # First, create a base structure
    # Use pairs that appear in exactly one triple (Steiner-like)
    used_pairs = set()
    
    for _ in range(200):
        # Pick 3 points, preferring those with low count
        candidates = []
        for _ in range(50):
            a, b, c = rnd.sample(range(16), 3)
            # Check pair constraints
            pairs = frozenset((a, b)), frozenset((a, c)), frozenset((b, c))
            if any(p in used_pairs for p in pairs):
                continue
            # Check against existing triples
            valid = True
            for t in triples:
                common = len(set([a, b, c]) & set(t))
                if common > 1:
                    valid = False
                    break
            if valid:
                # Score by how balanced it is
                s = -(point_counts[a] + point_counts[b] + point_counts[c])
                candidates.append((s, [a, b, c]))
        
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            # Pick from top candidates with some randomness
            top_k = min(5, len(candidates))
            _, chosen = rnd.choice(candidates[:top_k])
            triples.append(chosen)
            for p in chosen:
                point_counts[p] += 1
            for i in range(3):
                for j in range(i+1, 3):
                    used_pairs.add(frozenset((chosen[i], chosen[j])))
    
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    rnd = random.Random(42)
    
    best_triples = None
    best_score = 16  # Worst case: no triples, independence number = 16
    
    # Phase 1: Try structured builds with different seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        r = random.Random(seed)
        triples = _structured_build(r)
        if triples:
            s = _independence_number(triples)
            if s < best_score:
                best_score = s
                best_triples = [list(t) for t in triples]
    
    # Phase 2: Local search from best solution
    if best_triples is not None:
        if time.time() < deadline - 5:
            improved, s = _local_search(best_triples, random.Random(123), 
                                         time_limit=min(30, deadline - time.time() - 5))
            if s < best_score:
                best_score = s
                best_triples = improved
    
    # Phase 3: More random restarts with greedy
    while time.time() < deadline - 10:
        triples = _greedy_build(rnd)
        if triples:
            s = _independence_number(triples)
            if s < best_score:
                best_score = s
                best_triples = [list(t) for t in triples]
                # Quick local search
                if time.time() < deadline - 15:
                    improved, s2 = _local_search(best_triples, rnd, 
                                                   time_limit=min(10, deadline - time.time() - 15))
                    if s2 < best_score:
                        best_score = s2
                        best_triples = improved
    
    # Phase 4: Final polish - try to add more triples to best solution
    if best_triples is not None and time.time() < deadline - 5:
        final = [list(t) for t in best_triples]
        # Try adding triples
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            a, b, c = rnd.sample(range(16), 3)
            candidate = final + [[a, b, c]]
            if _check_validity(candidate):
                s = _independence_number(candidate)
                if s < best_score:
                    best_score = s
                    best_triples = candidate
                    final = [list(t) for t in candidate]
    
    if best_triples is None:
        # Fallback
        best_triples = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    
    return best_triples
