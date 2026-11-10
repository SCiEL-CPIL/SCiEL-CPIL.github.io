"""Construct a linear triple system on 16 points minimizing independence number."""
import random
import time
from itertools import combinations
import sys

# EVOLVE-BLOCK-START

def _all_triples():
    """Generate all C(16,3) triples as frozensets."""
    return [frozenset((a, b, c)) for a in range(16) for b in range(a + 1, 16) for c in range(b + 1, 16)]

def _compatibility_mask(triples):
    """
    For each triple i, compute a bitmask of triples j that are compatible with i
    (i.e., |i ∩ j| <= 1).
    """
    n = len(triples)
    masks = [0] * n
    for i in range(n):
        ti = triples[i]
        for j in range(i + 1, n):
            tj = triples[j]
            if len(ti & tj) <= 1:
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)
    return masks

def _independence_number(triples):
    """
    Compute the independence number: largest subset of points containing no triple.
    A set S is independent if no triple is fully contained in S.
    Equivalent: find max |S| such that for all triples t, t ⊄ S.
    
    We use a branch-and-bound / DP approach over subsets.
    Since 2^16 = 65536, we can precompute which subsets are "triple-free".
    """
    n_points = 16
    # A subset mask is "bad" if it contains any triple.
    # Precompute: for each triple, mark all supersets as bad.
    bad = [False] * (1 << n_points)
    for t in triples:
        t_mask = 0
        for p in t:
            t_mask |= (1 << p)
        # Mark all supersets of t_mask as bad
        # Iterate over all supersets: for each bit not in t_mask, it can be 0 or 1
        remaining = ((1 << n_points) - 1) ^ t_mask
        sub = remaining
        while True:
            bad[t_mask | sub] = True
            if sub == 0:
                break
            sub = (sub - 1) & remaining
    
    # Find max popcount of a non-bad subset
    best = 0
    for mask in range(1 << n_points):
        if not bad[mask]:
            cnt = bin(mask).count('1')
            if cnt > best:
                best = cnt
    return best

def _independence_number_fast(triples):
    """
    Faster independence number computation using a different approach.
    We want the largest set S with no triple fully inside S.
    Equivalently, the minimum number of points to remove so that no triple remains.
    This is a hitting set / vertex cover problem on the hypergraph.
    
    For 16 points, we can do DP or use the fact that we need to check subsets.
    Actually the previous method is O(3^16) which is too slow. Let me think again.
    
    Alternative: The independence number is the maximum k such that there exists a k-subset 
    with no triple contained in it.
    
    We can iterate k from 16 down to 1, and check if any k-subset is triple-free.
    For a given k-subset S, check if any triple is contained in S.
    
    But checking all C(16,k) subsets for each k is expensive.
    
    Better: precompute for each triple its mask. A subset S is triple-free iff 
    for every triple t, t_mask & S != t_mask (i.e., t is not fully contained in S).
    
    We can do: for each subset S (2^16 of them), check if it's triple-free.
    To speed up: precompute for each triple t, the set of subsets that contain t.
    But that's what we did before.
    
    Actually, let's use a different approach: 
    The complement of the independence number is the minimum vertex cover.
    For small hypergraphs, we can use inclusion-exclusion or just iterate.
    
    Given 16 points, 2^16 = 65536 subsets. For each subset, checking if it's 
    triple-free requires checking all triples. If we have ~560 triples, that's 
    560 * 65536 ≈ 36M operations, which is fine in Python if optimized.
    
    Let me use a bitset approach: for each point p, precompute the set of triples 
    containing p. Then for a subset S, a triple is contained in S iff all three 
    points are in S.
    """
    n_points = 16
    n_triples = len(triples)
    
    # Precompute triple masks
    triple_masks = []
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks.append(m)
    
    # For each subset S, check if any triple mask is a subset of S
    # i.e., triple_mask & S == triple_mask
    best = 0
    for s in range(1 << n_points):
        is_independent = True
        for tm in triple_masks:
            if (tm & s) == tm:
                is_independent = False
                break
        if is_independent:
            cnt = bin(s).count('1')
            if cnt > best:
                best = cnt
    return best

def _greedy_construct(triples, masks, rng):
    """
    Greedy construction: repeatedly pick a compatible triple.
    Returns a list of triples forming a linear system.
    """
    n = len(triples)
    available = (1 << n) - 1  # all triples available initially
    selected = []
    
    while available:
        # Pick a random triple from available
        # Count bits to get index
        bits = bin(available).count('1')
        if bits == 0:
            break
        idx = rng.randrange(bits)
        # Find the idx-th set bit
        tmp = available
        for _ in range(idx):
            tmp &= tmp - 1
        bit = tmp & (-tmp)
        i = bit.bit_length() - 1
        
        selected.append(triples[i])
        available &= masks[i]  # keep only compatible triples
        available &= ~(1 << i)  # remove i itself (should already be removed by mask if i not in mask[i])
    
    return selected

def _local_search(system, triples, masks, rng, max_iters=1000):
    """
    Try to improve the system by removing and re-adding triples.
    Returns improved system or the same system.
    """
    current = list(system)
    current_set = set(current)
    
    # Build available mask for current system
    n = len(triples)
    # For each triple in current, the available ones are those compatible with all selected
    # This is expensive to recompute, so let's use a simpler approach:
    # Try removing each triple and see if we can add a better one.
    
    for _ in range(max_iters):
        improved = False
        for i in range(len(current)):
            # Remove current[i]
            saved = current[i]
            remaining = current[:i] + current[i+1:]
            remaining_set = set(remaining)
            
            # Find all triples compatible with all in remaining
            # A triple t is compatible if for all r in remaining, |t ∩ r| <= 1
            candidates = []
            for j, t in enumerate(triples):
                if t in remaining_set:
                    continue
                compatible = True
                for r in remaining:
                    if len(t & r) > 1:
                        compatible = False
                        break
                if compatible:
                    candidates.append(t)
            
            if not candidates:
                current[i] = saved
                continue
            
            # Pick a random candidate
            new_t = rng.choice(candidates)
            new_system = remaining + [new_t]
            
            # Check if this improves independence number
            # For efficiency, only compute if system size changed or we want to try
            # Actually, let's just accept it and compute at the end
            current[i] = new_t
            improved = True
            break
        
        if not improved:
            break
    
    return current

def construct():
    """
    Construct a linear triple system on 16 points minimizing independence number.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    rng = random.Random(42)
    
    # Precompute all triples and compatibility
    all_triples = _all_triples()
    n = len(all_triples)
    
    # Build compatibility masks
    masks = _compatibility_mask(all_triples)
    
    best_system = None
    best_score = 16  # worst case: no triples, independence number = 16
    
    # First, try a few greedy constructions
    for restart in range(50):
        if time.time() > deadline:
            break
        
        system = _greedy_construct(all_triples, masks, rng)
        
        if not system:
            continue
        
        score = _independence_number_fast(system)
        
        if score < best_score:
            best_score = score
            best_system = system
        
        # Try local search
        if time.time() < deadline - 10:
            improved = _local_search(system, all_triples, masks, rng, max_iters=200)
            if improved:
                imp_score = _independence_number_fast(improved)
                if imp_score < best_score:
                    best_score = imp_score
                    best_system = improved
    
    if best_system is None:
        # Fallback: return empty or minimal
        return []
    
    return [list(t) for t in best_system]

# EVOLVE-BLOCK-END
