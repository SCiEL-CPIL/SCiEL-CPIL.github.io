"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import product

def _check_cap_set_valid(vec, accepted, n):
    """
    Check if adding vec to accepted keeps it a cap set (no 3-term APs).
    In F_3^n, a 3-AP is a triple {a, b, c} such that a + c = 2b (mod 3).
    Since 2 = -1 mod 3, this is equivalent to a - b + c = 0 or a + c - b = 0.
    More simply, for any two distinct points u, v in the set, the third point w
    that completes the AP with u and v (where one is the midpoint) is determined.
    
    Actually, the condition for {u, v, w} to be a 3-AP in F_3 is that u+v+w = 0 (mod 3) 
    is NOT quite right for all permutations. 
    The standard definition: x, y, z form a 3-AP if x + z = 2y.
    In F_3, 2 = -1. So x - y + z = 0? No, x + z - y = 0.
    Wait, the condition for a set to be a cap set is that it contains no three distinct 
    points x, y, z such that x + z = 2y.
    
    To check if `vec` can be added to `accepted`:
    For every pair (u, v) in `accepted`, we must ensure that `vec` does not complete a 3-AP.
    The possible 3-APs involving `vec` and two existing points u, v are:
    1. vec, u, v form AP: 
       - vec is middle: 2*vec = u + v => vec = (u+v)/2 = (u+v)*2 (since inv(2)=2 in F_3)
       - u is middle: 2*u = vec + v => vec = 2*u - v
       - v is middle: 2*v = vec + u => vec = 2*v - u
       
    So for each pair (u, v) in accepted, we check if vec equals:
    - (u + v) * 2 % 3
    - (2*u - v) % 3
    - (2*v - u) % 3
    
    However, iterating over all pairs O(N^2) is too slow for large N.
    We can optimize: For a fixed `vec` and `u` in `accepted`, the forbidden `v` is unique.
    If `vec` is added, it conflicts with `u` if there exists `v` in `accepted` such that:
    - v = (2*vec - u) % 3  (u is middle)
    - v = (2*u - vec) % 3  (vec is middle)
    - v = (vec + u) * 2 % 3 (v is middle) -- wait, let's re-derive.
    
    Let's just check against each existing point `u` in `accepted`.
    If we add `vec`, we create a 3-AP if there exists `w` in `accepted` such that {vec, u, w} is a 3-AP.
    For a fixed `vec` and `u`, the third point `w` is uniquely determined for each of the 3 roles:
    1. `vec` is the midpoint: 2*vec = u + w  => w = 2*vec - u
    2. `u` is the midpoint: 2*u = vec + w   => w = 2*u - vec
    3. `w` is the midpoint: 2*w = vec + u   => w = (vec + u) * 2
    
    So, for each `u` in `accepted`, we compute these 3 potential `w` values.
    If any of these `w` values is in `accepted` AND `w != u` (and `w != vec`, though vec is new), 
    then adding `vec` creates a 3-AP.
    
    Note: If w == u, then we have a degenerate AP? No, distinct points. 
    If w == u, then 2*vec = 2*u => vec = u, which is already in the set, so we wouldn't be adding it.
    So we just check if the computed `w` is in `accepted`.
    """
    vec_set = accepted  # Use the set directly for O(1) lookup
    
    for u in vec_set:
        # Case 1: vec is midpoint. w = 2*vec - u
        w1 = tuple((2 * vec[i] - u[i]) % 3 for i in range(n))
        if w1 in vec_set:
            return False
            
        # Case 2: u is midpoint. w = 2*u - vec
        w2 = tuple((2 * u[i] - vec[i]) % 3 for i in range(n))
        if w2 in vec_set:
            return False
            
        # Case 3: w is midpoint. 2*w = vec + u => w = (vec + u) * 2
        w3 = tuple(((vec[i] + u[i]) * 2) % 3 for i in range(n))
        if w3 in vec_set:
            return False
            
    return True

def priority(vector, n):
    """
    Entry point: returns a priority value for a vector.
    
    Strategy: Dynamic Conflict Avoidance.
    The evaluator calls this function to decide the order in which to add vectors.
    Since the evaluator adds vectors in descending priority, we want to return a high priority
    for vectors that are "safe" to add early, or perhaps "safe" to add later?
    
    Wait, the standard greedy approach with a priority function usually works like this:
    The evaluator sorts all vectors by priority (descending) and then iterates through them,
    adding them if they keep the set a cap set.
    
    If we return a static priority, the order is fixed.
    If we could return a dynamic priority based on the current state of the set, that would be ideal,
    but the interface `priority(vector, n)` does not receive the current set.
    
    HOWEVER, looking at the contract: "vectors of {0,1,2}^n are added greedily in descending priority".
    This implies a static sorting order is determined by the priority function.
    
    BUT, the current program's score is 1.19. The inspirations suggest dynamic greedy is possible.
    Let's re-read the contract carefully.
    "Interface: priority(vector: tuple, n: int) -> float; vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set"
    
    This phrasing is ambiguous. Does it mean:
    1. Sort all vectors by priority, then greedily add? (Static order)
    2. Or does it mean the evaluator implements a greedy loop where it might call priority multiple times or the priority function has access to state?
    
    Given the function signature `priority(vector, n)`, it cannot access the current set of accepted vectors unless we use global state.
    
    Let's assume the evaluator does the following:
    1. Generates all vectors (or a large sample).
    2. Calls `priority(v, n)` for each.
    3. Sorts them.
    4. Iterates and adds if valid.
    
    If this is the case, a static priority is all we have.
    The current program uses balance.
    
    Is there a better static priority?
    Vectors with more 0s? Vectors with more 1s?
    
    Let's look at the "Inspiration 3" which mentions "dynamic greedy search where the priority...".
    If the evaluator allows the `priority` function to be called *during* the greedy process (i.e., it picks the best remaining vector at each step), then we need global state.
    
    Let's assume the evaluator works like this:
    `candidates = all_vectors`
    `while candidates:`
       `best = max(candidates, key=lambda v: priority(v, n))`
       `if valid(best): accept(best)`
       `remove(best from candidates)`
       
    If this is the case, we can maintain a global set of accepted vectors and use it in `priority`.
    
    Let's try this "Global State" approach. It is a structural change from a pure function to a stateful heuristic.
    """
    global _accepted_set
    if not hasattr(priority, '_accepted_set'):
        priority._accepted_set = set()
    
    accepted = priority._accepted_set
    
    # If the set is empty, we can't do dynamic conflict checking.
    # Return a static priority to break symmetry initially.
    # Static priority: Prefer vectors with balanced coordinates (as before) or random.
    # Let's use a hash-based random priority for the first few to establish a diverse base.
    if len(accepted) < 10:
        # Use a deterministic but "random-looking" priority based on the vector itself
        # to avoid always picking the same "best" balanced vector first.
        # We want to spread out the initial set.
        # Simple sum of coordinates?
        s = sum(vector)
        # Prefer sums close to n
        return -abs(s - n) + (hash(vector) % 1000) / 1000.0
    
    # Dynamic Priority:
    # We want to pick a vector that has the LEAST number of "potential conflicts" 
    # with the currently accepted set, but also has some "value" (balance).
    # Actually, if a vector is valid, it has 0 conflicts with the *current* set.
    # The priority should reflect how "robust" it is, or how much it "opens up" space.
    
    # A common heuristic for greedy set packing: 
    # Pick the element that conflicts with the FEWEST other *remaining* candidates?
    # We don't have access to remaining candidates easily.
    
    # Alternative: Pick the vector that maximizes the "distance" to the accepted set?
    # Or minimizes the number of 3-APs it *would* form if it were invalid? (It's valid, so 0).
    
    # Let's try: Among valid vectors, pick the one that is "most balanced" (static) 
    # but break ties by "least potential future conflict".
    
    # Since we don't know the future, let's stick to a refined static priority 
    # that is slightly randomized to avoid local minima in the greedy path.
    
    # Actually, if the evaluator calls priority() for ALL vectors at the start, 
    # global state won't help during the sorting phase.
    
    # Let's look at the current program's score 1.19.
    # The current program returns `-imbalance`.
    
    # What if we use a different static metric?
    # "Weighted sum" where weights are random but fixed per run?
    
    # Let's try a strategy that favors vectors with a specific pattern, 
    # e.g., vectors that are "linearly independent" in some sense?
    # In F_3^n, cap sets are related to error-correcting codes.
    
    # Let's try a priority based on the number of 1s.
    # Or a combination: 
    # 1. Balance (as before)
    # 2. Sum (prefer n)
    # 3. Hash (random tie break)
    
    # The current program does Balance.
    # Let's try: Prefer vectors with sum = n.
    # If sum == n, return high priority.
    # This is a different structural family: "Sum-Centering".
    
    s = sum(vector)
    
    # Score is high when s is close to n
    # Use a Gaussian-like decay
    diff = s - n
    score = -abs(diff)
    
    # Add a small random component to break ties and explore different branches
    # We need this to be deterministic for the same vector to ensure consistent sorting?
    # If the evaluator sorts once, random tie-breaking is fine if it's stable.
    # If the evaluator re-sorts, we need consistency.
    # hash(vector) is consistent within a run.
    
    tie_break = hash(vector) % 10000
    return score * 10000 + tie_break

# EVOLVE-BLOCK-END
