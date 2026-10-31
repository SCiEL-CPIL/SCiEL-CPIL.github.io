import math
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 2.0

# State for adaptive heuristic
_set_members = []
_priority_cache = {}
_n_current = None
_time_exceeded = False

def _time_check():
    global _time_exceeded
    if _time_exceeded:
        return True
    if (time.time() - _START_TIME) > (_TIME_LIMIT - _SAFETY_MARGIN):
        _time_exceeded = True
        return True
    return False

def priority(vector, n):
    global _set_members, _priority_cache, _n_current
    
    key = (vector, n)
    if key in _priority_cache:
        return _priority_cache[key]
    
    if _time_check():
        # Fallback to a simple deterministic score if we're running out of time
        result = 0.0
        _priority_cache[key] = result
        return result
    
    # Update state if n changes
    if _n_current != n:
        _n_current = n
        _set_members = []
        _priority_cache.clear()
    
    # Compute adaptive priority:
    # 1. Count how many existing members would form an affine line with this vector.
    #    An affine line in {0,1,2}^n is a set {a, b, c} where a + c = 2b (mod 3)
    #    or equivalently, for any two points p, q, the third point r is determined by r = 2q - p (mod 3)
    #    Actually, the condition for a cap set is that no three distinct points are collinear.
    #    In {0,1,2}^n, three points a, b, c are collinear if a + c = 2b (component-wise mod 3).
    #    Alternatively, for any pair (u, v), the third point w is (2v - u) mod 3.
    #    If we add vector v, it forms a line with u and w if w is already in the set and u is already in the set.
    #    More efficiently: for each existing member u, check if the "other" point that would complete the line with u and v is in the set.
    #    Let's iterate through existing members u.
    #    The line through u and v has a third point w. 
    #    In F_3^3 geometry, the three points on a line are {u, v, w} where w = 2v - u (mod 3) is NOT correct for all orderings.
    #    Actually, the set of 3 points on a line through u and v is {u, v, 2v-u} only if we consider the specific order.
    #    Better: A set {a,b,c} is a line if a+b+c = 0 (mod 3) is NOT the condition. 
    #    The condition is that they are distinct and one is the midpoint of the other two.
    #    In {0,1,2}^n, the sum of the three points on a line is 0 mod 3? No.
    #    Example: (0,0), (1,1), (2,2). Sum = (3,3) = (0,0) mod 3. Yes.
    #    Example: (0,0), (0,1), (0,2). Sum = (0,3) = (0,0) mod 3. Yes.
    #    So, three points a, b, c form a line iff a + b + c = (0, ..., 0) mod 3.
    
    # So, if we add v, it creates a line with u and w if u + w + v = 0 mod 3.
    # This implies w = -v - u mod 3.
    # So for each existing u, we check if w = (-v - u) mod 3 is also in the set.
    # If it is, then adding v creates a violation.
    
    # To make this fast, we store set members in a set of tuples for O(1) lookup.
    # But we can't easily use a set if we don't know the type. Vector is a tuple of ints.
    
    # Let's maintain a set of members for fast lookup.
    # Since we can't modify the global set in a thread-safe way or if the evaluator calls priority in parallel (unlikely for greedy),
    # we assume sequential calls.
    
    # Optimization: If the set is large, this O(|S|) check per vector might be slow.
    # But n is small (usually 8-10 in these tasks). |S| is at most 3^n / something.
    # For n=8, 3^8 = 6561. Max cap set size is ~ 43? No, larger.
    # Actually, for n=8, max cap set size is 25.
    # So |S| is small. O(|S|) is fine.
    
    # However, we need to be careful: the greedy algorithm adds vectors one by one.
    # The priority function is called for ALL vectors to sort them.
    # This means priority is called 3^n times.
    # If we do O(|S|) work for each call, total work is O(3^n * |S|).
    # For n=8, 3^8 * 25 ~ 160,000 operations. Very fast.
    # For n=10, 3^10 * 100 ~ 600,000. Still fast.
    
    # But wait, the set changes as vectors are added.
    # The greedy algorithm sorts vectors by priority.
    # If priority depends on the current set, the order of sorting changes as the set grows.
    # The standard greedy algorithm for cap sets usually:
    # 1. Pre-compute a static order or
    # 2. Iteratively pick the best vector from the remaining candidates.
    # The contract says: "vectors ... are added greedily in descending priority".
    # This implies a static sort order based on the initial priority values?
    # OR does it mean that at each step, we pick the highest priority vector from the remaining?
    # "added greedily in descending priority" usually means:
    #   Sort all vectors by priority.
    #   Iterate through the sorted list.
    #   If adding the vector keeps the set a cap set, add it.
    #   Else, skip it.
    # If this is the case, the priority is STATIC (computed once, or computed based on the vector alone).
    # The current program computes priority based ONLY on the vector, not the set.
    # If I make priority depend on the set, the evaluator might not update the set between calls to priority?
    # Or does the evaluator call priority for each vector, then sort, then add?
    # If the evaluator calls priority for all 3^n vectors first, the set is empty for all calls.
    # Then my adaptive heuristic would see an empty set for all vectors, resulting in identical priorities (or just the fallback).
    # This would break the strategy.
    
    # Let's re-read the contract: "vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set".
    # This phrasing is ambiguous.
    # Interpretation A: Sort all vectors by priority. Iterate. Add if valid.
    # Interpretation B: While there are candidates, pick the one with highest priority that is valid.
    
    # If Interpretation A is true, the set is empty when priority is called. My adaptive strategy fails.
    # If Interpretation B is true, the set grows, and my strategy works.
    
    # The current program (score 1.2344) uses a static heuristic. It doesn't use the set.
    # The inspirations mention "stateful, adaptive heuristic".
    # If the evaluator uses Interpretation A, stateful heuristics that depend on the set will fail because the set is empty during the sort phase.
    # However, if the evaluator uses Interpretation B, it works.
    
    # Let's look at the "seed program" or common FunSearch patterns.
    # Usually, in these greedy tasks, the priority is static.
    # BUT, the inspiration explicitly says "Replace the static... with a stateful, adaptive heuristic".
    # This suggests that the evaluator might support stateful priorities, or the "state" is something else (like a counter of calls).
    # If the evaluator calls priority for each vector in a loop, and adds it immediately if valid, then the set IS updated.
    # Let's assume Interpretation B (dynamic selection) or that the evaluator allows the priority to depend on the current set state.
    
    # To be safe, I will implement the adaptive heuristic but include a fallback to the static balance score if the set is empty,
    # or if the time is running out.
    # Actually, if the set is empty, the "danger" is 0 for all vectors. So they all tie.
    # We need a tie-breaker. The static balance score is a good tie-breaker.
    
    # So, priority = -danger_count * LARGE_WEIGHT + balance_score
    
    # Let's implement this.
    
    # Check if vector is already in set? The evaluator shouldn't ask for priority of a vector already in the set,
    # but just in case, return -inf or low score.
    if tuple(vector) in _set_members:
        _priority_cache[key] = -1e18
        return -1e18
    
    # Calculate danger
    v = tuple(vector)
    danger = 0
    
    # For each existing member u, check if w = (-v - u) mod 3 is in the set.
    # We need a set of members for O(1) lookup.
    # We maintain _set_members as a list, but also a set for lookup.
    # Let's maintain a set globally.
    
    # I need to update the global set.
    # Wait, the priority function is called by the evaluator.
    # How does the evaluator know when a vector is added?
    # The evaluator adds the vector to its own internal set.
    # My function doesn't know when the evaluator adds a vector.
    # Unless... the evaluator calls a separate function to add? No, the contract only defines `priority`.
    # This implies that the evaluator DOES NOT update my internal state.
    # My internal state `_set_members` will never be updated by the evaluator.
    # So `_set_members` will always be empty.
    # Therefore, the adaptive heuristic based on the set is IMPOSSIBLE unless I can infer the set state from the call history or something else.
    
    # Can I infer the set?
    # If the evaluator uses Interpretation A (static sort), the set is always empty during priority calls.
    # If the evaluator uses Interpretation B (dynamic pick), it might call priority multiple times for the same vector?
    # Or does it call priority for all, sort, pick best, add, then call priority for remaining?
    # If it re-sorts, it calls priority again.
    
    # Given the uncertainty, the safest "adaptive" strategy that works with a static sort is to use a heuristic that is STATIC but better than balance.
    # The instruction says: "Strategy to try...: Priority based on the sum of squares of coordinate values...".
    # This is a STATIC strategy. It doesn't use the set.
    # So I should follow the specific strategy suggested in the directive: "Priority based on the sum of squares of coordinate values".
    
    # Let's implement the Sum of Squares strategy.
    # Sum of squares: sum(v_i^2 for v_i in vector).
    # Values: 0->0, 1->1, 2->4.
    # This favors vectors with more 2s.
    # Why? "exploit the non-linear distribution of affine lines".
    
    # Let's combine Sum of Squares with the Balance score.
    # score = -deviation * W1 + sum_sq * W2 + hash * W3
    
    # Let's try to make the Sum of Squares dominant.
    
    c0 = 0
    c1 = 0
    c2 = 0
    sum_sq = 0
    for val in vector:
        if val == 0:
            c0 += 1
        elif val == 1:
            c1 += 1
        else:
            c2 += 1
            sum_sq += 4
        if val == 1:
            sum_sq += 1
        # Wait, 0^2=0, 1^2=1, 2^2=4.
        # So sum_sq = c1*1 + c2*4.
        
    # Recalculate sum_sq correctly
    sum_sq = c1 + 4*c2
    
    ideal = n / 3.0
    deviation = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # We want to maximize sum_sq? Or minimize?
    # "favoring vectors with more 1s and 2s over 0s" -> Maximize sum_sq.
    
    # Let's try: score = sum_sq * 1000 - deviation * 10 + hash
    
    h = 0
    base = 31
    for val in vector:
        h = (h * base + val) % (10**9 + 7)
    
    hash_score = h / (10**9 + 7)
    
    result = float(sum_sq * 1000.0 - deviation * 10.0 + hash_score)
    
    _priority_cache[key] = result
    return result
