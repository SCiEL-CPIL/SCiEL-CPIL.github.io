"""
Cap set construction via greedy insertion with heuristic priority.

The evaluator calls priority(vector, n) for each candidate vector in {0,1,2}^n,
and inserts it into the current set if doing so keeps the set a cap set (no 3-term
AP). We want to design a priority function that leads to a large cap set.

Key insight: A 3-term AP in {0,1,2}^n is a triple (a, b, c) where a + c = 2b
coordinate-wise (with values in {0,1,2}). This means for each coordinate:
- If b_i = 0, then a_i = c_i = 0
- If b_i = 1, then a_i = c_i = 1 OR a_i, c_i are {0, 2} in some order
- If b_i = 2, then a_i = c_i = 2

So a conflict arises when there exist a, c in the set such that (a+c)/2 is also in the set
or would be in the set.

Strategy: We maintain the current cap set and for each candidate, we compute a "conflict score"
based on how many existing vectors would form a 3-AP with the candidate. We want to prefer
vectors with low conflict potential.

However, the priority function is called BEFORE the vector is added, and we don't know
the current state of the set inside the priority function (unless we use global state).

Let's use global state to track the current set and compute a dynamic priority.

The priority should be: prefer vectors that have fewer potential conflicts with existing vectors.
For a candidate v, the conflict count is: number of pairs (a, c) in current_set such that
(a + c) / 2 == v (i.e., v is the midpoint of a and c), OR equivalently, for each existing a,
check if 2v - a is in the set (and 2v - a != v, and 2v - a is valid in {0,1,2}^n).

Actually, a 3-AP (a, b, c) with b as midpoint means a + c = 2b. So for candidate b,
conflicts exist if there's a pair (a, c) in the set with a + c = 2b.
Equivalently, for each a in the set, c = 2b - a must be in the set and c != a (to avoid trivial a=a=c=b).

Wait, actually if a = c, then 2a = 2b, so a = b, which is trivial. So we need a != c.

So conflict_count(v) = number of a in current_set such that:
  c = 2*v - a is in current_set, c != a, and c is a valid vector in {0,1,2}^n.

But this is O(|S|) per query, which could be slow. Let's think about efficiency.

Alternative: We can precompute for the current set a hash set for O(1) lookup.
For each candidate v, iterate over a in S, compute c = 2v - a, check if c is in S and c != a.
This is O(|S|) per priority call. With |S| potentially in the thousands and 3^n candidates,
this might be too slow.

But wait - the evaluator adds vectors greedily in descending priority. So it sorts all 3^n vectors
by priority and tries to add them in that order. The priority function is called once per vector
to determine the order. So we need to compute priority for ALL 3^n vectors upfront (or as they are queried).

Actually, re-reading: "vectors of {0,1,2}^n are added greedily in descending priority". This likely
means the evaluator computes priority for all vectors, sorts them, then tries to add each one
in that order if it doesn't create a 3-AP.

So priority is called for each vector exactly once. We can use global state that accumulates
as vectors are added. But the priority for all vectors is determined before any are added (since
we sort first). Hmm, that's a problem for dynamic approaches.

Let me re-read: "added greedily in descending priority when they keep the set a cap set". This
suggests: sort by priority descending, then iterate through the sorted list, adding each vector
if it doesn't create a 3-AP. So priority is static - determined before the greedy process starts.

This changes things. We can't use the current set state in priority. We need a static heuristic.

Static heuristics for cap set:
1. Balanced count: prefer vectors with count of 1s close to n/2 (since extremes are more likely
   to form APs).
2. Spatial spread: prefer vectors where 1s are spread out.
3. Hamming weight balance.

From the knowledge: "Adding spatial spread of 1-positions as a secondary heuristic breaks ties
in the balanced-count score... Measured: 1.215 -> 1.242."

So a good static priority is:
- Primary: closeness of count_of_1s to n/2 (prefer balanced)
- Secondary: spatial spread of 1s (prefer more spread out)

Let's implement this.

For a vector v in {0,1,2}^n:
- count_1 = number of coordinates where v_i = 1
- balance_score = -abs(count_1 - n/2)  (higher is better, so negate the distance)
- For spatial spread: we can measure how spread out the 1s are. One way: for each pair of positions
  where v_i = 1, compute the distance, and sum or average. More spread = larger distances.
  Alternatively, use the variance of positions of 1s.

Actually, a simpler measure of spread: the sum of squared distances from the centroid of 1-positions.
Or just: for positions p_1 < p_2 < ... < p_k where v has 1s, compute sum of (p_i - mean)^2.

But this is O(k^2) or O(k) per vector, which is fine.

Let me code this up.
"""

import time
import math

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290  # Safety margin before 330s limit

def _check_deadline():
    return time.time() < _DEADLINE

def priority(vector, n):
    """
    Static priority for cap set construction.
    
    Primary: balance of 1-count around n/2
    Secondary: spatial spread of 1-positions
    Tertiary: small deterministic tie-breaker for stability
    """
    if not _check_deadline():
        return 0.0
    
    # Count 1s
    count_1 = sum(1 for x in vector if x == 1)
    
    # Primary: balance score (higher when count_1 is close to n/2)
    target = n / 2.0
    balance_score = -abs(count_1 - target)
    
    # Secondary: spatial spread of 1-positions
    positions = [i for i, x in enumerate(vector) if x == 1]
    if len(positions) > 1:
        mean_pos = sum(positions) / len(positions)
        spread = sum((p - mean_pos) ** 2 for p in positions)
    elif len(positions) == 1:
        spread = 0.0
    else:
        spread = 0.0
    
    # Normalize spread to be comparable to balance_score
    # Max possible spread for k positions in [0, n-1] is roughly k * (n/2)^2
    max_spread = max(1, len(positions) * (n / 2.0) ** 2)
    normalized_spread = spread / max_spread
    
    # Combine: balance is primary, spread is secondary
    # Weight spread so it acts as a tie-breaker
    combined = balance_score + 0.1 * normalized_spread
    
    # Tertiary: small hash-based tie-breaker for determinism
    # Use a simple hash of the vector to break ties
    h = hash(vector) % 1000000
    combined += h * 1e-9
    
    return combined

# EVOLVE-BLOCK-START
# The above priority function is the main logic
# EVOLVE-BLOCK-END
