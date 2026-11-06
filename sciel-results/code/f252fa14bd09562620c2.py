"""
Strategy: Set Cover / Dominance-Based Construction
Instead of greedily adding points in a fixed/random order and then trying to remove them,
we treat the problem as a hitting set or set cover variant.
We want to select a subset S such that:
1. S is an independent set in the "isosceles conflict graph" (no triangle in S is isosceles).
2. S is maximal: every point v not in S conflicts with S (i.e., S U {v} is not independent).

Condition 2 is equivalent to: For every v not in S, there exists a pair (u, w) in S
such that {u, w, v} forms an isosceles triangle.
Alternatively, for every v not in S, adding v creates a conflict.
A conflict is created if:
a) v forms an isosceles triangle with some pair in S.
b) v is "blocked" by distance constraints? No, the definition is simply "no further grid point can be added".
Adding v is valid if S U {v} is no-isosceles.
So v is blocked if S U {v} IS isosceles.
This means there exist u, w in S such that triangle(u, w, v) is isosceles.

So the condition for maximality is:
For all v not in S, exists u, w in S such that triangle(u, w, v) is isosceles.

We can reframe this:
Let C(v) be the set of pairs (u, w) such that triangle(u, w, v) is isosceles.
We need to select S such that S is independent, and for every v not in S, S contains at least one pair from C(v).
This is a complex constraint satisfaction problem.

However, we can use a greedy approach based on "efficiency".
Start with S = empty.
While S is not maximal:
    Find a point v not in S such that S U {v} is independent.
    Among all such v, pick the one that "blocks" the most other points (i.e., maximizes the number of points w not in S such that S U {v} U {w} is NOT independent, or more simply, increases the coverage of the grid).
    
Actually, a simpler proxy for "blocking power" is:
When we add a point p to S, it blocks a set of points B(p, S) which are points q not in S such that S U {p, q} is not independent.
We want to choose p to maximize |B(p, S)|.

This is a greedy set cover-like algorithm.
1. Precompute for every triple (i, j, k) if it is isosceles.
2. Maintain S.
3. At each step, for each candidate p not in S:
   a. Check if S U {p} is independent. If not, skip.
   b. If yes, calculate how many other points q not in S would be blocked by adding p.
      A point q is blocked by p (given S) if S U {p, q} is not independent.
      This happens if:
      - q forms an isosceles triangle with p and some s in S.
      - OR q forms an isosceles triangle with p and p? No.
      - OR q forms an isosceles triangle with two points in S? No, that would already block q regardless of p.
      
      Wait, if q is already blocked by S, we don't care. We only care about points q that are CURRENTLY unblocked (i.e., S U {q} is independent).
      Adding p blocks q if S U {p, q} is NOT independent.
      Since S U {q} IS independent, the conflict must involve p.
      So the conflict is an isosceles triangle with vertices from {p} U S and q.
      Specifically, there must be s1, s2 in S U {p} such that triangle(s1, s2, q) is isosceles, AND at least one of s1, s2 is p.
      So either:
      - triangle(p, s, q) is isosceles for some s in S.
      - triangle(p, p, q)? No, distinct points.
      
      So q is newly blocked by p if there exists s in S such that triangle(p, s, q) is isosceles.
      
   c. Score(p) = number of currently unblocked q (q not in S, S U {q} independent) such that exists s in S with triangle(p, s, q) isosceles.
   d. Pick p with max Score(p). Add p to S.
4. Repeat until no unblocked points remain (S is maximal).

This approach directly optimizes for "blocking efficiency" at each step, which should lead to a smaller S than random/greedy order.

Complexity:
V = 256.
Number of steps in S is likely small (10-20).
At each step, we evaluate up to 256 candidates.
For each candidate p, we check independence with S (O(|S|^2) or O(|S|) with precomputation).
Then we calculate Score(p).
To calculate Score(p), we need to know which q are currently unblocked.
We can maintain a set of Unblocked points.
Initially, all points are unblocked? No, S is empty, so all points can be added.
Wait, if S is empty, S U {q} is always independent. So all q are unblocked.
When we add p, we need to update the Unblocked set.
A point q becomes blocked if it was unblocked and is now blocked by the new S.
It is blocked if S U {q} is not independent.
Since it was unblocked before, S_old U {q} was independent.
So the conflict must involve p.
So q is blocked if there exists s in S_old such that triangle(p, s, q) is isosceles.
OR if triangle(p, s1, s2) is isosceles for s1, s2 in S_old? No, that would mean S_old U {p} is not independent, so we wouldn't have added p.

So the update is:
For each q in Unblocked:
    if exists s in S_old such that isosceles(p, s, q):
        remove q from Unblocked.

This is O(|Unblocked| * |S|) per step.
|Unblocked| <= 256. |S| <= 20.
Steps <= 20.
Total ops ~ 20 * 256 * 20 = 100k. Very fast.

We can add randomization: if multiple p have the same score, pick one randomly.
Or add a small random noise to the score to break ties and explore different solutions.
We can also run this multiple times with different random seeds and keep the best (smallest S).

Let's implement this.
"""

import itertools
import time
import random
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
V = len(POINTS)

# Precompute isosceles property for all triples
# isosceles[i, j, k] = True if triangle(i, j, k) is isosceles
# We can use a 3D boolean array or a set of frozensets.
# 256^3 is 16 million. A numpy bool array is 16 MB. Acceptable.
# But we can be smarter.
# For a fixed pair (i, j), the points k that form an isosceles triangle with i, j are those on the perpendicular bisector or on the circles centered at i or j with radius d(i,j).
# Actually, let's just precompute a set of blocked pairs.
# Or better: for each point p, and each point s, what points q are such that triangle(p, s, q) is isosceles?
# Let's precompute a matrix `conflict[p, s]` which is a bitset or list of q.
# Since V=256, we can use a 2D array of bitmasks (using Python ints or numpy uint64 if V<=64, but V=256 so we need 4 uint64s or just a set/list).
# Let's use a list of sets for simplicity and speed in Python if sets are small.
# Or a numpy boolean matrix of shape (V, V, V)? 16M booleans is 16MB. Fast to index.

# Let's try the numpy boolean matrix approach for speed.
# dist2[i, j]
dist2 = np.zeros((V, V), dtype=np.int64)
for i in range(V):
    xi, yi = POINTS[i]
    for j in range(i + 1, V):
        xj, yj = POINTS[j]
        d = (xi - xj) ** 2 + (yi - yj) ** 2
        dist2[i, j] = d
        dist2[j, i] = d

# isosceles[i, j, k]
# We can compute this on the fly or precompute.
# Precomputing a 3D array is O(V^3) = 16M ops. Fast.
# But memory is 16M bytes.
# Let's use a 2D array of sets instead, which might be more memory efficient if sparse.
# For each pair (i, j), store the set of k such that triangle(i, j, k) is isosceles.
# There are V(V-1)/2 pairs.
# For each pair, how many k?
# The locus of points k such that d(i,k)=d(j,k) is a line (perp bisector).
# The locus of points k such that d(i,k)=d(i,j) is a circle.
# The locus of points k such that d(j,k)=d(i,j) is a circle.
# In a grid, these are finite sets.
# Average number of k per pair is likely small (< 10-20).
# So a list of sets is good.

iso_conflicts = [set() for _ in range(V * V)] # index i*V + j

for i in range(V):
    for j in range(i + 1, V):
        d_ij = dist2[i, j]
        for k in range(V):
            if k == i or k == j:
                continue
            d_ik = dist2[i, k]
            d_jk = dist2[j, k]
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                iso_conflicts[i * V + j].add(k)
                iso_conflicts[j * V + i].add(k)

def _construct_greedy_coverage(seed=0):
    """
    Greedy set-cover style construction.
    """
    random.seed(seed)
    S = set()
    # Unblocked points: points q such that S U {q} is independent.
    # Initially S is empty, so all points are unblocked.
    unblocked = set(range(V))
    
    while unblocked:
        # Find the best point p to add
        best_p = -1
        best_score = -1
        best_tie_breaker = -1
        
        candidates = []
        for p in unblocked:
            # Check if S U {p} is independent
            # This requires that for all s1, s2 in S, triangle(p, s1, s2) is not isosceles.
            # And for all s in S, triangle(p, s, p) is not isosceles (trivially true).
            # Actually, we just need to check pairs in S U {p}.
            # Since S is independent, we only need to check triangles involving p.
            # So check if there exist s1, s2 in S such that triangle(p, s1, s2) is isosceles.
            valid = True
            s_list = list(S)
            for idx1 in range(len(s_list)):
                s1 = s_list[idx1]
                for idx2 in range(idx1 + 1, len(s_list)):
                    s2 = s_list[idx2]
                    if p in iso_conflicts[s1 * V + s2]:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Calculate score: how many unblocked q will be blocked by adding p?
            # q is blocked if exists s in S such that triangle(p, s, q) is isosceles.
            # Note: q must be in unblocked.
            score = 0
            for q in unblocked:
                if q == p:
                    continue
                # Check if q is blocked by p given S
                # Is there an s in S such that q in iso_conflicts[p * V + s]?
                blocked = False
                for s in s_list:
                    if q in iso_conflicts[p * V + s]:
                        blocked = True
                        break
                if blocked:
                    score += 1
            
            # Add small random noise to break ties and encourage diversity
            noise = random.random() * 0.1
            total_score = score + noise
            
            if total_score > best_score:
                best_score = total_score
                best_p = p
        
        if best_p == -1:
            # No valid point found? This shouldn't happen if unblocked is not empty
            # because S is independent, so any p in unblocked should be valid?
            # Wait, unblocked means S U {p} is independent.
            # So valid should always be true for p in unblocked.
            # If best_p is -1, it means no p in unblocked is valid?
            # That implies S is not maximal? No, it implies my check is wrong.
            # Let's just break if this happens.
            break
            
        # Add best_p to S
        S.add(best_p)
        
        # Update unblocked
        # Remove best_p from unblocked
        unblocked.remove(best_p)
        
        # Remove any q in unblocked that are now blocked
        # q is blocked if exists s in S (including best_p) such that triangle(s, s2, q) is isosceles?
        # No, q is blocked if S_new U {q} is NOT independent.
        # Since S_old U {q} WAS independent (q was in unblocked), the conflict must involve best_p.
        # So q is blocked if there exists s in S_old such that triangle(best_p, s, q) is isosceles.
        # OR if triangle(best_p, s1, s2) is isosceles for s1, s2 in S_old?
        # No, that would mean S_old U {best_p} is not independent, which we checked and it was valid.
        
        new_unblocked = set()
        for q in unblocked:
            # Check if q is blocked by best_p
            blocked = False
            for s in S:
                if s == best_p:
                    continue
                if q in iso_conflicts[best_p * V + s]:
                    blocked = True
                    break
            if not blocked:
                new_unblocked.add(q)
        
        unblocked = new_unblocked

    return S

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_set = None
    best_size = V
    
    # Run multiple times with different seeds
    # The greedy coverage is deterministic for a given seed (due to noise).
    # We can run it for, say, 50 seeds.
    
    for seed in range(100):
        if time.time() > deadline:
            break
            
        S = _construct_greedy_coverage(seed)
        
        # Verify maximality and independence just in case
        # Independence check
        is_valid = True
        s_list = list(S)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                for k in range(j + 1, len(s_list)):
                    if s_list[k] in iso_conflicts[s_list[i] * V + s_list[j]]:
                        is_valid = False
                        break
                if not is_valid:
                    break
            if not is_valid:
                break
        
        if not is_valid:
            continue
            
        # Maximality check
        is_maximal = True
        for p in range(V):
            if p in S:
                continue
            # Check if S U {p} is independent
            valid_add = True
            for s1 in s_list:
                for s2 in s_list:
                    if s1 == s2:
                        continue
                    if p in iso_conflicts[s1 * V + s2]:
                        valid_add = False
                        break
                if not valid_add:
                    break
            if valid_add:
                is_maximal = False
                break
                
        if not is_maximal:
            continue
            
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
            
            if best_size <= 10:
                break
                
    if best_set is None:
        # Fallback: simple greedy
        S = _construct_greedy_coverage(0)
        best_set = S
        
    return [POINTS[i] for i in best_set]

# EVOLVE-BLOCK-END
