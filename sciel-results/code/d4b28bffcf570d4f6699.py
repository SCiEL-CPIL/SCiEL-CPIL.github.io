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

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
V = len(POINTS)

# Precompute squared distances
dist2 = [[0] * V for _ in range(V)]
for i in range(V):
    xi, yi = POINTS[i]
    for j in range(i + 1, V):
        xj, yj = POINTS[j]
        d = (xi - xj) ** 2 + (yi - yj) ** 2
        dist2[i][j] = d
        dist2[j][i] = d

# Precompute conflict bitsets using Python ints (arbitrary precision)
# conflict_pair[i][j] = bitset of all k such that triangle(i,j,k) is isosceles
conflict_pair = [[0] * V for _ in range(V)]
for i in range(V):
    for j in range(i + 1, V):
        d_ij = dist2[i][j]
        mask = 0
        for k in range(V):
            if k == i or k == j:
                continue
            d_ik = dist2[i][k]
            d_jk = dist2[j][k]
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                mask |= (1 << k)
        conflict_pair[i][j] = mask
        conflict_pair[j][i] = mask

# Precompute for each point p, the bitset of points that conflict with p given a set S
# This depends on S, so we compute on the fly using bitset operations

def _is_independent_with_S(p, S_bits):
    """Check if adding p to S keeps S independent.
    S_bits is a bitset of points in S.
    We need: no pair (s1, s2) in S such that p in conflict_pair[s1][s2].
    Equivalently: for all s in S, p not in (union over s2 in S, s2 != s of conflict_pair[s][s2])
    But that's complex. Simpler: check all pairs in S.
    With bitsets, we can precompute for each p a bitset of "forbidden" points.
    Actually, let's just iterate over S members.
    """
    # For each s in S, check if there exists s2 in S such that p in conflict_pair[s][s2]
    # This is equivalent to: p is in the "pair-conflict neighborhood" of S
    # Let's compute: for s in S, the set of s2 in S where p conflicts with pair (s,s2)
    # = for s in S, (S & conflict_pair[s][p] | S & conflict_pair[p][s])... no
    # p in conflict_pair[s][s2] means triangle(s, s2, p) is isosceles
    # So we need: exists s, s2 in S, s != s2, such that p in conflict_pair[s][s2]
    # = exists s in S such that (S & conflict_pair[s][p]) is non-empty? No.
    # conflict_pair[s][s2] contains k if triangle(s,s2,k) is iso.
    # So p in conflict_pair[s][s2] iff triangle(s,s2,p) is iso.
    # We need to check all pairs (s, s2) in S.
    # With |S| <= ~20, this is at most 190 checks. Acceptable.
    s_list = []
    tmp = S_bits
    while tmp:
        lsb = tmp & (-tmp)
        idx = lsb.bit_length() - 1
        s_list.append(idx)
        tmp ^= lsb
    
    for i in range(len(s_list)):
        for j in range(i + 1, len(s_list)):
            if (conflict_pair[s_list[i]][s_list[j]] >> p) & 1:
                return False
    return True

def _construct_greedy_coverage(seed=0):
    """
    Greedy set-cover style construction with bitset optimization.
    """
    random.seed(seed)
    S_bits = 0  # bitset of points in S
    # Unblocked points: points q such that S U {q} is independent.
    # Initially S is empty, so all points are unblocked.
    all_bits = (1 << V) - 1
    unblocked = all_bits
    
    while unblocked:
        # Find the best point p to add
        best_p = -1
        best_score = -1
        
        # Get list of unblocked points
        ub_list = []
        tmp = unblocked
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            ub_list.append(idx)
            tmp ^= lsb
        
        for p in ub_list:
            # Check if S U {p} is independent
            if not _is_independent_with_S(p, S_bits):
                continue
            
            # Calculate score: how many unblocked q will be blocked by adding p?
            # q is blocked if exists s in S such that triangle(p, s, q) is isosceles
            # = q in conflict_pair[p][s] for some s in S
            # Compute the bitset of points blocked by p given S:
            blocked_by_p = 0
            tmp_s = S_bits
            while tmp_s:
                lsb = tmp_s & (-tmp_s)
                s = lsb.bit_length() - 1
                blocked_by_p |= conflict_pair[p][s]
                tmp_s ^= lsb
            
            # Count how many unblocked points (excluding p) are in blocked_by_p
            score = (unblocked & blocked_by_p & ~(1 << p)).bit_count()
            
            # Add small random noise to break ties
            noise = random.random() * 0.1
            total_score = score + noise
            
            if total_score > best_score:
                best_score = total_score
                best_p = p
        
        if best_p == -1:
            break
            
        # Add best_p to S
        S_bits |= (1 << best_p)
        
        # Update unblocked: remove best_p and any q that is now blocked
        # q is blocked if exists s in S (s != best_p) such that q in conflict_pair[best_p][s]
        # OR if q was already blocked (but we maintain unblocked correctly, so it's just the new blocking)
        blocked_by_new = 0
        tmp_s = S_bits & ~(1 << best_p)  # S without best_p
        while tmp_s:
            lsb = tmp_s & (-tmp_s)
            s = lsb.bit_length() - 1
            blocked_by_new |= conflict_pair[best_p][s]
            tmp_s ^= lsb
        
        # Remove best_p and all blocked points from unblocked
        unblocked &= ~(1 << best_p)
        unblocked &= ~blocked_by_new

    return S_bits

def _local_search_polish(S_bits, max_iters=50, time_limit=None):
    """
    Try to reduce S by swapping points: remove one point, add another if it maintains
    independence and maximality.
    Returns improved S_bits.
    """
    current = S_bits
    current_size = current.bit_count()
    
    # Precompute S list
    def bits_to_list(bits):
        lst = []
        tmp = bits
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            lst.append(idx)
            tmp ^= lsb
        return lst
    
    for iteration in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        
        s_list = bits_to_list(current)
        improved = False
        
        # Try removing each point and see if we can add a different one
        for remove_idx in range(len(s_list)):
            r = s_list[remove_idx]
            S_without_r = current & ~(1 << r)
            
            # Find candidate points to add
            all_bits = (1 << V) - 1
            candidates = all_bits & ~S_without_r
            
            # For each candidate, check if S_without_r U {c} is independent and maximal
            # We want to find a c such that the new set is still maximal
            # and ideally blocks at least as many points
            
            # First check independence of S_without_r U {c}
            valid_candidates = []
            tmp_c = candidates
            while tmp_c:
                lsb = tmp_c & (-tmp_c)
                c = lsb.bit_length() - 1
                tmp_c ^= lsb
                
                if _is_independent_with_S(c, S_without_r):
                    valid_candidates.append(c)
            
            if not valid_candidates:
                continue
            
            # For each valid candidate, check if S_without_r U {c} is maximal
            # Maximality: every point not in S_without_r U {c} is blocked
            best_c = -1
            best_block_count = -1
            
            for c in valid_candidates:
                new_S = S_without_r | (1 << c)
                new_s_list = bits_to_list(new_S)
                
                # Check maximality
                is_maximal = True
                for p in range(V):
                    if (new_S >> p) & 1:
                        continue
                    # Check if S U {p} is independent
                    if _is_independent_with_S(p, new_S):
                        is_maximal = False
                        break
                
                if not is_maximal:
                    continue
                
                # Count how many points are blocked (for tie-breaking)
                # A point q is blocked if exists s1, s2 in new_S with q in conflict_pair[s1][s2]
                # This is expensive; use as tie-breaker only
                block_count = 0
                for i in range(len(new_s_list)):
                    for j in range(i + 1, len(new_s_list)):
                        block_count += conflict_pair[new_s_list[i]][new_s_list[j]].bit_count()
                
                if block_count > best_block_count:
                    best_block_count = block_count
                    best_c = c
            
            if best_c != -1:
                # Perform the swap
                current = S_without_r | (1 << best_c)
                if current.bit_count() < current_size:
                    current_size = current.bit_count()
                    improved = True
                    break  # Restart with new S
        
        if not improved:
            break
    
    return current

def _verify_independent(S_bits):
    """Check that S is an independent no-isosceles set."""
    s_list = []
    tmp = S_bits
    while tmp:
        lsb = tmp & (-tmp)
        idx = lsb.bit_length() - 1
        s_list.append(idx)
        tmp ^= lsb
    
    for i in range(len(s_list)):
        for j in range(i + 1, len(s_list)):
            for k in range(j + 1, len(s_list)):
                if (conflict_pair[s_list[i]][s_list[j]] >> s_list[k]) & 1:
                    return False
    return True

def _verify_maximal(S_bits):
    """Check that S is maximal."""
    for p in range(V):
        if (S_bits >> p) & 1:
            continue
        if _is_independent_with_S(p, S_bits):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_bits = None
    best_size = V
    
    # Run multiple greedy constructions with different seeds
    num_greedy = 30
    
    for seed in range(num_greedy):
        if time.time() > deadline:
            break
            
        S_bits = _construct_greedy_coverage(seed)
        
        # Verify
        if not _verify_independent(S_bits) or not _verify_maximal(S_bits):
            continue
        
        size = S_bits.bit_count()
        if size < best_size:
            best_size = size
            best_bits = S_bits
            if best_size <= 10:
                break
    
    # Local search polish on the best solution found
    if best_bits is not None:
        polish_deadline = min(deadline, start_time + 285.0)
        polished = _local_search_polish(best_bits, max_iters=20, time_limit=polish_deadline)
        
        if _verify_independent(polished) and _verify_maximal(polished):
            if polished.bit_count() < best_size:
                best_size = polished.bit_count()
                best_bits = polished
    
    if best_bits is None:
        # Fallback
        best_bits = _construct_greedy_coverage(0)
    
    # Convert to list of points
    result = []
    tmp = best_bits
    while tmp:
        lsb = tmp & (-tmp)
        idx = lsb.bit_length() - 1
        result.append(POINTS[idx])
        tmp ^= lsb
    
    return result

# EVOLVE-BLOCK-END
