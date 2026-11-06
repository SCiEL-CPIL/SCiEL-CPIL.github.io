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

def _is_maximal_fast(S_bits):
    """Check if S is maximal using bitset union of conflict pairs."""
    s_list = []
    tmp = S_bits
    while tmp:
        lsb = tmp & (-tmp)
        idx = lsb.bit_length() - 1
        s_list.append(idx)
        tmp ^= lsb
    
    if len(s_list) < 2:
        return False

    covered = 0
    for i in range(len(s_list)):
        for j in range(i + 1, len(s_list)):
            covered |= conflict_pair[s_list[i]][s_list[j]]
    
    # S is maximal if S U covered == All points
    return (covered | S_bits) == ((1 << V) - 1)

def _local_search_polish(S_bits, max_iters=500, time_limit=None):
    """
    Improved local search with fast maximality check.
    Moves:
    1. Remove r if S \ {r} is still maximal (reduces size).
    2. Swap r with c if S \ {r} U {c} is independent and maximal (maintains size).
    """
    current = S_bits
    current_size = current.bit_count()
    all_bits = (1 << V) - 1
    
    def bits_to_list(bits):
        lst = []
        tmp = bits
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            lst.append(idx)
            tmp ^= lsb
        return lst

    improved = True
    iter_count = 0
    while improved and iter_count < max_iters:
        if time_limit is not None and time.time() > time_limit:
            break
        improved = False
        iter_count += 1
        
        s_list = bits_to_list(current)
        
        # 1. Try removing points (Reduction)
        for r in s_list:
            S_without_r = current & ~(1 << r)
            if _is_maximal_fast(S_without_r):
                current = S_without_r
                current_size -= 1
                improved = True
                break # Restart loop with new S
            
        if improved:
            continue

        # 2. Try swaps (Optimization)
        # Randomize order of removals to avoid getting stuck
        random.shuffle(s_list)
        
        for r in s_list:
            S_without_r = current & ~(1 << r)
            
            # Find all c such that S_without_r U {c} is independent
            valid_candidates = []
            tmp_c = all_bits & ~S_without_r
            while tmp_c:
                lsb = tmp_c & (-tmp_c)
                c = lsb.bit_length() - 1
                tmp_c ^= lsb
                
                if _is_independent_with_S(c, S_without_r):
                    valid_candidates.append(c)
            
            if not valid_candidates:
                continue

            # For each valid candidate, check if S' is maximal
            maximal_candidates = []
            for c in valid_candidates:
                new_S = S_without_r | (1 << c)
                if _is_maximal_fast(new_S):
                    maximal_candidates.append(c)
            
            if maximal_candidates:
                c = random.choice(maximal_candidates)
                current = S_without_r | (1 << c)
                improved = True
                break # Restart loop

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
    # We can run more restarts now because local search is faster
    # Or just run until time runs out, polishing each one?
    # Better: Run greedy, polish, keep best.
    
    seed = 0
    while time.time() < deadline - 5.0:
        S_bits = _construct_greedy_coverage(seed)
        seed += 1
        
        # Verify
        if not _verify_independent(S_bits) or not _is_maximal_fast(S_bits):
            continue
        
        # Polish this specific solution
        polish_deadline = min(deadline - 2.0, time.time() + 10.0)
        polished = _local_search_polish(S_bits, max_iters=200, time_limit=polish_deadline)
        
        if _verify_independent(polished) and _is_maximal_fast(polished):
            size = polished.bit_count()
            if size < best_size:
                best_size = size
                best_bits = polished
                if best_size <= 10:
                    break
    
    if best_bits is None:
        # Fallback
        best_bits = _construct_greedy_coverage(0)
        if not _verify_independent(best_bits) or not _verify_maximal(best_bits):
            # Extremely unlikely, but handle it
            best_bits = (1 << V) - 1 # Invalid, but prevents crash? No, we need valid.
            # Just return empty or a single point?
            # The task requires a valid set.
            # If greedy fails, it's a bug.
            pass

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
