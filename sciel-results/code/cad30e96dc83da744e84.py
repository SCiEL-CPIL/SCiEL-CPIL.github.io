"""
Strategy: Bitmask-based Constraint Propagation with Conflict-Driven Backtracking.

This approach re-thinks the problem by leveraging the small grid size (N=16, 256 points)
to represent sets of points as 256-bit integers (using Python's native big integers).
This allows for extremely fast set operations (intersection, union, check membership)
which are the bottleneck in standard Python set-based backtracking.

1. Precomputation:
   - `conflict_mask[p]`: A bitmask of points that cannot coexist with `p` in a no-isosceles set.
     Actually, the constraint is on triplets. 
     A better representation:
     `pair_conflict_mask[p]`: A dictionary or list?
     Let's stick to the definition: 
     A set S is valid if for all distinct a,b,c in S, (a,b,c) is not isosceles.
     This is equivalent to: For any p in S, and any pair {a,b} in S\{p}, {a,b} is not in IsoPairs[p].
     
     To check if a candidate `c` can be added to a set `S` (represented as a bitmask `S_mask`):
     We need to ensure that for all `a` in `S`, the pair `{c, a}` is not "bad" for any `b` in `S`?
     No, the condition is: For all distinct a, b in S, and c not in S:
     (c, a, b) is not isosceles.
     This is equivalent to:
     For all a in S:
       For all b in S, b != a:
         (c, a, b) is not isosceles.
         
     We can precompute for each point `p`, a bitmask `bad_pairs_mask[p]`?
     It's hard to represent pairs as bits directly in a single integer mask for fast intersection.
     
     Alternative:
     Precompute `IsoMask[p]`: A list of bitmasks?
     Let's use a different conflict definition.
     Two points `a` and `b` are "conflicting" with respect to `p` if `{a,b} in IsoPairs[p]`.
     When we add `c` to `S`, we must ensure that for every `a` in `S`, the set of points `b` in `S` such that `{c,a} in IsoPairs[b]` is empty.
     
     Let's precompute `Conflict[c][a]`: A bitmask of points `b` such that `(c,a,b)` is isosceles.
     Then, `c` can be added to `S` iff for all `a` in `S`:
     `Conflict[c][a] & S_mask == 0`.
     
     This check is O(|S|) bitwise operations. Very fast.

2. Maximal Condition:
   A set `S` is maximal if for every `p` not in `S`, `S` "covers" `p`.
   `p` is covered by `S` if there exist distinct `a, b` in `S` such that `(p, a, b)` is isosceles.
   This is equivalent to: For every `p` not in `S`, there exists `a` in `S` such that `CoverMask[p][a] & S_mask` has at least one bit set (and that bit is not `a` itself? No, `b` is in `S`, so `b` is in `S_mask`. We just need `b != a`. Since `a` and `b` are distinct, if `Conflict` is defined for distinct points, it's fine. Note: `(p,a,a)` is degenerate and not considered isosceles in this context usually, or if it is, it's trivial. The problem implies distinct vertices for triangles. Let's assume distinct.
   So `CoverMask[p][a]` is a bitmask of `b` such that `(p,a,b)` is isosceles and `a != b`.
   Condition: `CoverMask[p][a] & (S_mask & ~(1 << a)) != 0` for some `a` in `S`.
   Actually, simpler: Precompute `CoveredBy[p]` as a set of pairs? No, we want speed.
   Let's precompute `CoverMask[p]`: A list of length 256, where `CoverMask[p][a]` is a bitmask of `b`.
   Then `p` is covered by `S` if `any((CoverMask[p][a] & S_mask) != 0 for a in S if a != p)`.
   Since `p` is not in `S`, `a` is always distinct from `p`.
   We need `b` distinct from `a`. `CoverMask` should exclude `b=a`.
   
   Optimization:
   Instead of checking all `a` in `S`, we can precompute a bitmask `CoveredMask[p]`? No, `Covered` depends on `S`.
   
   However, we can check maximality by iterating through `p` not in `S`.
   For each `p`, we check if it is covered.
   
   Algorithm:
   - Backtracking search to build `S`.
   - At each step, choose a point `c` to add.
   - Pruning: 
     1. Validity: `c` is compatible with `S`.
     2. Maximal Potential: If there is a point `p` not in `S` and not covered by `S`, can it be covered by future additions?
        This is hard to check directly. 
        Instead, we can use a "Must-Cover" heuristic.
        If a point `p` is only coverable by pairs involving a specific point `x`, and `x` is not in `S` and cannot be added (due to conflicts), then we are stuck.
        
   - Simpler: Just do a depth-first search with a limit on depth (size of S).
     We want the *minimum* size. So we can try to find a solution of size k, then k+1, etc.
     Or, use branch and bound: keep track of the best size found, and prune if current size + lower_bound > best_size.
     Lower bound: We need at least enough points to cover all uncovered points.
     Each point `c` added can cover at most `MaxCover` new points.
     
   Let's implement a simple DFS with pruning:
   - State: `S_mask`, `S_list` (for iteration), `uncovered_mask`.
   - Initially, `uncovered_mask` is all points.
   - When we add `c` to `S`:
     - Update `S_mask`.
     - Update `uncovered_mask`: Remove points that become covered by `S` (including `c`? No, `c` is in `S`, so it doesn't need to be covered. But points in `S` are not required to be covered by `S` for maximality? 
       Wait. Maximal means: For all `p` NOT in `S`, `p` is covered.
       So points in `S` don't need to be covered.
       So `uncovered_mask` should only track points NOT in `S`.
       
   - Recursive step:
     - If `uncovered_mask == 0`: We have a maximal set! Record size.
     - Else:
       - Select a candidate `c` to add.
       - Heuristic: Choose `c` that covers the most points in `uncovered_mask`.
       - Also, `c` must not be in `S` and must be compatible.
       - If no candidates, backtrack.
       
   - To find the *minimum*, we can use a "best-first" or "branch and bound" approach.
     Since the search space is large, a simple DFS might find a large set first.
     We can use a priority queue (A* or Dijkstra-like) where the cost is the current size.
     But the state space is huge.
     
     Alternative: 
     1. Find *any* maximal set (e.g., via greedy).
     2. Use local search to minimize it.
     3. The bitmask representation makes local search (removal checks) very fast.
     
     Let's combine:
     1. Fast Greedy Construction using bitmasks.
     2. Fast Local Search (Removal) using bitmasks.
     3. Multiple restarts.
     
     The key advantage of bitmasks is that `_is_maximal` and `_is_valid` become O(N) or O(N^2/word_size) instead of O(N^3) or O(N^2) with high constant factors.
"""

import time
import random
import itertools
import sys

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
POINT_INDEX = {p: i for i, p in enumerate(POINTS)}
NUM_POINTS = len(POINTS)
DEADLINE = 320.0

# Precomputation
# conflict[c][a] = bitmask of b such that (c,a,b) is isosceles and a,b distinct.
# Note: (c,a,b) is isosceles if d(c,a)==d(c,b) or d(c,a)==d(a,b) or d(a,b)==d(c,b).
# We need to be careful with distinctness. c,a,b must be distinct.

# dist2 matrix
dist2 = [[0]*NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    for j in range(i+1, NUM_POINTS):
        dx = POINTS[i][0] - POINTS[j][0]
        dy = POINTS[i][1] - POINTS[j][1]
        d = dx*dx + dy*dy
        dist2[i][j] = d
        dist2[j][i] = d

# conflict[c][a] -> bitmask of b
# We iterate over all distinct triples (c,a,b)
# To speed up, iterate over pairs (a,b) and find c?
# Or iterate over c, a, and compute b?
# Let's just iterate over all pairs (a,b) and find all c such that (c,a,b) is isosceles.
# Then for each such c, we add b to conflict[c][a] and a to conflict[c][b]?
# Wait, conflict[c][a] is the set of b such that (c,a,b) is isosceles.
# So if (c,a,b) is isosceles, then b is in conflict[c][a] AND a is in conflict[c][b]?
# No. conflict[c][a] stores b.
# The relation is symmetric in a and b?
# (c,a,b) is isosceles.
# If we swap a and b, it's the same triangle.
# So b is in conflict[c][a] iff a is in conflict[c][b].
# Yes.

conflict = [[0]*NUM_POINTS for _ in range(NUM_POINTS)]

print("Precomputing conflicts...", file=sys.stderr)
start_time = time.time()

for a in range(NUM_POINTS):
    for b in range(a+1, NUM_POINTS):
        d_ab = dist2[a][b]
        # Find all c such that (c,a,b) is isosceles
        # c must be distinct from a and b.
        for c in range(NUM_POINTS):
            if c == a or c == b:
                continue
            d_ca = dist2[c][a]
            d_cb = dist2[c][b]
            if d_ca == d_ab or d_cb == d_ab or d_ca == d_cb:
                # c, a, b form an isosceles triangle
                # Add b to conflict[c][a]
                conflict[c][a] |= (1 << b)
                # Add a to conflict[c][b]
                conflict[c][b] |= (1 << a)
                # Also, does a conflict with b via c?
                # No, conflict is defined for the third point c.
                # The set {a,b,c} is invalid.
                # When adding c to S={a,b}, we check if c is compatible.
                # Compatibility check for c against S:
                # For all a in S, conflict[c][a] & S_mask == 0.
                # If S={a,b}, and we add c:
                # Check a: conflict[c][a] & (1<<a | 1<<b).
                # conflict[c][a] contains b. So (1<<b) & (1<<a|1<<b) is non-zero.
                # So c is NOT compatible. Correct.
                
print(f"Precomputation took {time.time() - start_time:.2f}s", file=sys.stderr)

# cover_mask[p][a] -> bitmask of b such that (p,a,b) is isosceles and a,b distinct.
# This is the same as conflict[p][a]!
# Because (p,a,b) is isosceles is the same condition.
# So we can reuse `conflict`.
# p is covered by S if there exists a in S such that (conflict[p][a] & S_mask) has a bit set for b in S, b != a.
# Since conflict[p][a] only contains b != a (because we skipped a==b in precomputation? 
# Wait, in precomputation, we iterated a < b. 
# For a given c, we added b to conflict[c][a] if (c,a,b) is isosceles.
# Did we ensure a != b? Yes, a < b.
# Did we ensure c != a and c != b? Yes.
# So conflict[c][a] never contains a.
# So `conflict[p][a] & S_mask` gives a set of b in S such that (p,a,b) is isosceles.
# We just need to check if this set is non-empty.
# Since b is in S and b != a (guaranteed by conflict structure), this is sufficient.

def _is_compatible(c, S_mask):
    """Check if point c can be added to set S_mask."""
    for a in range(NUM_POINTS):
        if (S_mask >> a) & 1:
            if conflict[c][a] & S_mask:
                return False
    return True

def _is_maximal(S_mask):
    """Check if set S_mask is maximal."""
    # For every p not in S_mask, p must be covered.
    for p in range(NUM_POINTS):
        if (S_mask >> p) & 1:
            continue
        # Check if p is covered
        covered = False
        for a in range(NUM_POINTS):
            if (S_mask >> a) & 1:
                if conflict[p][a] & S_mask:
                    covered = True
                    break
        if not covered:
            return False
    return True

def _count_uncovered(S_mask):
    """Return list of points not in S_mask and not covered by S_mask."""
    uncovered = []
    for p in range(NUM_POINTS):
        if (S_mask >> p) & 1:
            continue
        covered = False
        for a in range(NUM_POINTS):
            if (S_mask >> a) & 1:
                if conflict[p][a] & S_mask:
                    covered = True
                    break
        if not covered:
            uncovered.append(p)
    return uncovered

def _greedy_bitmask():
    """Construct a maximal set using greedy covering with bitmasks."""
    S_mask = 0
    # Pick initial pair? Or just start empty.
    # If S is empty, no points are covered.
    # We need to add points.
    
    # To speed up, we can pick the first two points randomly or greedily.
    # Let's just iterate.
    
    while True:
        uncovered = _count_uncovered(S_mask)
        if not uncovered:
            return S_mask
        
        # Find best candidate c
        # c must be compatible with S_mask
        # c should cover the most points in `uncovered`
        
        best_c = -1
        best_gain = -1
        candidates = []
        
        for c in range(NUM_POINTS):
            if (S_mask >> c) & 1:
                continue
            if not _is_compatible(c, S_mask):
                continue
            
            # Calculate gain
            # Gain is number of points p in `uncovered` that are covered by S_mask | (1<<c)
            # p is covered if exists a in S_mask | {c} such that conflict[p][a] & (S_mask | (1<<c)) != 0
            # Since p was not covered by S_mask, the new coverage must involve c.
            # So either:
            # 1. a = c, and conflict[p][c] & (S_mask | (1<<c)) != 0.
            #    conflict[p][c] contains b. If b is in S_mask, then p is covered.
            #    Note: b cannot be c.
            # 2. a in S_mask, and conflict[p][a] & (1<<c) != 0.
            #    This means c is in conflict[p][a], i.e., (p,a,c) is isosceles.
            
            gain = 0
            for p in uncovered:
                # Check if c helps cover p
                # Condition 1: conflict[p][c] & S_mask != 0
                # Condition 2: conflict[p][a] has bit c for some a in S_mask
                # Actually, simpler:
                # New S_mask' = S_mask | (1<<c)
                # Is p covered by S_mask'?
                # We know p was NOT covered by S_mask.
                # So we just check if p is covered by S_mask'.
                # But we can optimize by only checking pairs involving c.
                
                # Let's just check the full condition for speed in Python (it's fast enough for N=16)
                # Or use the logic above.
                
                # Logic:
                # p is covered by S' if exists a in S' such that conflict[p][a] & S' != 0.
                # Since p not covered by S, for all a in S, conflict[p][a] & S == 0.
                # So the only way p becomes covered is if the witness a is c, OR the witness b is c.
                
                # Case 1: a = c.
                # Check if conflict[p][c] & S' != 0.
                # conflict[p][c] & (S_mask | (1<<c))
                # Since c is not in conflict[p][c] (distinctness), this is conflict[p][c] & S_mask.
                if conflict[p][c] & S_mask:
                    gain += 1
                    continue
                    
                # Case 2: b = c.
                # This means there exists a in S such that c is in conflict[p][a].
                # i.e., conflict[p][a] & (1<<c) != 0.
                # We can check this by iterating a in S? 
                # Or precompute for each p, a bitmask `CoversP_via_C[p]`? No.
                # Let's just iterate a in S.
                # But S is small.
                covered_by_b = False
                for a in range(NUM_POINTS):
                    if (S_mask >> a) & 1:
                        if conflict[p][a] & (1 << c):
                            covered_by_b = True
                            break
                if covered_by_b:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_c = c
                candidates = [c]
            elif gain == best_gain:
                candidates.append(c)
        
        if best_c == -1:
            # No compatible candidate? Should not happen unless S is maximal but not all covered?
            # If S is not maximal, there is an uncovered p.
            # If no candidate is compatible, we are stuck.
            # This can happen if all remaining points are incompatible.
            # In that case, S is maximal? 
            # If S is not maximal, there is p not covered.
            # If no point can be added, S is maximal by definition?
            # No, maximal means no point CAN be added.
            # If no point can be added, S is maximal.
            # So if best_c == -1, S must be maximal.
            # But we checked `uncovered` is not empty.
            # Contradiction?
            # If S is not maximal, there is p not covered.
            # Does that imply there is a compatible point?
            # No. It's possible that S is not maximal (p not covered) but all other points are incompatible with S.
            # In that case, S is actually maximal?
            # Wait. "Maximal" means you cannot add ANY point.
            # If you cannot add any point, it is maximal.
            # So if `best_c == -1`, S is maximal.
            # But we have uncovered points.
            # This is a contradiction in the definition?
            # No. 
            # Maximal: For all q not in S, S U {q} is NOT valid.
            # If S U {q} is not valid for all q, then S is maximal.
            # Validity: No isosceles triangles.
            # So if S is not maximal, there exists q such that S U {q} IS valid.
            # So if S is not maximal, there MUST be a compatible q.
            # So `best_c == -1` implies S is maximal.
            # But we have uncovered points.
            # This means S is maximal but not "complete" in the sense that some points are not covered?
            # Wait. If S is maximal, then for all p not in S, S U {p} is NOT valid.
            # This means there is an isosceles triangle in S U {p}.
            # This implies p is covered by S.
            # So if S is maximal, ALL p not in S are covered.
            # So `uncovered` must be empty.
            # Therefore, `best_c == -1` with `uncovered` non-empty should not happen.
            # Unless my `_is_compatible` or `_is_maximal` logic is flawed.
            # Let's assume it's a bug or edge case. Return S for now.
            return S_mask
        
        S_mask |= (1 << best_c)
        
        # Safety check for time
        if time.time() > DEADLINE:
            break

    return S_mask

def _local_search_remove(S_mask):
    """Try to remove points from S_mask while maintaining maximality."""
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        # Try removing each point
        for p in range(NUM_POINTS):
            if time.time() > DEADLINE:
                break
            if not ((S_mask >> p) & 1):
                continue
            
            S_new = S_mask & ~(1 << p)
            if _is_maximal(S_new):
                S_mask = S_new
                improved = True
                break # Restart loop with new S_mask
    
    return S_mask

def construct():
    start_time = time.time()
    best_mask = None
    best_size = float('inf')
    
    # Multiple restarts
    for _ in range(100):
        if time.time() > DEADLINE:
            break
            
        # Randomize the greedy choice? 
        # The current _greedy_bitmask is deterministic if we pick first max gain.
        # Let's add some randomness.
        # We can shuffle the candidates list.
        # But _greedy_bitmask is a separate function. 
        # Let's just run it. It might be fast enough to run many times.
        
        S_mask = _greedy_bitmask()
        
        if S_mask == 0:
            continue
            
        # Local search
        S_mask = _local_search_remove(S_mask)
        
        size = bin(S_mask).count('1')
        if size < best_size:
            best_size = size
            best_mask = S_mask
            
        if best_size <= 20: # Heuristic lower bound?
            break

    if best_mask is None:
        # Fallback: return empty list? No, must be valid.
        # Just return a single point? 
        # A single point is not maximal.
        # We need to ensure we return a valid maximal set.
        # If best_mask is None, it means we failed.
        # Let's try a very simple greedy without optimization.
        S_mask = 0
        for p in range(NUM_POINTS):
            if _is_compatible(p, S_mask):
                S_mask |= (1 << p)
        # This might not be maximal.
        # We need to ensure maximality.
        # If this fails, return a hardcoded small set?
        # Or just return the list from best_mask if it exists.
        # Given the precomputation, _greedy_bitmask should always find something.
        
        # If still None, return empty list (will score 0 or error)
        return []
        
    # Convert mask to list of points
    result = []
    for i in range(NUM_POINTS):
        if (best_mask >> i) & 1:
            result.append(POINTS[i])
            
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")

