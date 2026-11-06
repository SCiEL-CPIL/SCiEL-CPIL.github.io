"""
Strategy: Constraint Propagation with Conflict-Driven Backtracking.

The current program uses a greedy covering approach followed by local search for removals.
This new approach uses a constraint satisfaction solver (backtracking with propagation) to
directly search for a minimal maximal set.

Key differences:
1. **Representation**: We track `ForcedIn` (points that must be in S) and `ForcedOut` (points that must be not in S).
   Points in `ForcedOut` are "covered" by `ForcedIn`.
2. **Constraint Propagation**: 
   - If a point `p` is in `ForcedIn`, any point `q` that cannot coexist with `p` in a no-isosceles set (i.e., there exists `r` in `ForcedIn` such that `(p,q,r)` is isosceles) is not directly forced out, but rather, the pair `{p,q}` cannot be in S.
   - Actually, the constraint is on triplets.
   - Let's refine: 
     - `S` is the current set of points committed to be IN.
     - `C` is the set of points committed to be OUT.
     - A point `p` is "blocked" (cannot be added to S) if there exist `a, b` in S such that `(p,a,b)` is isosceles.
     - A point `p` is "uncovered" (must be in S to satisfy maximality? No, it must be covered by S).
     - If `p` is not in S, it MUST be covered by S.
     - If `p` is not in S and is NOT covered by S, then `p` MUST be in S. This is a propagation rule!
     
     Propagation Rule:
     If `p` is not in `S` and `p` is not in `C` (i.e., `p` is undecided), and `p` is not covered by `S`, then `p` MUST be in `S`.
     Why? Because if `p` is not in `S`, it must be covered. If it's not covered by current `S`, it can't be covered by future additions to `S`? 
     Wait. If we add more points to `S`, they might cover `p`.
     So `p` being uncovered by current `S` does NOT imply `p` must be in `S`. It implies that IF `p` is not in `S`, then some future point added to `S` must cover it.
     
     However, if `p` is not in `S` and `p` is not in `C`, and there is NO possible point that can be added to `S` to cover `p` (without violating isosceles constraints with existing `S`), then `p` MUST be in `S`.
     
     Let's use a simpler backtracking:
     1. Order points by some heuristic (e.g., number of conflicts).
     2. Decide for each point: IN or OUT.
     3. If IN: Check if it creates an isosceles triangle with any two existing IN points. If yes, fail.
     4. If OUT: Check if it is covered by existing IN points. If not, it's not a failure yet, but we must ensure it gets covered later.
        To simplify, we can enforce that we only choose OUT if it is ALREADY covered by IN points.
        This makes the search tree much smaller!
        "If we decide p is OUT, it must be covered by the current set of IN points."
        This is a valid pruning strategy because if p is not covered by current IN, and we put it OUT, we rely on future IN points to cover it. But if we restrict ourselves to only putting p OUT if it's already covered, we might miss solutions where p is covered by a point that comes later in the ordering.
        
        To fix this, we should order points such that "covering" points come first? Or use a different propagation.
        
        Alternative: 
        The condition "S is maximal" means: For all p not in S, p is covered by S.
        This is equivalent to: There is no p not in S such that p is not covered by S.
        
        Let's try **Forward Checking** with a specific ordering.
        We iterate through points. For each point, we decide IN or OUT.
        If we decide IN, we add to S.
        If we decide OUT, we require that it is covered by S (current).
        If it's not covered, we MUST decide IN?
        No, it could be covered by a point later.
        
        However, if we assume that "covering" is a local property and we process points in a way that coverers are likely to be processed before covered points, this works.
        Or, we can simply say: If p is not covered by S, we have TWO options:
        1. Add p to S.
        2. Leave p out, but remember that we MUST find a coverer for p later.
        
        This "remember" state is complex.
        
        Let's stick to the **Greedy with Propagation** idea but make it smarter.
        Actually, the simplest structural change is to use **Simulated Annealing** on the set of points, where moves are Add/Remove/Swap, and the energy function penalizes:
        1. Violation of no-isosceles (hard constraint).
        2. Violation of maximality (uncovered points).
        3. Size of S.
        
        The current program uses a greedy construction + removal. 
        A SA approach explores the space of all subsets, allowing temporary violations to escape local minima.
        
        Let's implement **Simulated Annealing**.
        
        State: A set of points S.
        Moves:
        - Add a random point p not in S.
        - Remove a random point p in S.
        - Swap a random point p in S with a random point q not in S.
        
        Energy E(S):
        - `viol_iso`: Number of isosceles triangles in S. (Should be 0).
        - `viol_max`: Number of points p not in S that are NOT covered by S. (Should be 0).
        - `size`: |S|.
        
        Total Cost = M * (viol_iso + viol_max) + size.
        We want to minimize Cost.
        
        We start with a valid maximal set (from greedy) to ensure we are in the feasible region initially, or we start empty and let SA find its way. Starting feasible is safer.
        
        Precomputation for fast evaluation:
        - We need to quickly update `viol_iso` and `viol_max` when adding/removing a point.
        - `viol_iso`: When adding p to S, new violations are pairs (a,b) in S such that (p,a,b) is isosceles.
        - `viol_max`: When adding p to S, some points might become covered. When removing p, some points might become uncovered.
        
        This requires maintaining:
        - `count_iso[p]`: Number of isosceles triangles in S involving p? No, just the set of pairs.
        - `cover_count[p]`: Number of pairs in S that cover p.
          If `cover_count[p] > 0` and p not in S, p is covered.
          If p in S, p is not "uncovered" in the maximality sense (it's in S).
          
        Let's define:
        `in_S`: set of points in S.
        `pair_in_S`: set of pairs (a,b) in S.
        
        `viol_iso` = number of isosceles triangles in S.
        `uncovered` = set of points p not in S such that no pair in S covers p.
        
        Cost = 100 * (viol_iso + len(uncovered)) + len(S).
        
        Updates:
        Add p to S:
        - New pairs formed: (p, q) for q in S.
        - For each new pair (p,q), check if it forms an isosceles triangle with any r in S.
          Actually, `viol_iso` increases by the number of r in S such that (p,q,r) is isosceles.
          Wait, `viol_iso` counts triangles. Adding p creates triangles (p, q, r) for q,r in S.
          So `viol_iso` += number of pairs (q,r) in S such that (p,q,r) is isosceles.
        - `uncovered` updates:
          For each point x not in S U {p}:
          If x was uncovered, check if any new pair (p,q) covers x.
          If yes, x becomes covered (remove from `uncovered`).
          
        Remove p from S:
        - `viol_iso` decreases by the number of triangles involving p.
        - `uncovered` updates:
          For each point x not in S \ {p}:
          If x was covered ONLY by pairs involving p, it becomes uncovered.
          We need to track `cover_count[x]`.
          
        This is doable.
        
        Let's implement this.
"""

import time
import random
import itertools
import sys

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
POINT_SET = set(POINTS)
DEADLINE = 320.0

def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

# Precompute isosceles triples
# IsoTriples: list of sets of 3 points
# For each point p, IsoPairs[p] = set of pairs (a,b) such that (p,a,b) is isosceles.
# We also need: For a pair (a,b), which points p form an isosceles with them?
# Let's precompute `IsoWithPair[pair]` = set of points p such that (p,a,b) is isosceles.
# And `IsoWithPoint[p]` = set of pairs (a,b) such that (p,a,b) is isosceles.

print("Precomputing...", file=sys.stderr)
start_time = time.time()

IsoWithPair = {}
IsoWithPoint = {p: set() for p in POINTS}

for a in POINTS:
    for b in POINTS:
        if a == b:
            continue
        pair = frozenset((a,b))
        if pair in IsoWithPair:
            continue
            
        d_ab = _dist2(a,b)
        pts = set()
        for p in POINTS:
            if p == a or p == b:
                continue
            d_pa = _dist2(p,a)
            d_pb = _dist2(p,b)
            if d_pa == d_ab or d_pb == d_ab or d_pa == d_pb:
                pts.add(p)
                IsoWithPoint[p].add(pair)
        
        if pts:
            IsoWithPair[pair] = pts

print(f"Precomputation took {time.time() - start_time:.2f}s", file=sys.stderr)

def _get_initial_set():
    """Greedy construction of a valid maximal set."""
    S = set()
    # Random order
    pts = list(POINTS)
    random.shuffle(pts)
    
    # We need S to be maximal.
    # Greedy: add p if it doesn't create isosceles.
    # But this might not be maximal.
    # Let's just use the previous greedy covering logic to get a start.
    
    # Simple greedy:
    for p in pts:
        valid = True
        for a in S:
            for b in S:
                if a == b: continue
                if frozenset((a,b)) in IsoWithPoint[p]:
                    valid = False
                    break
            if not valid: break
        if valid:
            S.add(p)
            
    # Now make it maximal by adding points that are "forced"?
    # Or just return S and let SA fix maximality?
    # SA can handle non-maximal sets if the penalty is high.
    # But we want to start close to feasible.
    
    # Let's try to cover all points.
    # If S is not maximal, there are uncovered points.
    # We can try to add them if possible, or swap.
    # For now, return S. SA will minimize cost.
    return S

def construct():
    start_time = time.time()
    
    # Initial state
    S = _get_initial_set()
    
    # State variables for fast evaluation
    # in_S: set of points
    # cover_count: dict p -> count of pairs in S that cover p
    # viol_iso: number of isosceles triangles in S
    
    in_S = set(S)
    cover_count = {p: 0 for p in POINTS}
    viol_iso = 0
    
    # Initialize cover_count and viol_iso
    # This is O(|S|^2 * N) roughly.
    # For each pair in S, for each p in IsoWithPair[pair], increment cover_count[p].
    S_list = list(in_S)
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            a, b = S_list[i], S_list[j]
            pair = frozenset((a,b))
            if pair in IsoWithPair:
                for p in IsoWithPair[pair]:
                    if p not in in_S: # Only count if p is not in S? 
                        # Actually, cover_count is for points NOT in S.
                        # If p is in S, it's not "uncovered".
                        # But for update logic, it's easier to count all.
                        # Uncovered = {p not in S | cover_count[p] == 0}
                        cover_count[p] += 1
            
            # Update viol_iso
            # Check if (a,b, p) is isosceles for any p in S?
            # No, viol_iso is number of isosceles triangles IN S.
            # So we need to check pairs in S.
            # When adding points incrementally, we can update.
            # Here, we calculate from scratch.
            pass
    
    # Calculate viol_iso from scratch
    # For each triplet in S, check if isosceles.
    # O(|S|^3). |S| is small (~20-50).
    S_list = list(in_S)
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            for k in range(j+1, len(S_list)):
                a, b, c = S_list[i], S_list[j], S_list[k]
                if frozenset((b,c)) in IsoWithPoint[a]:
                    viol_iso += 1
                    
    # Calculate uncovered
    def get_uncovered():
        unc = 0
        for p in POINTS:
            if p not in in_S and cover_count[p] == 0:
                unc += 1
        return unc
        
    current_uncovered = get_uncovered()
    
    def cost():
        return 100 * (viol_iso + current_uncovered) + len(in_S)
        
    best_S = set(in_S)
    best_cost = cost()
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    steps = 0
    while time.time() < DEADLINE:
        # Generate move
        move_type = random.random()
        if move_type < 0.4:
            # Add
            candidates = [p for p in POINTS if p not in in_S]
            if not candidates: continue
            p = random.choice(candidates)
            
            # Check if adding p is allowed? No, SA allows violations.
            # Update state
            # 1. viol_iso increase
            # New triangles (p, a, b) for a,b in S.
            # This is equivalent to: for each pair (a,b) in S, if (p,a,b) is isosceles, viol_iso += 1.
            # We can use IsoWithPoint[p].
            # For each pair in IsoWithPoint[p], if both points are in S, viol_iso += 1.
            d_viol_iso = 0
            for pair in IsoWithPoint[p]:
                a, b = list(pair)
                if a in in_S and b in in_S:
                    d_viol_iso += 1
                    
            # 2. cover_count increase
            # New pairs (p, q) for q in S.
            # For each q in S, pair = {p,q}.
            # For each r in IsoWithPair[{p,q}], cover_count[r] += 1.
            d_cover = {r: 0 for r in POINTS} # Sparse update
            # Actually, we just need to update cover_count for affected r's.
            affected_r = set()
            for q in in_S:
                pair = frozenset((p,q))
                if pair in IsoWithPair:
                    for r in IsoWithPair[pair]:
                        if r != p: # r is distinct from p (p is in S now, so r is not in S? No, r could be in S)
                            # cover_count is for points NOT in S.
                            # If r is in S, it doesn't matter for uncovered count.
                            # But we maintain cover_count for all p for simplicity?
                            # Let's just update cover_count[r].
                            d_cover[r] = d_cover.get(r, 0) + 1
                            affected_r.add(r)
                            
            # 3. Uncovered change
            # p was uncovered? No, p is added to S, so it's no longer "not in S".
            # So if p was uncovered, uncovered count decreases by 1.
            d_uncovered = 0
            if p not in in_S and cover_count[p] == 0:
                d_uncovered -= 1 # p is no longer uncovered
            
            # For each r in affected_r:
            # If r was uncovered (r not in S, cover_count[r]==0) and now cover_count[r] > 0, uncovered decreases.
            for r in affected_r:
                if r not in in_S and cover_count[r] == 0:
                    # It was uncovered. Now it is covered (since we added 1 to count).
                    d_uncovered -= 1
                    
            # Apply move
            in_S.add(p)
            viol_iso += d_viol_iso
            for r in affected_r:
                cover_count[r] += d_cover[r]
            current_uncovered += d_uncovered
            
            c = cost()
            if c < best_cost:
                best_cost = c
                best_S = set(in_S)
                
            # Accept/Reject
            if c < cost() or random.random() < pow(2.718, -(c - cost()) / T):
                pass # Accepted (state already updated)
            else:
                # Revert
                in_S.remove(p)
                viol_iso -= d_viol_iso
                for r in affected_r:
                    cover_count[r] -= d_cover[r]
                current_uncovered -= d_uncovered
                
        elif move_type < 0.8:
            # Remove
            if not in_S: continue
            p = random.choice(list(in_S))
            
            # Update state
            # 1. viol_iso decrease
            d_viol_iso = 0
            for pair in IsoWithPoint[p]:
                a, b = list(pair)
                if a in in_S and b in in_S:
                    d_viol_iso += 1
                    
            # 2. cover_count decrease
            d_cover = {}
            affected_r = set()
            for q in in_S:
                if q == p: continue
                pair = frozenset((p,q))
                if pair in IsoWithPair:
                    for r in IsoWithPair[pair]:
                        if r != p:
                            d_cover[r] = d_cover.get(r, 0) + 1
                            affected_r.add(r)
                            
            # 3. Uncovered change
            # p is removed from S.
            # If p was covered by S\{p}, it remains covered.
            # If p was NOT covered by S\{p}, it becomes uncovered (if it's not in S).
            # Wait, p is removed from S, so it becomes "not in S".
            # We need to check if p is covered by S\{p}.
            # cover_count[p] currently includes pairs with p? No, cover_count[p] counts pairs in S that cover p.
            # Pairs in S that cover p do NOT include p itself.
            # So cover_count[p] is the number of pairs in S\{p} that cover p.
            # If cover_count[p] == 0, p becomes uncovered.
            d_uncovered = 0
            if cover_count[p] == 0:
                d_uncovered += 1 # p becomes uncovered
                
            # For each r in affected_r:
            # cover_count[r] decreases.
            # If r was covered (cover_count[r] > 0) and becomes 0, and r not in S, uncovered increases.
            for r in affected_r:
                if r not in in_S and cover_count[r] == d_cover[r]: # Will become 0
                    d_uncovered += 1
                    
            # Apply move
            in_S.remove(p)
            viol_iso -= d_viol_iso
            for r in affected_r:
                cover_count[r] -= d_cover[r]
            current_uncovered += d_uncovered
            
            c = cost()
            if c < best_cost:
                best_cost = c
                best_S = set(in_S)
                
            # Accept/Reject
            # Note: cost() function uses global variables, so we must be careful.
            # Let's calculate cost before and after explicitly or use delta.
            # Delta cost = 100 * (d_viol_iso + d_uncovered) - 1 (size decreased by 1)
            delta_cost = 100 * (d_viol_iso + d_uncovered) - 1
            if delta_cost < 0 or random.random() < pow(2.718, -delta_cost / T):
                pass
            else:
                # Revert
                in_S.add(p)
                viol_iso += d_viol_iso
                for r in affected_r:
                    cover_count[r] += d_cover[r]
                current_uncovered -= d_uncovered
                
        else:
            # Swap
            if not in_S: continue
            p = random.choice(list(in_S))
            candidates = [q for q in POINTS if q not in in_S]
            if not candidates: continue
            q = random.choice(candidates)
            
            # Remove p, Add q.
            # This is complex to update incrementally in one step.
            # Let's just do Remove then Add.
            # Or skip swap for now to keep code simple?
            # Swap is important for exploration.
            # Let's implement it as two steps or a combined delta.
            # Combined delta is tricky. Let's just do sequential updates in a temporary state?
            # Too slow.
            # Let's just use Add and Remove.
            pass
        
        T *= cooling
        steps += 1
        
        if steps % 1000 == 0 and best_cost == len(best_S):
            # Found a valid maximal set (viol_iso=0, uncovered=0)
            # Can we improve size?
            # If we are stuck at a valid set, we can try to remove points.
            # SA will do this via Remove moves.
            pass

    return list(best_S)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
