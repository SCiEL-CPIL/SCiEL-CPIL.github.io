"""Improved program for minimum maximal no-isosceles set."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import sys

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)

# Precomputation
# all_points: list of (x, y)
all_points = list(itertools.product(range(N), repeat=2))
point_index = {p: i for i, p in enumerate(all_points)}
num_points = len(all_points)

# Precompute isosceles relations
# iso_pairs[p] = set of indices q such that p, q are part of an isosceles triple with some r?
# Actually, we need: given p and a, what b's make (p, a, b) isosceles?
# Let's precompute a 3D boolean or just a set of sets.
# iso_conflict[p][a] = set of b such that (p, a, b) is isosceles.
# This is 256 * 256 * 256 = 16M entries. Might be heavy but manageable in memory?
# 16M booleans is 16MB. OK.
# Or, for each p, store a set of (a, b) pairs?
# Let's compute:
# For each triple (p, a, b), if isosceles, mark it.
# We want fast lookup: "Does adding p to chosen block any new points?"
# When we add p, a point q is blocked if there exists a in chosen such that (p, a, q) is isosceles.
# So for a fixed p and a, we need the set of q's that form isosceles with p and a.
# Let's create `iso_q[p][a]` as a set of q indices.

iso_q = [[set() for _ in range(num_points)] for _ in range(num_points)]

# Calculate
for i in range(num_points):
    pi = all_points[i]
    for j in range(num_points):
        pj = all_points[j]
        if i == j:
            continue
        # Find all k such that (pi, pj, pk) is isosceles
        # d(pi, pj) == d(pi, pk) or d(pi, pj) == d(pj, pk) or d(pi, pk) == d(pj, pk)
        d_ij = (pi[0]-pj[0])**2 + (pi[1]-pj[1])**2
        for k in range(num_points):
            if k == i or k == j:
                continue
            pk = all_points[k]
            d_ik = (pi[0]-pk[0])**2 + (pi[1]-pk[1])**2
            d_jk = (pj[0]-pk[0])**2 + (pj[1]-pk[1])**2
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                iso_q[i][j].add(k)
                iso_q[j][i].add(k)
                iso_q[k][i].add(j)
                iso_q[k][j].add(i)

# Note: The above loop is O(N^3) = 16^3 = 4096 iterations? No, 256^3 = 16,777,216.
# This might take a few seconds in Python.
# Let's optimize the precomputation or do it on the fly?
# 16M iterations in pure Python is ~1-2 seconds. Acceptable for 300s limit.

def construct():
    best_size = float('inf')
    best_set = None
    
    # Heuristic lower bound for k
    # A random greedy set has size ~ 256 / (1 + avg_degree)?
    # Let's just start from k=5 and go up.
    
    # To make it faster, we can use a heuristic to pick the next point.
    # In backtracking, we need to choose which point to try adding.
    
    def solve(k, chosen, blocked, available):
        nonlocal best_size, best_set
        
        if len(chosen) > k:
            return None
        
        # If we have chosen k points, check if it's maximal
        if len(chosen) == k:
            if not available:
                # Maximal set found
                return list(chosen)
            else:
                return None

        # If available is empty, we have a maximal set, but size < k
        # So this branch doesn't yield a set of size k
        if not available:
            return None

        # Pruning: if even adding all available points doesn't reach k, fail
        if len(chosen) + len(available) < k:
            return None
            
        # Heuristic: Pick a point from available
        # To find a small set, we want to block as much as possible?
        # In backtracking for size k, we just need *a* solution.
        # Order doesn't matter for correctness, but for speed.
        # Let's pick the point that blocks the most available points.
        
        best_p = -1
        max_blocks = -1
        # Sample a few points to avoid O(N^2) per node?
        # Available size is at most 256.
        # For each p in available, count how many q in available are in iso_q[p][a] for a in chosen.
        # This is expensive.
        # Let's just pick the first one or random one to keep it simple?
        # Random might get stuck in deep branches.
        # Let's pick the one with highest "potential" to block.
        
        # Simplified heuristic: Just pick the first available point.
        # To improve, we can sort available by some static metric.
        
        p = next(iter(available))
        
        # Branch 1: Add p
        new_chosen = set(chosen)
        new_chosen.add(p)
        
        new_blocked = set(blocked)
        # Add p to blocked? No, p is in chosen.
        # Points blocked by p:
        # A point q is blocked if there exists a in chosen (including p?) 
        # Definition: q is blocked if (q, a, b) is isosceles for some a,b in chosen.
        # So if we add p, new pairs are (p, a) for a in old_chosen.
        # Also (p, p) is not a pair.
        # So we need to check q for each a in old_chosen: if q in iso_q[p][a], then q is blocked.
        
        for a in chosen:
            new_blocked.update(iso_q[p][a])
        
        # Also, p itself is in chosen, so it's not available.
        # What about points blocked by (p, p)? No such thing.
        
        new_available = available - new_blocked - new_chosen
        
        res = solve(k, new_chosen, new_blocked, new_available)
        if res:
            return res
            
        # Branch 2: Do not add p
        # p remains available? No, we must decide.
        # If we don't add p, p must be blocked by future additions.
        # So we just remove p from available?
        # No, if we don't add p, p is NOT in chosen.
        # Is p blocked? Only if it was already blocked. But p was in available, so it was NOT blocked.
        # So if we don't add p, and we don't block it later, the final set won't be maximal.
        # So we MUST ensure p gets blocked later.
        # This constraint is hard to propagate simply.
        
        # Alternative view:
        # We are searching for a set S of size k such that S is maximal.
        # Maximal means: for all q not in S, q is blocked by S.
        
        # If we skip p (don't add to S), then p is not in S.
        # So p MUST be blocked by S.
        # This means there must be some pair (a, b) in S such that (p, a, b) is isosceles.
        
        # This is a global constraint.
        
        # Let's simplify: Just try adding p.
        # If we only try adding, we will only find sets that are "greedy-like" in the backtracking order?
        # No, if we only add, we will never skip a point.
        # Then S will be the set of points we picked.
        # If we pick k points, and they block all others, we are done.
        # But what if we need to skip a point to make room for a better one?
        # Or what if we need to skip a point because it doesn't help block others?
        
        # Actually, if we just greedily add points, we might end up with a set that is NOT maximal (if some points are not blocked).
        # But if we stop at k points, and some are not blocked, it's invalid.
        
        # The "Do not add p" branch is crucial.
        # But how to handle the "must be blocked" constraint?
        
        # Let's try a different tactic:
        # Generate a candidate set of size k using a heuristic greedy, then check if maximal.
        # If not, try another permutation.
        
        # This is basically what the current program does, but with fixed size k?
        
        # Let's stick to the random greedy but with a twist:
        # Use the `iso_q` data to make a "smart" random choice.
        
        pass # Placeholder for the actual implementation below

    # Let's implement a "Smart Greedy" instead of full backtracking for now, as it's more robust.
    # Heuristic: At each step, pick the point from available that blocks the most other available points.
    
    def smart_greedy():
        chosen = set()
        blocked = set()
        available = set(range(num_points))
        
        while available:
            # Pick p in available that maximizes number of blocked points
            best_p = -1
            max_new_blocks = -1
            
            # To speed up, don't check all 256 every time?
            # Available shrinks.
            for p in available:
                # Count how many q in available are blocked if we add p
                # q is blocked if q in iso_q[p][a] for some a in chosen
                new_blocks = set()
                for a in chosen:
                    new_blocks.update(iso_q[p][a])
                # Intersect with available to see how many we remove
                count = len(new_blocks & available)
                if count > max_new_blocks:
                    max_new_blocks = count
                    best_p = p
            
            if best_p == -1:
                break # Should not happen if available is not empty
            
            chosen.add(best_p)
            # Update blocked
            for a in chosen:
                blocked.update(iso_q[best_p][a])
            
            available -= blocked
            available -= chosen
            
        # Check if maximal
        # By construction, available is empty, so it is maximal.
        return list(chosen)

    # Run smart greedy multiple times with different random initializations?
    # The "pick best" is deterministic.
    # To vary, we can add random noise to the "count".
    
    best_size = float('inf')
    best_set = None
    
    for _ in range(100):
        # Perturb the heuristic
        # We can't easily perturb the deterministic "max blocks" without changing the logic.
        # Let's just run it once?
        # If it's deterministic, it gives one result.
        
        # Let's combine with random:
        # Pick top 3 candidates, choose one randomly.
        
        chosen = set()
        blocked = set()
        available = set(range(num_points))
        
        while available:
            candidates = []
            for p in available:
                new_blocks = set()
                for a in chosen:
                    new_blocks.update(iso_q[p][a])
                count = len(new_blocks & available)
                candidates.append((count, p))
            
            # Sort by count descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Pick from top 5
            top_k = min(5, len(candidates))
            _, best_p = random.choice(candidates[:top_k])
            
            chosen.add(best_p)
            for a in chosen:
                blocked.update(iso_q[best_p][a])
            
            available -= blocked
            available -= chosen
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_set = [all_points[i] for i in chosen]
            
    return best_set
# EVOLVE-BLOCK-END
