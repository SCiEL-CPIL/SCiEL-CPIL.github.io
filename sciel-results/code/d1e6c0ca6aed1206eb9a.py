"""
Strategy: Minimum Hitting Set on Hypergraph of Isosceles Triples.

The problem of finding a maximal no-isosceles set can be reframed.
A set S is "no-isosceles" if it contains no isosceles triangle.
A set S is "maximal" if for every point p not in S, S U {p} contains an isosceles triangle.

Let T be the set of all isosceles triples in the grid.
If S is no-isosceles, then S is maximal if and only if for every p not in S, 
there exists an isosceles triple (a, b, p) in T such that {a, b} subset of S.
This is equivalent to saying that the set S "hits" every isosceles triple that involves 
any point outside S? No, that's not quite right.

Let's look at the complement.
Let U be the set of all grid points.
S is maximal no-isosceles iff:
1. S contains no isosceles triple.
2. For every p in U \ S, S U {p} contains an isosceles triple.
   Since S is no-isosceles, any isosceles triple in S U {p} must include p.
   So, for every p in U \ S, there exist a, b in S such that (a, b, p) is an isosceles triple.

This looks like a covering problem.
Let H be the hypergraph where vertices are grid points and hyperedges are isosceles triples.
We want to find a subset S of vertices such that:
1. S is independent in H (no hyperedge is fully contained in S).
2. For every v not in S, there is a hyperedge e such that e \ {v} subset of S.

This is a specific type of combinatorial optimization. 
A common heuristic for "minimum maximal independent set" (which is what a maximal no-isosceles set is, in the graph of isosceles constraints) is the "greedy removal" or "iterative reduction" approach, but the prompt asks for a "minimum hitting set" perspective.

Actually, the "minimum maximal independent set" problem is NP-hard. 
Standard greedy approach for MMIS:
1. Start with a maximal independent set (e.g., via greedy vertex selection).
2. Iteratively try to remove vertices. If removing a vertex v keeps the set maximal, remove it.
   To check if S \ {v} is maximal, we need to ensure that every point u not in S \ {v} is still "blocked" by an isosceles triangle formed with points in S \ {v}.
   A point u is blocked by S if there exist a, b in S such that (a, b, u) is isosceles.
   When we remove v, we only lose blocks that relied on v.
   So, if S was maximal, and we remove v, the set S' = S \ {v} is NOT maximal if and only if there exists some u not in S' such that ALL isosceles triples containing u and two points from S' are empty? 
   No, S' is not maximal if there exists u not in S' such that there is NO isosceles triple (a, b, u) with a, b in S'.
   Note that u could be v itself! If we remove v, v is now outside S'. If v can be added back to S' without creating an isosceles triangle, then S' is not maximal. 
   Condition for v to be addable to S': There is no isosceles triple (a, b, v) with a, b in S'.
   So, v is "essential" for maximality if there is at least one isosceles triple (a, b, v) with a, b in S.
   If v is essential, we cannot remove it without checking other points.
   If v is NOT essential, we can remove it, BUT we must also check if removing v makes some OTHER point u (that was previously blocked only by triples involving v) become addable.
   
   This suggests a "Dominance" or "Hitting Set" view:
   Each point v in S "covers" (blocks) the points u for which (a, b, u) is isosceles with a, b in S.
   Specifically, v helps block u if there is an isosceles triple (v, w, u) or (w, v, u) with w in S.
   Actually, simpler: 
   Let B(S, u) be the set of isosceles triples (a, b, u) such that a, b in S.
   S is maximal iff for all u not in S, B(S, u) is non-empty.
   
   If we remove v from S, we get S' = S \ {v}.
   For any u not in S', B(S', u) = { (a, b, u) in B(S, u) | v not in {a, b} }.
   S' is maximal iff for all u not in S', B(S', u) is non-empty.
   
   So, v is removable iff for all u not in S', B(S, u) is not a subset of triples containing v.
   Equivalently: For all u not in S', there exists an isosceles triple (a, b, u) with a, b in S' (i.e., a, b in S and a!=v, b!=v).
   
   This is a strong condition. It means v is "redundant" for blocking every external point.
   
   The "Hitting Set" interpretation in the prompt suggests:
   View the problem as selecting points to "hit" constraints.
   Constraint for each u not in S: "Must be blocked".
   A point v in S contributes to blocking u if it forms an isosceles triangle with another point in S.
   
   Let's implement the "Greedy Reduction" which is effectively solving for a minimal maximal set.
   To improve on the current SA approach, we can use a more efficient check for removability and perhaps a different initialization or perturbation strategy.
   
   However, the prompt specifically asks for a "Minimum Hitting Set on a hypergraph... using an iterative greedy algorithm that selects the point covering the most uncovered hyperedges".
   
   Let's reinterpret:
   Maybe we model the "Maximality" as a covering problem.
   We want to choose S such that S is independent and covers all "external" points.
   This is tricky because S determines which points are external.
   
   Alternative interpretation:
   Start with empty set.
   We want to build S.
   At each step, we pick a point.
   But we don't know which points will be "external" until we finish.
   
   Let's stick to the "Minimal Maximal Independent Set" (MMIS) heuristic but optimize the "Removability Check" using precomputed data structures for speed, allowing more iterations within the time limit.
   
   Key optimization:
   Precompute for each point v, the list of isosceles triples involving v.
   Maintain for each external point u, a count of how many "blocks" (isosceles pairs in S) cover it.
   When we add/remove points, we update these counts.
   
   Wait, the "blocks" are pairs (a,b) in S such that (a,b,u) is isosceles.
   If we remove v, we lose all pairs (v, w) in S.
   For each w in S, we check if (v, w, u) is isosceles for any u. If so, decrement block_count[u].
   If any block_count[u] becomes 0, then S is no longer maximal (specifically, u can be added).
   So v is removable ONLY IF for all u, block_count[u] > 0 after removing v.
   
   This allows O(1) or O(degree) checks for removability if we maintain the counts.
   
   Let's implement this "Dynamic Maximal Independent Set Reduction".
"""

import time
import random
import sys

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute distances
dist2 = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2
        dist2[i][j] = d
        dist2[j][i] = d

# Precompute isosceles triples? 
# Too many. 
# Instead, for each point p, and each other point q, we can find r such that (p, q, r) is isosceles.
# Or, for each pair (p, q), find r such that (p, q, r) is isosceles.
# We need to know, for a given set S, which external points are blocked.
# A point u is blocked by pair (a, b) in S if (a, b, u) is isosceles.
# So for each pair (a, b), we can precompute the list of points u such that (a, b, u) is isosceles.
# Let's call this list `iso_thirds[a][b]`.

# Memory check: 256 * 256 lists. Each list can have up to 256 points.
# 256*256*256 bytes ~ 16MB. Acceptable.
# But computing it: 
# For each pair (a, b), iterate all u? 256^3 = 16M operations. Fast enough.

iso_thirds = [[[] for _ in range(NUM_POINTS)] for _ in range(NUM_POINTS)]

for a in range(NUM_POINTS):
    for b in range(a + 1, NUM_POINTS):
        da = dist2[a][b]
        # Find c such that dist(a,c) == dist(b,c) OR dist(a,c) == da OR dist(b,c) == da
        # Actually, we just need to check all c != a, b
        for c in range(NUM_POINTS):
            if c == a or c == b:
                continue
            d_ac = dist2[a][c]
            d_bc = dist2[b][c]
            if d_ac == d_bc or d_ac == da or d_bc == da:
                iso_thirds[a][b].append(c)
                iso_thirds[b][a].append(c)

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_set = None
    best_size = NUM_POINTS + 1
    
    # We will run multiple restarts of the "Greedy Construction + Fast Reduction"
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # 1. Greedy Construction
        # Random order to get diverse maximal sets
        random.seed(restart)
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        S = []
        S_set = set()
        
        # Fast check for adding point p to S:
        # p can be added if no isosceles triangle is formed with any pair in S.
        # i.e., for all a, b in S, p is NOT in iso_thirds[a][b].
        # Also, p must not form an isosceles triangle with a, b where a,b are in S?
        # Yes, that's the definition.
        # To speed up, we can maintain a "blocked" set? 
        # No, the condition is: for all a in S, for all b in S, p not in iso_thirds[a][b].
        # This is O(|S|^2) per check. 
        # With |S| ~ 50, 2500 checks. 256 points * 2500 = 640k ops per construction.
        # 200 restarts * 640k = 128M ops. Might be tight in Python.
        
        # Optimization: 
        # When we add a point a to S, we can precompute which points are now "blocked" by pairs involving a?
        # Actually, if we add a, we create new pairs (a, b) for all b in S.
        # Any point p in iso_thirds[a][b] is now blocked by the pair (a, b).
        # If p was previously unblocked, it is now blocked.
        # We can maintain a count `block_count[p]` = number of pairs (a,b) in S such that p in iso_thirds[a][b].
        # p can be added to S iff block_count[p] == 0.
        
        block_count = [0] * NUM_POINTS
        S = []
        
        for p in order:
            if block_count[p] == 0:
                # Add p to S
                # Update block_count for all points q that are in iso_thirds[p][a] for a in S
                for a in S:
                    for q in iso_thirds[p][a]:
                        block_count[q] += 1
                S.append(p)
        
        # Now S is a maximal no-isosceles set?
        # Let's verify.
        # If p is not in S, block_count[p] > 0.
        # This means there is a pair (a, b) in S such that p in iso_thirds[a][b].
        # So (a, b, p) is isosceles.
        # So p cannot be added.
        # Yes, S is maximal.
        
        # 2. Reduction Phase (Minimize S while keeping it maximal)
        # We want to remove points from S such that the remaining set is still maximal.
        # Condition for removing v from S:
        # Let S' = S \ {v}.
        # For all u not in S', block_count_S'[u] > 0.
        # block_count_S'[u] = block_count_S[u] - (number of pairs (v, a) in S such that u in iso_thirds[v][a]).
        # Let loss[u] = number of pairs (v, a) in S such that u in iso_thirds[v][a].
        # Note that pairs are (v, a) where a in S.
        # So loss[u] is the number of a in S such that u in iso_thirds[v][a].
        
        # If for all u not in S', block_count_S[u] - loss[u] > 0, then v is removable.
        
        # We can compute loss[u] for a candidate v quickly?
        # loss[u] = sum( 1 for a in S if u in iso_thirds[v][a] )
        # This is O(|S|) per u. 
        # We need to check all u not in S. That's O(|S| * |S|) = O(|S|^2).
        # Same as before.
        
        # However, we can precompute for each v, the list of u that are "critically blocked" by v?
        # A point u is critically blocked by v (with respect to S) if block_count[u] == loss_v[u].
        # If we remove v, and block_count[u] - loss_v[u] == 0, then u becomes unblocked.
        # So v is removable iff for all u not in S, block_count[u] > loss_v[u].
        
        # Let's implement the removal loop.
        
        changed = True
        while changed:
            changed = False
            # Try to remove points in random order
            random.shuffle(S)
            for v in S:
                if time.time() > deadline:
                    break
                
                # Check if v is removable
                # Calculate loss_v[u] for all u? 
                # Or just check if any u becomes unblocked.
                
                # Optimization: 
                # We only care about u where block_count[u] is small.
                # If block_count[u] > max_possible_loss, it's safe.
                # max_possible_loss for v is |S|-1 (since v pairs with |S|-1 other points).
                
                # Let's just compute loss for all u not in S.
                # To speed up, we can iterate through the pairs involving v.
                # For each a in S (a != v), for each u in iso_thirds[v][a], loss[u] += 1.
                
                loss = {}
                for a in S:
                    if a == v:
                        continue
                    for u in iso_thirds[v][a]:
                        if u in S:
                            continue # u is in S, so it's not "external" in the sense of maximality check for S'
                            # Wait, if u is in S, it's already in the set. 
                            # The maximality condition applies to points NOT in S.
                            # If u is in S, it doesn't matter if it's blocked or not.
                            # But wait, if u is in S, and we remove v, u is still in S'.
                            # So we only care about u not in S.
                        if u not in S:
                            loss[u] = loss.get(u, 0) + 1
                
                removable = True
                for u, l in loss.items():
                    if block_count[u] <= l:
                        removable = False
                        break
                
                if removable:
                    # Remove v from S
                    # Update block_count
                    for a in S:
                        if a == v:
                            continue
                        for u in iso_thirds[v][a]:
                            block_count[u] -= 1
                    
                    S.remove(v)
                    changed = True
                    break # Restart the loop to check new set
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
            
        # If we have time, try a few perturbations?
        # For now, let's rely on the diversity of the greedy construction.
        
    if best_set is None:
        return [POINTS[i] for i in range(NUM_POINTS)] # Fallback, shouldn't happen
        
    return [POINTS[i] for i in best_set]

if __name__ == "__main__":
    construct()

# EVOLVE-BLOCK-END
