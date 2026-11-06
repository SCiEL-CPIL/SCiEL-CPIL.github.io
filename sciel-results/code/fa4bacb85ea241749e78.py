"""
Maximum independent set in a 3-uniform hypergraph via branch-and-bound with bitset
conflict tracking. The hypergraph vertices are [1,128]; a hyperedge is a bad triple
(a,b,c) where lcm(a,b)==lcm(a,c)==lcm(b,c). We want the largest subset with no
hyperedge fully contained.

Strategy:
- Precompute all bad triples and, for each vertex x, a bitset of vertices y such that
  there exists a bad triple containing both x and y (i.e., x and y cannot both be in
  an independent set if some third vertex z is also present... actually we need a
  different formulation).

Actually, for a 3-uniform hypergraph independent set (no hyperedge fully contained),
the constraint is: for every bad triple (a,b,c), NOT (a in S AND b in S AND c in S).

This is equivalent to: S is a valid solution iff for every bad triple, at least one
element is excluded.

Branch-and-bound:
- State: (candidates set, current selected set, current size)
- At each step, pick a vertex from candidates. Branch: include it or exclude it.
- If we include vertex x, we must ensure that no bad triple has x and two other
  already-selected vertices. Actually, we need to track which pairs of selected
  vertices would complete a bad triple with x.

Better formulation using the "forbidden pairs" structure:
- If x is in S, then for every pair (a,b) in forbidden_pairs[x], we cannot have both
  a and b in S.

So the constraint is: for each x in S, the selected set must not contain any pair from
forbidden_pairs[x].

This means: if we build S incrementally, when we add x, we need to check that for every
(a,b) in forbidden_pairs[x], not both a and b are already in S.

For branch-and-bound with bitsets:
- Represent the candidate set as a bitset (Python int with 128 bits).
- When we decide to include x, we need to "forbid" certain pairs. But this is complex.

Alternative: Use the fact that the problem is small (128 vertices). We can use a
recursive branch-and-bound where:
- We maintain a set of "active" candidates.
- We pick the vertex with highest degree (most constraints) to branch on.
- Upper bound: current_size + number of remaining candidates that can still be added
  (using a greedy coloring or packing bound).

Let me think about a cleaner approach. Since n=128, we can use bitmasks (Python ints).

For each vertex x, let conflict_mask[x] be a bitmask where bit y is set if there exists
a bad triple containing both x and y. Wait, that's not quite right either.

Actually, the constraint is: for any bad triple (a,b,c), we cannot have all three in S.
Equivalently, for each x in S, the set S \ {x} must not contain any pair (a,b) such that
(a,b,x) is a bad triple.

So when adding x to a partial solution P, we need: for all (a,b) in forbidden_pairs[x],
not (a in P and b in P).

For branch-and-bound:
- State: (P, C) where P is the set of selected vertices, C is the set of candidate vertices
  not yet decided.
- Pick a vertex v from C.
- Branch 1: Include v. Check if v is compatible with P (i.e., no pair in forbidden_pairs[v]
  is fully in P). If compatible, new state is (P ∪ {v}, C \ {v}).
- Branch 2: Exclude v. New state is (P, C \ {v}).
- Upper bound: |P| + |C| (trivial) or better: |P| + (number of vertices in C that are
  compatible with P and with each other in a greedy sense).

A better upper bound: For the remaining candidates C, compute a greedy independent set
size (in the induced hypergraph) and add to |P|. But computing that is expensive.

Simpler upper bound: |P| + |C| is too loose. Let's use: |P| + max_independent_set_bound(C).

For a 3-uniform hypergraph, a simple bound is: |C| - (number of edges in C) / 2, but
that's not tight.

Alternative: Use a greedy coloring bound. Color the vertices of the induced subgraph on C
where two vertices are adjacent if they appear together in some bad triple within C. Then
the independence number of the hypergraph is at most the chromatic number of this graph...
no, that's not right either.

Let me try a different approach: since n=128 is small, I'll use a recursive search with
memoization on the candidate set (as a bitmask). The state is just the candidate set, and
we track the best solution found so far.

Actually, let's use a simpler but effective approach:
1. Precompute all bad triples.
2. For each vertex x, compute forbidden_pairs[x] as a list of pairs.
3. Use branch-and-bound with:
   - Ordering: pick the vertex with the most constraints first.
   - Pruning: if current_size + remaining_count <= best_found, prune.
   - Feasibility check: when adding x, verify no forbidden pair is fully in current set.

Let me implement this with bitmasks for efficiency.
"""
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    n = 128
    universe = list(range(1, n + 1))

    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    # Find all bad triples
    from collections import defaultdict
    lcm_to_pairs = defaultdict(list)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = lcm_table[i][j]
            lcm_to_pairs[l].append((i, j))

    bad_triples = []
    for l, pairs in lcm_to_pairs.items():
        # Candidates are divisors of l
        candidates = [x for x in range(1, n + 1) if l % x == 0]
        if len(candidates) < 3:
            continue
        for idx_a in range(len(candidates)):
            a = candidates[idx_a]
            for idx_b in range(idx_a + 1, len(candidates)):
                b = candidates[idx_b]
                if lcm_table[a][b] != l:
                    continue
                for idx_c in range(idx_b + 1, len(candidates)):
                    c = candidates[idx_c]
                    if lcm_table[a][c] == l and lcm_table[b][c] == l:
                        bad_triples.append((a, b, c))

    # Build forbidden_pairs: for each x, list of (a,b) such that {a,b,x} is a bad triple
    forbidden_pairs = [[] for _ in range(n + 1)]
    for (a, b, c) in bad_triples:
        forbidden_pairs[c].append((a, b))
        forbidden_pairs[a].append((b, c))
        forbidden_pairs[b].append((a, c))

    # Convert to sets for O(1) membership, and also create bitmask representation
    forbidden_pairs_set = [set(fp) for fp in forbidden_pairs]

    # For each x, create a bitmask of "conflict partners": vertices that appear in any
    # forbidden pair with x. This helps with a looser but faster bound.
    conflict_mask = [0] * (n + 1)
    for x in range(1, n + 1):
        for (a, b) in forbidden_pairs[x]:
            conflict_mask[x] |= (1 << (a - 1)) | (1 << (b - 1))

    # Constraint degree
    constraint_degree = [0] * (n + 1)
    for x in range(1, n + 1):
        constraint_degree[x] = len(forbidden_pairs[x])

    # Greedy baseline
    def greedy(order):
        selected = set()
        for x in order:
            ok = True
            for (a, b) in forbidden_pairs[x]:
                if a in selected and b in selected:
                    ok = False
                    break
            if ok:
                selected.add(x)
        return selected

    # Get initial solutions from multiple greedy orderings
    strategies = [
        lambda: sorted(universe, key=lambda x: (constraint_degree[x], x)),
        lambda: sorted(universe, key=lambda x: (-constraint_degree[x], x)),
        lambda: list(reversed(universe)),
        lambda: sorted(universe, key=lambda x: (x % 2, x)),
        lambda: sorted(universe, key=lambda x: (-x.bit_count(), x)),
        lambda: sorted(universe, key=lambda x: (x.bit_count(), x)),
        lambda: sorted(universe, key=lambda x: (constraint_degree[x] % 13, x)),
    ]

    best_size = 0
    best_solution = set()

    for strat in strategies:
        order = strat()
        sol = greedy(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = set(sol)

    # Randomized greedy restarts
    import random
    for i in range(200):
        if time.time() - start_time > DEADLINE - 10:
            break
        random.seed(i * 12345 + 67890)
        order = list(universe)
        random.shuffle(order)
        sol = greedy(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = set(sol)

    # Branch-and-bound to try to improve
    # State: (selected_set, candidate_bitmask)
    # We use a recursive search with pruning.

    # Precompute: for each vertex x, a bitmask of all vertices y such that (x,y) appears
    # in some forbidden pair. This is conflict_mask[x] already.

    # For the B&B, we need a fast compatibility check. When adding x to selected set P,
    # we need: for all (a,b) in forbidden_pairs[x], not (a in P and b in P).
    # We can precompute for each x a list of forbidden pairs, and check each one.

    # Upper bound function: given candidate set C (as bitmask), estimate max additional
    # vertices we can add. Simple bound: popcount(C). Better: use a greedy packing.
    def popcount(x):
        return bin(x).count('1')

    def upper_bound_greedy(candidates_mask, current_selected):
        """Estimate how many more vertices can be added from candidates_mask."""
        # Greedy: repeatedly pick the vertex with fewest conflicts with current_selected
        # and with other candidates, add it, remove its conflicts.
        remaining = candidates_mask
        count = 0
        # Work with a copy of current_selected for checking
        sel = set(current_selected)
        
        while remaining:
            # Find vertex with minimum "conflict score"
            best_v = -1
            best_score = float('inf')
            temp = remaining
            while temp:
                lb = temp & (-temp)
                v = lb.bit_length()  # 1-indexed
                temp ^= lb
                # Score: number of forbidden pairs of v that are fully in sel
                score = 0
                for (a, b) in forbidden_pairs[v]:
                    if a in sel and b in sel:
                        score += 1
                        break
                # Also count conflicts with other remaining vertices (loose)
                # Simplify: just use score above
                if score < best_score:
                    best_score = score
                    best_v = v
                    if score == 0:
                        break
            
            if best_v == -1:
                break
            
            # Check if best_v is actually compatible
            compatible = True
            for (a, b) in forbidden_pairs[best_v]:
                if a in sel and b in sel:
                    compatible = False
                    break
            
            if not compatible:
                # Remove best_v from remaining and continue
                remaining &= ~(1 << (best_v - 1))
                continue
            
            # Add best_v to sel, count it
            sel.add(best_v)
            count += 1
            remaining &= ~(1 << (best_v - 1))
            
            # Remove from remaining any vertex that conflicts with best_v in a way
            # that would prevent addition (i.e., vertices y such that some pair (a,b)
            # in forbidden_pairs[y] has both a and b in sel after adding best_v)
            # This is expensive; skip for speed and rely on the greedy loop.
        
        return count

    # Branch-and-bound
    # To avoid deep recursion, we'll use an iterative approach with a stack.
    # But for n=128, recursion depth is at most 128, which is fine.

    # We'll order vertices by constraint degree (highest first) for better pruning.
    vertex_order = sorted(universe, key=lambda x: (-constraint_degree[x], x))
    # Map vertex to its position in the order
    vertex_pos = {v: i for i, v in enumerate(vertex_order)}

    # For B&B, we process vertices in order. At each step, we decide include/exclude.
    # State: (index, selected_set, candidates_mask)
    # candidates_mask contains vertices from vertex_order[index:] that haven't been decided.

    # Actually, let's use a simpler recursive B&B:
    # - Pick the next vertex to branch on (highest degree among remaining).
    # - Branch include/exclude.
    # - Prune if current_size + popcount(remaining) <= best_size.

    # For efficiency, we'll limit the B&B to a certain number of nodes.
    node_count = 0
    max_nodes = 500000

    def branch_and_bound(selected, remaining_mask):
        nonlocal best_size, best_solution, node_count
        
        if time.time() - start_time > DEADLINE - 2:
            return
        
        node_count += 1
        if node_count > max_nodes:
            return
        
        current_size = len(selected)
        
        # Prune: if we can't beat best, return
        remaining_count = popcount(remaining_mask)
        if current_size + remaining_count <= best_size:
            return
        
        # If no remaining, update best
        if remaining_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_solution = set(selected)
            return
        
        # Use a tighter upper bound
        ub = upper_bound_greedy(remaining_mask, selected)
        if current_size + ub <= best_size:
            return
        
        # Pick vertex to branch on: highest constraint degree among remaining
        # Find the vertex with highest degree in remaining
        best_v = -1
        best_deg = -1
        temp = remaining_mask
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length()
            temp ^= lb
            if constraint_degree[v] > best_deg:
                best_deg = constraint_degree[v]
                best_v = v
        
        if best_v == -1:
            return
        
        # Branch 1: Include best_v
        # Check compatibility
        compatible = True
        for (a, b) in forbidden_pairs[best_v]:
            if a in selected and b in selected:
                compatible = False
                break
        
        if compatible:
            selected.add(best_v)
            new_remaining = remaining_mask & ~(1 << (best_v - 1))
            branch_and_bound(selected, new_remaining)
            selected.remove(best_v)
        
        # Branch 2: Exclude best_v
        new_remaining = remaining_mask & ~(1 << (best_v - 1))
        branch_and_bound(selected, new_remaining)

    # Start B&B from the best greedy solution's complement
    # Actually, start from empty and let B&B find the optimum.
    # But that's expensive. Instead, start from a good lower bound.
    
    # Let's try B&B from scratch with a time limit
    initial_remaining = (1 << n) - 2  # bits 1..128 set (0-indexed: bits 0..127)
    # Actually, bit i represents vertex i+1. So bits 0 to 127 for vertices 1 to 128.
    initial_remaining = (1 << 128) - 1  # all 128 vertices
    
    # Start with empty selected set
    branch_and_bound(set(), initial_remaining)

    # Final local search polish on the best solution found
    if time.time() - start_time < DEADLINE - 1:
        # Simple local search: try to add any vertex that fits
        changed = True
        while changed and time.time() - start_time < DEADLINE - 0.5:
            changed = False
            for x in universe:
                if x in best_solution:
                    continue
                ok = True
                for (a, b) in forbidden_pairs[x]:
                    if a in best_solution and b in best_solution:
                        ok = False
                        break
                if ok:
                    best_solution.add(x)
                    best_size = len(best_solution)
                    changed = True
    
    if not best_solution:
        best_solution = {1}

    return list(best_solution)
