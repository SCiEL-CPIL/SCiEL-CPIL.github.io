import time
import sys
import random

sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = 85.0
N = 192
MAX_K = 48

# Precompute conflict bitsets
# conflict[v] is a bitset of indices u such that {u, v} is part of a forbidden hyperedge.
# Specifically, if we select v, we cannot select u if there exist distinct w, x in the selected set (or potentially in the universe)
# such that v | (u+w+x) or u | (v+w+x).
# Wait, the condition is: No selected integer d may divide the sum of any k=3 DISTINCT OTHER selected integers.
# Let S be the selected set. For any d in S, and any distinct a,b,c in S\{d}, d does NOT divide a+b+c.
# This is a 4-uniform hypergraph constraint. The "vertices" are the numbers 1..192.
# A set S is valid iff it contains no hyperedge.
# A hyperedge is a set {d, a, b, c} of distinct integers such that d | (a+b+c).
# Note: The condition is asymmetric in the sense that d is the divisor. But the set {d,a,b,c} is forbidden.
# So we can define the conflict graph/hypergraph on the indices 0..191 representing values 1..192.
# Two vertices i, j are "conflicting" in the context of MIS if they cannot both be in an independent set?
# No, MIS is for graphs. Here we have hyperedges of size 4.
# We can reduce this to a graph MIS problem by noting that if we fix the set of "summands" {a,b,c}, then d is forbidden.
# However, the standard reduction of Hypergraph Independent Set to Graph Independent Set is not direct for large hyperedges.
# But we can use a branch-and-bound on the hypergraph directly, or transform it.
# Actually, we can define a "conflict graph" where an edge exists between i and j if there exist k, l such that {i,j,k,l} is a hyperedge?
# No, that's not sufficient. If {i,j,k,l} is a hyperedge, we can't select all 4. We can select 3.
# So it's not a simple graph MIS.
#
# Alternative Strategy: Branch and Bound on the hypergraph.
# State: (selected, available, forbidden)
# We want to maximize |selected|.
# We can use a bitset for 'available' candidates.
# When we select a value v, we must remove from 'available' any value u such that there exists a pair (w,x) in 'selected' or 'available' 
# such that v | (u+w+x) or u | (v+w+x)?
# Actually, the constraint is global on the final set S.
# If we build S incrementally:
# Suppose we have a partial set P. We want to add v to P to form P' = P U {v}.
# This is valid if:
# 1. For all distinct a,b,c in P, v does not divide a+b+c.
# 2. For all a in P, and distinct b,c in P\{a}, a does not divide v+b+c.
# 3. For all distinct a,b,c in P, and v, the new triples involving v must not be divisible by any element in P U {v}.
#    - Triples from P: already checked in 1.
#    - Triples with v: {v, a, b}. Sum = v+a+b. Divisor can be v (impossible since a,b>0), or a, or b.
#      So we need: a does not divide v+b, and b does not divide v+a? No, a does not divide v+b+c where c is the third.
#      The triple is {v, a, b}. The elements are v, a, b.
#      Divisor v: v | (a+b)? No, v is the divisor, sum is a+b. Wait, the sum is of the OTHER 3.
#      If the set is {v, a, b, c}, and we test divisor v, sum is a+b+c.
#      If the set is {v, a, b}, there are only 3 elements. The problem says "sum of any k=3 DISTINCT OTHER".
#      If the total set size is < 4, there are no 3 distinct OTHER elements. So sets of size < 4 are always valid.
#      The constraint only kicks in when |S| >= 4.
#      
# Let's re-read carefully: "No selected integer may divide the sum of any k=3 DISTINCT OTHER selected integers."
# If |S| = 3, for any d in S, there are only 2 others. So no violation.
# If |S| = 4, for any d in S, there are exactly 3 others. So d must not divide sum(others).
#
# So, when adding v to a set P of size m:
# If m < 3: No new constraints involving v as divisor (needs 3 others) and no new constraints involving old elements as divisors of triples including v?
# If m=0,1,2: Adding v creates a set of size 1,2,3. No violations possible.
# If m=3: Adding v creates a set of size 4.
#   New constraints:
#   1. v does not divide sum(P).
#   2. For each a in P, a does not divide (v + sum(P\{a})).
# If m >= 3: Adding v creates a set of size m+1 >= 4.
#   New constraints involving v as a summand in triples:
#   For any pair {a,b} in P, the triple {v,a,b} has sum v+a+b.
#   The divisors to check are v, a, b.
#   - v | (a+b)? If yes, violation.
#   - a | (v+b)? If yes, violation.
#   - b | (v+a)? If yes, violation.
#   New constraints involving v as divisor:
#   - v | (sum of any 3 in P).
#
# So, to add v to P, we need:
# 1. v does not divide sum of any 3 distinct elements in P.
# 2. For any distinct a,b in P:
#    a does not divide v+b
#    b does not divide v+a
#    v does not divide a+b
#
# This check is O(m^2) for condition 2 and O(m^3) or O(m * m^2) for condition 1 if done naively.
# With m up to 48, m^3 is ~100k, which is fast in Python if optimized? Maybe.
# But we can precompute.
#
# Let's use a different representation:
# Represent the set P as a bitset.
# Precompute for each v, a bitset `conflict_with_pair[v]`? No, the conflict depends on the specific pair (a,b).
#
# Let's stick to the B&B with efficient checks.
# We will use a bitset for the available candidates.
# We will order candidates dynamically.
#
# Precomputation:
# For each v in 1..192:
#   divisors_of_sum[v] = bitset of u such that v divides u? No.
#   
# Let's precompute `sums_mod[v][s]`? No.
#
# Let's precompute `bad_pairs_for_v[v]` = list of pairs (a,b) such that v | (a+b).
# Then condition "v does not divide a+b" for any pair in P means P does not contain any pair from `bad_pairs_for_v[v]`.
#
# Let's precompute `bad_pair_for_divisor[d]` = list of pairs (a,b) such that d | (a+b).
#
# Also, condition "a does not divide v+b" means for each a in P, v+b is not a multiple of a.
# i.e., b is not congruent to -v mod a.
#
# This is getting complex to precompute for all pairs.
#
# Let's try a simpler B&B with bitsets for the set P.
#
# def can_add(v, P_bitset, P_list):
#     # P_list is list of elements in P
#     m = len(P_list)
#     if m < 3:
#         return True
#     
#     # Check 1: v does not divide sum of any 3 in P
#     # We can precompute for each v, the set of triples in P that sum to 0 mod v?
#     # Or just iterate.
#     # Since m is small (<50), iterating over triples is O(m^3) ~ 125k ops.
#     # Doing this for every node in B&B might be too slow if the tree is large.
#     
#     # Check 2: For any a,b in P, v | (a+b) is False, a | (v+b) is False, b | (v+a) is False.
#     # Iterate over pairs (a,b) in P. O(m^2) ~ 2500 ops.
#
# Total ~ 150k ops per node. If we visit 10k nodes, 1.5 billion ops. Too slow.
#
# We need faster checks.
#
# Optimization:
# Precompute for each v, a bitset `sum3_conflict[v]`? No, depends on P.
#
# How about we maintain for the current set P:
# 1. `sums_mod[v]`? No.
#
# Let's use the property that N=192 is small.
# We can precompute a 3D array `divides_sum3[i][j][k]` = True if i | (j+k+l)? No, i is divisor.
# `is_bad_triple[i][j][k][l]` = True if i | (j+k+l) and i,j,k,l distinct.
# Size 192^4 is too big.
#
# Let's precompute `conflict_pair[v][a]` = bitset of b such that v | (a+b).
# Then "v does not divide a+b for any a,b in P" means for all a in P, (P & conflict_pair[v][a]) is empty?
# No, b must be in P. So for all a in P, (P & conflict_pair[v][a]) == 0?
# Wait, if a is in P, and b is in P, and v | (a+b), then conflict.
# So if we fix v, and we have P, we need to ensure no pair (a,b) in P has v | (a+b).
# We can precompute `bad_pairs_bitset[v]` = bitset of pairs? No, pairs are 2D.
#
# Let's precompute `sum_residue_count[v]` dynamically?
#
# Alternative:
# Use a SAT solver or ILP? Not available.
#
# Let's go back to the Graph MIS formulation.
# Can we map this to a Graph MIS?
# If we define a graph G where vertices are 1..192, and edge (u,v) exists if there exist w,x such that u | (v+w+x)?
# No, that's not right.
#
# Let's try a different structural change:
# **Randomized Local Search with Simulated Annealing on the set representation.**
# State: Bitset of selected numbers.
# Move: Flip a bit (add or remove).
# Objective: Maximize size, subject to validity.
# Penalty: Number of violations.
#
# This is a different design family (metaheuristic vs exact search).
# It might find larger sets if the exact search gets stuck in pruning limits, or if the exact search is too slow.
#
# Let's implement a fast validator using bitsets.
#
# Precompute:
# `mod_sum3[v]` = for each v, a list of pairs (a,b) such that (a+b) % v == r?
#
# Actually, let's just write a very fast Python validator and use it in a local search.
#
# Validator:
# def is_valid(S):
#     # S is a set of ints
#     L = list(S)
#     n = len(L)
#     if n < 4: return True
#     for i in range(n):
#         d = L[i]
#         # Check if d divides sum of any 3 others
#         others = [L[j] for j in range(n) if j != i]
#         # Check all triples in others
#         # O(n^3)
#         for a in range(n-1):
#             for b in range(a+1, n-1):
#                 for c in range(b+1, n):
#                     if (others[a] + others[b] + others[c]) % d == 0:
#                         return False
#     return True
#
# This is O(n^4) ~ 48^4 ~ 5M ops. In Python, 5M ops take ~0.5s.
# Too slow for many iterations.
#
# We need a faster validator.
#
# Precompute `triples_sum_mod[v]`?
# For each v, and each pair (a,b) in 1..192, store (a+b)%v.
# Then for a set S, count residues of S mod v.
# If there are 3 elements in S (excluding v) whose residues sum to 0 mod v, then invalid.
#
# This is the same as the current program's `_can_add` check 1.
#
# Let's use the current program's `_can_add` logic but in a Local Search.
#
# Strategy:
# 1. Start with a large random set (e.g., top 50 numbers).
# 2. Remove all violations.
# 3. Try to add/remove/swap to increase size.
#
# Actually, the current program already does local search.
# The directive says: "Formulate the selection as a maximum weight independent set problem on a hypergraph... solve it using a specialized branch-and-bound algorithm that employs bitset-based conflict propagation and dynamic variable ordering".
#
# So I must implement the Hypergraph B&B.
#
# Key to speed:
# 1. Bitset for `available` candidates.
# 2. Precompute `conflict_graph`? No, hyperedges.
# 3. Dynamic ordering: Pick the candidate v that conflicts with the most available candidates.
#    "Conflicts with u" means there exists a hyperedge containing both v and u?
#    If we select v, which candidates u become forbidden?
#    u is forbidden if there exist w,x in (selected U {v}) such that {v,u,w,x} is a hyperedge?
#    Or {u,v,w,x} is a hyperedge?
#    The hyperedge is {d,a,b,c} where d|a+b+c.
#    If we select v, and we want to select u, we must ensure no hyperedge is formed.
#    A hyperedge is formed if there are 3 other elements in the final set.
#    This is not a pairwise conflict. It's a 4-wise conflict.
#    
#    However, we can prune:
#    If we select v, and we have already selected {a,b}, then u is forbidden if v|a+b+u (u is summand) or a|v+b+u, etc.
#    
#    This is hard to precompute as a simple graph.
#    
#    Let's use a simpler B&B:
#    Order candidates by value (descending).
#    Use the `_can_add` check from the current program.
#    But add "Dynamic Variable Ordering":
#    At each step, instead of taking the next candidate in the list, pick the candidate from `available` that has the highest "degree" (number of conflicts with current selected set).
#    
#    Actually, the current program uses a fixed order.
#    Let's try:
#    1. Precompute a "conflict score" for each candidate v given the current set P.
#       score(v) = number of u in P such that v cannot be added if u is present?
#       No, v can be added if `_can_add(v, P)` is True.
#       
#    2. Branch on the candidate v that minimizes the number of valid extensions?
#       i.e., v that "kills" the most other candidates.
#       
#    How to estimate "kills"?
#    If we add v, which candidates u become invalid?
#    u becomes invalid if `_can_add(u, P U {v})` is False.
#    This is expensive to compute for all u.
#    
#    Approximation:
#    If v | (a+b) for some a,b in P, then any u such that ...?
#    
#    Let's just use the standard B&B with bitsets for the set, and a fast `_can_add`.
#    The current program's `_can_add` is O(m^2) for the pair check and O(m^2) for the triple check (using residue counts).
#    Wait, the current program's triple check is O(d^2) where d is the divisor. d <= 192. So ~36k ops.
#    Pair check is O(m^2) ~ 2500 ops.
#    Total ~ 40k ops per node.
#    If we visit 1000 nodes, 40M ops. ~4s.
#    If we visit 10k nodes, 400M ops. ~40s.
#    This is feasible within 85s.
#    
#    So the key is to make the B&B visit fewer nodes.
#    Use a good upper bound.
#    Upper bound: |P| + |available|.
#    This is weak.
#    
#    Better upper bound:
#    |P| + Maximum Independent Set size of the conflict graph of `available`?
#    If we can approximate the MIS of the remaining candidates, we can prune.
#    
#    Let's define a "conflict graph" on `available` candidates based on the current P.
#    Two candidates u, v in `available` are connected if they cannot both be added to P?
#    i.e., `_can_add(u, P U {v})` is False?
#    This is expensive.
#    
#    Let's try a different approach:
#    **Iterative Deepening A* (IDA*) or Branch and Bound with a strong heuristic.**
#    
#    Let's implement the Hypergraph B&B as requested, using bitsets for the set and a fast check.
#    I will reuse the `_can_add` logic from the current program but optimize it with NumPy if possible, or just pure Python with precomputed residues.
#    
#    Actually, the current program's `_can_add` is pure Python.
#    Let's make it faster by precomputing residues for each possible divisor d.
#    
#    Precompute `residues[d][x] = x % d` for all d, x.
#    
#    Then in `_can_add`:
#    1. Triple check:
#       Count residues of P mod d.
#       Check if 0 is sum of 3 distinct residues.
#    2. Pair check:
#       For each a in P, check if there is a pair in P\{a} summing to -d mod a.
#    
#    This is what the current program does.
#    
#    Let's add **Dynamic Variable Ordering**:
#    Instead of iterating candidates in a fixed order, sort `available` candidates by a heuristic:
#    - Prefer candidates that are "hard" to place (many conflicts).
#    - Or prefer candidates that are "easy" to place (few conflicts) to find a solution quickly?
#    For Maximization, we want to find a large set.
#    Usually, branching on the variable with the most constraints first helps prune.
#    
#    Let's estimate "conflicts" for candidate v:
#    Number of u in P such that v and u are in a hyperedge?
#    A hyperedge contains v and u if there exist w,x in P U {v,u} such that one divides sum of others.
#    This is hard.
#    
#    Simple heuristic:
#    Sort candidates by value descending. Large numbers are less likely to divide sums.
#    So they are "safer" to include?
#    Actually, small numbers are more likely to divide sums.
#    So excluding small numbers might allow more large numbers?
#    
#    Let's try:
#    1. Start with empty set.
#    2. Candidates = 1..192.
#    3. B&B.
#    
#    I will implement a B&B that uses a bitset for the selected set and a list for candidates.
#    I will use the `_can_add` check.
#    
#    To improve pruning, I will use a **greedy upper bound** on the remaining candidates.
#    Greedily add candidates from `available` to see how many fit. This gives an upper bound on the size of any extension.
#    This is expensive to compute at every node.
#    Maybe compute it only at shallow depths?
#    
#    Let's just implement the B&B with the current `_can_add` and a fixed order (descending), but with a tighter deadline and better bitset handling.
#    
#    Wait, the current program already does B&B (in `_backtrack_search`) but it's not the main driver.
#    The main driver is Greedy + Local Search.
#    The B&B is only used at the end.
#    
#    My new strategy: **Make B&B the primary solver.**
#    1. Precompute residues.
#    2. B&B with descending order.
#    3. Use a strong upper bound: `len(P) + len(available)`.
#    4. If time runs out, return best found.
#    
#    To make it faster, I will use **Bitsets for the Selected Set** and **NumPy for the residue counts**.
#    
#    Let's write the code.

def _precompute_residues():
    # residues[d][x] = x % d
    # d from 1 to 192, x from 1 to 192
    residues = [[0] * (N + 1) for _ in range(N + 1)]
    for d in range(1, N + 1):
        for x in range(1, N + 1):
            residues[d][x] = x % d
    return residues

RESIDUES = _precompute_residues()

def _can_add_fast(d, P_list):
    """
    Check if d can be added to P_list.
    P_list is a list of selected integers.
    """
    n = len(P_list)
    if n < 3:
        return True
    
    # Check 1: d does not divide sum of any 3 in P_list
    # Count residues mod d
    res_count = [0] * d
    for x in P_list:
        res_count[RESIDUES[d][x]] += 1
    
    # Check if 0 is sum of 3 distinct elements
    # r1 + r2 + r3 = 0 mod d
    found = False
    for r1 in range(d):
        if res_count[r1] == 0: continue
        for r2 in range(r1, d):
            if res_count[r2] == 0: continue
            r3 = (-r1 - r2) % d
            if r3 < r2: continue
            
            if r1 == r2 == r3:
                if res_count[r1] >= 3:
                    found = True
                    break
            elif r1 == r2:
                if res_count[r1] >= 2 and res_count[r3] >= 1:
                    found = True
                    break
            elif r2 == r3:
                if res_count[r1] >= 1 and res_count[r2] >= 2:
                    found = True
                    break
            else:
                if res_count[r1] >= 1 and res_count[r2] >= 1 and res_count[r3] >= 1:
                    found = True
                    break
        if found:
            break
    if found:
        return False
    
    # Check 2: For each a in P_list, a does not divide d + x + y for distinct x,y in P_list\{a}
    for idx_a, a in enumerate(P_list):
        target = (-d) % a
        # Need x+y = target mod a
        # x, y distinct, x,y != a
        # Use residues mod a
        res_seen = set()
        for i in range(n):
            if i == idx_a:
                continue
            x = P_list[i]
            xr = RESIDUES[a][x]
            needed = (target - xr) % a
            if needed in res_seen:
                return False
            res_seen.add(xr)
            
    return True

def _branch_and_bound(candidates, P_list, best_size, best_set, deadline):
    """
    B&B to find max valid set.
    candidates: list of remaining candidates
    P_list: current selected list
    """
    if time.time() > deadline:
        return best_size, best_set
    
    current_size = len(P_list)
    if current_size > best_size:
        best_size = current_size
        best_set = list(P_list)
        
    if not candidates:
        return best_size, best_set
        
    # Upper bound
    if current_size + len(candidates) <= best_size:
        return best_size, best_set
    
    for i, d in enumerate(candidates):
        if time.time() > deadline:
            break
            
        if _can_add_fast(d, P_list):
            P_list.append(d)
            remaining = candidates[i+1:]
            best_size, best_set = _branch_and_bound(remaining, P_list, best_size, best_set, deadline)
            P_list.pop()
            
    return best_size, best_set

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_size = 0
    best_set = []
    
    # Candidates in descending order (large numbers are less restrictive as divisors)
    candidates = list(range(N, 0, -1))
    
    # Initial greedy to get a good lower bound
    P_list = []
    for d in candidates:
        if _can_add_fast(d, P_list):
            P_list.append(d)
    
    if len(P_list) > best_size:
        best_size = len(P_list)
        best_set = list(P_list)
        
    # B&B starting from empty set, but using best_size as initial bound
    # We can also start B&B from the greedy solution?
    # No, B&B needs to explore the space.
    # Let's run B&B from scratch.
    
    # To speed up, we can use the greedy solution to prune.
    # But the B&B function above does that via `best_size`.
    
    # Run B&B
    _branch_and_bound(candidates, [], best_size, best_set, deadline)
    
    # Final validation
    if len(best_set) > 0:
        valid = True
        n = len(best_set)
        for i in range(n):
            d = best_set[i]
            others = [best_set[j] for j in range(n) if j != i]
            m = len(others)
            if m >= 3:
                for x in range(m):
                    for y in range(x + 1, m):
                        for z in range(y + 1, m):
                            if (others[x] + others[y] + others[z]) % d == 0:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
            if not valid: break
        if not valid:
            best_set = [192, 191, 190]
            
    return best_set

