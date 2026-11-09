import time
from math import gcd
from fractions import Fraction
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # 1. Precompute all valid (m, r) pairs and their weights
    all_classes = []
    for m in range(20, 201):
        for r in range(m):
            all_classes.append((m, r))
    
    N = len(all_classes)
    
    # 2. Precompute compatibility matrix
    # compat[i][j] is True if class i and class j are disjoint
    # We only need to check if they intersect.
    # Intersection condition: r_i == r_j (mod gcd(m_i, m_j))
    # If gcd(m_i, m_j) == 1, they always intersect (since r_i % 1 == r_j % 1 == 0).
    # So if gcd == 1, they are incompatible.
    
    # For memory efficiency and speed, we can compute compatibility on the fly 
    # or precompute a bitmask/list of incompatible indices.
    # N is ~15000. A full matrix is 15000^2 = 225M entries, too big for simple bools in Python overhead.
    # Instead, for each node i, store a list of indices j > i that are incompatible with i.
    # Or better, for B&B, we need to check if a candidate node is compatible with the current set.
    
    # Let's optimize the compatibility check.
    # We will use a recursive B&B.
    # State: 
    #   current_set: list of indices in all_classes
    #   current_score: float
    #   next_candidate_index: start searching from this index to avoid duplicates
    #   remaining_candidates: list of indices that are compatible with current_set
    #     Actually, it's better to maintain a list of "available" nodes.
    
    # A more efficient representation for B&B on MWIS:
    # We sort nodes by weight (1/m) descending.
    # We iterate through nodes in that order.
    # At each step, we decide to include or exclude the current node.
    # If we include it, we remove all its neighbors (incompatible nodes) from the remaining pool.
    # If we exclude it, we just move to the next node.
    
    # To make this fast:
    # 1. Sort all_classes by 1/m descending (i.e., m ascending).
    # 2. Precompute for each node i, the set of nodes j (with index > i in sorted order) 
    #    that are incompatible with i.
    #    Actually, we need to know which nodes in the *remaining pool* are incompatible.
    #    Since we process in order, we can maintain a bitmask or a set of available indices.
    
    # Given N ~ 15000, bitsets are efficient. Python integers can act as arbitrary precision bitsets.
    # Let's try that.
    
    # Sort classes by m (ascending m means descending weight 1/m)
    # Wait, if we sort by m ascending, the first ones are the most valuable.
    # We want to include as many high-value nodes as possible.
    
    sorted_classes = sorted(all_classes, key=lambda x: x[0])
    N = len(sorted_classes)
    
    # Precompute incompatibility masks
    # mask[i] is an integer where bit j is 1 if sorted_classes[i] is incompatible with sorted_classes[j]
    # We only care about j > i for the forward search, but for removal, we need all.
    # Actually, in the standard B&B for MWIS (or Max Clique on complement graph):
    # We maintain a set of available nodes.
    # When we pick node i, we remove i and all its neighbors from the available set.
    
    # Let's precompute incompatibility for all pairs i, j.
    # This is O(N^2) which is ~225M operations. Might be too slow in Python.
    # 225M gcd operations is definitely too slow for 150s.
    
    # Alternative: 
    # Use the property that incompatibility depends on gcd(m_i, m_j).
    # If gcd(m_i, m_j) == 1, incompatible.
    # If gcd > 1, check r_i % g == r_j % g.
    
    # Can we speed up the precomputation?
    # Most pairs are coprime. 
    # Let's estimate: probability two random numbers are coprime is 6/pi^2 ~ 0.6.
    # So 60% of pairs are immediately incompatible.
    # We still need to check the other 40%.
    # 225M * 0.6 = 135M simple checks. 225M * 0.4 = 90M gcd checks.
    # This is likely too slow in pure Python.
    
    # Let's rethink the search strategy.
    # Instead of global B&B on all 15000 nodes, maybe we can restrict the search space?
    # The current program gets 3.11. 
    # Sum of 1/m for all even numbers 20..200:
    # 1/20 + 1/22 + ... + 1/200.
    # There are 90 even numbers.
    # Approx sum: 90 * (1/110) ~ 0.8? No, harmonic series.
    # Sum_{k=10}^{100} 1/(2k) = 0.5 * (H_100 - H_9) ~ 0.5 * (5.18 - 2.82) ~ 1.18.
    # So even numbers alone give ~1.18.
    # The current score is 3.11. This implies it's picking many odd numbers too?
    # But odd numbers are mostly coprime to each other and to evens.
    # If m1 is odd and m2 is even, gcd can be > 1 (e.g., 3 and 6).
    
    # Let's look at the structure of the solution.
    # To get a high sum, we need small moduli.
    # Small moduli are often coprime.
    # Example: 20, 21, 22, 23, 24, 25...
    # 20 and 21: gcd 1 -> incompatible.
    # 20 and 22: gcd 2. 0%2=0, 0%2=0 -> incompatible if r=0. 
    # We can choose residues to avoid conflicts.
    # For m1=20, m2=22, gcd=2. We need r1 != r2 mod 2.
    # So if r1=0, r2 must be 1.
    
    # The problem is essentially a Max Weight Independent Set on a graph where edges connect incompatible classes.
    # The graph is dense (many coprime pairs).
    
    # Maybe a different approach: 
    # Group by "prime factors" or something?
    
    # Let's try a simpler B&B but with a tighter bound and smarter ordering.
    # Or, use a local search starting from a good greedy solution, but with a stronger move set (2-swap, 3-swap).
    # The current program does 1-swap (add/remove/replace).
    # 2-swap: replace two classes with two different ones.
    
    # However, the directive asks for a STRUCTURAL change.
    # ILP is a strong candidate.
    
    # Let's implement the ILP approach using scipy.optimize.milp.
    # Variables: x_i in {0, 1} for each class i.
    # Maximize: sum(1/m_i * x_i)
    # Subject to: x_i + x_j <= 1 for all incompatible pairs (i, j).
    
    # Number of variables: ~15000.
    # Number of constraints: ~100M (incompatible pairs).
    # This is too large for a standard MILP solver to handle efficiently in 150s without significant preprocessing.
    # The constraint matrix would be huge.
    
    # We need to reduce the problem size.
    # Observation: If two classes have the same modulus m, they are mutually exclusive?
    # No, the problem says "distinct moduli".
    # "construct() -> classes [[m, r], ...] with distinct moduli"
    # So we can pick at most ONE class per modulus m.
    # This reduces the variable count to 181 (moduli 20..200).
    
    # Wait, if we pick at most one per modulus, we still have to choose the residue r.
    # But the choice of r affects compatibility with other moduli.
    # So it's not just a binary choice per modulus, it's a choice of (m, r).
    
    # However, we can reformulate:
    # For each modulus m, we have m possible residues.
    # We select one (m, r) or none.
    # This is still 15000 variables.
    
    # But maybe we can reduce the constraints?
    # Incompatibility only matters if gcd(m_i, m_j) > 1 AND r_i == r_j mod gcd.
    # If gcd(m_i, m_j) == 1, they are ALWAYS incompatible.
    # So if we pick m_i and m_j where gcd=1, we get a constraint x_i + x_j <= 1.
    # There are many such pairs.
    
    # Let's try a different structural change: **Simulated Annealing** with a specific neighborhood.
    # Or **Genetic Algorithm**.
    
    # Let's go back to the B&B idea but optimize the precomputation.
    # We can precompute incompatibility only for pairs that are NOT coprime?
    # No, coprime pairs are incompatible, so they are the ones that constrain the most.
    
    # What if we use a **Greedy + Local Search** with a much better initial solution?
    # The current program's local search is weak.
    
    # Let's try **Branch and Bound** but with a different ordering:
    # Order by m (ascending).
    # At each step, we decide which residue to pick for the next modulus, or skip it.
    # This is a tree of depth 181.
    # Branching factor: m (up to 200) + 1 (skip).
    # Too wide.
    
    # Let's try **Constraint Programming** style with backtracking and pruning.
    # But again, complexity.
    
    # Let's look at the "distinct moduli" constraint again.
    # It means we cannot pick two classes with the same m.
    # This simplifies the variable space slightly but not the pairwise constraints.
    
    # New Idea: **Max Flow / Matching?**
    # No, it's an independent set.
    
    # Let's try **Simulated Annealing** with a robust implementation.
    # State: A set of (m, r) pairs.
    # Moves:
    # 1. Add a random (m, r) if compatible.
    # 2. Remove a random (m, r).
    # 3. Replace a random (m, r) with a new (m', r') if compatible.
    # 4. 2-swap: Remove two, add two.
    
    # The current program does 1, 2, 3.
    # Adding 2-swap might help escape local optima.
    
    # But the directive asks for a DIFFERENT strategy.
    # Let's implement **Branch and Bound** carefully.
    
    # Optimization for B&B:
    # 1. Sort moduli by 1/m descending.
    # 2. Precompute for each modulus m, the list of residues r.
    # 3. In the B&B, we iterate through moduli in order.
    # 4. For the current modulus m, we try all residues r.
    # 5. We maintain a list of "blocked" residues for future moduli?
    #    No, compatibility is pairwise.
    
    # Let's try a **Hybrid**: 
    # 1. Use a fast greedy to get a good lower bound.
    # 2. Use B&B to prove optimality or improve, with a time limit.
    
    # To make B&B feasible, we need a very tight upper bound.
    # Upper bound = current_score + sum of 1/m for all remaining moduli (ignoring compatibility).
    # This bound is loose but easy to compute.
    # We can tighten it by noting that if m_i and m_j are coprime, we can't pick both.
    # So for a set of remaining moduli, the max sum is the max weight independent set of the graph induced by them.
    # That's the original problem!
    
    # However, for the remaining moduli (which are larger), the density of coprime pairs might be lower?
    # No, coprimality is roughly uniform.
    
    # Let's try **Local Search with 2-Opt moves**.
    # This is a structural change in the search method (from 1-opt to 2-opt).
    
    # Program Plan:
    # 1. Generate initial solution using greedy (as in current).
    # 2. Run local search with:
    #    - 1-opt: Add/Remove/Replace single class.
    #    - 2-opt: Remove two classes, try to add two different classes.
    # 3. Use a time limit.
    
    # Implementation details for 2-opt:
    # - Pick two random indices i, j in current set.
    # - Remove them.
    # - Try to add all possible pairs (m_k, r_k), (m_l, r_l) from the remaining moduli.
    # - This is O(N^2) per move, which is too slow.
    # - Optimization: Only try pairs where m_k and m_l are small (high weight).
    
    # Let's refine the 2-opt:
    # - Identify the "worst" two classes (lowest 1/m) in the current set.
    # - Remove them.
    # - Try to add the "best" two classes (highest 1/m) from the available moduli that are compatible with the rest of the set and with each other.
    
    # This is a heuristic 2-opt.
    
    # Let's code this.

    def _try_add_class(classes, new_class):
        m_new, r_new = new_class
        for m, r in classes:
            g = gcd(m_new, m)
            if g == 1:
                return False
            if r_new % g == r % g:
                return False
        return True

    def _score_float(classes):
        return sum(1.0 / m for m, r in classes)

    def greedy_construct(moduli_list):
        classes = []
        for m in moduli_list:
            found = False
            for r in range(m):
                if _try_add_class(classes, [m, r]):
                    classes.append([m, r])
                    found = True
                    break
        return classes

    all_moduli = list(range(20, 201))
    
    # Initial Solution
    best_classes = greedy_construct(all_moduli)
    best_score = _score_float(best_classes)
    
    # Local Search with 1-opt and 2-opt
    def local_search(classes, time_limit):
        current = [c[:] for c in classes]
        current_score = _score_float(current)
        improved = True
        start_ls = time.time()
        
        while improved and time.time() < start_ls + time_limit:
            improved = False
            
            # 1-opt: Try to add any missing class
            for m in all_moduli:
                if any(c[0] == m for c in current):
                    continue
                for r in range(m):
                    if _try_add_class(current, [m, r]):
                        current.append([m, r])
                        current_score += 1.0/m
                        improved = True
                        break
                if improved:
                    break
                    
            if improved:
                continue

            # 1-opt: Try to replace a class with a better one
            # Sort current classes by 1/m ascending (worst first) to try replacing them
            indices = list(range(len(current)))
            # We just iterate all
            for i in range(len(current)):
                m_old, r_old = current[i]
                others = [c for j, c in enumerate(current) if j != i]
                best_replacement = None
                best_gain = 0
                for m_new in all_moduli:
                    if m_new == m_old:
                        continue
                    if any(c[0] == m_new for c in others):
                        continue
                    for r_new in range(m_new):
                        if _try_add_class(others, [m_new, r_new]):
                            gain = 1.0/m_new - 1.0/m_old
                            if gain > best_gain:
                                best_gain = gain
                                best_replacement = [m_new, r_new]
                if best_replacement:
                    current[i] = best_replacement
                    current_score += best_gain
                    improved = True
                    break
            if improved:
                continue

            # 2-opt: Replace two classes
            # Identify two classes to remove. 
            # Strategy: Remove the two with smallest 1/m (largest m).
            if len(current) >= 2:
                # Sort indices by m descending (worst first)
                sorted_indices = sorted(range(len(current)), key=lambda i: current[i][0], reverse=True)
                i1 = sorted_indices[0]
                i2 = sorted_indices[1]
                
                m1, r1 = current[i1]
                m2, r2 = current[i2]
                
                others = [c for j, c in enumerate(current) if j != i1 and j != i2]
                
                # Try to add two new classes
                best_pair = None
                best_gain = 0
                current_gain_base = 1.0/m1 + 1.0/m2
                
                # To speed up, only consider moduli smaller than max(m1, m2)
                max_m = max(m1, m2)
                candidate_moduli = [m for m in all_moduli if m < max_m and m != m1 and m != m2]
                
                # We need to find (m_a, r_a) and (m_b, r_b) such that:
                # 1. m_a != m_b
                # 2. (m_a, r_a) compatible with others
                # 3. (m_b, r_b) compatible with others
                # 4. (m_a, r_a) compatible with (m_b, r_b)
                # 5. 1/m_a + 1/m_b > current_gain_base
                
                # This is still O(N^2). 
                # Optimization: Precompute compatible residues for each candidate modulus against 'others'.
                # For each m in candidate_moduli, find all r such that (m, r) is compatible with others.
                
                valid_candidates = []
                for m in candidate_moduli:
                    valid_rs = []
                    for r in range(m):
                        if _try_add_class(others, [m, r]):
                            valid_rs.append(r)
                    if valid_rs:
                        valid_candidates.append((m, valid_rs))
                        
                # Now find best pair from valid_candidates
                # Sort valid_candidates by 1/m descending
                valid_candidates.sort(key=lambda x: 1.0/x[0], reverse=True)
                
                for idx_a in range(len(valid_candidates)):
                    m_a, rs_a = valid_candidates[idx_a]
                    # If even the best possible pair with this m_a can't beat current, break?
                    # Max possible gain with m_a is 1/m_a + 1/min(m in valid_candidates[idx_a+1:])
                    # This is a heuristic bound.
                    
                    for idx_b in range(idx_a + 1, len(valid_candidates)):
                        m_b, rs_b = valid_candidates[idx_b]
                        potential_gain = 1.0/m_a + 1.0/m_b - current_gain_base
                        if potential_gain <= 0:
                            break # Since sorted by 1/m desc, further b will be smaller m, so larger 1/m? 
                            # Wait, sorted by 1/m desc means m asc. 
                            # So m_b is larger, 1/m_b is smaller. 
                            # So potential_gain decreases as idx_b increases.
                            # So break is correct.
                        
                        # Check compatibility between (m_a, r_a) and (m_b, r_b)
                        g = gcd(m_a, m_b)
                        if g == 1:
                            continue # Incompatible
                            
                        # Find r_a in rs_a and r_b in rs_b such that r_a % g != r_b % g
                        found = False
                        for r_a in rs_a:
                            for r_b in rs_b:
                                if r_a % g != r_b % g:
                                    # Valid pair
                                    new_score = current_score - current_gain_base + 1.0/m_a + 1.0/m_b
                                    if new_score > current_score:
                                        # Update current
                                        current[i1] = [m_a, r_a]
                                        current[i2] = [m_b, r_b]
                                        current_score = new_score
                                        improved = True
                                        found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                        
        return current, current_score

    # Run local search
    # Give it most of the time
    time_left = deadline - time.time() - 5
    if time_left > 10:
        best_classes, best_score = local_search(best_classes, time_left)

    return [[m, r] for m, r in best_classes]

