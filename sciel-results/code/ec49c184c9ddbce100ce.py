"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Moduli range [20, 200]
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    
    # Precompute gcd and conflict structure
    # For each pair (i, j), the conflict is that r_i == r_j (mod gcd(m_i, m_j)).
    # This is a non-linear constraint for ILP directly, but we can linearize it
    # or use a different formulation.
    
    # Alternative Strategy: 
    # The problem is equivalent to finding a maximum weight independent set in a graph where 
    # nodes are (m, r) and edges connect conflicting pairs.
    # However, the graph is huge (sum of m for m in 20..200 ~ 20000 nodes).
    
    # Better Strategy: 
    # Use a greedy algorithm with a smarter heuristic: 
    # 1. Sort moduli by 1/m (descending).
    # 2. For each modulus, we want to pick a residue that minimizes future conflicts.
    #    A residue r for m conflicts with a future modulus m' if r == r' (mod gcd(m, m')).
    #    This suggests we should look at the "congestion" of residues modulo small gcds.
    
    # Let's try a different deterministic approach:
    # Since the moduli are restricted to [20, 200], the number of distinct moduli is 181.
    # The maximum possible score is sum(1/m for m in 20..200) which is approx ln(200/19) ~ 2.35? 
    # Wait, sum(1/m) for m=20..200 is H_200 - H_19.
    # H_200 ~ ln(200) + gamma + 1/(400) ~ 5.3 + 0.577 + 0.0025 ~ 5.88
    # H_19 ~ ln(19) + gamma + 1/(38) ~ 2.94 + 0.577 + 0.026 ~ 3.54
    # Sum ~ 2.34.
    # The current score is 3.1069. This is IMPOSSIBLE if we are just summing 1/m.
    # Let me re-read the task.
    # "The score is the sum of 1/m".
    # If the current score is 3.1069, and the max sum of reciprocals is ~2.34, something is wrong.
    
    # Ah, the task says: "Pairwise disjoint residue classes with distinct moduli in [20, 200]".
    # "construct() -> classes [[m, r], ...] with distinct moduli 20 <= m <= 200"
    # "The score is the sum of 1/m".
    
    # Let's check the seed program score.
    # Seed: [[20 * j, j - 1] for j in range(1, 11)] -> moduli 20, 40, 60, 80, 100, 120, 140, 160, 180, 200.
    # Sum 1/m: 1/20 + 1/40 + ... + 1/200.
    # 1/20 = 0.05
    # 1/40 = 0.025
    # 1/60 = 0.0166
    # ...
    # Sum is definitely less than 1.
    # But the current program score is 3.1069.
    # How is that possible?
    
    # Re-reading: "The evaluator measures the raw value".
    # "search score is value/seed".
    # If seed score is ~0.15, and current is 3.1, then raw value is ~0.46?
    # No, "score 3.1069" usually refers to the value reported by the evaluator.
    # If the evaluator reports the raw sum, then 3.1069 is the sum.
    # This implies I am misunderstanding the problem or the constraints.
    
    # "Pairwise disjoint residue classes with distinct moduli in [20, 200]"
    # Maybe "distinct moduli" doesn't mean m_i != m_j?
    # "distinct moduli 20 <= m <= 200" usually means m_i are distinct.
    # If m_i are distinct, max sum is ~2.34.
    
    # Is it possible the moduli are NOT distinct?
    # "distinct moduli" is explicit.
    
    # Let's look at Inspiration 1: score 3.1143.
    # If the max sum is 2.34, how can they get 3.1?
    # Maybe the moduli can be repeated?
    # "distinct moduli" -> m1, m2, ... must be distinct.
    
    # Wait, look at the seed program again.
    # `return [[20 * j, j - 1] for j in range(1, 11)]`
    # Moduli: 20, 40, 60, 80, 100, 120, 140, 160, 180, 200.
    # These are distinct.
    
    # Is it possible the score is NOT sum(1/m)?
    # "The score is the sum of 1/m (exact rational, reported as a float)."
    
    # There is a contradiction between the max theoretical sum (~2.34) and the observed scores (~3.1).
    # Unless... "moduli in [20, 200]" allows multiple classes with the SAME modulus?
    # "distinct moduli" -> No.
    
    # Let's check if the range is different. [20, 200] is 181 numbers.
    # Sum 1/m for m=20..200 is indeed ~2.34.
    
    # Is it possible the task allows moduli OUTSIDE [20, 200]?
    # "distinct moduli 20 <= m <= 200". No.
    
    # Is it possible the score is sum(1/r)? No.
    
    # Let's assume the observed scores are correct and my calculation of the upper bound is wrong.
    # Sum(1/m for m in 20..200).
    # 1/20 + 1/21 + ... + 1/200.
    # Integral from 20 to 200 of 1/x dx = ln(10) = 2.302.
    # Discrete sum is slightly higher, maybe 2.35.
    
    # How can the score be 3.1?
    # Maybe the "seed" score is not the sum?
    # "search score is value/seed".
    # If seed value is V_seed, and current value is V_cur.
    # Score = V_cur / V_seed.
    # If V_cur ~ 2.3, and Score = 3.1, then V_seed ~ 0.74.
    # Let's check the seed value.
    # Seed moduli: 20, 40, ..., 200.
    # Sum 1/m = 1/20 * (1 + 1/2 + 1/3 + ... + 1/10) = 0.05 * H_10.
    # H_10 = 2.928968.
    # 0.05 * 2.929 = 0.146.
    # If V_cur = 2.3, Score = 2.3 / 0.146 = 15.7.
    # This doesn't match 3.1.
    
    # What if the score reported IS the raw value?
    # Then V_cur = 3.1.
    # This is impossible with distinct moduli in [20, 200].
    
    # Hypothesis: The moduli do NOT have to be distinct?
    # "distinct moduli" is in the text.
    # But maybe "distinct moduli" means "the set of moduli used is a subset of [20, 200]"?
    # No, "distinct" usually implies uniqueness.
    
    # Let's look at the inspiration code if available? No, only description.
    # "uses fractions, math, time; defines _try_add_class..."
    
    # Let's try to interpret "distinct moduli" as "the moduli in the list are distinct".
    # If the score is really 3.1, maybe the range is [2, 200]?
    # Sum 1/m for m=2..200 is H_200 - 1 ~ 4.88.
    # If the range was [2, 200], 3.1 is plausible.
    # But the text says [20, 200].
    
    # Let's assume the text is correct and there is a trick.
    # "Pairwise disjoint residue classes".
    # Maybe the score is defined differently?
    # "The score is the sum of 1/m".
    
    # I will proceed with the assumption that the goal is to maximize sum(1/m) under the constraint that m_i are distinct in [20, 200] and residues are pairwise disjoint.
    # If the max possible is ~2.34, and the current score is 3.1, I am missing something fundamental.
    # 
    # Wait! "bounds epsilon_19 from below".
    # This refers to a specific mathematical problem (Euclid-Mullin? No, likely related to covering systems or Egyptian fractions).
    # 
    # Let's look at the "seed" again.
    # `[[20 * j, j - 1] for j in range(1, 11)]`
    # 
    # What if I just run a greedy algorithm that tries to include as many small moduli as possible?
    # Since 1/m decreases, we want small m.
    # 
    # Strategy:
    # 1. Iterate m from 20 to 200.
    # 2. For each m, try to find a residue r such that (m, r) is disjoint from all previously added classes.
    # 3. If found, add it.
    # 
    # This is essentially what the current program does, but it uses randomization and local search.
    # A deterministic greedy might be faster and allow more time for a smarter local search.
    # 
    # However, if the current score is 3.1 and the max is 2.3, I might be failing to understand the scoring.
    # Let's assume the scoring is correct and I should just maximize the sum.
    # 
    # New Strategy: 
    # Use a bitset representation for residues to speed up conflict checking.
    # For each modulus m, the conflict with existing set S is:
    # For each (m', r') in S, r must NOT be congruent to r' mod gcd(m, m').
    # 
    # We can precompute for each m, a bitmask of forbidden residues?
    # No, the forbidden residues depend on the specific r' chosen.
    # 
    # Let's stick to a robust greedy + local search but optimized with NumPy for vectorized conflict checks.
    
    def is_disjoint_set(classes):
        """Check if all pairs are disjoint."""
        for i in range(len(classes)):
            m1, r1 = classes[i]
            for j in range(i + 1, len(classes)):
                m2, r2 = classes[j]
                g = gcd(m1, m2)
                if (r1 - r2) % g == 0:
                    return False
        return True

    def can_add(classes, m, r):
        """Check if (m, r) can be added to classes without conflict."""
        for m2, r2 in classes:
            g = gcd(m, m2)
            if (r - r2) % g == 0:
                return False
        return True

    def score(classes):
        """Calculate exact score as Fraction."""
        if not classes:
            return Fraction(0, 1)
        s = Fraction(0, 1)
        for m, r in classes:
            s += Fraction(1, m)
        return s

    # Precompute gcd table
    gcd_table = {}
    for i in range(20, 201):
        for j in range(i, 201):
            gcd_table[(i, j)] = gcd(i, j)
            gcd_table[(j, i)] = gcd(i, j)

    def get_gcd(m1, m2):
        return gcd_table[(m1, m2)]

    def greedy_construct_deterministic():
        """
        Deterministic greedy:
        Try moduli in ascending order (20, 21, 22...).
        For each m, try residues in a specific order (e.g., 0, 1, 2...) or randomized.
        To break symmetry, we can try a few different residue orderings.
        """
        best_set = []
        best_score = Fraction(0, 1)
        
        # Try a few different deterministic orders for residues
        # 1. Natural order 0..m-1
        # 2. Reverse order m-1..0
        # 3. Prime order?
        
        strategies = [
            lambda m: list(range(m)),
            lambda m: list(range(m-1, -1, -1)),
            lambda m: [x for x in range(m) if x % 2 == 0] + [x for x in range(m) if x % 2 == 1],
            lambda m: [x for x in range(m) if x % 2 == 1] + [x for x in range(m) if x % 2 == 0],
        ]
        
        for strat in strategies:
            current_set = []
            for m in range(20, 201):
                residues = strat(m)
                added = False
                for r in residues:
                    if can_add(current_set, m, r):
                        current_set.append([m, r])
                        added = True
                        break
                if not added:
                    continue
                
            s = score(current_set)
            if s > best_score:
                best_score = s
                best_set = [x[:] for x in current_set]
                
            if time.time() > deadline:
                break
                
        return best_set

    def local_search(initial_set):
        """
        Improve the set by swapping or adding.
        1. Try adding any missing modulus with a valid residue.
        2. Try swapping an existing modulus with a missing one if it improves score.
        """
        current_set = [x[:] for x in initial_set]
        current_score = score(current_set)
        
        current_mods = {m for m, r in current_set}
        candidates = [m for m in range(20, 201) if m not in current_mods]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try adding a candidate
            # Sort candidates by 1/m descending (smallest m first)
            candidates_sorted = sorted(candidates)
            
            for m_new in candidates_sorted:
                # Try to find a valid residue for m_new
                # Try all residues? m_new is at most 200. 200 checks is fast.
                found = False
                for r_new in range(m_new):
                    if can_add(current_set, m_new, r_new):
                        current_set.append([m_new, r_new])
                        current_score += Fraction(1, m_new)
                        current_mods.add(m_new)
                        candidates.remove(m_new)
                        improved = True
                        found = True
                        break
                
                if found:
                    # Adding one might block others, so restart the candidate loop
                    break
            
            if improved:
                continue
                
            # 2. Try swapping
            # For each existing class, try replacing it with a candidate
            # Only swap if 1/m_new > 1/m_old (i.e., m_new < m_old)
            if time.time() < deadline - 10:
                for i, (m_old, r_old) in enumerate(current_set):
                    swap_candidates = [m for m in candidates if m < m_old]
                    if not swap_candidates:
                        continue
                    
                    for m_new in swap_candidates:
                        # Try to find residue for m_new in the set without m_old
                        temp_set = current_set[:i] + current_set[i+1:]
                        found = False
                        for r_new in range(m_new):
                            if can_add(temp_set, m_new, r_new):
                                # Perform swap
                                current_set[i] = [m_new, r_new]
                                current_score += Fraction(1, m_new) - Fraction(1, m_old)
                                current_mods.remove(m_old)
                                current_mods.add(m_new)
                                candidates.remove(m_new)
                                candidates.append(m_old)
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if improved:
                        break
            
        return current_set

    # 1. Deterministic Greedy
    best_set = greedy_construct_deterministic()
    
    # 2. Local Search Refinement
    if time.time() < deadline:
        refined_set = local_search(best_set)
        if score(refined_set) > score(best_set):
            best_set = refined_set

    # 3. If time permits, try a randomized greedy restart to escape local optima
    if time.time() < deadline - 10:
        import random
        random.seed(42)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            current_set = []
            # Shuffle moduli? No, we want small ones first.
            # But shuffle residues.
            for m in range(20, 201):
                residues = list(range(m))
                random.shuffle(residues)
                for r in residues:
                    if can_add(current_set, m, r):
                        current_set.append([m, r])
                        break
            
            refined = local_search(current_set)
            s_refined = score(refined)
            s_best = score(best_set)
            if s_refined > s_best:
                best_set = refined

    return best_set
# EVOLVE-BLOCK-END
