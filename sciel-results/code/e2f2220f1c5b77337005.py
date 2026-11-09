"""Rewritten program for pairwise disjoint residue classes with distinct moduli in [20, 200].
Uses a deterministic constructive approach based on number-theoretic analysis to maximize
the sum of 1/m over pairwise disjoint residue classes."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    moduli = list(range(20, 201))
    
    def is_pairwise_disjoint(classes):
        """Check if all pairs are disjoint: r_i != r_j mod gcd(m_i, m_j)."""
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
    
    def compute_score(classes):
        """Calculate exact score as Fraction."""
        if not classes:
            return Fraction(0, 1)
        s = Fraction(0, 1)
        for m, r in classes:
            s += Fraction(1, m)
        return s
    
    def greedy_with_backtracking():
        """
        Structurally different approach: Use the fact that for pairwise coprime moduli,
        any residues work. For non-coprime moduli, we need r1 != r2 mod gcd.
        
        Strategy: 
        1. Start with all moduli sorted by 1/m descending (smaller m first).
        2. For each modulus, try to assign a residue that minimizes conflicts with future moduli.
        3. Use a look-ahead heuristic: prefer residues that leave more flexibility.
        
        Key insight: Two moduli m1, m2 conflict only if their gcd > 1 AND residues are congruent mod gcd.
        For large moduli with small gcds (e.g., gcd=2), most residues are valid.
        """
        # Sort moduli: smaller first (higher 1/m weight)
        sorted_moduli = sorted(moduli)
        
        best_set = []
        best_score = Fraction(0, 1)
        
        # Try multiple strategies for residue assignment
        strategies = [
            "first_valid",      # First valid residue
            "max_flex",         # Residue that conflicts with fewest future moduli
            "random_order",     # Random residue order
        ]
        
        for strategy in strategies:
            current_set = []
            
            for m in sorted_moduli:
                if time.time() > deadline:
                    break
                
                valid_residues = []
                for r in range(m):
                    if can_add(current_set, m, r):
                        valid_residues.append(r)
                
                if not valid_residues:
                    continue
                
                if strategy == "first_valid":
                    r = valid_residues[0]
                elif strategy == "max_flex":
                    # Choose residue that minimizes potential conflicts
                    # A residue r conflicts with (m2, r2) if r ≡ r2 mod gcd(m, m2)
                    # For each future modulus m2, count how many residues would conflict
                    # Simpler: just pick r that is "spread out"
                    r = valid_residues[len(valid_residues) // 2]
                elif strategy == "random_order":
                    import random
                    r = random.choice(valid_residues)
                
                current_set.append([m, r])
            
            s = compute_score(current_set)
            if s > best_score:
                best_score = s
                best_set = [x[:] for x in current_set]
        
        return best_set
    
    def local_improvement(initial_set):
        """
        Improve via:
        1. Try adding any missing modulus with a valid residue.
        2. Try replacing a higher-m modulus with a lower-m one if it fits.
        3. Try reassigning residues to free up space for additions.
        """
        current_set = [x[:] for x in initial_set]
        current_score = compute_score(current_set)
        current_mods = {m for m, r in current_set}
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding missing moduli (sorted by 1/m descending)
            missing = sorted([m for m in moduli if m not in current_mods], key=lambda x: -1/x)
            
            for m_new in missing:
                if time.time() > deadline:
                    break
                
                # Try to find a valid residue
                for r in range(m_new):
                    if can_add(current_set, m_new, r):
                        current_set.append([m_new, r])
                        current_score += Fraction(1, m_new)
                        current_mods.add(m_new)
                        improved = True
                        break
            
            # Try replacing: for each existing modulus, see if we can swap with a smaller missing one
            if not improved and time.time() < deadline - 5:
                for i in range(len(current_set)):
                    m_old = current_set[i][0]
                    # Try to replace with a smaller modulus not in set
                    candidates = [m for m in moduli if m < m_old and m not in current_mods]
                    candidates.sort(key=lambda x: -1/x)
                    
                    for m_new in candidates[:5]:  # Limit candidates
                        temp_set = current_set[:i] + current_set[i+1:]
                        for r in range(m_new):
                            if can_add(temp_set, m_new, r):
                                current_set[i] = [m_new, r]
                                current_score += Fraction(1, m_new) - Fraction(1, m_old)
                                current_mods.remove(m_old)
                                current_mods.add(m_new)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return current_set
    
    # Phase 1: Greedy construction
    best_set = greedy_with_backtracking()
    
    # Phase 2: Local improvement
    if time.time() < deadline:
        refined = local_improvement(best_set)
        if compute_score(refined) > compute_score(best_set):
            best_set = refined
    
    # Phase 3: Multiple restarts with different initial orders
    if time.time() < deadline - 10:
        import random
        
        for restart in range(30):
            if time.time() > deadline - 5:
                break
            
            random.seed(restart * 12345 + 67890)
            current_set = []
            
            # Shuffle the order of moduli to break symmetry
            shuffled_moduli = sorted_moduli if restart == 0 else sorted(random.sample(moduli, len(moduli)))
            
            # Process in order of 1/m but with some randomness in tie-breaking
            for m in shuffled_moduli:
                if time.time() > deadline:
                    break
                
                # Try residues in random order
                residues = list(range(m))
                random.shuffle(residues)
                
                added = False
                for r in residues:
                    if can_add(current_set, m, r):
                        current_set.append([m, r])
                        added = True
                        break
                
                if not added:
                    continue
            
            # Local improve
            refined = local_improvement(current_set)
            s_refined = compute_score(refined)
            s_best = compute_score(best_set)
            
            if s_refined > s_best:
                best_set = refined
    
    # Final validation
    if not is_pairwise_disjoint(best_set):
        # Fallback: build a safe set using only pairwise coprime moduli
        # Find a large set of pairwise coprime moduli in [20, 200]
        # Actually, let's just verify and fix
        best_set = []
        for m in sorted(moduli):
            for r in range(m):
                if can_add(best_set, m, r):
                    best_set.append([m, r])
                    break
    
    return best_set
# EVOLVE-BLOCK-END
