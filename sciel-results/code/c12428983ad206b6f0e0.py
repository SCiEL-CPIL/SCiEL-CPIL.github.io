import math
import random

def construct():
    best_classes = []
    best_score = 0.0
    
    # Try multiple random seeds for the greedy choice to avoid local optima
    # in residue assignment.
    num_attempts = 100
    
    for seed in range(num_attempts):
        random.seed(seed)
        classes = []
        current_score = 0.0
        
        for m in range(20, 201):
            # Find all valid residues for m
            # A residue r is valid if for all (m_j, r_j) in classes:
            # r != r_j mod gcd(m, m_j)
            
            # Optimization: If there is any m_j with gcd(m, m_j)==1, 
            # then no r is valid (since r != r_j mod 1 is impossible).
            # So we can quickly check for coprime conflicts.
            has_coprime_conflict = False
            for m_j, r_j in classes:
                if math.gcd(m, m_j) == 1:
                    has_coprime_conflict = True
                    break
            
            if has_coprime_conflict:
                continue
            
            # If no coprime conflicts, find valid residues
            valid_res = [True] * m
            for m_j, r_j in classes:
                g = math.gcd(m, m_j)
                target = r_j % g
                # Mark all residues congruent to target mod g as invalid
                for k in range(target, m, g):
                    valid_res[k] = False
            
            candidates = [i for i, v in enumerate(valid_res) if v]
            
            if candidates:
                # Randomly pick one to diversify the search
                r = random.choice(candidates)
                classes.append([m, r])
                current_score += 1.0 / m
            # else: no valid residue, skip m
        
        if current_score > best_score:
            best_score = current_score
            best_classes = classes
            
    return best_classes
