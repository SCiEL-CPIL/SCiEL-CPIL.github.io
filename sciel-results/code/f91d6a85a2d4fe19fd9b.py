import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    moduli = list(range(20, 201))
    
    def can_add(classes, m, r):
        for (m2, r2) in classes:
            g = gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True
    
    def score(classes):
        s = Fraction(0)
        for m, r in classes:
            s += Fraction(1, m)
        return s
    
    # Greedy construction: add moduli in ascending order
    classes = []
    for m in moduli:
        if time.time() > deadline - 5:
            break
        for r in range(m):
            if can_add(classes, m, r):
                classes.append([m, r])
                break
    
    best_score = score(classes)
    best_classes = [list(c) for c in classes]
    
    # Local search: try to improve
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(len(best_classes)):
            if time.time() > deadline - 5:
                break
            m_i, r_i = best_classes[i]
            for r_new in range(m_i):
                if r_new == r_i:
                    continue
                valid = True
                for j in range(len(best_classes)):
                    if j == i:
                        continue
                    m_j, r_j = best_classes[j]
                    g = gcd(m_i, m_j)
                    if r_new % g == r_j % g:
                        valid = False
                        break
                if not valid:
                    continue
                
                trial = [list(c) for c in best_classes]
                trial[i][1] = r_new
                for m_add in moduli:
                    if any(m_add == cm for cm, _ in trial):
                        continue
                    for r_add in range(m_add):
                        if can_add(trial, m_add, r_add):
                            trial.append([m_add, r_add])
                            s = score(trial)
                            if s > best_score:
                                best_score = s
                                best_classes = trial
                                improved = True
                            break
                    if improved:
                        break
            if improved:
                break
        
        if not improved:
            break
    
    return best_classes
