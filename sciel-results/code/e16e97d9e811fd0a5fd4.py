import numpy as np
import time
import random

def construct():
    start = time.time()
    DEADLINE = start + 140.0

    base = 51975
    divs = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    def time_left():
        return DEADLINE - time.time()

    def greedy_build(order):
        cover_count = np.zeros(base, dtype=np.int32)
        classes = []
        for m in order:
            # Find residue r that covers most currently uncovered
            zeros = np.where(cover_count == 0)[0]
            if len(zeros) == 0:
                classes.append([m, 0])
                continue
            
            # Count uncovered per residue class mod m
            # A[r] = count of x in zeros such that x % m == r
            A = np.bincount(zeros % m, minlength=m)
            best_r = int(np.argmax(A))
            classes.append([m, best_r])
            
            # Update cover_count
            cover_count[best_r::m] += 1
            
        # Calculate score
        score = base - int(np.sum(cover_count > 0))
        return classes, score, cover_count

    def local_search(classes, cover_count, cur_score):
        moduli = [c[0] for c in classes]
        residues = [c[1] for c in classes]
        
        improved = True
        while improved and time_left() > 5:
            improved = False
            # Shuffle order of moduli to avoid bias? Or just iterate.
            # Iterating is fine.
            for i in range(len(moduli)):
                if time_left() < 2: break
                m = moduli[i]
                r = residues[i]
                
                # Compute A: count of uncovered (cover_count==0) per residue mod m
                zeros = np.where(cover_count == 0)[0]
                A = np.bincount(zeros % m, minlength=m)
                
                # Compute L_fixed: count of positions in current class P_r that are covered ONLY by this class
                # These are positions x in P_r where cover_count[x] == 1
                old_pos = np.arange(r, base, m)
                L_fixed = int(np.sum(cover_count[old_pos] == 1))
                
                # Find best new residue r' != r
                # Gain for r' is A[r'] - L_fixed (since P_r and P_r' are disjoint)
                # We want to maximize A[r'] for r' != r
                A_temp = A.copy()
                A_temp[r] = -1
                best_r_prime = int(np.argmax(A_temp))
                best_gain = A_temp[best_r_prime] - L_fixed
                
                if best_gain > 0:
                    # Update
                    new_pos = np.arange(best_r_prime, base, m)
                    cover_count[old_pos] -= 1
                    cover_count[new_pos] += 1
                    residues[i] = best_r_prime
                    cur_score -= best_gain
                    improved = True
        
        return [[moduli[i], residues[i]] for i in range(len(moduli))], cur_score

    best_score = base
    best_classes = None
    best_cover_count = None

    # Try several orderings
    orderings = [
        sorted(divs),
        sorted(divs, reverse=True),
        sorted(divs, key=lambda x: random.random()),
        sorted(divs, key=lambda x: random.random()),
        sorted(divs, key=lambda x: random.random()),
    ]
    
    for order in orderings:
        if time_left() < 10: break
        classes, score, cover_count = greedy_build(order)
        if score < best_score:
            best_score = score
            best_classes = classes
            best_cover_count = cover_count.copy()
        
        # Local search
        if time_left() > 10:
            classes, score = local_search(classes, cover_count, score)
            if score < best_score:
                best_score = score
                best_classes = classes
                # Reconstruct cover_count for best_classes if we want to continue?
                # Or just keep the one from local search?
                # local_search returns classes and score. It modifies cover_count in place.
                # So cover_count is already updated.
                best_cover_count = cover_count.copy()

    return best_classes if best_classes is not None else [[3, 0]]
