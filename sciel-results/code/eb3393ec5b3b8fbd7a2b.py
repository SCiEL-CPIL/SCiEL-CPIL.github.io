import random
import time

def construct():
    n = 128
    target_lo = 128
    target_hi = 256
    
    # Start with a reasonable initial set
    # Must include 128 and some elements <= 64
    A = set()
    A.add(128)
    A.add(0)
    A.add(64)
    
    # Greedily add elements to cover all targets
    # Compute current coverage
    def get_representations(A_set, lo, hi):
        reps = {s: 0 for s in range(lo, hi+1)}
        elems = sorted(A_set)
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                s = elems[i] + elems[j]
                if lo <= s <= hi:
                    reps[s] += 1
        return reps
    
    def peak_mult(A_set):
        reps = get_representations(A_set, target_lo, target_hi)
        return max(reps.values()) if reps else 0
    
    def is_covered(A_set):
        reps = get_representations(A_set, target_lo, target_hi)
        return all(v > 0 for v in reps.values())
    
    # Greedy: add elements that reduce uncovered targets
    uncovered = set(range(target_lo, target_hi+1))
    
    # Try adding elements from [0, 128]
    candidates = list(range(0, 129))
    random.shuffle(candidates)
    
    start_time = time.time()
    while not is_covered(A) and time.time() - start_time < 10:
        best_elem = None
        best_uncovered_after = len(uncovered)
        for e in candidates:
            if e in A:
                continue
            A.add(e)
            reps = get_representations(A, target_lo, target_hi)
            new_uncovered = sum(1 for s in range(target_lo, target_hi+1) if reps[s] == 0)
            if new_uncovered < best_uncovered_after:
                best_uncovered_after = new_uncovered
                best_elem = e
            A.remove(e)
        if best_elem is None:
            break
        A.add(best_elem)
    
    # Now do local search to minimize peak multiplicity
    # Try swapping elements
    for iteration in range(200):
        if time.time() - start_time > 50:
            break
        # Try adding an element not in A
        if len(A) < 80:  # keep size reasonable
            best_swap = None
            best_peak = peak_mult(A)
            for e in range(0, 129):
                if e in A:
                    continue
                A.add(e)
                p = peak_mult(A)
                if p < best_peak:
                    best_peak = p
                    best_swap = ('add', e, None)
                A.remove(e)
            if best_swap:
                A.add(best_swap[1])
                continue
        
        # Try removing an element
        if len(A) > 20:
            best_swap = None
            best_peak = peak_mult(A)
            for e in list(A):
                A.remove(e)
                if is_covered(A):
                    p = peak_mult(A)
                    if p < best_peak:
                        best_peak = p
                        best_swap = ('remove', e, None)
                A.add(e)
            if best_swap:
                A.remove(best_swap[1])
                continue
        
        # Try swapping: remove e1, add e2
        if len(A) > 20:
            best_swap = None
            best_peak = peak_mult(A)
            for e1 in list(A):
                for e2 in range(0, 129):
                    if e2 in A or e2 == e1:
                        continue
                    A.remove(e1)
                    A.add(e2)
                    if is_covered(A):
                        p = peak_mult(A)
                        if p < best_peak:
                            best_peak = p
                            best_swap = ('swap', e1, e2)
                    A.remove(e2)
                    A.add(e1)
                    if best_swap:
                        break
                if best_swap:
                    break
            if best_swap:
                A.remove(best_swap[1])
                A.add(best_swap[2])
                continue
    
    return sorted(A)
