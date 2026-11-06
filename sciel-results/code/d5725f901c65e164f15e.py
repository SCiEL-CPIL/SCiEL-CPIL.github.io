"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _compute_peak(A_set, A_list):
    """Compute the maximum number of representations for any s in [128,256]."""
    max_rep = 0
    for s in range(128, 257):
        count = 0
        for a in A_list:
            b = s - a
            if b < a:
                continue
            if b in A_set:
                count += 1
        if count > max_rep:
            max_rep = count
    return max_rep

def _is_valid(A_set, A_list):
    """Check that every integer in [128,256] is a sum of two elements of A."""
    for s in range(128, 257):
        found = False
        for a in A_list:
            b = s - a
            if b >= a and b in A_set:
                found = True
                break
        if not found:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial solution: contiguous block
    A = set(range(64, 129))
    A_list = sorted(A)
    best_peak = _compute_peak(A, A_list)
    best_A = set(A)
    
    random.seed(42)
    
    while time.time() < deadline:
        # Compute current peak and representation counts
        A_list = sorted(A)
        rep_counts = {}
        for s in range(128, 257):
            count = 0
            for a in A_list:
                b = s - a
                if b < a:
                    continue
                if b in A:
                    count += 1
            rep_counts[s] = count
        
        current_peak = max(rep_counts.values())
        
        if current_peak < best_peak:
            best_peak = current_peak
            best_A = set(A)
        
        # Try to reduce peak by removing elements involved in peak sums
        high_sums = [s for s in range(128, 257) if rep_counts[s] >= current_peak]
        
        # Collect elements in pairs for high sums
        involved = set()
        for s in high_sums:
            for a in A_list:
                b = s - a
                if b >= a and b in A:
                    involved.add(a)
                    involved.add(b)
        
        # Try removing each involved element
        candidates = list(involved)
        random.shuffle(candidates)
        
        improved = False
        for elem in candidates:
            if time.time() > deadline - 5.0:
                break
            new_A = A - {elem}
            if len(new_A) < 1:
                continue
            new_A_list = sorted(new_A)
            if not _is_valid(new_A, new_A_list):
                continue
            new_peak = _compute_peak(new_A, new_A_list)
            if new_peak < current_peak:
                A = new_A
                current_peak = new_peak
                improved = True
                break
        
        if not improved:
            # Try adding elements to balance representations
            not_in_A = [x for x in range(0, 129) if x not in A]
            if not_in_A:
                random.shuffle(not_in_A)
                for elem in not_in_A[:5]:
                    if time.time() > deadline - 5.0:
                        break
                    new_A = A | {elem}
                    new_A_list = sorted(new_A)
                    new_peak = _compute_peak(new_A, new_A_list)
                    if new_peak < current_peak:
                        A = new_A
                        current_peak = new_peak
                        break
    
    return list(best_A)
# EVOLVE-BLOCK-END

