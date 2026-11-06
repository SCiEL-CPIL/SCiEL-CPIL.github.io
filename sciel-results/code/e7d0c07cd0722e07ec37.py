import time
import numpy as np

N = 5000

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _get_residue_classes_np(points_arr, p):
    """Vectorized: return dict mapping residue -> count of points in that residue class."""
    if len(points_arr) == 0:
        return {}
    residues = np.mod(points_arr, p)
    unique_res, counts = np.unique(residues, return_counts=True)
    return dict(zip(unique_res.tolist(), counts.tolist()))

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    points = np.arange(N + 1, dtype=np.int32)
    primes = _primes(N + 1)
    
    idx = 0
    while idx < len(primes):
        if time.time() > deadline:
            break
            
        p = primes[idx]
        if p > len(points):
            break
            
        classes = _get_residue_classes_np(points, p)
        
        if len(classes) < p:
            # Already admissible for p
            idx += 1
            continue
            
        # Need to remove one class
        # Sort classes by count (ascending)
        sorted_classes = sorted(classes.items(), key=lambda item: item[1])
        
        best_points = None
        best_score = float('-inf')
        
        # Lookahead depth: try top K smallest classes, evaluate with multi-step lookahead
        K = min(7, len(sorted_classes))
        
        # Precompute next few primes for lookahead
        lookahead_primes = []
        for j in range(idx + 1, min(idx + 4, len(primes))):
            q = primes[j]
            if q <= len(points):
                lookahead_primes.append(q)
        
        for r, cnt in sorted_classes[:K]:
            # Simulate removal: remove all points with residue r mod p
            mask = np.mod(points, p) != r
            new_points = points[mask]
            
            if len(new_points) == 0:
                score = float('-inf')
            else:
                # Estimate future cost using soft-min over next few primes
                future_cost = 0.0
                for q in lookahead_primes:
                    q_classes = _get_residue_classes_np(new_points, q)
                    if len(q_classes) >= q:
                        # Need to remove something; estimate min class size
                        q_counts = list(q_classes.values())
                        # Use soft-min (log-sum-exp) for smooth surrogate
                        min_val = min(q_counts)
                        T = 2.0
                        neg_counts = np.array([-c / T for c in q_counts])
                        max_neg = np.max(neg_counts)
                        log_sum = np.log(np.sum(np.exp(neg_counts - max_neg))) + max_neg
                        soft_min = -log_sum * T
                        future_cost += soft_min
                    else:
                        future_cost += 0.0
                
                # Total cost = current removal + estimated future removal
                total_cost = cnt + future_cost
                score = -total_cost
            
            if score > best_score:
                best_score = score
                best_points = new_points
        
        if best_points is not None:
            points = best_points
            
        idx += 1
        
    return list(points)
