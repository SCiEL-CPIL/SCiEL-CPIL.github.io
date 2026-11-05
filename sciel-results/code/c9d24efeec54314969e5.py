import random
import time

def construct():
    N = 51975
    # Find divisors
    divisors = [d for d in range(2, N + 1) if N % d == 0]
    # Sort them, maybe smaller first? Or random?
    # Using all divisors is likely best.
    
    # Initialize residues
    residues = {}
    for m in divisors:
        residues[m] = random.randint(0, m - 1)
        
    count_covered = [0] * N
    score = N
    
    def update_counts(m, r, delta):
        # delta is +1 or -1
        for x in range(r, N, m):
            count_covered[x] += delta
    
    # Initial counts
    for m in divisors:
        update_counts(m, residues[m], 1)
        
    # Calculate initial score
    for x in range(N):
        if count_covered[x] == 0:
            score -= 1 # No, score is count of uncovered.
            # Let's just count uncovered directly.
    uncovered_count = sum(1 for x in range(N) if count_covered[x] == 0)
    
    best_score = uncovered_count
    best_residues = residues.copy()
    
    start_time = time.time()
    timeout = 115.0 # Leave 5s for safety
    
    # Local Search
    while time.time() - start_time < timeout:
        # Pick a random modulus
        m = random.choice(divisors)
        old_r = residues[m]
        new_r = random.randint(0, m - 1)
        
        if new_r == old_r:
            continue
            
        # Calculate Delta
        # Delta = |S_old ∩ A| - |S_new ∩ U|
        # A: count_covered[x] == 1
        # U: count_covered[x] == 0
        
        # Note: S_old and S_new are disjoint if old_r != new_r
        # So we just sum over the sets.
        
        sum_old = 0
        for x in range(old_r, N, m):
            if count_covered[x] == 1:
                sum_old += 1
                
        sum_new = 0
        for x in range(new_r, N, m):
            if count_covered[x] == 0:
                sum_new += 1
                
        delta = sum_old - sum_new
        
        if delta < 0:
            # Accept
            update_counts(m, old_r, -1)
            update_counts(m, new_r, 1)
            residues[m] = new_r
            uncovered_count += delta
            if uncovered_count < best_score:
                best_score = uncovered_count
                best_residues = residues.copy()
        elif delta == 0:
            # Accept with some probability to allow exploration
            if random.random() < 0.5:
                update_counts(m, old_r, -1)
                update_counts(m, new_r, 1)
                residues[m] = new_r
                # uncovered_count unchanged
        # else reject
        
    # Convert best_residues to list of [m, r]
    classes = [[m, best_residues[m]] for m in divisors]
    return classes
