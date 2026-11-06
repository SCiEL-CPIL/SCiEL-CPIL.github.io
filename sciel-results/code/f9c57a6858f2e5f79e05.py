import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 128
    target_lo = 128
    target_hi = 256
    
    def compute_counts(A):
        """
        Compute representation counts for sums in [0, 256].
        Uses the identity: N[s] = 0.5 * (C[s] + D[s])
        where C is the ordered sum counts (convolution) and D is the diagonal count.
        """
        A_arr = np.array(A, dtype=np.int32)
        if len(A_arr) == 0:
            return np.zeros(257, dtype=np.int32)
        
        # Indicator vector for A
        ind = np.zeros(257, dtype=np.int32)
        ind[A_arr] = 1
        
        # Ordered convolution C
        C = np.convolve(ind, ind, mode='full')
        
        # Diagonal count D: 1 if s is even and s/2 is in A
        D = np.zeros(257, dtype=np.int32)
        even_sums = np.arange(0, 257, 2)
        half_sums = even_sums // 2
        # Check which half_sums are in A
        # Since ind is 0/1, ind[half_sums] works if half_sums are valid indices
        # half_sums are in [0, 128], which is valid for ind (size 257)
        D[even_sums] = ind[half_sums]
        
        # N[s] = 0.5 * (C[s] + D[s])
        # Since C and D are integers, and C[s] + D[s] is always even, 
        # we can do integer division or float then cast.
        counts = (C + D) // 2
        
        return counts
    
    def get_score(counts):
        """Return (uncovered, peak, variance)."""
        target_counts = counts[target_lo:target_hi+1]
        uncovered = np.sum(target_counts == 0)
        peak = np.max(target_counts) if len(target_counts) > 0 else 0
        var = float(np.var(target_counts))
        return (int(uncovered), int(peak), var)
    
    def evaluate(A):
        counts = compute_counts(A)
        return get_score(counts)

    best_A = None
    best_score = (float('inf'), float('inf'), float('inf'))
    
    # Initial Candidates
    candidates = []
    
    # 1. Greedy Construction
    def greedy_build():
        A = set()
        # Start with a spread
        for i in range(0, 129, 10):
            A.add(i)
        
        counts = compute_counts(list(A))
        
        for _ in range(100): # Add up to 100 more elements?
            if time.time() > deadline - 5:
                break
            # Find the sum with max deficit (low count)
            target_counts = counts[target_lo:target_hi+1]
            # We want to add an element that increases counts for low-count sums
            # Simple heuristic: try adding each y in [0, 128] not in A
            # and see which one minimizes the new score
            best_y = None
            best_new_score = None
            
            # Sample candidates to speed up
            avail = [y for y in range(129) if y not in A]
            random.shuffle(avail)
            avail = avail[:20]
            
            for y in avail:
                new_A = list(A) + [y]
                new_counts = compute_counts(new_A)
                new_score = get_score(new_counts)
                
                # Prefer lower uncovered, then lower peak
                if best_new_score is None or (new_score[0], new_score[1]) < (best_new_score[0], best_new_score[1]):
                    best_new_score = new_score
                    best_y = y
            
            if best_y is not None:
                A.add(best_y)
                counts = compute_counts(list(A))
                
            if best_score[0] == 0 and best_score[1] <= 2:
                break
                
        return list(A)

    # Generate a few greedy sets
    for _ in range(5):
        candidates.append(greedy_build())
        
    # Random sets
    for k in [40, 50, 60]:
        for _ in range(5):
            candidates.append(random.sample(range(129), k))
            
    # Evaluate all candidates
    scored_candidates = []
    for A in candidates:
        score = evaluate(A)
        scored_candidates.append((score, A))
        
    scored_candidates.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]))
    
    # Local Search / Simulated Annealing
    def local_search(initial_A, max_steps=20000):
        A = list(initial_A)
        A_set = set(A)
        counts = compute_counts(A)
        curr_score = get_score(counts)
        
        best_A_local = A[:]
        best_score_local = curr_score
        
        temperature = 1.0
        cooling = 0.999
        
        for step in range(max_steps):
            if time.time() > deadline - 2:
                break
                
            # Choose move
            move_type = random.random()
            if move_type < 0.7:
                # Swap
                if len(A) == 0:
                    continue
                x = random.choice(A)
                y = random.randint(0, 128)
                if y in A_set:
                    continue
                
                new_A = A[:]
                new_A.remove(x)
                new_A.append(y)
                
            elif move_type < 0.85:
                # Add
                y = random.randint(0, 128)
                if y in A_set:
                    continue
                new_A = A + [y]
            else:
                # Remove
                if len(A) <= 10:
                    continue
                x = random.choice(A)
                new_A = A[:]
                new_A.remove(x)
                
            new_counts = compute_counts(new_A)
            new_score = get_score(new_counts)
            
            # Acceptance criteria
            # Primary: Uncovered, Peak
            # Secondary: Variance
            # SA: Accept worse if within temperature
            
            # Calculate delta in "score"
            # We map score to a "cost" for SA
            # Cost = Uncovered * 10000 + Peak * 100 + Variance
            curr_cost = curr_score[0] * 10000 + curr_score[1] * 100 + curr_score[2]
            new_cost = new_score[0] * 10000 + new_score[1] * 100 + new_score[2]
            
            delta_cost = new_cost - curr_cost
            
            if delta_cost < 0:
                # Better
                A = new_A
                A_set = set(A)
                counts = new_counts
                curr_score = new_score
                temperature *= cooling
            else:
                # Worse
                prob = np.exp(-delta_cost / (temperature * 100))
                if random.random() < prob:
                    A = new_A
                    A_set = set(A)
                    counts = new_counts
                    curr_score = new_score
                    temperature *= cooling
                else:
                    temperature *= cooling
            
            # Update best
            if (curr_score[0], curr_score[1]) < (best_score_local[0], best_score_local[1]) or \
               (curr_score[0] == best_score_local[0] and curr_score[1] == best_score_local[1] and curr_score[2] < best_score_local[2]):
                best_score_local = curr_score
                best_A_local = A[:]
                
        return best_A_local, best_score_local

    # Run local search on top candidates
    for score, A in scored_candidates[:10]:
        if time.time() > deadline - 5:
            break
        
        A_res, score_res = local_search(A)
        
        if (score_res[0], score_res[1], score_res[2]) < best_score:
            best_score = score_res
            best_A = A_res
            
        # Early exit if perfect
        if best_score[0] == 0 and best_score[1] <= 2:
            break
            
    if best_A is None:
        return list(range(64, 129))
        
    return best_A
