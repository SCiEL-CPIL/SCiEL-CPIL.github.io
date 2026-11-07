import time
import random
import math
import numpy as np

P = {'n': 12, 'm': 40}

def construct():
    deadline = time.time() + 85
    n = P['n']
    m = P['m']
    N = 1 << n
    
    # Precompute comparability matrix as NumPy array
    # comp_mat_np[a][b] = True if a and b are comparable (a subset b or b subset a)
    # Note: a is comparable to a, but we deal with distinct sets.
    comp_mat_np = np.zeros((N, N), dtype=bool)
    for a in range(N):
        for b in range(N):
            if (a & b) == a or (a & b) == b:
                comp_mat_np[a][b] = True
    
    # For faster access in Python loops if needed, but we will use numpy vectorization
    # Let's precompute the comparable masks for each node as a list of indices? 
    # No, numpy boolean indexing is fast enough for N=4096.

    def count_comparable(family):
        count = 0
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                if comp_mat_np[family[i]][family[j]]:
                    count += 1
        return count

    def greedy_init(seed=42):
        random.seed(seed)
        F = []
        fam_set = set()
        # comp_in_F[y] = number of elements in F comparable to y
        comp_in_F = np.zeros(N, dtype=int)
        
        # Start with a random set
        start_mask = random.randint(0, N-1)
        F.append(start_mask)
        fam_set.add(start_mask)
        comp_in_F[comp_mat_np[:, start_mask]] += 1
        
        while len(F) < m:
            scores = comp_in_F.copy()
            # Mask out current family
            for y in fam_set:
                scores[y] = -1000000
            
            best_y = np.argmax(scores)
            best_val = scores[best_y]
            
            if best_val < 0:
                # Should not happen as there are 4096 sets and we only pick 40
                break
                
            F.append(best_y)
            fam_set.add(best_y)
            comp_in_F[comp_mat_np[:, best_y]] += 1
            
        return F

    def local_search(family, max_iters, deadline, use_sa=False, start_temp=10.0, end_temp=0.1):
        # family is a list of masks
        comp_in_F = np.zeros(N, dtype=int)
        
        for z in family:
            comp_in_F[comp_mat_np[:, z]] += 1
        
        current_score = count_comparable(family)
        best_score = current_score
        best_family = family[:]
        
        fam_set = set(family)
        
        if use_sa:
            current_temp = start_temp
            cooling_factor = (end_temp / start_temp) ** (1.0 / max(1, max_iters))
        
        for iteration in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # Randomly pick an element to swap out
            i = random.randint(0, m - 1)
            x = family[i]
            
            # Calculate gain for removing x: k_x = comp_in_F[x] - 1
            k_x = comp_in_F[x] - 1
            
            # Calculate scores for all y not in F
            # Score(y) = (comp_in_F[y] - (1 if y comparable to x else 0)) - k_x
            # We want to maximize this.
            
            # Vectorized calculation
            # comp_in_F is the count of comparable sets in F for each y.
            # If we remove x, the count for y becomes comp_in_F[y] - (1 if x~y else 0).
            # The gain from adding y is that new count.
            # The loss from removing x is k_x.
            # Delta = (comp_in_F[y] - (1 if x~y else 0)) - k_x
            
            # We need to mask out y in F.
            scores = comp_in_F.copy()
            
            # Subtract 1 if y is comparable to x
            # comp_mat_np[:, x] is a boolean vector where True means y is comparable to x
            scores[comp_mat_np[:, x]] -= 1
            
            # Subtract k_x from all scores (constant shift, doesn't change argmax, but useful for SA delta)
            # Actually, for argmax, we can just ignore k_x.
            # For SA, we need delta.
            
            # Mask out current family
            for y in fam_set:
                scores[y] = -1000000
            
            best_y = np.argmax(scores)
            best_val = scores[best_y]
            
            # The actual delta is best_val - k_x
            delta = best_val - k_x
            
            if delta > 0:
                # Accept
                # Update comp_in_F
                comp_in_F[comp_mat_np[:, best_y]] += 1
                comp_in_F[comp_mat_np[:, x]] -= 1
                
                fam_set.remove(x)
                fam_set.add(best_y)
                family[i] = best_y
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
                
                if use_sa:
                    current_temp *= cooling_factor
            elif use_sa and random.random() < math.exp(delta / current_temp):
                # Accept worse solution
                comp_in_F[comp_mat_np[:, best_y]] += 1
                comp_in_F[comp_mat_np[:, x]] -= 1
                
                fam_set.remove(x)
                fam_set.add(best_y)
                family[i] = best_y
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
                
                current_temp *= cooling_factor
            else:
                # Reject (or just move on in SA)
                if use_sa:
                    current_temp *= cooling_factor
                else:
                    # In Hill Climbing, if the best possible swap doesn't improve, we might be stuck.
                    # We can try to pick a random y to escape? No, standard HC stops or tries all.
                    # This implementation tries the best y. If it fails, we break or continue?
                    # If the best y doesn't improve, no other y will improve.
                    # So for pure HC, we can break.
                    # But we are doing random i. So we continue to next i.
                    pass
                    
        return best_family, best_score

    best_overall = None
    best_score_overall = 0
    
    # Run multiple restarts
    # Use a mix of Greedy Init and SA/HC Local Search
    
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Alternate between Greedy and Random init
        if restart % 2 == 0:
            family = greedy_init(seed=100 + restart)
        else:
            # Random init
            random.seed(200 + restart)
            family = random.sample(range(N), m)
        
        # Run local search
        # Use SA for some, HC for others
        use_sa = (restart % 3 == 0)
        max_iters = 1000
        fam, score = local_search(family, max_iters, deadline, use_sa=use_sa)
        
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam[:]
            
    if best_overall is None:
        best_overall = list(range(m))
        
    result = []
    for mask in best_overall:
        result.append([b for b in range(n) if (mask >> b) & 1])
        
    return result
