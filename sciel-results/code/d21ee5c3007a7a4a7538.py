import time
import random
import numpy as np

P = {'n': 12, 'm': 40}

def construct():
    deadline = time.time() + 85
    n = P['n']
    m = P['m']
    N = 1 << n
    
    submasks = [[] for _ in range(N)]
    supermasks = [[] for _ in range(N)]
    
    for z in range(N):
        sub = z
        while True:
            submasks[z].append(sub)
            if sub == 0:
                break
            sub = (sub - 1) & z
        complement = (~z) & (N - 1)
        s = complement
        while True:
            supermasks[z].append(z | s)
            if s == 0:
                break
            s = (s - 1) & complement
    
    # Precompute comparability matrix: comp_mat[a][b] = True if a and b are comparable
    comp_mat = np.zeros((N, N), dtype=bool)
    for a in range(N):
        for b in submasks[a]:
            comp_mat[a][b] = True
        for b in supermasks[a]:
            comp_mat[a][b] = True
    
    def count_comparable(family):
        count = 0
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                if comp_mat[family[i]][family[j]]:
                    count += 1
        return count
    
    def build_comp_in_F(family):
        comp_in_F = np.zeros(N, dtype=np.int32)
        for z in family:
            comp_in_F[submasks[z]] += 1
            comp_in_F[supermasks[z]] += 1
        fam_set = set(family)
        for y in fam_set:
            comp_in_F[y] -= 1
        return comp_in_F
    
    def local_search(family, max_iters, deadline):
        comp_in_F = build_comp_in_F(family)
        current_score = count_comparable(family)
        best_score = current_score
        best_family = family[:]
        fam_set = set(family)
        
        # Precompute numpy array of family masks for vectorized operations
        fam_arr = np.array(family, dtype=np.int32)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            improved = False
            for i in range(m):
                x = family[i]
                k_x = int(comp_in_F[x]) - 1
                
                # Vectorized candidate evaluation
                # For each y not in fam_set, compute val = comp_in_F[y] - (1 if comp_mat[y][x] else 0)
                # We need to exclude y that are in fam_set
                
                # Get indices not in fam_set
                in_fam = np.zeros(N, dtype=bool)
                in_fam[list(fam_set)] = True
                
                # Compute adjusted values
                adj_vals = comp_in_F.copy()
                adj_vals[comp_mat[:, x]] -= 1
                adj_vals[in_fam] = -999999  # Exclude family members
                
                best_idx = np.argmax(adj_vals)
                best_val = int(adj_vals[best_idx])
                
                if best_val <= k_x:
                    continue
                
                best_y = int(best_idx)
                
                # Update comp_in_F for removal of x
                comp_in_F[submasks[x]] -= 1
                comp_in_F[supermasks[x]] -= 1
                comp_in_F[x] += 1
                
                # Update comp_in_F for addition of best_y
                comp_in_F[submasks[best_y]] += 1
                comp_in_F[supermasks[best_y]] += 1
                comp_in_F[best_y] -= 1
                
                fam_set.remove(x)
                fam_set.add(best_y)
                family[i] = best_y
                current_score += best_val - k_x
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
                
                improved = True
            
            if not improved:
                break
                
        return best_family, best_score
    
    def greedy_construct(seed_val):
        random.seed(seed_val)
        family = []
        in_cand = np.ones(N, dtype=bool)
        comp_counts = np.zeros(N, dtype=np.int32)
        
        for _ in range(m):
            # Vectorized: find max comp_counts among candidates
            scores = comp_counts.copy()
            scores[~in_cand] = -999999
            max_score = int(np.max(scores))
            candidates = np.where(scores == max_score)[0]
            
            if len(candidates) == 0:
                break
                
            best_y = int(random.choice(candidates))
            
            family.append(best_y)
            in_cand[best_y] = False
            
            comp_counts[submasks[best_y]] += 1
            comp_counts[supermasks[best_y]] += 1
            # Don't double-count best_y itself (it's excluded from candidates anyway)
            
        return family

    best_overall = None
    best_score_overall = 0
    
    # More restarts possible due to speedup
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        family = greedy_construct(42 + restart)
        
        fam, score = local_search(family, 300, deadline)
        
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam
    
    if best_overall is not None and time.time() < deadline - 5:
        best_overall, best_score_overall = local_search(best_overall, 800, deadline)
    
    if best_overall is None:
        best_overall = list(range(m))
    
    result = []
    for mask in best_overall:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
