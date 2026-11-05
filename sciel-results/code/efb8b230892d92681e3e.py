import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    def get_idx(i, j, n):
        if i > j:
            i, j = j, i
        return i * (i - 1) // 2 + (j - i - 1)
    
    def try_local_search(n, initial_colors, max_steps=30000):
        colors = list(initial_colors)
        
        # neighbors[c][v] = set of neighbors of v in color c
        neighbors = [set() for _ in range(5)]
        for c in range(5):
            neighbors[c] = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                neighbors[c][i].add(j)
                neighbors[c][j].add(i)
        
        # common_count[c][u][v] = |N_c(u) ∩ N_c(v)| for u < v
        # bad_count = sum over c, u<v of C(common_count[c][u][v], 2)
        
        # Precompute common counts
        common = [[[0]*n for _ in range(n)] for _ in range(5)]
        bad_total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    cnt = len(neighbors[c][u] & neighbors[c][v])
                    common[c][u][v] = cnt
                    if cnt >= 2:
                        bad_total += cnt * (cnt - 1) // 2
        
        if bad_total == 0:
            return colors, True
        
        best_bad = bad_total
        best_colors = list(colors)
        
        step = 0
        while step < max_steps:
            if time.time() > deadline - 10:
                break
            
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            idx = get_idx(i, j, n)
            
            old_c = colors[idx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            # Compute delta in bad_total
            # Removing edge (i,j) from color old_c:
            #   For each other vertex k, common[old_c][i][k] and common[old_c][j][k] change
            # Adding edge (i,j) to color new_c:
            #   For each other vertex k, common[new_c][i][k] and common[new_c][j][k] change
            
            delta = 0
            
            # Effect of removing (i,j) from old_c
            # For each k != i,j: 
            #   if k in neighbors[old_c][i]: common[old_c][i][k] decreases by 1 (if j was in N(old_c,k) and i in N(old_c,k))
            #   Actually: common[old_c][i][k] = |N_c(i) ∩ N_c(k)|. Removing j from N_c(i) affects this only if j in N_c(k).
            
            for k in range(n):
                if k == i or k == j:
                    continue
                # old_c: remove j from N(i), remove i from N(j)
                # common[old_c][i][k]: was |Ni ∩ Nk|. j removed from Ni. If j in Nk, count decreases by 1.
                if j in neighbors[old_c][k]:
                    old_cnt = common[old_c][i][k]
                    new_cnt = old_cnt - 1
                    delta -= (old_cnt*(old_cnt-1)//2) - (new_cnt*(new_cnt-1)//2)
                    common[old_c][i][k] = new_cnt
                    common[old_c][k][i] = new_cnt
                
                # common[old_c][j][k]: i removed from Nj. If i in Nk, count decreases by 1.
                if i in neighbors[old_c][k]:
                    old_cnt = common[old_c][j][k]
                    new_cnt = old_cnt - 1
                    delta -= (old_cnt*(old_cnt-1)//2) - (new_cnt*(new_cnt-1)//2)
                    common[old_c][j][k] = new_cnt
                    common[old_c][k][j] = new_cnt
                
                # common[old_c][i][j]: both i and j lose a neighbor? No, we're removing edge (i,j) itself.
                # common[old_c][i][j] = |Ni ∩ Nj|. This doesn't directly change from removing (i,j), 
                # but Ni and Nj change. Actually removing (i,j) means j not in Ni and i not in Nj.
                # common[i][j] = |Ni ∩ Nj|. j is in Ni? No, j was in Ni (since (i,j) edge). After removal, j not in Ni.
                # So if j was in Nj (no, j not in Nj since no self-loops). 
                # Actually common[i][j] counts vertices w such that w in Ni and w in Nj.
                # Removing (i,j): j removed from Ni, i removed from Nj.
                # j in Nj? No. i in Ni? No. So common[i][j] unchanged? 
                # Wait: common[i][j] = |{w : w in Ni AND w in Nj}|. 
                # Before: j in Ni (yes), is j in Nj? No. i in Nj (yes), is i in Ni? No.
                # So removing (i,j) doesn't change common[i][j] because j wasn't in Nj and i wasn't in Ni.
                # But wait, we need to also handle the pair (i,j) itself.
                # common[old_c][i][j]: the intersection Ni ∩ Nj. Removing j from Ni: if j in Nj? No. 
                # Removing i from Nj: if i in Ni? No. So no change.
                
                # new_c: add j to N(i), add i to N(j)
                # common[new_c][i][k]: j added to Ni. If j in Nk, count increases by 1.
                if j in neighbors[new_c][k]:
                    old_cnt = common[new_c][i][k]
                    new_cnt = old_cnt + 1
                    delta += (new_cnt*(new_cnt-1)//2) - (old_cnt*(old_cnt-1)//2)
                    common[new_c][i][k] = new_cnt
                    common[new_c][k][i] = new_cnt
                
                if i in neighbors[new_c][k]:
                    old_cnt = common[new_c][j][k]
                    new_cnt = old_cnt + 1
                    delta += (new_cnt*(new_cnt-1)//2) - (old_cnt*(old_cnt-1)//2)
                    common[new_c][j][k] = new_cnt
                    common[new_c][k][j] = new_cnt
            
            # Also handle common[i][j] for old_c and new_c
            # old_c: removing (i,j). common[old_c][i][j] = |Ni ∩ Nj|.
            # j in Ni (yes, being removed), j in Nj? No. i in Nj (yes, being removed), i in Ni? No.
            # So common[old_c][i][j] unchanged.
            # new_c: adding (i,j). common[new_c][i][j] = |Ni ∩ Nj|.
            # j added to Ni, is j in Nj? No. i added to Nj, is i in Ni? No.
            # So common[new_c][i][j] unchanged.
            
            new_bad = bad_total + delta
            
            if new_bad < bad_total:
                # Accept
                bad_total = new_bad
                neighbors[old_c][i].discard(j)
                neighbors[old_c][j].discard(i)
                neighbors[new_c][i].add(j)
                neighbors[new_c][j].add(i)
                colors[idx] = new_c
                if bad_total == 0:
                    return colors, True
                if bad_total < best_bad:
                    best_bad = bad_total
                    best_colors = list(colors)
            else:
                # Revert common counts
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if j in neighbors[old_c][k]:
                        common[old_c][i][k] += 1
                        common[old_c][k][i] += 1
                    if i in neighbors[old_c][k]:
                        common[old_c][j][k] += 1
                        common[old_c][k][j] += 1
                    if j in neighbors[new_c][k]:
                        common[new_c][i][k] -= 1
                        common[new_c][k][i] -= 1
                    if i in neighbors[new_c][k]:
                        common[new_c][j][k] -= 1
                        common[new_c][k][j] -= 1
                # No neighbor changes, no color change
            
            step += 1
        
        if best_bad < bad_total:
            return best_colors, (best_bad == 0)
        return colors, (bad_total == 0)
    
    for n in range(12, 18):
        if time.time() > deadline - 40:
            break
        
        found = False
        for restart in range(10):
            if time.time() > deadline - 15:
                break
            
            random.seed(n * 99991 + restart * 31)
            initial = [random.randint(0, 4) for _ in range(n*(n-1)//2)]
            
            result_colors, is_valid = try_local_search(n, initial, max_steps=25000)
            
            if is_valid:
                best_n = n
                best_colors = result_colors
                found = True
                break
        
        if not found:
            break
    
    if best_n < 11:
        best_n = 11
        best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    return {'n': best_n, 'colors': best_colors}
