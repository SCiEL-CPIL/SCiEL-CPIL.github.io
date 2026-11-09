import time
import random
import numpy as np

N = 9
MAX_VERTS = 1 << N
DEADLINE = 118.0

def construct():
    start_time = time.time()
    
    # Precompute neighbors
    neighbors = np.empty((MAX_VERTS, N), dtype=np.uint32)
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v, b] = v ^ (1 << b)
    
    # Also keep a list version for the refinement phase
    nbr_lists = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            nbr_lists[v].append(v ^ (1 << b))
    
    best_path = None
    best_len = 0
    rng = np.random.default_rng(seed=42)
    
    def greedy_build(start_v, strategy="min_degree"):
        """Fast greedy snake construction using numpy."""
        path = [start_v]
        used = np.zeros(MAX_VERTS, dtype=bool)
        used[start_v] = True
        forbidden = np.zeros(MAX_VERTS, dtype=bool)
        cur = start_v
        
        while True:
            nbrs = neighbors[cur]
            valid = []
            for n in nbrs:
                if not used[n] and not forbidden[n]:
                    valid.append(n)
            
            if not valid:
                break
            
            if strategy == "min_degree":
                best_score = 100
                best_cands = []
                for n in valid:
                    score = 0
                    for nn in neighbors[n]:
                        if not used[nn] and not forbidden[nn]:
                            score += 1
                    if score < best_score:
                        best_score = score
                        best_cands = [n]
                    elif score == best_score:
                        best_cands.append(n)
                next_v = best_cands[rng.integers(len(best_cands))]
            elif strategy == "max_degree":
                best_score = -1
                best_cands = []
                for n in valid:
                    score = 0
                    for nn in neighbors[n]:
                        if not used[nn] and not forbidden[nn]:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_cands = [n]
                    elif score == best_score:
                        best_cands.append(n)
                next_v = best_cands[rng.integers(len(best_cands))]
            else:  # random
                next_v = valid[rng.integers(len(valid))]
            
            path.append(next_v)
            used[next_v] = True
            for nc in nbrs:
                forbidden[nc] = True
            cur = next_v
        
        return path
    
    def refine_path(path):
        """
        Systematic refinement via gap insertion and bidirectional regrowth.
        Repeatedly:
          1. For each gap in the path, try inserting an unvisited vertex.
          2. After any insertion, try to regrow both ends.
          3. Repeat until no improvement or time runs out.
        """
        if path is None or len(path) < 2:
            return path
        
        current = list(path)
        used = set(current)
        adj_count = {}
        for v in current:
            for w in nbr_lists[v]:
                adj_count[w] = adj_count.get(w, 0) + 1
        
        improved = True
        max_rounds = 200
        
        while improved and time.time() - start_time < DEADLINE - 2:
            improved = False
            
            # Build used set and adj_count efficiently
            used_set = set(current)
            adj_cnt = {}
            for v in current:
                for w in nbr_lists[v]:
                    adj_cnt[w] = adj_cnt.get(w, 0) + 1
            
            # Try gap insertions: for each pair of consecutive vertices (a, b) in path,
            # find unvisited vertex u adjacent to both a and b that can be inserted between them.
            # Actually, in a snake, consecutive vertices are adjacent. We want to insert
            # a vertex u between path[i] and path[i+1] such that:
            #   - u is unvisited
            #   - u is adjacent to path[i] and path[i+1]  (Hamming distance 1 to each)
            #   - u is NOT adjacent to any other path vertex (induced path constraint)
            #   - path[i] and path[i+1] are NOT adjacent to u's other neighbors... 
            #   Wait: in the induced path, non-consecutive vertices must differ in >= 2 bits.
            #   So u must not be adjacent to path[j] for j != i, i+1.
            #   But u IS adjacent to path[i] and path[i+1] (they become consecutive with u).
            #   The old edge path[i]-path[i+1] is removed; new edges are path[i]-u and u-path[i+1].
            #   We need: for all j not in {i, i+1}, dist(u, path[j]) >= 2.
            #   Also, path[i] and path[i+1] must still be valid: they're consecutive with u now.
            #   But path[i] was previously consecutive with path[i-1] (if exists) and path[i+1].
            #   After insertion: path[i-1]-path[i]-u-path[i+1]-path[i+2]
            #   So path[i] is now consecutive with path[i-1] and u.
            #   We need dist(path[i], path[j]) >= 2 for j not in {i-1, i+1} (old constraint) 
            #   and now also for j not in {i-1, i+1}... wait, the constraint is:
            #   non-consecutive vertices in the NEW path must differ in >= 2 bits.
            #   The only new non-consecutive pairs involving u are (u, path[j]) for j not in {i, i+1}.
            #   And (path[i], path[i+1]) are no longer consecutive, so they must differ in >= 2 bits.
            #   But they were consecutive before (differ in 1 bit), so this insertion is INVALID
            #   unless we also ensure path[i] and path[i+1] are not adjacent... but they ARE adjacent.
            #   
            #   So gap insertion between two ADJACENT path vertices is impossible because
            #   removing their edge makes them non-consecutive but they're still at distance 1.
            #
            #   The correct approach: we can only insert u if u is adjacent to path[i] and path[i+1],
            #   AND path[i] and path[i+1] are NOT adjacent (distance >= 2). But in a snake,
            #   consecutive vertices ARE adjacent. So we can't insert between consecutive vertices.
            #
            #   Instead, we should look for "gaps" where we can extend the path by inserting
            #   a vertex that connects to an END of the path, or we do a more complex operation.
            #
            #   Actually, the standard snake improvement is:
            #   - Try to extend one end
            #   - Try to "slide" or "shift" parts of the path
            #
            #   A simpler and effective approach: try to extend both ends greedily.
            #   And try local search: remove last k vertices, regrow.
            
            # Phase 1: Try to extend the right end
            v_end = current[-1]
            valid_ext = []
            for u in nbr_lists[v_end]:
                if u not in used_set:
                    # Check induced path constraint: u must not be adjacent to any
                    # path vertex except v_end (which is the last)
                    ok = True
                    for j in range(len(current) - 1):
                        if current[j] in nbr_lists[u]:
                            ok = False
                            break
                    if ok:
                        # Count onward moves
                        cnt = 0
                        for w in nbr_lists[u]:
                            if w not in used_set and w != v_end:
                                # w must not be adjacent to any path vertex except u's future predecessor
                                ok2 = True
                                for j in range(len(current)):
                                    if current[j] in nbr_lists[w]:
                                        # w is adjacent to current[j]; if j == len(current)-1 (v_end), that's fine
                                        # but w is being considered as next after u, so w must not be adj to current[j] for j <= len(current)-1
                                        # Actually w is new, so it just needs to not be adj to any existing path vertex
                                        # except it will be adj to u (its predecessor)
                                        if j != len(current) - 1:  # not v_end
                                            ok2 = False
                                            break
                                        # if j == len(current)-1, w is adj to v_end, but v_end is not u's predecessor
                                        # u's predecessor is current[-2] (if exists) or none
                                        # w's predecessor will be u, so w must not be adj to current[j] for j != index_of_u
                                        # Since u is new, w must not be adj to any current[j]
                                        ok2 = False
                                        break
                                if not ok2:
                                    continue
                                cnt += 1
                        valid_ext.append((cnt, u))
            
            if valid_ext:
                valid_ext.sort()
                best_score = valid_ext[0][0]
                candidates = [x[1] for x in valid_ext if x[0] <= best_score + 1]
                next_v = random.choice(candidates)
                current.append(next_v)
                used_set.add(next_v)
                improved = True
                continue
            
            # Phase 2: Try to extend the left end
            v_start = current[0]
            valid_ext_l = []
            for u in nbr_lists[v_start]:
                if u not in used_set:
                    ok = True
                    for j in range(1, len(current)):
                        if current[j] in nbr_lists[u]:
                            ok = False
                            break
                    if ok:
                        cnt = 0
                        for w in nbr_lists[u]:
                            if w not in used_set and w != v_start:
                                ok2 = True
                                for j in range(1, len(current)):
                                    if current[j] in nbr_lists[w]:
                                        ok2 = False
                                        break
                                if ok2:
                                    cnt += 1
                        valid_ext_l.append((cnt, u))
            
            if valid_ext_l:
                valid_ext_l.sort()
                best_score = valid_ext_l[0][0]
                candidates = [x[1] for x in valid_ext_l if x[0] <= best_score + 1]
                next_v = random.choice(candidates)
                current.insert(0, next_v)
                used_set.add(next_v)
                improved = True
                continue
            
            # Phase 3: Local search - remove last k, regrow
            for k in range(1, 6):
                if len(current) <= k:
                    break
                orig = current[:]
                truncated = current[:-k]
                used_trunc = set(truncated)
                
                new_path = truncated[:]
                new_used = set(truncated)
                
                # Regrow right end
                v_end = new_path[-1]
                for _ in range(50):  # limited regrow steps
                    valid = []
                    for u in nbr_lists[v_end]:
                        if u not in new_used:
                            ok = True
                            for j in range(len(new_path) - 1):
                                if new_path[j] in nbr_lists[u]:
                                    ok = False
                                    break
                            if ok:
                                cnt = 0
                                for w in nbr_lists[u]:
                                    if w not in new_used and w != v_end:
                                        ok2 = True
                                        for j in range(len(new_path)):
                                            if new_path[j] in nbr_lists[w]:
                                                ok2 = False
                                                break
                                        if ok2:
                                            cnt += 1
                                valid.append((cnt, u))
                    if not valid:
                        break
                    valid.sort()
                    best_score = valid[0][0]
                    candidates = [x[1] for x in valid if x[0] <= best_score + 1]
                    next_v = random.choice(candidates)
                    new_path.append(next_v)
                    new_used.add(next_v)
                    v_end = next_v
                
                if len(new_path) > len(current):
                    current = new_path
                    used_set = new_used
                    improved = True
                    break
            
            # Phase 4: Try inserting a vertex in the middle by "expanding" a gap
            # Find a vertex u not in path that is adjacent to exactly two consecutive path vertices
            # This is complex; skip for now and rely on end-extension + local search
        
        return current
    
    # Phase 1: Multiple greedy starts
    num_starts = 30
    for i in range(num_starts):
        if time.time() - start_time > DEADLINE - 30:
            break
        start_v = int(rng.integers(MAX_VERTS))
        r = rng.random()
        if r < 0.4:
            strategy = "min_degree"
        elif r < 0.7:
            strategy = "max_degree"
        else:
            strategy = "random"
        
        path = greedy_build(start_v, strategy)
        
        if best_path is None or len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Phase 2: Refine the best path
    if best_path is not None:
        refined = refine_path(best_path)
        if len(refined) > best_len:
            best_len = len(refined)
            best_path = refined
    
    # Phase 3: More greedy starts + refinement, using remaining time
    while time.time() - start_time < DEADLINE - 5:
        start_v = int(rng.integers(MAX_VERTS))
        r = rng.random()
        if r < 0.5:
            strategy = "min_degree"
        elif r < 0.8:
            strategy = "max_degree"
        else:
            strategy = "random"
        
        path = greedy_build(start_v, strategy)
        
        # Quick refine
        if len(path) > best_len - 5:  # only refine if close to best
            refined = refine_path(path)
            if len(refined) > best_len:
                best_len = len(refined)
                best_path = refined[:]
        elif len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Final polish: one more refinement pass
    if best_path is not None and time.time() - start_time < DEADLINE - 3:
        final = refine_path(best_path)
        if len(final) > best_len:
            best_len = len(final)
            best_path = final
    
    if best_path is None:
        # Fallback
        best_path = [0]
        used = {0}
        forbidden = set()
        while True:
            cur = best_path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used or v in forbidden:
                    continue
                nxt = v
                break
            if nxt is None:
                break
            best_path.append(nxt)
            used.add(nxt)
            for b in range(N):
                forbidden.add(cur ^ (1 << b))
    
    return best_path
