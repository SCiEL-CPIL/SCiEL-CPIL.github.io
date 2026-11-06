import time
import random
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj_mask = [0] * n
    adj_list = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    # Precompute neighbor sets for faster access
    for i in range(n):
        adj_list[i] = [v for v in range(n) if (adj_mask[i] >> v) & 1]
    
    def is_edge(a, b):
        return (adj_mask[a] >> b) & 1
    
    def warnsdorff_score(v, used_mask):
        """Number of unvisited neighbors - lower is more constrained."""
        m = adj_mask[v] & ~used_mask
        score = 0
        while m:
            m &= m - 1
            score += 1
        return score
    
    def greedy_extend_both_ends(path, used_mask, max_iter=1000):
        """Extend path greedily from both ends using Warnsdorff heuristic."""
        for _ in range(max_iter):
            changed = False
            # Try extending from the end
            end = path[-1]
            candidates = [v for v in adj_list[end] if not (used_mask >> v) & 1]
            if candidates:
                candidates.sort(key=lambda v: (warnsdorff_score(v, used_mask), v))
                path.append(candidates[0])
                used_mask |= (1 << candidates[0])
                changed = True
            # Try extending from the start
            start = path[0]
            candidates = [v for v in adj_list[start] if not (used_mask >> v) & 1]
            if candidates:
                candidates.sort(key=lambda v: (warnsdorff_score(v, used_mask), v))
                path.insert(0, candidates[0])
                used_mask |= (1 << candidates[0])
                changed = True
            if not changed:
                break
        return path, used_mask
    
    def beam_search(start, beam_width=50, max_depth=100):
        """Beam search for longest path from start."""
        # Each beam entry: (path, used_mask)
        beams = [([start], 1 << start)]
        best_path = [start]
        
        for depth in range(max_depth):
            if not beams:
                break
            new_beams = []
            seen_masks = set()
            
            for path, mask in beams:
                # Try extending from both ends
                for end_pos in [-1, 0]:
                    end_node = path[end_pos]
                    candidates = [v for v in adj_list[end_node] if not (mask >> v) & 1]
                    if not candidates:
                        continue
                    
                    # Score candidates by Warnsdorff
                    scored = [(warnsdorff_score(v, mask), v) for v in candidates]
                    scored.sort()
                    
                    # Keep top candidates
                    for score, v in scored[:beam_width // 2]:
                        new_mask = mask | (1 << v)
                        if new_mask in seen_masks:
                            continue
                        seen_masks.add(new_mask)
                        
                        new_path = path[:]
                        if end_pos == -1:
                            new_path.append(v)
                        else:
                            new_path.insert(0, v)
                        
                        new_beams.append((new_path, new_mask))
                        if len(new_path) > len(best_path):
                            best_path = new_path[:]
            
            # Keep top beams by path length, then by minimum degree
            new_beams.sort(key=lambda x: (-len(x[0]), min(warnsdorff_score(v, x[1]) for v in x[0] if (x[1] >> v) & 1 and v not in x[0][:1] and v not in x[0][-1:])) if len(x[0]) > 2 else len(x[0]))
            beams = new_beams[:beam_width]
        
        return best_path
    
    def local_search_2opt(path, deadline):
        """Apply 2-opt improvements."""
        improved = True
        while improved:
            improved = False
            lp = len(path)
            for i in range(lp - 2):
                if time.time() > deadline:
                    return path
                for j in range(i + 2, lp):
                    if time.time() > deadline:
                        return path
                    # Check if reversing path[i+1..j] maintains validity
                    if is_edge(path[i], path[j]):
                        if j + 1 < lp:
                            if is_edge(path[i+1], path[j+1]):
                                path[i+1:j+1] = path[i+1:j+1][::-1]
                                improved = True
                                break
                        else:
                            path[i+1:j+1] = path[i+1:j+1][::-1]
                            improved = True
                            break
                if improved:
                    break
        return path
    
    def find_longest_path_with_backtracking(start, used_mask, time_limit, max_depth=100):
        """Iterative DFS with Warnsdorff ordering and backtracking."""
        best_path = [start]
        stack = [(start, 0, [start], used_mask)]
        
        while stack:
            if time.time() > time_limit:
                break
            
            node, idx, path, cur_mask = stack[-1]
            neighbors = [v for v in adj_list[node] if not (cur_mask >> v) & 1]
            neighbors.sort(key=lambda v: (warnsdorff_score(v, cur_mask), v))
            
            if idx < len(neighbors):
                v = neighbors[idx]
                stack[-1] = (node, idx + 1, path, cur_mask)
                new_path = path + [v]
                new_mask = cur_mask | (1 << v)
                if len(new_path) > len(best_path):
                    best_path = new_path[:]
                stack.append((v, 0, new_path, new_mask))
            else:
                stack.pop()
        
        return best_path
    
    deadline = time.time() + 80.0  # 10s safety margin before 90s limit
    best = [0]
    best_mask = 1
    
    # Phase 1: Greedy from every start vertex
    for start in range(n):
        if time.time() > deadline:
            break
        path = [start]
        mask = 1 << start
        path, mask = greedy_extend_both_ends(path, mask)
        if len(path) > len(best):
            best = path[:]
            best_mask = mask
    
    # Phase 2: 2-opt on best path
    if time.time() < deadline:
        best = local_search_2opt(best[:], deadline)
        best_mask = 0
        for v in best:
            best_mask |= (1 << v)
    
    # Phase 3: Extend after 2-opt
    if time.time() < deadline:
        path, mask = greedy_extend_both_ends(best[:], best_mask)
        if len(path) > len(best):
            best = path
            best_mask = mask
    
    # Phase 4: Beam search from multiple starts
    remaining = deadline - time.time()
    if remaining > 10.0:
        per_beam = min(5.0, remaining / 20)
        for start in range(n):
            if time.time() > deadline - 5.0:
                break
            t_limit = time.time() + per_beam
            beam_path = beam_search(start, beam_width=30, max_depth=100)
            if len(beam_path) > len(best):
                best = beam_path
                best_mask = 0
                for v in best:
                    best_mask |= (1 << v)
    
    # Phase 5: 2-opt + extend again
    if time.time() < deadline:
        best = local_search_2opt(best[:], deadline)
        best_mask = 0
        for v in best:
            best_mask |= (1 << v)
        path, mask = greedy_extend_both_ends(best[:], best_mask)
        if len(path) > len(best):
            best = path
            best_mask = mask
    
    # Phase 6: Randomized restarts with Warnsdorff
    rng = random.Random(748)
    remaining = deadline - time.time() - 3.0
    if remaining > 2.0:
        n_restarts = int(min(20, remaining / 1.0))
        for _ in range(n_restarts):
            if time.time() > deadline - 3.0:
                break
            start = rng.randint(0, n - 1)
            path = [start]
            mask = 1 << start
            # Randomized Warnsdorff greedy
            while True:
                end = path[-1]
                candidates = [v for v in adj_list[end] if not (mask >> v) & 1]
                if not candidates:
                    break
                candidates.sort(key=lambda v: (warnsdorff_score(v, mask), v))
                top_k = min(3, len(candidates))
                v = candidates[rng.randint(0, top_k - 1)]
                path.append(v)
                mask |= (1 << v)
            # Extend from both ends
            path, mask = greedy_extend_both_ends(path, mask)
            if len(path) > len(best):
                best = path
                best_mask = mask
    
    # Phase 7: Final 2-opt polish
    if time.time() < deadline - 2.0:
        best = local_search_2opt(best[:], deadline)
        best_mask = 0
        for v in best:
            best_mask |= (1 << v)
        path, mask = greedy_extend_both_ends(best[:], best_mask)
        if len(path) > len(best):
            best = path
    
    return best
