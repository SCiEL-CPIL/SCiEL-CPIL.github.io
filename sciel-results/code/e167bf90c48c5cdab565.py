import time
import sys
from random import Random
from collections import deque
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _greedy_path_cover_fast(edges, n):
        rem_edges = set(tuple(sorted(e)) for e in edges)
        adj = {i: set() for i in range(n)}
        for u, v in rem_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        paths = []
        
        while rem_edges:
            min_deg = float('inf')
            start_v = -1
            for v in range(n):
                deg = len(adj[v])
                if deg > 0 and deg < min_deg:
                    min_deg = deg
                    start_v = v
            
            if start_v == -1:
                break

            path = [start_v]
            visited_in_path = {start_v}
            current_v = start_v
            
            while True:
                candidates = []
                for nb in adj[current_v]:
                    if nb not in visited_in_path:
                        e_key = tuple(sorted((current_v, nb)))
                        if e_key in rem_edges:
                            candidates.append((len(adj[nb]), nb))
                
                if not candidates:
                    break
                
                candidates.sort()
                next_v = candidates[0][1]
                
                e_key = tuple(sorted((current_v, next_v)))
                rem_edges.discard(e_key)
                
                adj[current_v].discard(next_v)
                adj[next_v].discard(current_v)
                
                path.append(next_v)
                visited_in_path.add(next_v)
                current_v = next_v
            
            if len(path) > 1:
                paths.append(path)
            else:
                break

        return paths

    def _is_valid(paths, edges, n):
        edge_set = set(tuple(sorted(e)) for e in edges)
        used = set()
        for p in paths:
            if len(p) < 2:
                return False
            seen = set()
            for v in p:
                if v in seen:
                    return False
                seen.add(v)
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i + 1])))
                if e not in edge_set:
                    return False
                if e in used:
                    return False
                used.add(e)
        return used == edge_set

    def _get_endpoints(path):
        if not path:
            return None, None
        return path[0], path[-1]

    def _try_merge(paths, i, j):
        """
        Try to merge paths[i] and paths[j].
        Returns True if merged, False otherwise.
        Modifies paths in place if successful.
        """
        p1 = paths[i]
        p2 = paths[j]
        
        e1_s, e1_e = _get_endpoints(p1)
        e2_s, e2_e = _get_endpoints(p2)
        
        # Check if they share an endpoint
        # Case 1: e1_s == e2_s
        # Case 2: e1_s == e2_e
        # Case 3: e1_e == e2_s
        # Case 4: e1_e == e2_e
        
        # We need to ensure the merged path is simple (no repeated vertices)
        
        def check_simple(p):
            return len(p) == len(set(p))
            
        # Try all 4 combinations of joining endpoints
        # Join e1_e to e2_s: p1 + p2
        # Join e1_e to e2_e: p1 + p2_reversed
        # Join e1_s to e2_s: p1_reversed + p2
        # Join e1_s to e2_e: p1_reversed + p2_reversed
        
        candidates = []
        
        # 1. p1 + p2 (if e1_e == e2_s)
        if e1_e == e2_s:
            merged = p1 + p2[1:]
            if check_simple(merged):
                candidates.append(merged)
                
        # 2. p1 + p2_reversed (if e1_e == e2_e)
        if e1_e == e2_e:
            merged = p1 + p2[::-1][1:]
            if check_simple(merged):
                candidates.append(merged)
                
        # 3. p1_reversed + p2 (if e1_s == e2_s)
        if e1_s == e2_s:
            merged = p1[::-1] + p2[1:]
            if check_simple(merged):
                candidates.append(merged)
                
        # 4. p1_reversed + p2_reversed (if e1_s == e2_e)
        if e1_s == e2_e:
            merged = p1[::-1] + p2[::-1][1:]
            if check_simple(merged):
                candidates.append(merged)
                
        if not candidates:
            return False
            
        # Pick the first valid one (or longest, doesn't matter much)
        paths[i] = candidates[0]
        # Remove paths[j]
        paths.pop(j)
        return True

    def _local_search(paths, n, deadline):
        rng = Random(42)
        current_paths = [list(p) for p in paths]
        current_count = len(current_paths)
        
        # Precompute edge set for validity check if needed, but we trust the structure
        # We only need to ensure simplicity and edge existence.
        
        while time.time() < deadline:
            # 1. Greedy Merge: Try to find any mergeable pair
            merged_any = False
            for i in range(len(current_paths)):
                for j in range(i + 1, len(current_paths)):
                    if _try_merge(current_paths, i, j):
                        current_count -= 1
                        merged_any = True
                        break
                if merged_any:
                    break
                    
            if merged_any:
                continue
                
            # 2. If no greedy merge, try Split-Merge
            # Pick a random path to split
            if not current_paths:
                break
                
            # Prioritize splitting longer paths or paths with high degree vertices?
            # Just random for now
            idx = rng.randint(0, len(current_paths) - 1)
            p = current_paths[idx]
            
            if len(p) < 3:
                # Cannot split a path of length 2 (2 vertices, 1 edge) into two non-empty paths
                # Actually, splitting [u, v] at u or v is not possible for internal split.
                # We need an internal vertex.
                # If len(p) == 2, no internal vertex.
                # If len(p) == 3, u-v-w. Split at v -> [u,v] and [v,w].
                continue
                
            # Pick a random internal vertex
            # Vertices are p[1:-1]
            internal_verts = p[1:-1]
            if not internal_verts:
                continue
                
            split_v = rng.choice(internal_verts)
            
            # Find index of split_v in p
            # It might appear multiple times? No, path is simple.
            split_idx = p.index(split_v)
            
            # Split p into p1 and p2
            p1 = p[:split_idx+1]
            p2 = p[split_idx:]
            
            # Replace p with p1 and p2
            current_paths[idx] = p1
            current_paths.insert(idx + 1, p2)
            current_count += 1
            
            # Now try to merge p1 or p2 with others
            # Try to merge p1 (at idx)
            merged_p1 = False
            for j in range(len(current_paths)):
                if j == idx:
                    continue
                if _try_merge(current_paths, idx, j):
                    current_count -= 1
                    merged_p1 = True
                    break
                    
            if not merged_p1:
                # Try to merge p2 (at idx+1 if p1 wasn't merged, or idx if p1 was merged and p2 shifted?)
                # If p1 was merged, p2 is now at idx (if j < idx) or idx+1 (if j > idx)?
                # _try_merge pops j.
                # If j < idx, p1 is at idx, p2 is at idx+1. Merging idx and j pops j.
                # p1 is now at idx. p2 is at idx.
                # If j > idx, p1 is at idx, p2 is at idx+1. Merging idx and j pops j.
                # p1 is at idx. p2 is at idx+1.
                
                # Let's just re-scan all pairs if we want to be safe, or track indices carefully.
                # Given small size, re-scanning is fine.
                
                # If p1 didn't merge, try p2
                # p2 is at idx+1
                idx_p2 = idx + 1
                for j in range(len(current_paths)):
                    if j == idx_p2:
                        continue
                    if _try_merge(current_paths, idx_p2, j):
                        current_count -= 1
                        break
                        
            # If we split and couldn't merge back to reduce count, we have increased count.
            # We should accept this only if it leads to a better state later.
            # For now, if count increased, we might want to revert if no merge happened.
            # But if a merge happened, count is same as before split.
            # If no merge happened, count is +1.
            
            # Heuristic: If count increased, revert?
            # If we always revert when count increases, we get stuck.
            # We need to allow temporary increases.
            # But for simplicity in this time-boxed script, let's just keep it if it allows future merges.
            # Actually, if we split and don't merge, we have a worse solution.
            # We should probably revert if no merge occurred.
            
            # Check if any merge occurred
            # If merged_p1 is True, count decreased by 1 from the split increase. Net 0.
            # If merged_p1 is False, we tried p2.
            # This logic is getting messy with indices.
            
            # Simpler: Just keep the state. The greedy merge loop at the start of the while
            # will try to fix it.
            
        return current_paths

    def path_cover(edges, n):
        if not edges:
            return []
        
        res = _greedy_path_cover_fast(edges, n)
        
        if res and _is_valid(res, edges, n):
            # Run local search
            deadline = time.time() + 100 # Leave 10s for other graphs
            res = _local_search(res, n, deadline)
            if _is_valid(res, edges, n):
                return res
            else:
                return _greedy_path_cover_fast(edges, n)
        
        fallback = []
        for e in edges:
            fallback.append([e[0], e[1]])
        return fallback

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE - 10:
            while len(results) < 3:
                edges = instance(*spec)
                results.append([list(e) for e in edges])
            break
        
        edges = instance(*spec)
        n = spec[0]
        paths = path_cover(edges, n)
        results.append(paths)
        
    return results

if __name__ == "__main__":
    res = construct()
    print("Number of paths:", [len(r) for r in res])
    print("Total paths:", sum(len(r) for r in res))
