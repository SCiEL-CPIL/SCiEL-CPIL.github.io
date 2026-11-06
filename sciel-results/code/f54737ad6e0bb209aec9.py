import time
import random
import sys

N = 9
NUM_VERTICES = 1 << N

# Precompute neighbor masks for all vertices
nbr_masks = [0] * NUM_VERTICES
for v in range(NUM_VERTICES):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    nbr_masks[v] = m

# Precompute the list of neighbor bit values (the bit positions)
nbr_bits = [0] * NUM_VERTICES
for v in range(NUM_VERTICES):
    bits = []
    for b in range(N):
        bits.append(1 << (v ^ (1 << b)))
    nbr_bits[v] = bits

def construct():
    start_time = time.time()
    deadline = start_time + 114.0  # 6s safety margin
    
    best_path = []
    best_len = 0
    
    def score_path(path):
        """
        Score a path based on the number of available moves from the end,
        weighted by the availability from the start as well (to ensure symmetry considerations).
        Actually, for extension, only the end matters.
        """
        if not path:
            return 0
        last = path[-1]
        used_mask = 0
        for v in path:
            used_mask |= 1 << v
        
        # Disallowed: neighbors of all vertices except the last one
        disallowed = 0
        for v in path[:-1]:
            disallowed |= nbr_masks[v]
        
        # Available moves from the end
        avail_mask = nbr_masks[last] & ~used_mask & ~disallowed
        score = avail_mask.bit_count()
        
        # Heuristic: also consider the "bottleneck" in the middle?
        # No, too slow.
        return score

    def try_extend(path):
        """
        Attempt to extend the path using a heuristic greedy/beam approach.
        Returns the extended path.
        """
        used_mask = 0
        for v in path:
            used_mask |= 1 << v
        
        disallowed = 0
        for v in path[:-1]:
            disallowed |= nbr_masks[v]
        
        cur_path = list(path)
        
        # Use a small beam search to extend
        # State: (path, used, disallowed)
        # Start with the current path
        # Beam width 10-20 for quick extension
        
        beam = [(cur_path, used_mask, disallowed)]
        max_depth = 50 # Try to extend by at most 50 steps quickly
        
        for _ in range(max_depth):
            if time.time() > deadline:
                break
            
            next_beam = []
            for p, u, d in beam:
                last = p[-1]
                cand_mask = nbr_masks[last] & ~u & ~d
                
                # Generate candidates
                cands = []
                cm = cand_mask
                while cm:
                    lsb = cm & -cm
                    n = lsb.bit_length() - 1
                    cands.append(n)
                    cm ^= lsb
                
                if not cands:
                    continue
                
                # Score candidates:
                # For each candidate n, calculate the number of moves from n
                # New state: p + [n], u | (1<<n), d | nbr_masks[last]
                new_d_base = d | nbr_masks[last]
                
                scored_cands = []
                for n in cands:
                    new_u = u | (1 << n)
                    new_d = new_d_base
                    # Moves from n
                    avail = nbr_masks[n] & ~new_u & ~new_d
                    s = avail.bit_count()
                    scored_cands.append((s, n))
                
                scored_cands.sort(key=lambda x: x[0], reverse=True)
                
                # Keep top K candidates (e.g. 3) to diversify
                for s, n in scored_cands[:3]:
                    new_u = u | (1 << n)
                    new_d = new_d_base
                    next_beam.append((p + [n], new_u, new_d))
            
            if not next_beam:
                break
            
            # Prune beam
            # Score states in next_beam
            scored_states = []
            for p, u, d in next_beam:
                last = p[-1]
                avail = nbr_masks[last] & ~u & ~d
                s = avail.bit_count()
                scored_states.append((s, p, u, d))
            
            scored_states.sort(key=lambda x: x[0], reverse=True)
            beam = [(p, u, d) for s, p, u, d in scored_states[:20]]
            
            # Update best
            if len(beam[0][0]) > len(cur_path):
                cur_path = beam[0][0]
            
            # If no state has moves left, stop
            if all(scored_states[i][0] == 0 for i in range(len(scored_states))):
                break
        
        return cur_path

    # Main loop: Random restarts with local search
    
    # 1. Generate a random initial path of length 10-20
    # 2. Extend it
    # 3. Try to "swap" or "repair" if possible?
    #    For now, just random restarts + extension.
    
    # To get diversity, we can also try starting from different vertices.
    # Due to symmetry, starting from 0 is WLOG for the first restart.
    # For subsequent restarts, random start.
    
    # Also, we can use the "island" idea: keep a few best paths and try to merge them?
    # Too complex.
    
    # Let's stick to:
    # 1. Random start vertex
    # 2. Greedy/Beam extension
    # 3. Keep best
    
    # To improve further, we can use a "perturbation" step:
    # Take the best path, remove a random vertex from the middle, and try to reconnect?
    # This is a 2-opt style move.
    # If we remove vertex v from path P = ... u, v, w ...
    # We need to connect u to w? No, u and w are not necessarily neighbors.
    # We need to find a path from u to w that avoids all other vertices in P.
    # This is a shortest path problem in the remaining graph.
    # If a path exists of length L, we replace v with that path.
    # The length changes by L - 1.
    # If L > 1, we gain length.
    
    def try_2opt(path):
        """
        Try to improve the path by removing a vertex and reconnecting via a shortest path.
        """
        if len(path) < 5:
            return path
        
        # Pick a random vertex to remove (not the ends)
        idx = random.randint(1, len(path) - 2)
        v = path[idx]
        u = path[idx - 1]
        w = path[idx + 1]
        
        # Build used mask
        used_mask = 0
        for p in path:
            used_mask |= 1 << p
        
        # We want a path from u to w avoiding all vertices in `path` except u and w?
        # No, avoiding all vertices in `path` except u and w is correct for an induced path.
        # The new path will be ... u, ..., w ...
        # The vertices in the new subpath must be distinct from all other vertices in the original path.
        # So forbidden = used_mask ^ (1 << u) ^ (1 << w)
        
        forbidden = used_mask & ~(1 << u) & ~(1 << w)
        
        # BFS from u to w in the graph induced by V \ forbidden
        # Note: u and w are allowed.
        
        # Since the hypercube is small (512 nodes), BFS is fast.
        # But we do this many times.
        
        # Optimization: Only try this if we have time.
        
        from collections import deque
        
        queue = deque([(u, [u])])
        visited = {u}
        
        found = None
        
        while queue:
            if time.time() > deadline:
                return path
            node, p = queue.popleft()
            if node == w:
                found = p
                break
            
            for n in nbr_bits[node]:
                if n in forbidden:
                    continue
                if n in visited:
                    continue
                visited.add(n)
                queue.append((n, p + [n]))
        
        if found:
            # found is a path from u to w.
            # Check if found has length > 1 (i.e. we replaced v with a longer path)
            # Actually, even if length is 1 (u=w? no, u!=w), length >= 2.
            # If found is [u, w], it means u and w are neighbors.
            # Then we remove v and connect u-w directly. Length change: -1. Bad.
            # We want length > 1.
            if len(found) > 1:
                # Replace path[idx] with found
                # But wait, `found` starts with u and ends with w.
                # So new path = path[:idx] + found[1:] + path[idx+1:]
                # Wait, path[:idx] ends at u? No, path[idx-1] is u.
                # path = [..., u, v, w, ...]
                # path[:idx] = [..., u]
                # found[1:] = [..., w] (skips u)
                # path[idx+1:] = [w, ...]
                # So: path[:idx] + found[1:-1] + path[idx+1:] ?
                # No.
                # path[:idx] contains u at the end? Yes, index idx-1.
                # We want to remove v (index idx) and insert the path between u and w.
                # The path is u, ..., w.
                # So we replace v with the intermediate nodes of `found`.
                # New segment: u, found[1], ..., found[-2], w
                # So we insert found[1:-1] at index idx.
                
                if len(found) > 2: # i.e. there are intermediate nodes
                    new_path = path[:idx] + found[1:-1] + path[idx+1:]
                    # Verify validity?
                    # The BFS ensures distinctness from forbidden.
                    # Consecutiveness is guaranteed by BFS.
                    # So it should be valid.
                    return new_path
        
        return path

    # Loop
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # Random start
        start_v = random.randint(0, NUM_VERTICES - 1)
        
        # Initial path
        path = [start_v]
        
        # Extend
        path = try_extend(path)
        
        # Try some 2-opt moves
        for _ in range(5):
            if time.time() > deadline:
                break
            new_path = try_2opt(path)
            if len(new_path) > len(path):
                path = new_path
            else:
                # If it failed, try again with a different random idx?
                # try_2opt already picks random idx.
                pass
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
            
        # Print progress? No, too slow.
        
    return best_path

if __name__ == "__main__":
    p = construct()
    print(f"Length: {len(p)}")
    # Quick verification
    valid = True
    for i in range(len(p)-1):
        if bin(p[i] ^ p[i+1]).count('1') != 1:
            valid = False
            break
    for i in range(len(p)):
        for j in range(i+2, len(p)):
            if bin(p[i] ^ p[j]).count('1') == 1:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
