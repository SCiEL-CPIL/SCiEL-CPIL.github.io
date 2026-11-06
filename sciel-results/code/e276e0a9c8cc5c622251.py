"""
Improved Snake-in-the-Box solver for Q_9.
Uses a hybrid approach: Fast Greedy Extension + Limited Backtracking (DFS).
"""
N = 9
import time
import random
import sys

def construct():
    deadline = time.time() + 145.0  # Safety margin
    best_path = None
    best_len = 0
    
    # Precompute neighbors for each vertex
    neighbors = [[] for _ in range(1 << N)]
    for v in range(1 << N):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    def greedy_extend(start_v, path, used, block_count, time_limit):
        """
        Extends the path greedily from start_v.
        path: list of vertices, path[-1] is current end.
        used: boolean array of used vertices.
        block_count: array tracking how many path vertices are at distance 1 from v.
        """
        cur = start_v
        while time.time() < time_limit:
            candidates = []
            for v in neighbors[cur]:
                if not used[v] and block_count[v] == 0:
                    candidates.append(v)
            
            if not candidates:
                break
                
            # Pick a random candidate
            v = random.choice(candidates)
            
            # Add v to path
            path.append(v)
            used[v] = True
            
            # Update block_count: neighbors of the previous vertex (cur) are now blocked
            # because they are now at distance 1 from the new vertex v? No.
            # The rule: non-consecutive vertices must differ by >= 2 bits.
            # If we add v, then all neighbors of the *previous* vertex (path[-2]) become "blocked" 
            # in the sense that they cannot be added next, because they would be at distance 1 
            # from v? No, they are at distance 1 from path[-2]. 
            # Wait, the condition is: new vertex v must be at distance >= 2 from ALL previous vertices except the last one.
            # So, when we add v, we must ensure v is not a neighbor of any path[i] for i < len(path)-1.
            # The block_count[v] tracks the number of neighbors of v that are in the path.
            # If block_count[v] > 0, then v is a neighbor of some path[i].
            # If that path[i] is path[-1] (the current end), it's allowed (distance 1).
            # If that path[i] is any other path[j] (j < len(path)-1), it's forbidden.
            
            # Let's re-evaluate the block_count logic.
            # block_count[v] = number of u in path such that dist(v, u) == 1.
            # When we add v to path, for any neighbor w of the *previous* end (path[-2]), 
            # w is now at distance 1 from v? No.
            # The new constraint is on the *next* vertex to be added.
            # Let the current path be P = [p0, ..., pk]. We add pk+1.
            # pk+1 must not be a neighbor of p0, ..., pk-1.
            # It IS allowed to be a neighbor of pk.
            
            # So, for a candidate vertex v, it is valid if:
            # 1. not used[v]
            # 2. The only neighbor of v in the current path is pk (the current end).
            
            # How to track this?
            # Let count[v] be the number of neighbors of v that are in the path.
            # If count[v] == 0, v is not a neighbor of any path vertex. Valid? 
            # If we add v, the new end is v. The next vertex x must not be a neighbor of p0...pk.
            # x can be a neighbor of v.
            
            # Actually, the standard optimization is:
            # When we move from u to v (u=path[-1], v=new), 
            # the neighbors of u become "blocked" for future steps?
            # No. The neighbors of u are now at distance 1 from v? No, dist(u, v)=1.
            # dist(neighbor_of_u, v) can be 0, 1, 2.
            
            # Let's stick to the simple check:
            # A vertex v is a valid next step if:
            # - v is not used
            # - v is not a neighbor of any p_i for i < len(path)-1
            # This is equivalent to: 
            # - v is a neighbor of path[-1]
            # - AND the number of neighbors of v in the path is exactly 1 (which is path[-1]).
            
            # So we maintain `block_count[v]` = number of neighbors of v in the path.
            # Initially, block_count is 0 for all.
            # When we add a vertex u to the path, we increment block_count[w] for all w in neighbors[u].
            # Then, a vertex v is a valid candidate if:
            # - v is a neighbor of current_end
            # - block_count[v] == 1 (meaning its only neighbor in the path is current_end)
            #   Wait, if v is a neighbor of current_end, and block_count[v] is 1, 
            #   that 1 neighbor MUST be current_end, because current_end is in the path and is a neighbor of v.
            #   If block_count[v] > 1, then v is a neighbor of some other path vertex, which is forbidden.
            #   If block_count[v] == 0, that's impossible if v is a neighbor of current_end (since current_end is in path).
            
            # So the condition is simply:
            # v is in neighbors[current_end] AND block_count[v] == 1.
            
            # Let's re-verify the update step.
            # We have path P. We add v.
            # New path P' = P + [v].
            # For any w in neighbors[v], block_count[w] should increase by 1, 
            # because v is now in the path and is a neighbor of w.
            # So, after adding v, we do:
            # for w in neighbors[v]: block_count[w] += 1
            
            # Let's trace:
            # Start: path=[s]. block_count[s] = 0 (initially). 
            # Wait, if path=[s], neighbors of s are not blocked yet?
            # If we add s, we should increment block_count for neighbors of s.
            # So, init: path=[s], used[s]=True. 
            # for w in neighbors[s]: block_count[w] += 1.
            # Now, for any neighbor v of s, block_count[v] is 1 (because s is a neighbor of v and s is in path).
            # So v is a valid candidate? Yes, dist(v, s)=1, and no other path vertices.
            
            # Next step: add v.
            # path=[s, v]. used[v]=True.
            # for w in neighbors[v]: block_count[w] += 1.
            # Now, consider a neighbor u of v.
            # If u == s, block_count[s] becomes 2.
            # If u != s, block_count[u] becomes 1 (if u wasn't blocked before) or more.
            # For a new candidate x (neighbor of v):
            # We check block_count[x].
            # If x is a neighbor of s, block_count[x] was incremented when s was added.
            # So if x is a neighbor of both s and v, block_count[x] would be 2.
            # Then x is NOT a valid candidate. Correct, because dist(x, s)=1.
            # If x is a neighbor of v but not s, block_count[x] is 1 (from v).
            # Then x IS a valid candidate. Correct.
            
            # So the logic holds.
            
            # Back to the loop:
            # We already added v above. We need to update block_count for neighbors of v.
            for w in neighbors[v]:
                block_count[w] += 1
                
            cur = v
            
        return path

    def backtrack_extend(path, used, block_count, time_limit, max_depth=10):
        """
        Tries to extend the path by backtracking.
        Returns the longest path found.
        """
        best_local_path = list(path)
        best_local_len = len(path)
        
        # We will do a DFS. To avoid deep recursion issues and allow time checks,
        # we implement it iteratively or with a strict depth limit.
        # Given the time limit, a recursive DFS with a small depth limit is fine.
        
        def dfs(current_path, current_used, current_block_count, depth, time_limit):
            nonlocal best_local_path, best_local_len
            
            if time.time() > time_limit:
                return
            
            if len(current_path) > best_local_len:
                best_local_len = len(current_path)
                best_local_path = list(current_path)
                
            if depth >= max_depth:
                return
                
            cur = current_path[-1]
            
            candidates = []
            for v in neighbors[cur]:
                if not current_used[v] and current_block_count[v] == 1:
                    candidates.append(v)
                    
            if not candidates:
                return
                
            random.shuffle(candidates)
            
            for v in candidates:
                # Add v
                current_path.append(v)
                current_used[v] = True
                for w in neighbors[v]:
                    current_block_count[w] += 1
                    
                dfs(current_path, current_used, current_block_count, depth + 1, time_limit)
                
                # Backtrack
                for w in neighbors[v]:
                    current_block_count[w] -= 1
                current_used[v] = False
                current_path.pop()
                
                if time.time() > time_limit:
                    return

        # We need to copy the state to avoid modifying the original if we want to keep it?
        # Actually, we can just modify the passed lists. If we find a longer path, 
        # we update best_local_path. The caller can then use best_local_path.
        # But we need to be careful: the `path` passed in is the greedy path.
        # If we modify it, we lose the greedy path if the backtracking fails?
        # No, we update best_local_path. We can return best_local_path.
        
        # To be safe and clean, let's copy the state for the DFS.
        # But copying 512-element arrays is fast.
        
        # Actually, we can just pass the lists and let them be modified, 
        # but we must restore them if we want to try other restarts?
        # No, we are doing a restart anyway. We can just let the backtracking 
        # modify the state, and if it finds a longer path, we update the global best.
        # The "current" path for the next restart will be a new one.
        
        # However, the `path` passed here is the result of the greedy step.
        # If we modify it, we lose the original greedy path. 
        # But we already saved it to `best_path` if it was better.
        # And we are going to start a new restart.
        # So it's safe to modify.
        
        dfs(path, used, block_count, 0, time_limit)
        
        return best_local_path

    # Main Loop
    for restart in range(100000):
        if time.time() > deadline - 5.0:  # Leave 5s for final cleanup/return
            break
            
        rng = random.Random(restart * 7919 + 1)
        start_v = rng.randint(0, (1 << N) - 1)
        
        path = [start_v]
        used = [False] * (1 << N)
        used[start_v] = True
        block_count = [0] * (1 << N)
        
        # Initialize block_count for start_v
        for w in neighbors[start_v]:
            block_count[w] += 1
            
        # Greedy Extend
        # We give the greedy step a generous time limit, as it's fast.
        # But we also want to do backtracking.
        # Let's split the time: 80% for greedy, 20% for backtracking?
        # No, greedy is O(L). Backtracking is exponential in depth.
        # Let's just run greedy until stuck. It's very fast.
        greedy_extend(start_v, path, used, block_count, deadline - 2.0)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            
        # Backtracking Extend
        # We give backtracking a small time slice, e.g., 0.05s
        # If the greedy path is already very long, maybe skip backtracking?
        # No, backtracking can often add 10-20 more vertices.
        
        # We need to be careful with the time limit for backtracking.
        # If we have 100s left, we don't want to spend 100s on one backtracking.
        # We want to try many restarts.
        # So we limit backtracking to a small fraction of the remaining time per restart.
        
        # Let's say we allow 0.02s for backtracking per restart.
        bt_time_limit = time.time() + 0.02
        if time.time() < deadline - 0.1:
            improved_path = backtrack_extend(path, used, block_count, bt_time_limit, max_depth=15)
            
            if len(improved_path) > best_len:
                best_len = len(improved_path)
                best_path = list(improved_path)
                
        # Early exit if we find a very long snake
        if best_len >= 190:
            break
            
    if best_path is None:
        return [0]
        
    return best_path
