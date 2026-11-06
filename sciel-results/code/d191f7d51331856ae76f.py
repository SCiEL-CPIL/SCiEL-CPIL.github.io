import time
import random
import numpy as np

N = 9
NUM_VERTICES = 1 << N
TIME_LIMIT = 115.0  # Safety margin

# Precompute neighbors for all vertices
NEIGHBORS = [[] for _ in range(NUM_VERTICES)]
for v in range(NUM_VERTICES):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

def construct():
    start_time = time.time()
    best_path = []
    
    # Symmetry: Fix start at 0, first move bit 0, second move bit 1
    # Path: 0 -> 1 -> 3
    # We can vary the order of subsequent bits or use different starting triples if we break symmetry more, 
    # but 0-1-3 is a canonical start.
    
    # To explore different branches, we can randomize the order in which we consider bit flips 
    # or use different initial "blocked" states if we didn't fix the prefix.
    # Since we fix 0->1->3, the variation comes from the beam search tie-breaking.
    
    # Constraint-based backtracking with MRV heuristic
    
    def dfs(path, used, blocked, depth):
        nonlocal best_path, start_time
        
        if len(path) > len(best_path):
            best_path = path[:]
            
        # Time check
        if time.time() - start_time > TIME_LIMIT:
            return False # Signal to stop
        
        cur = path[-1]
        
        # Find all valid moves from cur
        valid_moves = []
        for nb in NEIGHBORS[cur]:
            if not used[nb] and not blocked[nb]:
                valid_moves.append(nb)
                
        if not valid_moves:
            return True # Dead end, backtrack
            
        # Heuristic: Sort moves by the number of valid moves they would leave for the next step (MRV)
        # This helps prune branches that lead to immediate dead ends
        def heuristic_score(nb):
            # Count valid moves from nb, given the new blocked state
            # New blocked state would be blocked | neighbors(cur)
            # We can approximate or compute exactly. Exact is better.
            count = 0
            # Pre-compute neighbors of cur to avoid repeated list access in inner loop? 
            # NEIGHBORS[cur] is already a list.
            nbrs_cur = NEIGHBORS[cur]
            for n2 in NEIGHBORS[nb]:
                if not used[n2] and not blocked[n2] and not (n2 in nbrs_cur or n2 == cur):
                    # Note: n2 cannot be cur if nb is neighbor of cur? 
                    # No, n2 is neighbor of nb. cur is neighbor of nb. 
                    # So n2 can be cur. But cur is in used. 
                    # The condition `not used[n2]` handles cur.
                    # The condition `not blocked[n2]` handles neighbors of path[:-1].
                    # We need to check against neighbors of cur (which will be blocked in the next step).
                    # So if n2 is neighbor of cur, it will be blocked.
                    # So we check `not (n2 in nbrs_cur)`.
                    # Also n2 cannot be nb (used).
                    count += 1
            return count
            
        # Optimize heuristic: 
        # Instead of full count, we can just check if the move is "tight".
        # A simple MRV: sort by count of available neighbors.
        
        # To speed up, we can limit the sort if the list is large? 
        # N=9, max 9 moves. Sorting 9 items is cheap.
        
        # Calculate scores
        scored_moves = []
        nbrs_cur_set = set(NEIGHBORS[cur]) # For faster check in heuristic
        for nb in valid_moves:
            count = 0
            for n2 in NEIGHBORS[nb]:
                if not used[n2] and not blocked[n2] and n2 not in nbrs_cur_set:
                    count += 1
            scored_moves.append((count, nb))
            
        # Sort by count ascending (MRV: most constrained first)
        scored_moves.sort(key=lambda x: x[0])
        
        # Randomize ties to explore different branches in different restarts
        # Group by score
        if scored_moves:
            min_score = scored_moves[0][0]
            # We will iterate through all, but prioritize lower scores.
            # Actually, for backtracking, we usually pick the best one first.
            # If we want to find *longest* path, picking the most constrained first 
            # helps detect dead ends early, but might not lead to the longest path immediately.
            # However, it prunes the tree significantly.
            
            # Let's try to shuffle items with the same score to add diversity
            # But primarily follow MRV order.
            
            # To allow different paths in restarts, we can shuffle the list of moves 
            # that have the minimum score, or just shuffle all if we want random walk?
            # Standard backtracking follows the heuristic order.
            # To get diversity, we can randomize the order of moves with equal scores.
            
            # Simple approach: Shuffle the scored_moves list, but stable sort by score?
            # No, let's just iterate in the sorted order.
            # To get different paths, we rely on the fact that if the first branch fails, 
            # we backtrack and try the next.
            
            # However, if we always pick the same "best" move, we might miss a long path 
            # that starts with a "less constrained" move.
            # Beam search explores multiple. Backtracking explores depth-first.
            # To improve, we can use "random restarts" with different tie-breaking.
            
            # Let's implement a simple randomization: 
            # If there are multiple moves with the same minimum score, pick one randomly.
            # Then recurse. If it fails, try the next one.
            
            # Implementation:
            # 1. Identify min_score.
            # 2. Collect all moves with min_score.
            # 3. Shuffle them.
            # 4. Try them one by one.
            # 5. If all fail, try moves with min_score + 1, etc.
            
            # This is a bit complex. 
            # Simpler: Just shuffle the `valid_moves` list before scoring?
            # No, MRV is the key.
            
            # Let's stick to strict MRV order but allow randomization in the initial setup 
            # or by slightly perturbing the score?
            
            # Alternative: 
            # In each restart, we can shuffle the NEIGHBORS lists? No, that changes the graph.
            # We can shuffle the order of bits considered in the initial prefix?
            # The prefix is fixed 0-1-3.
            
            # Let's just iterate through the sorted moves.
            # To add diversity between restarts, we can randomly swap moves with equal scores.
            
            # For now, let's just do the DFS.
            # To improve diversity, we can randomize the order of `scored_moves` 
            # among those with the same score.
            
            # Let's group by score
            groups = {}
            for s, m in scored_moves:
                if s not in groups:
                    groups[s] = []
                groups[s].append(m)
                
            # Sort scores
            sorted_scores = sorted(groups.keys())
            
            for s in sorted_scores:
                moves_in_group = groups[s]
                random.shuffle(moves_in_group)
                for nb in moves_in_group:
                    # Check time
                    if time.time() - start_time > TIME_LIMIT:
                        return False
                        
                    # Make move
                    used[nb] = 1
                    # Update blocked: add neighbors of cur
                    # We need to restore state after recursion
                    # To avoid copying sets, we can use a "undo" list
                    # Or just copy the bytearray? 512 bytes is small.
                    # But recursion depth can be ~200. 200 * 512 bytes = 100KB. Fine.
                    # However, copying is O(N). 
                    # Let's use an undo list for blocked.
                    
                    # Actually, `blocked` is a bytearray. 
                    # We can just set the bits and unset them later.
                    
                    changed_blocked = []
                    for nb_cur in NEIGHBORS[cur]:
                        if not blocked[nb_cur]:
                            blocked[nb_cur] = 1
                            changed_blocked.append(nb_cur)
                            
                    path.append(nb)
                    
                    # Recurse
                    # If time limit exceeded, dfs returns False
                    if not dfs(path, used, blocked, depth + 1):
                        # Time limit reached, unwind quickly
                        path.pop()
                        used[nb] = 0
                        for v in changed_blocked:
                            blocked[v] = 0
                        return False
                        
                    path.pop()
                    used[nb] = 0
                    for v in changed_blocked:
                        blocked[v] = 0
                        
                    # If we found a path longer than current best, we continue searching 
                    # to see if we can go even longer.
                    # But if we hit a dead end, we backtrack.
                    
            return True # All moves tried
            
    # Initialize state
    # Path: 0, 1, 3
    # Used: 0, 1, 3
    # Blocked: neighbors of 0, 1
    
    # We will run multiple restarts to explore different branches due to randomization in ties
    max_restarts = 50 # We can do more if fast
    
    for restart in range(max_restarts):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        path = [0, 1, 3]
        used = bytearray(NUM_VERTICES)
        used[0] = 1
        used[1] = 1
        used[3] = 1
        
        blocked = bytearray(NUM_VERTICES)
        # Blocked for next step (finding v3) should contain neighbors of v0 and v1
        # Wait, the definition of blocked in my DFS:
        # `blocked` contains neighbors of vertices that are NOT the last one?
        # Let's re-verify the logic in `dfs`.
        # In `dfs`, we pick `nb` from `NEIGHBORS[cur]`.
        # Condition: `not used[nb] and not blocked[nb]`.
        # Then we update `blocked` by adding `NEIGHBORS[cur]`.
        # So at the start of `dfs(path)`, `blocked` should contain neighbors of `path[0:-1]`.
        # Let's check initialization.
        # Path = [0, 1, 3].
        # We are about to find the next vertex after 3.
        # `blocked` should contain neighbors of 0 and 1.
        
        for nb in NEIGHBORS[0]:
            blocked[nb] = 1
        for nb in NEIGHBORS[1]:
            blocked[nb] = 1
            
        # Note: 0 and 1 are in used, so they are not candidates anyway.
        # But their neighbors are blocked.
        
        dfs(path, used, blocked, 2)
        
        if time.time() - start_time > TIME_LIMIT:
            break
            
    return best_path
