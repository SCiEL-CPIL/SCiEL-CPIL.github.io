"""
Snake-in-the-box in the 9-dimensional hypercube.
Implements a Warnsdorff-based heuristic with backtracking and randomized restarts.
"""
import time
import random
import sys

N = 9
FULL_MASK = (1 << N) - 1
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _neighbors(v):
    """Yield all neighbors of vertex v in Q_N."""
    for b in range(N):
        yield v ^ (1 << b)

def _warnsdorff_search(start_vertex, max_backtracks=1000, randomize_ties=True):
    """
    Attempt to find a long snake starting from start_vertex using Warnsdorff's rule.
    Returns the list of vertices in the path.
    """
    path = [start_vertex]
    used = {start_vertex}
    
    # For backtracking, we store the state: (path, used, current_vertex)
    # But storing full 'used' sets is expensive. 
    # Instead, we implement a recursive/iterative backtracking that undoes moves.
    
    # To keep it efficient, we'll use a stack for backtracking.
    # State: (current_vertex, path_length, choices_tried_mask)
    # Actually, simple recursion with undo is cleaner for small N=9.
    
    def backtrack(idx, current, used_set, path_list):
        """
        idx: current index in path_list (len(path_list)-1)
        current: current vertex
        used_set: set of used vertices
        path_list: list of vertices
        """
        if len(path_list) > 200: # Safety cap, though we want max
            return True
        
        # Find all valid next moves
        nexts = []
        for n in _neighbors(current):
            if n not in used_set:
                # Check if n is at distance >= 2 from all previous vertices
                # Optimization: only need to check if n is adjacent to any vertex in path_list[:-1]
                # If popcount(n ^ path_list[i]) == 1 for any i < idx, it's invalid.
                valid = True
                for i in range(idx):
                    if (n ^ path_list[i]).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    # Calculate Warnsdorff score: number of unvisited neighbors of n
                    # that are also valid (distance >= 2 from all path vertices)
                    # Simplified score: just count unvisited neighbors.
                    # More precise score: count neighbors of n that are not in used_set 
                    # and are not adjacent to any vertex in path_list.
                    score = 0
                    for nn in _neighbors(n):
                        if nn not in used_set:
                            # Check if nn is adjacent to any vertex in path_list
                            # This is expensive. 
                            # Heuristic: just count unvisited neighbors.
                            score += 1
                    nexts.append((score, n))
        
        if not nexts:
            return False # Dead end
        
        # Sort by score (Warnsdorff: lowest score first)
        # Add randomization to break ties and explore different paths
        if randomize_ties:
            # Group by score, shuffle within groups
            nexts.sort(key=lambda x: x[0])
            # Simple randomization: shuffle the entire list slightly or just pick randomly from the best k?
            # Better: shuffle all, then stable sort? No, that destroys the order.
            # Let's just shuffle the list of nexts if they have similar scores?
            # For now, standard Warnsdorff: sort by score.
            # To allow exploration, we can randomly permute the list of nexts 
            # and then pick the one with min score? 
            # Or just add a small random noise to the score.
            for i in range(len(nexts)):
                nexts[i] = (nexts[i][0] + random.random() * 0.1, nexts[i][1])
            nexts.sort(key=lambda x: x[0])
        else:
            nexts.sort(key=lambda x: x[0])
        
        # Try each next move
        for score, n in nexts:
            # Check validity again (in case randomization changed order and we missed a constraint? No, constraint is static)
            # We already checked validity in the 'nexts' construction.
            
            # Make move
            path_list.append(n)
            used_set.add(n)
            
            if backtrack(len(path_list) - 1, n, used_set, path_list):
                return True
            
            # Undo move
            path_list.pop()
            used_set.remove(n)
        
        return False

    # Start the search
    # We limit the depth of backtracking implicitly by the recursion depth.
    # For N=9, the path can be up to 512, but we expect ~150-200.
    # Recursion depth ~200 is fine.
    
    # To avoid deep recursion issues, we can use a stack, but Python recursion limit is usually 1000.
    # We'll rely on the function returning False when stuck.
    
    # Initial call
    # We need to be careful with 'used_set' and 'path_list' being passed by reference.
    # backtrack modifies them in place and undoes.
    
    # One issue: backtrack explores ALL possibilities if it backtracks a lot.
    # We want to limit the number of backtracks to avoid exponential blowup.
    # We can add a global counter or pass a 'budget' parameter.
    
    # Let's implement a limited backtracking version.
    
    class BacktrackLimit:
        def __init__(self, limit):
            self.limit = limit
            self.count = 0
            self.exceeded = False
            
    bl = BacktrackLimit(max_backtracks)
    
    def backtrack_limited(idx, current, used_set, path_list):
        if bl.exceeded:
            return False
            
        nexts = []
        for n in _neighbors(current):
            if n not in used_set:
                valid = True
                for i in range(idx):
                    if (n ^ path_list[i]).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    # Score: number of unvisited neighbors
                    score = 0
                    for nn in _neighbors(n):
                        if nn not in used_set:
                            score += 1
                    nexts.append((score, n))
        
        if not nexts:
            return False
        
        # Randomize tie-breaking
        if randomize_ties:
            for i in range(len(nexts)):
                nexts[i] = (nexts[i][0] + random.random() * 0.5, nexts[i][1])
            nexts.sort(key=lambda x: x[0])
        else:
            nexts.sort(key=lambda x: x[0])
        
        for score, n in nexts:
            path_list.append(n)
            used_set.add(n)
            
            if backtrack_limited(len(path_list) - 1, n, used_set, path_list):
                return True
            
            path_list.pop()
            used_set.remove(n)
            
            bl.count += 1
            if bl.count > bl.limit:
                bl.exceeded = True
                return False
        
        return False

    path_list = [start_vertex]
    used_set = {start_vertex}
    backtrack_limited(0, start_vertex, used_set, path_list)
    
    return path_list

def _verify_snake(path):
    """Verify that the path is a valid induced path (snake)."""
    if not path:
        return False
    
    n = len(path)
    if len(set(path)) != n:
        return False
    
    for v in path:
        if v < 0 or v >= (1 << N):
            return False
    
    for i in range(n - 1):
        if (path[i] ^ path[i+1]).bit_count() != 1:
            return False
    
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_path = None
    best_len = 0
    
    # Try a few different starting vertices and random seeds
    # The structure of the hypercube is symmetric, but the search heuristic is not.
    
    # We will run multiple independent searches.
    # Each search uses Warnsdorff with random tie-breaking.
    
    num_restarts = 0
    while time.time() < deadline - 5.0:
        # Random start vertex
        start_v = random.randint(0, FULL_MASK)
        
        # Increase backtracking limit slightly for later iterations?
        # Or keep it fixed to ensure consistent time per run.
        # Let's use a moderate backtracking limit.
        max_bt = 200 + num_restarts * 10
        if max_bt > 1000:
            max_bt = 1000
            
        path = _warnsdorff_search(start_v, max_backtracks=max_bt, randomize_ties=True)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        num_restarts += 1
        
        # Check time
        if time.time() > deadline - 10.0:
            break
            
    # If no path found (should not happen), return a trivial one
    if best_path is None:
        best_path = [0]
        
    # Final verification
    if not _verify_snake(best_path):
        # Fallback to a very simple valid path if something went wrong
        best_path = [0, 1]
        
    return best_path

if __name__ == "__main__":
    # Quick test
    p = construct()
    print(f"Length: {len(p)}")
    print(f"Valid: {_verify_snake(p)}")
