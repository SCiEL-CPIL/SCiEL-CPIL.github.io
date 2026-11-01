import itertools
import collections

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    num_nodes = len(perms)
    
    # Greedy TSP approach
    # We want to visit all nodes.
    # Cost of moving from P to Q is n - overlap(P, Q).
    # We maximize overlap.
    
    # To speed up overlap calculation, we can use a trie or just linear scan.
    # Given N=5040, linear scan is 25M ops. 
    # Python might be slow for 25M tuple comparisons.
    # Let's optimize by grouping.
    
    # We'll maintain a set of unvisited nodes.
    # At each step, we find the best next node.
    
    # Optimization: 
    # Instead of checking all unvisited nodes against current,
    # we can check potential overlaps in descending order of k.
    # For k = n-1 down to 0:
    #   suffix = current[n-k:]
    #   Check if any unvisited node starts with suffix.
    #   If yes, pick one (maybe the one with best next step? or just first found).
    #   Break.
    
    # How to check "any unvisited node starts with suffix"?
    # We can maintain a dictionary: prefix -> set of nodes starting with prefix.
    # But prefixes change as we remove nodes.
    # Removing a node from a set in a dict is O(1) if we have the key.
    # But we need to check multiple prefixes (lengths n-1, n-2, ...).
    
    # Let's try a simpler heuristic first: 
    # Just pick the node with max overlap. 
    # If we use tuples, comparison is fast in C.
    
    unvisited = set(perms)
    current = perms[0]
    unvisited.remove(current)
    
    path = [current]
    
    # Helper to calculate overlap
    def get_overlap(p, q):
        # p and q are tuples of length n
        # Max k such that p[n-k:] == q[:k]
        # k can be 0 to n-1 (since p != q, k cannot be n)
        # Check n-1 down to 0
        # Using slicing is O(n). n=7. Very fast.
        for k in range(n - 1, -1, -1):
            if k == 0:
                return 0
            if p[n-k:] == q[:k]:
                return k
        return 0

    # To make it faster, we can precompute the "best candidate" logic?
    # No, let's just run the greedy loop.
    # 5040 iterations.
    # Each iteration scans ~5000 nodes.
    # 25 million overlap checks.
    # Each check involves slicing tuples of length 7.
    # This might take ~10-20 seconds in Python. Acceptable for 120s limit.
    
    while unvisited:
        best_node = None
        best_overlap = -1
        
        # Iterate unvisited
        for node in unvisited:
            ov = get_overlap(current, node)
            if ov > best_overlap:
                best_overlap = ov
                best_node = node
                # Early termination if we found max possible overlap (n-1)
                # But there might be multiple nodes with n-1 overlap?
                # No, for a fixed current, only one node has overlap n-1 (the rotation).
                # If we find n-1, we can break immediately.
                if best_overlap == n - 1:
                    break
        
        if best_node is None:
            # Should not happen
            break
            
        path.append(best_node)
        unvisited.remove(best_node)
        current = best_node

    # Construct the string
    result = list(path[0])
    for i in range(1, len(path)):
        prev = path[i-1]
        curr = path[i]
        ov = get_overlap(prev, curr)
        # Append curr[ov:]
        result.extend(curr[ov:])
        
    return "".join(map(str, result))
