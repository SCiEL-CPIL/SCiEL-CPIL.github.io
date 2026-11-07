"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 10
    # Generate candidate Latin squares of the form L[i][j] = (a*i + b*j) % n
    # where gcd(a, n) == 1 and gcd(b, n) == 1 to ensure Latin property.
    candidates = []
    for a in range(n):
        if a % 2 == 0 or a % 5 == 0:
            continue
        for b in range(n):
            if b % 2 == 0 or b % 5 == 0:
                continue
            sq = [[(a * i + b * j) % n for j in range(n)] for i in range(n)]
            candidates.append(sq)
    
    # Check orthogonality between all pairs
    # orth[i][j] = True if candidates[i] and candidates[j] are orthogonal
    num_cands = len(candidates)
    orth = [[False] * num_cands for _ in range(num_cands)]
    
    for i in range(num_cands):
        for j in range(i + 1, num_cands):
            # Check if candidates[i] and candidates[j] are orthogonal
            pairs = set()
            for r in range(n):
                for c in range(n):
                    p = (candidates[i][r][c], candidates[j][r][c])
                    if p in pairs:
                        break
                    pairs.add(p)
            else:
                if len(pairs) == n * n:
                    orth[i][j] = True
                    orth[j][i] = True
    
    # Find the maximum clique in the orthogonality graph
    # Since num_cands is small (16), we can use a simple greedy approach or backtracking
    # Greedy: pick a node, remove non-neighbors, repeat
    best_set = []
    remaining = list(range(num_cands))
    
    while remaining:
        # Pick the node with the most neighbors in the remaining set
        best_node = -1
        max_deg = -1
        for node in remaining:
            deg = sum(1 for other in remaining if node != other and orth[node][other])
            if deg > max_deg:
                max_deg = deg
                best_node = node
        
        if best_node == -1:
            break
            
        best_set.append(best_node)
        # Remove best_node and its non-neighbors from remaining
        neighbors = set([best_node])
        for other in remaining:
            if orth[best_node][other]:
                neighbors.add(other)
        
        remaining = [node for node in remaining if node in neighbors]
        
    # Convert indices to squares
    result = [candidates[i] for i in best_set]
    
    # Ensure types are correct (list of lists of ints)
    final_result = []
    for sq in result:
        final_sq = [[int(sq[i][j]) for j in range(n)] for i in range(n)]
        final_result.append(final_sq)
        
    return final_result

# EVOLVE-BLOCK-END
