import time
import itertools

_DEADLINE = time.time() + 85.0

def _crt(x5, x2):
    """Chinese Remainder Theorem: find x in 0..9 with x%5=x5, x%2=x2."""
    # x = x5 * 6 + x2 * 5 mod 10
    return (x5 * 6 + x2 * 5) % 10

def _build_square(a5, b5, a2, b2):
    """
    Build a Latin square of order 10 using parameters:
      L(i,j) = (a5*i5 + b5*j5 mod 5, a2*i2 + b2*j2 mod 2)
    where i=(i5,i2), j=(j5,j2) via CRT decomposition.
    """
    n = 10
    sq = [[0] * n for _ in range(n)]
    for i in range(n):
        i5, i2 = i % 5, i % 2
        for j in range(n):
            j5, j2 = j % 5, j % 2
            r5 = (a5 * i5 + b5 * j5) % 5
            r2 = (a2 * i2 + b2 * j2) % 2
            sq[i][j] = _crt(r5, r2)
    return sq

def _is_latin_square(sq):
    """Check if sq is a valid Latin square of order 10."""
    n = 10
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _are_orthogonal(sq1, sq2):
    """Check if two Latin squares are orthogonal."""
    n = 10
    pairs = set()
    for i in range(n):
        for j in range(n):
            pair = (sq1[i][j], sq2[i][j])
            if pair in pairs:
                return False
            pairs.add(pair)
    return True

def construct():
    """
    Return a list of mutually orthogonal Latin squares of order 10.
    """
    # Generate candidates
    candidates = []
    for a5 in range(5):
        for b5 in range(1, 5):
            for a2 in range(2):
                b2 = 1
                sq = _build_square(a5, b5, a2, b2)
                if _is_latin_square(sq):
                    candidates.append(sq)
    
    # If we have fewer than 9 candidates, we might need more, but 40 is enough.
    # Build orthogonality graph
    n_cands = len(candidates)
    adj = [[False] * n_cands for _ in range(n_cands)]
    for i in range(n_cands):
        for j in range(i + 1, n_cands):
            if _are_orthogonal(candidates[i], candidates[j]):
                adj[i][j] = True
                adj[j][i] = True
    
    # Find maximum clique
    # Since we expect 9, we can try to find a clique of size 9.
    # Use backtracking with pruning.
    best_clique = []
    
    def expand_clique(candidates_idx, current_clique):
        nonlocal best_clique
        if len(current_clique) > len(best_clique):
            best_clique = current_clique[:]
        
        if not candidates_idx:
            return
        
        # Pruning: if current size + remaining candidates <= best size, stop
        if len(current_clique) + len(candidates_idx) <= len(best_clique):
            return
            
        for k in range(len(candidates_idx)):
            v = candidates_idx[k]
            # Check if v is connected to all in current_clique
            is_connected = all(adj[v][u] for u in current_clique)
            if is_connected:
                # New candidates are those after k that are connected to v
                new_cands = [u for u in candidates_idx[k+1:] if adj[v][u]]
                expand_clique(new_cands, current_clique + [v])
                
    # Start with all nodes
    # To speed up, we can order nodes by degree
    degrees = [sum(adj[i]) for i in range(n_cands)]
    nodes = list(range(n_cands))
    nodes.sort(key=lambda x: degrees[x], reverse=True)
    
    expand_clique(nodes, [])
    
    # Extract the best clique
    result = [candidates[i] for i in best_clique]
    
    # Ensure types are correct
    for sq in result:
        for i in range(10):
            for j in range(10):
                sq[i][j] = int(sq[i][j])
                
    return result
