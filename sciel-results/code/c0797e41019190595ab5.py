"""Find a graph with no regular induced subgraph on 6+ vertices."""
import random
import time
import itertools

def _is_regular_degrees(degs):
    """Check if all degrees are equal."""
    first = degs[0]
    for d in degs[1:]:
        if d != first:
            return False
    return True

def _find_violations(adj, n):
    """Find all 6-subsets that induce a regular graph. Returns list of (subset, degree)."""
    violations = []
    for subset in itertools.combinations(range(n), 6):
        degs = []
        for v in subset:
            d = 0
            for u in subset:
                if u != v and adj[v][u]:
                    d += 1
            degs.append(d)
        if _is_regular_degrees(degs):
            violations.append((subset, degs[0]))
    return violations

def _count_violations_fast(adj, n, max_check=50000):
    """Count violations, stop early if too many."""
    count = 0
    checked = 0
    for subset in itertools.combinations(range(n), 6):
        degs = []
        for v in subset:
            d = 0
            for u in subset:
                if u != v and adj[v][u]:
                    d += 1
            degs.append(d)
        if _is_regular_degrees(degs):
            count += 1
        checked += 1
        if checked >= max_check:
            break
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    best_n = 5
    best_edges = []
    
    # Try to build graphs of increasing size
    for target_n in range(6, 23):
        if time.time() > deadline - 20:
            break
        
        found = False
        # Try multiple random seeds
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            random.seed(attempt * 1000 + target_n)
            
            # Start with random graph at moderate density
            p = 0.5
            adj = [[False] * target_n for _ in range(target_n)]
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if random.random() < p:
                        adj[i][j] = adj[j][i] = True
            
            # Iteratively repair violations
            for iteration in range(2000):
                if time.time() > deadline - 5:
                    break
                
                violations = _find_violations(adj, target_n)
                if not violations:
                    # Valid! Extract edges
                    edges = []
                    for i in range(target_n):
                        for j in range(i+1, target_n):
                            if adj[i][j]:
                                edges.append([i, j])
                    best_n = target_n
                    best_edges = edges
                    found = True
                    break
                
                # Pick a random violation and try to fix it
                subset, deg = random.choice(violations)
                # Try flipping an edge within the subset
                fixed = False
                for _ in range(20):
                    i, j = random.sample(subset, 2)
                    adj[i][j] = adj[j][i] = not adj[i][j]
                    
                    # Quick check: does this reduce violations?
                    # Check if this specific subset is still regular
                    still_regular = True
                    new_degs = []
                    for v in subset:
                        d = 0
                        for u in subset:
                            if u != v and adj[v][u]:
                                d += 1
                        new_degs.append(d)
                    if not _is_regular_degrees(new_degs):
                        # Good, this violation is fixed. Keep the flip.
                        fixed = True
                        break
                    else:
                        # Revert
                        adj[i][j] = adj[j][i] = not adj[i][j]
                
                if not fixed:
                    # Try flipping an edge between subset and outside
                    outside = [v for v in range(target_n) if v not in subset]
                    if outside:
                        for _ in range(10):
                            i = random.choice(subset)
                            j = random.choice(outside)
                            adj[i][j] = adj[j][i] = not adj[i][j]
                            # Check if this helps (revert if it creates new 6-reg violations involving i or j)
                            # Simple heuristic: just keep it, hope for the best
                            break
    
        if found:
            # Continue trying for larger n
            continue
        else:
            # Couldn't find for this n, stop
            break
    
    if best_n < 6:
        # Fallback: trivial graph
        return {'n': 5, 'edges': []}
    
    return {'n': best_n, 'edges': best_edges}
