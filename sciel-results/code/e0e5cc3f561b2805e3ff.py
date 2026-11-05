"""Local search to grow a (5,5)-free graph beyond the Paley seed."""
import time
import random
import numpy as np
from itertools import combinations

SEED_TIME_LIMIT = 280  # seconds, safety margin before 300s kill


def _paley_graph(q):
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, edges


def _build_adj_matrix(n, edges):
    """Build numpy adjacency matrix (0-indexed, symmetric)."""
    mat = np.zeros((n, n), dtype=np.uint8)
    for i, j in edges:
        mat[i, j] = 1
        mat[j, i] = 1
    return mat


def _mat_to_edges(mat):
    """Convert adjacency matrix to edge list."""
    n = mat.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i, j]:
                edges.append((i, j))
    return edges


def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _count_cliques_in_subset(mat, subset, k):
    """Count k-cliques within a subset of vertices using adjacency matrix."""
    if len(subset) < k:
        return 0
    count = 0
    for combo in combinations(subset, k):
        is_clique = True
        for a, b in combinations(combo, 2):
            if not mat[a, b]:
                is_clique = False
                break
        if is_clique:
            count += 1
    return count


def _has_k_clique_or_independent(n, adj, k=5):
    """Check if graph has a k-clique or k-independent set. Returns (is_valid, witness)."""
    # Check for k-clique
    for combo in combinations(range(n), k):
        is_clique = True
        for a, b in combinations(combo, 2):
            if b not in adj[a]:
                is_clique = False
                break
        if is_clique:
            return False, ('clique', combo)
    # Check for k-independent set
    for combo in combinations(range(n), k):
        is_indep = True
        for a, b in combinations(combo, 2):
            if b in adj[a]:
                is_indep = False
                break
        if is_indep:
            return False, ('independent', combo)
    return True, None


def _find_violation_fast(n, adj, k=5):
    """Find a k-clique or k-independent set using backtracking with pruning."""
    # Use a recursive search with early termination
    def _search_clique(start, current, remaining_needed):
        if len(current) == k:
            return tuple(current)
        if len(current) + (n - start) < k:
            return None
        for v in range(start, n):
            if len(current) + (n - v) < k:
                break
            # Check if v is connected to all in current
            if all(v in adj[u] for u in current):
                result = _search_clique(v + 1, current + [v], remaining_needed - 1)
                if result is not None:
                    return result
        return None

    def _search_independent(start, current, remaining_needed):
        if len(current) == k:
            return tuple(current)
        if len(current) + (n - start) < k:
            return None
        for v in range(start, n):
            if len(current) + (n - v) < k:
                break
            # Check if v is NOT connected to any in current
            if all(v not in adj[u] for u in current):
                result = _search_independent(v + 1, current + [v], remaining_needed - 1)
                if result is not None:
                    return result
        return None

    # Try to find a clique
    clique = _search_clique(0, [], k)
    if clique is not None:
        return 'clique', clique
    
    # Try to find an independent set
    indep = _search_independent(0, [], k)
    if indep is not None:
        return 'independent', indep
    
    return None, None


def _find_violation(n, adj, k=5):
    """Find a k-clique or k-independent set. Returns (type, set) or (None, None)."""
    return _find_violation_fast(n, adj, k)


def _count_violations_fast(n, adj, k=5):
    """Count number of k-cliques and k-independent sets (for scoring)."""
    cliques = 0
    indeps = 0
    for combo in combinations(range(n), k):
        is_clique = True
        is_indep = True
        for a, b in combinations(combo, 2):
            if b in adj[a]:
                is_indep = False
            else:
                is_clique = False
            if not is_clique and not is_indep:
                break
        if is_clique:
            cliques += 1
        if is_indep:
            indeps += 1
    return cliques + indeps


def _check_add_vertex_valid(n, adj, new_neighbours, k=5):
    """
    Check if adding a new vertex (index n) with given neighbours creates a k-clique or k-independent set.
    new_neighbours: set of existing vertices connected to the new vertex.
    Returns (is_valid, violation_type, witness)
    """
    # A k-clique involving the new vertex: new vertex + (k-1) vertices all connected to new vertex
    # and forming a (k-1)-clique among themselves
    # A k-independent set involving the new vertex: new vertex + (k-1) vertices NOT connected to new vertex
    # and forming a (k-1)-independent set among themselves
    
    # Check for k-clique with new vertex
    # Need (k-1) vertices from new_neighbours that form a (k-1)-clique
    for combo in combinations(sorted(new_neighbours), k - 1):
        is_clique = True
        for a, b in combinations(combo, 2):
            if b not in adj[a]:
                is_clique = False
                break
        if is_clique:
            return False, 'clique', tuple(list(combo) + [n])
    
    # Check for k-independent set with new vertex
    # Need (k-1) vertices from (all vertices NOT in new_neighbours) that form a (k-1)-independent set
    non_neighbours = set(range(n)) - new_neighbours
    for combo in combinations(sorted(non_neighbours), k - 1):
        is_indep = True
        for a, b in combinations(combo, 2):
            if b in adj[a]:
                is_indep = False
                break
        if is_indep:
            return False, 'independent', tuple(list(combo) + [n])
    
    return True, None, None


def _try_add_vertex(n, adj, deadline, rng, max_attempts=500):
    """
    Try to add a new vertex (index n) to the graph.
    Returns True if successful, False if all attempts failed.
    """
    # Precompute: for each 4-set, determine if it's a clique or independent
    # A new vertex connected to a 4-clique creates a 5-clique
    # A new vertex not connected to a 4-independent set creates a 5-independent set
    
    # We need: for every 4-clique, new vertex must NOT be connected to all 4
    # We need: for every 4-independent set, new vertex must be connected to at least 1
    
    # Build lists of 4-cliques and 4-independent sets
    four_cliques = []
    four_indeps = []
    for combo in combinations(range(n), 4):
        is_clique = True
        is_indep = True
        for a, b in combinations(combo, 2):
            if b in adj[a]:
                is_indep = False
            else:
                is_clique = False
            if not is_clique and not is_indep:
                break
        if is_clique:
            four_cliques.append(set(combo))
        if is_indep:
            four_indeps.append(set(combo))
    
    # Strategy 1: Try biased random assignments
    # Strategy 2: Use local search on the assignment
    
    # First, try a few random assignments
    for attempt in range(min(100, max_attempts)):
        if time.time() > deadline:
            return False
        
        # Random assignment with bias towards sparser connections
        p = 0.3 + 0.4 * rng.random()  # probability between 0.3 and 0.7
        new_adj = set()
        for v in range(n):
            if rng.random() < p:
                new_adj.add(v)
        
        # Check constraints
        valid = True
        for fc in four_cliques:
            if fc.issubset(new_adj):
                valid = False
                break
        if not valid:
            continue
        for fi in four_indeps:
            if not (fi & new_adj):
                valid = False
                break
        if not valid:
            continue
        
        # Valid! Add the vertex
        adj.append(new_adj)
        for v in new_adj:
            adj[v].add(n)
        return True
    
    # Strategy 2: Start from a random assignment and fix violations
    for attempt in range(max_attempts - 100):
        if time.time() > deadline:
            return False
        
        new_adj = set()
        for v in range(n):
            if rng.random() < 0.5:
                new_adj.add(v)
        
        # Try to fix violations
        for fix_iter in range(50):
            if time.time() > deadline:
                return False
            
            # Check for violations
            violation_found = False
            
            # Check 4-cliques fully connected
            for fc in four_cliques:
                if fc.issubset(new_adj):
                    # Remove one vertex from new_adj
                    to_remove = rng.choice(list(fc))
                    new_adj.discard(to_remove)
                    violation_found = True
                    break
            
            if not violation_found:
                # Check 4-independent sets with no connection
                for fi in four_indeps:
                    if not (fi & new_adj):
                        # Add one vertex to new_adj
                        to_add = rng.choice(list(fi))
                        new_adj.add(to_add)
                        violation_found = True
                        break
            
            if not violation_found:
                # No violations, we're done
                break
        
        # Final check
        valid = True
        for fc in four_cliques:
            if fc.issubset(new_adj):
                valid = False
                break
        if not valid:
            continue
        for fi in four_indeps:
            if not (fi & new_adj):
                valid = False
                break
        if not valid:
            continue
        
        # Valid! Add the vertex
        adj.append(new_adj)
        for v in new_adj:
            adj[v].add(n)
        return True
    
    return False


def _local_search_repair(n, adj, deadline, rng, max_iter=2000):
    """
    If the graph has violations, try to repair by flipping edges.
    Returns True if repaired, False otherwise.
    """
    for it in range(max_iter):
        if time.time() > deadline:
            return False
        
        vtype, witness = _find_violation_fast(n, adj, 5)
        if vtype is None:
            return True
        
        # Try to fix by flipping one edge in the witness
        # If it's a clique, remove an edge from the witness
        # If it's independent, add an edge to the witness
        if vtype == 'clique':
            candidates = list(combinations(witness, 2))
            rng.shuffle(candidates)
            for a, b in candidates:
                # Try removing edge (a,b)
                adj[a].remove(b)
                adj[b].remove(a)
                # Check if this fixes things without creating new violations
                vtype2, witness2 = _find_violation_fast(n, adj, 5)
                if vtype2 is None:
                    return True
                # Revert
                adj[a].add(b)
                adj[b].add(a)
        else:  # independent
            candidates = list(combinations(witness, 2))
            rng.shuffle(candidates)
            for a, b in candidates:
                # Try adding edge (a,b)
                adj[a].add(b)
                adj[b].add(a)
                vtype2, witness2 = _find_violation_fast(n, adj, 5)
                if vtype2 is None:
                    return True
                # Revert
                adj[a].remove(b)
                adj[b].remove(a)
    
    return False


def construct():
    start_time = time.time()
    deadline = start_time + SEED_TIME_LIMIT
    
    rng = random.Random(42)
    
    # Start with Paley graph on 37 vertices
    n, edges = _paley_graph(37)
    adj = _build_adj(n, edges)
    
    # Verify seed is valid
    valid, _ = _has_k_clique_or_independent(n, adj, 5)
    assert valid, "Paley graph should be (5,5)-free"
    
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Try to grow the graph by adding vertices one at a time
    for target_n in range(n + 1, 47):
        if time.time() > deadline - 30:
            break
        
        success = _try_add_vertex(n, adj, deadline, rng, max_attempts=300)
        if success:
            n += 1
            # Verify
            valid, _ = _has_k_clique_or_independent(n, adj, 5)
            if valid:
                best_n = n
                best_adj = [set(s) for s in adj]
            else:
                # Try to repair
                if _local_search_repair(n, adj, deadline, rng, max_iter=1000):
                    valid, _ = _has_k_clique_or_independent(n, adj, 5)
                    if valid:
                        best_n = n
                        best_adj = [set(s) for s in adj]
                    else:
                        # Revert
                        adj.pop()
                        for v in range(n - 1):
                            adj[v].discard(n - 1)
                        n -= 1
                else:
                    # Revert
                    adj.pop()
                    for v in range(n - 1):
                        adj[v].discard(n - 1)
                    n -= 1
        else:
            # Try a few more times with different random seeds
            for retry in range(5):
                if time.time() > deadline - 30:
                    break
                success = _try_add_vertex(n, adj, deadline, rng, max_attempts=200)
                if success:
                    n += 1
                    valid, _ = _has_k_clique_or_independent(n, adj, 5)
                    if valid:
                        best_n = n
                        best_adj = [set(s) for s in adj]
                        break
                    else:
                        if _local_search_repair(n, adj, deadline, rng, max_iter=500):
                            valid, _ = _has_k_clique_or_independent(n, adj, 5)
                            if valid:
                                best_n = n
                                best_adj = [set(s) for s in adj]
                                break
                        # Revert
                        adj.pop()
                        for v in range(n - 1):
                            adj[v].discard(n - 1)
                        n -= 1
    
    # If we couldn't grow beyond 37, try edge-flipping local search to create space
    if time.time() < deadline - 60 and best_n <= 37:
        # Try flipping random edges to create more flexibility for growth
        work_adj = [set(s) for s in best_adj]
        work_n = best_n
        
        for flip_iter in range(5000):
            if time.time() > deadline - 60:
                break
            
            # Pick a random pair and flip it
            v = rng.randint(0, work_n - 1)
            u = rng.randint(0, work_n - 1)
            if u == v:
                continue
            
            # Flip edge (v, u)
            if u in work_adj[v]:
                work_adj[v].remove(u)
                work_adj[u].remove(v)
            else:
                work_adj[v].add(u)
                work_adj[u].add(v)
            
            # Check validity
            valid, _ = _has_k_clique_or_independent(work_n, work_adj, 5)
            if not valid:
                # Revert
                if u in work_adj[v]:
                    work_adj[v].remove(u)
                    work_adj[u].remove(v)
                else:
                    work_adj[v].add(u)
                    work_adj[u].add(v)
                continue
            
            # Try to add a vertex to this modified graph
            if time.time() > deadline - 45:
                break
            
            saved_adj = [set(s) for s in work_adj]
            success = _try_add_vertex(work_n, work_adj, deadline, rng, max_attempts=100)
            if success:
                work_n += 1
                valid2, _ = _has_k_clique_or_independent(work_n, work_adj, 5)
                if valid2 and work_n > best_n:
                    best_n = work_n
                    best_adj = [set(s) for s in work_adj]
                else:
                    # Revert
                    work_adj = [set(s) for s in saved_adj]
                    work_n -= 1
            else:
                # Revert the flip
                work_adj = [set(s) for s in saved_adj]
    
    # Build final edge list
    final_edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                final_edges.append((i, j))
    
    return best_n, final_edges


if __name__ == '__main__':
    n, edges = construct()
    print(f"Constructed graph with {n} vertices and {len(edges)} edges")
