import random
import itertools
import time

n = 16
total_possible = n * (n - 1) // 2  # 120

# Precompute all 8-vertex subsets and their Q3 edge sets
q3_edge_sets = []  # list of frozensets of 12 edge indices
q3_subsets = []  # list of 8-vertex tuples

# Map (i,j) to edge index
edge_to_idx = {}
idx_to_edge = []
for i in range(n):
    for j in range(i + 1, n):
        edge_to_idx[(i, j)] = len(idx_to_edge)
        idx_to_edge.append((i, j))

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return edge_to_idx[(i, j)]

# Generate all 8-vertex subsets
all_8_subsets = list(itertools.combinations(range(n), 8))

# For each subset, check if it can contain Q3 and record the edge pattern
# Q3 has 12 edges. For an 8-vertex subset, we need to check if the induced subgraph
# contains Q3 as a subgraph.

# Precompute Q3 adjacency
q3_adj = [0] * 8
for i in range(8):
    for j in range(8):
        if i != j and bin(i ^ j).count('1') == 1:
            q3_adj[i] |= (1 << j)

# For each 8-vertex subset, find all ways to map Q3 vertices to these 8 vertices
# such that the required edges are within the subset
q3_patterns = {}  # subset tuple -> list of frozensets of edge indices

for subset in all_8_subsets:
    patterns = []
    # Try all permutations of mapping Q3 vertices to subset vertices
    # But we can prune: only try mappings where each Q3 vertex maps to a subset vertex
    # with sufficient degree potential
    
    # Actually, let's be smarter. The Q3 is a specific graph. We need to find
    # all isomorphic embeddings of Q3 into the 8-vertex subset.
    # Since Q3 is 3-regular on 8 vertices, and the subset has 8 vertices,
    # we're looking for whether the complete graph K8 on these vertices contains Q3.
    
    # For each subset, the set of possible edge sets that could form Q3:
    # We need to find all distinct sets of 12 edges among the 28 possible edges
    # of K8 that form a Q3.
    
    # Use backtracking to find all Q3 embeddings
    def find_all_q3_embeddings(sub):
        """Find all mappings from Q3 vertices 0-7 to subset vertices that form Q3."""
        results = []
        mapping = [-1] * 8
        used = [False] * 8  # used[local_idx] = True if local_idx is used
        
        def bt(qv):
            if qv == 8:
                # Extract edge set
                edges = set()
                for i in range(8):
                    for j in range(i + 1, 8):
                        if (q3_adj[i] >> j) & 1:
                            edges.add(edge_idx(sub[mapping[i]], sub[mapping[j]]))
                results.append(frozenset(edges))
                return
            
            for local_idx in range(8):
                if used[local_idx]:
                    continue
                # Check compatibility with already mapped Q3 neighbors
                valid = True
                for u in range(qv):
                    if (q3_adj[u] >> qv) & 1:
                        # u and qv are adjacent in Q3
                        # They must be adjacent in the graph, but we don't know the graph yet
                        # So we just need to record the constraint
                        pass
                # Actually for counting patterns, we just need all valid bijections
                # that respect the Q3 structure. But since we're looking at ALL possible
                # Q3s that could fit, any bijection works as long as the edges match.
                
                # Wait, I'm overcomplicating. Let me think differently.
                # For a given 8-vertex subset, a Q3 subgraph is determined by:
                # 1. A partition of the 8 vertices into 8 positions (a permutation)
                # 2. The 12 edges of Q3 mapped to edges of the subset
                # 
                # But different permutations can give the same set of 12 edges.
                # We just need the set of 12 edges.
                
                used[local_idx] = True
                mapping[qv] = local_idx
                bt(qv + 1)
                mapping[qv] = -1
                used[local_idx] = False
        
        bt(0)
        # Deduplicate
        unique = set(results)
        return list(unique)
    
    patterns = find_all_q3_embeddings(subset)
    if patterns:
        q3_patterns[subset] = patterns

# Now for each possible edge subset (represented as a bitmask or set),
# we need to count how many Q3 patterns are fully contained.

# For efficiency, let's precompute for each edge index, which Q3 patterns it belongs to
# Actually, let's just use a different approach for the local search.

def count_q3_violations(adj_bits):
    """Count how many 8-vertex subsets contain a Q3 subgraph."""
    violations = 0
    for subset in all_8_subsets:
        # Build local adjacency
        # Check if this subset contains Q3
        # Quick degree check first
        local_deg = [0] * 8
        for i in range(8):
            vi = subset[i]
            deg = 0
            for j in range(8):
                if i != j and (adj_bits[vi] >> subset[j]) & 1:
                    deg += 1
            local_deg[i] = deg
            if deg < 3:
                break
        else:
            # All degrees >= 3, do full check
            # Check all permutations is expensive. Use the precomputed patterns.
            if subset in q3_patterns:
                for pattern in q3_patterns[subset]:
                    # Check if all 12 edges in pattern are present
                    all_present = True
                    for eidx in pattern:
                        u, v = idx_to_edge[eidx]
                        if not ((adj_bits[u] >> v) & 1):
                            all_present = False
                            break
                    if all_present:
                        violations += 1
                        break  # Count each subset at most once
    
    return violations

def construct():
    start_time = time.time()
    time_limit = 115
    
    # Initialize with a random graph at medium density
    adj_bits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    current_edges = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
    current_violations = count_q3_violations(adj_bits)
    
    best_adj = adj_bits[:]
    best_edges = current_edges
    best_violations = current_violations
    
    # Simulated annealing
    T = 5.0
    cooling = 0.997
    
    iterations = 0
    while time.time() - start_time < time_limit - 5:
        iterations += 1
        
        # Choose a random edge to flip
        u, v = random.choice(idx_to_edge)
        
        # Flip it
        old_state = (adj_bits[u] >> v) & 1
        if old_state:
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
        else:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        
        new_edges = current_edges - (1 if old_state else -1)
        new_violations = count_q3_violations(adj_bits)
        
        # Compute energy difference
        # We want to minimize violations and maximize edges
        # Energy = violations * 1000 - edges
        old_energy = current_violations * 1000 - current_edges
        new_energy = new_violations * 1000 - new_edges
        delta = new_energy - old_energy
        
        # Accept if better or with probability exp(-delta/T)
        accept = False
        if delta < 0:
            accept = True
        elif T > 0.01:
            import math
            if random.random() < math.exp(-delta / T):
                accept = True
        
        if accept:
            current_edges = new_edges
            current_violations = new_violations
            if new_violations < best_violations or (new_violations == best_violations and new_edges > best_edges):
                best_adj = adj_bits[:]
                best_edges = new_edges
                best_violations = new_violations
        else:
            # Revert
            if old_state:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
            else:
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
        
        T *= cooling
    
    # Use the best solution found
    adj_bits = best_adj
    
    # Repair: remove edges to eliminate all Q3s
    while time.time() - start_time < time_limit - 1:
        if best_violations == 0:
            break
        
        # Find a violating subset
        found = False
        for subset in all_8_subsets:
            if subset not in q3_patterns:
                continue
            for pattern in q3_patterns[subset]:
                all_present = True
                present_edges = []
                for eidx in pattern:
                    u, v = idx_to_edge[eidx]
                    if (adj_bits[u] >> v) & 1:
                        present_edges.append((u, v))
                    else:
                        all_present = False
                        break
                if all_present:
                    # Remove a random edge from this Q3
                    if present_edges:
                        u, v = random.choice(present_edges)
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
                        best_violations -= 1
                        found = True
                    break
            if found:
                break
        
        if not found:
            break
    
    # Try to add back edges
    if time.time() - start_time < time_limit - 2:
        for eidx in range(total_possible):
            if time.time() - start_time > time_limit - 1:
                break
            u, v = idx_to_edge[eidx]
            if (adj_bits[u] >> v) & 1:
                continue
            
            # Try adding
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            
            if count_q3_violations(adj_bits) == 0:
                best_edges += 1
            else:
                # Revert
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
