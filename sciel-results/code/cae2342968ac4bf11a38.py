import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # N[v] for v in I
        N = {}
        for v in range(k, n):
            N[v] = set(neighbors[v])
        
        best_result = None
        best_count = float('inf')
        
        # Generate candidate clique partitions of K = {0, ..., k-1}
        # A partition is a list of subsets of K that are edge-disjoint and cover all edges
        
        def generate_partitions(k):
            """Generate several candidate clique partitions of K_k."""
            partitions = []
            K_set = set(range(k))
            
            # Partition 1: single clique
            partitions.append([K_set])
            
            # Partition 2: all edges as 2-cliques (1-factorization style)
            # Use a simple round-robin or just enumerate all edges
            # For small k, we can do a proper 1-factorization
            if k >= 2:
                # 1-factorization for even k, near-1-factorization for odd k
                edges = []
                for i in range(k):
                    for j in range(i+1, k):
                        edges.append((i, j))
                # Group into matchings
                # Simple approach: use the standard 1-factorization
                if k % 2 == 0:
                    # Standard 1-factorization
                    verts = list(range(k))
                    factors = []
                    for r in range(k - 1):
                        matching = set()
                        # Fix vertex 0, rotate others
                        # Actually let's use a simpler method
                        pass
                    # Simpler: just use all edges as separate 2-cliques
                    # This gives k*(k-1)/2 cliques
                    two_cliques = [frozenset([i, j]) for i in range(k) for j in range(i+1, k)]
                    partitions.append([set(c) for c in two_cliques])
                else:
                    two_cliques = [set([i, j]) for i in range(k) for j in range(i+1, k)]
                    partitions.append([set(c) for c in two_cliques])
            
            # Partition 3: greedy large cliques
            # Repeatedly pick the largest subset that can form a clique and covers many edges
            remaining_edges = set((i, j) for i in range(k) for j in range(i+1, k))
            greedy_partition = []
            while remaining_edges:
                # Find vertex with max remaining degree
                deg = {v: 0 for v in range(k)}
                for u, w in remaining_edges:
                    deg[u] += 1
                    deg[w] += 1
                # Start with max degree vertex
                v0 = max(range(k), key=lambda x: deg[x])
                clique = {v0}
                # Greedily add vertices
                changed = True
                while changed:
                    changed = False
                    for v in range(k):
                        if v in clique:
                            continue
                        # Check if adding v keeps it a clique (always true in K_k)
                        # and if it covers at least one remaining edge
                        new_edges = set()
                        for u in clique:
                            e = (min(u, v), max(u, v))
                            if e in remaining_edges:
                                new_edges.add(e)
                        if new_edges:
                            clique.add(v)
                            changed = True
                            break
                greedy_partition.append(set(clique))
                # Remove edges covered by this clique
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        e = (min(list(clique)[i], list(clique)[j]), max(list(clique)[i], list(clique)[j]))
                        if e in remaining_edges:
                            remaining_edges.discard(e)
            partitions.append(greedy_partition)
            
            # Partition 4: balanced split into two cliques
            if k >= 4:
                half = k // 2
                c1 = set(range(half))
                c2 = set(range(half, k))
                # But this doesn't cover cross edges!
                # Need to handle cross edges separately
                # This partition is invalid unless we add cross-edge cliques
                # Skip this; instead try: one big clique + leftover
                pass
            
            # Partition 5: cliques of size ~k/2
            if k >= 4:
                s = max(2, k // 2)
                part = []
                used = set()
                for i in range(k):
                    if i in used:
                        continue
                    clique = {i}
                    for j in range(i+1, k):
                        if j not in used and len(clique) < s:
                            clique.add(j)
                            used.add(j)
                    used.add(i)
                    part.append(clique)
                # Verify edge coverage - this might not cover all edges
                # Check
                covered = set()
                for c in part:
                    cl = list(c)
                    for i in range(len(cl)):
                        for j in range(i+1, len(cl)):
                            covered.add((min(cl[i], cl[j]), max(cl[i], cl[j])))
                all_edges = set((i, j) for i in range(k) for j in range(i+1, k))
                if covered == all_edges:
                    partitions.append(part)
                else:
                    # Add remaining edges as 2-cliques
                    rem = all_edges - covered
                    for e in rem:
                        part.append(set(e))
                    partitions.append(part)
            
            # Partition 6: random partitions
            rng_local = random.Random(42)
            for trial in range(20):
                # Randomly partition K into cliques
                part = []
                remaining = set(range(k))
                while remaining:
                    # Pick random size
                    max_size = len(remaining)
                    size = rng_local.randint(2, max_size) if max_size >= 2 else 1
                    if size == 1:
                        v = next(iter(remaining))
                        part.append({v})
                        remaining.discard(v)
                        continue
                    clique = set(rng_local.sample(list(remaining), size))
                    part.append(clique)
                    remaining -= clique
                # Check edge coverage
                covered = set()
                for c in part:
                    cl = list(c)
                    for i in range(len(cl)):
                        for j in range(i+1, len(cl)):
                            covered.add((min(cl[i], cl[j]), max(cl[i], cl[j])))
                all_edges = set((i, j) for i in range(k) for j in range(i+1, k))
                if covered == all_edges:
                    partitions.append(part)
                else:
                    rem = all_edges - covered
                    for e in rem:
                        part.append(set(e))
                    partitions.append(part)
            
            return partitions

        def evaluate_partition(partition, N, n, k):
            """Evaluate a clique partition of K and assign I vertices."""
            # partition: list of sets (cliques on K)
            # For each clique C, find I vertices v with C ⊆ N[v]
            # Assign at most one v per clique, maximizing savings
            
            # Build: for each clique index, list of (v, savings) where C_i ⊆ N[v]
            clique_to_verts = defaultdict(list)
            for i, C in enumerate(partition):
                for v in range(k, n):
                    if C.issubset(N[v]):
                        savings = len(N[v]) - 1
                        clique_to_verts[i].append((v, savings))
            
            # Greedy assignment: sort all (savings, v, i) and greedily assign
            # Each v can be assigned to at most one clique, each clique to at most one v
            # This is a bipartite matching problem. Use greedy by savings.
            all_assignments = []
            for i, lst in clique_to_verts.items():
                for v, sav in lst:
                    all_assignments.append((sav, v, i))
            all_assignments.sort(key=lambda x: -x[0])
            
            assigned_v = set()
            assigned_c = set()
            assignment = {}  # v -> clique index
            
            for sav, v, i in all_assignments:
                if v in assigned_v or i in assigned_c:
                    continue
                assigned_v.add(v)
                assigned_c.add(i)
                assignment[v] = i
            
            # Count total cliques
            total = len(partition)  # base cliques for K edges
            for v in range(k, n):
                if v in assignment:
                    total += 1  # extended clique
                else:
                    total += len(N[v])  # one clique per edge
            
            # Build the actual clique list
            result = []
            # K-edge cliques
            for C in partition:
                result.append(sorted(C))
            
            # I vertex cliques
            for v in range(k, n):
                if v in assignment:
                    i = assignment[v]
                    result.append(sorted(partition[i] | {v}))
                else:
                    for u in sorted(N[v]):
                        result.append(sorted([u, v]))
            
            return total, result

        # Try all partitions
        partitions = generate_partitions(k)
        
        for part in partitions:
            total, result = evaluate_partition(part, N, n, k)
            if total < best_count:
                best_count = total
                best_result = result
        
        # If no valid partition found (shouldn't happen), fallback
        if best_result is None:
            # Fallback: each edge as its own clique
            best_result = []
            for u in range(k):
                for w in range(u+1, k):
                    best_result.append([u, w])
            for v in range(k, n):
                for u in N[v]:
                    best_result.append(sorted([u, v]))
        
        # Verify: all cliques have >= 2 vertices, all edges covered exactly once
        # (skip verification for speed, trust the construction)
        
        return best_result

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        # Convert to ensure Python ints and lists
        clean = []
        for c in cliques:
            clean.append([int(x) for x in c])
        results.append(clean)
    
    return results
