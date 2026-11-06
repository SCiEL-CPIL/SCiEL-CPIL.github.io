import random
import time
import sys

def construct():
    """
    Construct a maximum edge set of Q7 with no 2-dimensional face having all 4 edges.
    Uses Simulated Annealing with multiple restarts.
    """
    d = 7
    n_vertices = 1 << d
    total_edges = n_vertices * d // 2  # 448
    
    # Precompute all edges and their "conflicts" (edges that form a C4 with them)
    # An edge is (u, v). A C4 is formed by (u, v), (u^k, v^k), (u, u^k), (v, v^k) for some k not in {i} where i is the bit flipped in (u,v).
    # Actually, simpler: For each edge e=(u,v), it conflicts with edge f=(x,y) if adding f to a set containing e completes a C4.
    # A C4 consists of 4 edges. If we have 3 edges of a C4, the 4th is forbidden.
    # Precompute for each edge index, the list of other edge indices that form a C4 with it.
    
    edges = []
    edge_to_idx = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                edge_to_idx[(v, u)] = idx # just in case, though we always use u<v

    # Build conflict graph: conflict[i] is a set of indices j such that edge i and edge j are part of the same C4.
    # Actually, we need to know: if I add edge e, which existing edges would create a C4?
    # A C4 is defined by two vertices u, v and two bit flips i, j (i != j).
    # The 4 edges are: (u, u^i), (u, u^j), (v, v^i), (v, v^j) where v = u ^ (1<<k)? No.
    # A 2D face is defined by a vertex u and two distinct bit positions i, j.
    # The vertices are u, u^i, u^j, u^(i^j).
    # The edges are: (u, u^i), (u, u^j), (u^i, u^(i^j)), (u^j, u^(i^j)).
    
    # Let's precompute for each edge, the other 3 edges in the faces it belongs to.
    # An edge (u, v) where v = u ^ (1<<i) belongs to faces defined by (u, i, j) and (v, i, j) for all j != i.
    # For a face (u, i, j):
    # Edges: e1=(u, u^i), e2=(u, u^j), e3=(u^i, u^(i^j)), e4=(u^j, u^(i^j)).
    # If we have e1, adding e2, e3, e4 creates C4s.
    # Specifically, if we have e1 and e2, we cannot add e3 or e4? No.
    # If we have e1, e2, and e3, then e4 is forbidden.
    # If we have e1, e2, and e4, then e3 is forbidden.
    # If we have e1, e3, and e4, then e2 is forbidden.
    # If we have e2, e3, and e4, then e1 is forbidden.
    
    # So, for each edge e, we can precompute a list of "forbidden sets".
    # If e is added, and the set contains ANY pair from its associated pairs, it's invalid?
    # No, it's invalid if it completes a C4.
    # Let's just use a fast validity check.
    
    # Precompute for each edge index, the list of other edge indices that share a face with it.
    # face_conflicts[e_idx] = list of (other_idx1, other_idx2) such that {e, other1, other2} is 3 edges of a C4.
    # If e is added, and both other1 and other2 are present, it's a C4.
    # Wait, if e is added, it conflicts with (other1, other2) if all three are present?
    # Yes. If I have edges A, B, C forming 3 sides of a square, D is forbidden.
    # So if I add D, I check if (A,B), (A,C), (B,C) are present? No, I check if any pair of the other 3 is present?
    # If I have A and B, and I add D, do I have a C4? Only if C is also there.
    # So adding D is valid unless (A and B) are present, OR (A and C) are present, OR (B and C) are present.
    
    # Let's build: for each edge e, a list of pairs (p, q) of other edges such that {e, p, q} is a subset of a C4.
    # If e is added, it is invalid if any (p, q) in this list has both p and q in the current set.
    
    conflicts = [[] for _ in range(total_edges)]
    
    # Iterate over all faces
    # Face defined by base u, and bits i, j (i < j)
    # Vertices: u, u^i, u^j, u^(i^j)
    # Edges:
    # e_ii = (u, u^i)
    # e_jj = (u, u^j)
    # e_ij_i = (u^i, u^(i^j))  # flips j
    # e_ij_j = (u^j, u^(i^j))  # flips i
    
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(u, v1), max(u, v1))
                e2 = (min(u, v2), max(u, v2))
                e3 = (min(v1, v3), max(v1, v3))
                e4 = (min(v2, v3), max(v2, v3))
                
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                # For each edge in this face, the other 3 form pairs that are forbidden if the edge is added.
                # If I add idx1, I can't have (idx2, idx3) or (idx2, idx4) or (idx3, idx4).
                # Wait, if I have idx2 and idx3, and I add idx1, do I have a C4?
                # Edges: idx1, idx2, idx3. The 4th is idx4.
                # A C4 requires all 4.
                # So having idx1, idx2, idx3 is NOT a C4.
                # The constraint is: No 2-dimensional face may have ALL FOUR edges selected.
                # So if I add idx1, I violate if idx2, idx3, AND idx4 are all present.
                
                # My previous logic was slightly off.
                # Constraint: |{e1, e2, e3, e4} \cap S| < 4.
                # So adding e is invalid if the other 3 are already in S.
                
                # So for each edge, I just need to know the other 3 edges in the face.
                # If all 3 others are in S, adding e is invalid.
                
                # Let's store for each edge, a list of "triples" of other edges.
                # conflicts[e_idx].append((other1, other2, other3))
                
                conflicts[idx1].append((idx2, idx3, idx4))
                conflicts[idx2].append((idx1, idx3, idx4))
                conflicts[idx3].append((idx1, idx2, idx4))
                conflicts[idx4].append((idx1, idx2, idx3))
    
    # Remove duplicates in conflicts list (an edge belongs to multiple faces)
    for i in range(total_edges):
        conflicts[i] = list(set(conflicts[i]))
    
    def is_valid_addition(edge_idx, current_set):
        # current_set is a set of edge indices
        for triple in conflicts[edge_idx]:
            if all(x in current_set for x in triple):
                return False
        return True

    def get_violations(current_set):
        # Returns list of (edge_idx, triple) where triple is in current_set
        # This is expensive, so we only use it for validation or local search moves that might break things.
        # For SA, we want to allow invalid states to escape local minima.
        # Energy = number of C4s.
        # We can calculate energy incrementally.
        pass

    # Simulated Annealing State
    # We will maintain:
    # - set of edges (as a boolean array or set)
    # - count of edges
    # - count of C4s (violations)
    
    # To efficiently update C4 count:
    # When adding edge e:
    #   For each triple T in conflicts[e]:
    #      if all(T) in S: violation_count += 1
    # When removing edge e:
    #   For each triple T in conflicts[e]:
    #      if all(T) in S (excluding e, but e is not in T): violation_count -= 1
    
    # Wait, if I have a C4 {e1, e2, e3, e4}, and I remove e1, the C4 is broken.
    # So violation_count decreases by 1 for each C4 containing e1.
    # How many C4s contain e1? It's the number of triples in conflicts[e1] that are fully present in S.
    
    # Let's implement a fast SA.
    
    def run_sa(seed, time_limit):
        random.seed(seed)
        
        # Initialize with random subset of edges
        # Start with empty or random?
        # Random subset of size ~300 is a good start.
        p = 300 / total_edges
        S = set()
        for i in range(total_edges):
            if random.random() < p:
                S.add(i)
        
        # Calculate initial violations
        violations = 0
        for e in S:
            for triple in conflicts[e]:
                if all(x in S for x in triple):
                    violations += 1
        # Each C4 is counted 4 times (once for each edge). So actual C4s = violations // 4.
        # But for energy, we can just use violations.
        
        current_energy = violations
        current_size = len(S)
        
        best_S = S.copy()
        best_size = current_size
        best_energy = current_energy
        
        T = 50.0
        cooling = 0.999
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if T < 1e-6:
                break
                
            # Choose a random move
            # 1. Add a random edge
            # 2. Remove a random edge
            # 3. Swap: remove one, add another
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Add
                e = random.randint(0, total_edges - 1)
                if e in S:
                    continue
                # Check delta energy
                delta_viol = 0
                for triple in conflicts[e]:
                    if all(x in S for x in triple):
                        delta_viol += 1
                # Delta size = +1
                # We want to maximize size, minimize violations.
                # Score = size - lambda * violations
                # Or just accept if valid and larger, or invalid but better?
                # Standard SA on (Energy, Size).
                # Let's define Cost = violations * 1000 - size. Minimize Cost.
                # Delta Cost = delta_viol * 1000 - 1
                
                delta_cost = delta_viol * 1000 - 1
                if delta_cost < 0 or random.random() < (10 ** (-delta_cost / T)):
                    S.add(e)
                    current_energy += delta_viol
                    current_size += 1
                    if current_energy == 0 and current_size > best_size:
                        best_S = S.copy()
                        best_size = current_size
                        best_energy = 0
            elif move_type < 0.8:
                # Remove
                if not S:
                    continue
                e = random.choice(list(S))
                # Remove e
                # Delta violations: count how many triples in conflicts[e] are fully in S (excluding e)
                # Since e is in S, and we are removing it, any C4 containing e is broken.
                # How many C4s contain e?
                # It is the number of triples T in conflicts[e] such that T is subset of S.
                # Note: T does not contain e.
                delta_viol = 0
                for triple in conflicts[e]:
                    if all(x in S for x in triple):
                        delta_viol += 1
                # Removing e reduces violations by delta_viol
                # Delta size = -1
                # Delta Cost = -delta_viol * 1000 + 1
                
                delta_cost = -delta_viol * 1000 + 1
                if delta_cost < 0 or random.random() < (10 ** (-delta_cost / T)):
                    S.remove(e)
                    current_energy -= delta_viol
                    current_size -= 1
                    # No need to update best if we removed an edge, unless we are in a weird state, but best is always valid
            else:
                # Swap
                if not S:
                    continue
                e_remove = random.choice(list(S))
                e_add = random.randint(0, total_edges - 1)
                if e_add in S:
                    continue
                if e_add == e_remove:
                    continue
                
                # Calculate delta for removing e_remove and adding e_add
                # This is complex because the removal affects the validity of the addition.
                # Simpler: just do two separate moves in sequence? No, that changes the state.
                # Let's just do Add or Remove. Swaps are less critical for this problem.
                pass

            T *= cooling
            
            # Safety check
            if current_energy == 0 and current_size > best_size:
                best_S = S.copy()
                best_size = current_size
                best_energy = 0

        return best_S, best_size

    # Main loop: Run multiple SA instances
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    best_overall_S = set()
    best_overall_size = 0
    
    # Initial greedy to get a decent start?
    # Let's just run SA.
    
    num_restarts = 0
    while time.time() < deadline:
        # Allocate time for this run
        # If we have 100s left, do a run of 10s?
        # Or just run until deadline - 5s?
        # Better: run for a fixed duration, e.g., 5-10s per run, restart.
        run_time = min(5.0, deadline - time.time() - 2.0)
        if run_time <= 0:
            break
        
        seed = num_restarts
        S, size = run_sa(seed, run_time)
        
        if size > best_overall_size:
            best_overall_size = size
            best_overall_S = S
        
        num_restarts += 1
        
        # If we hit 308, we can stop early? No, upper bound is 308.
        # If we hit 308, it's optimal.
        if best_overall_size >= 308:
            break

    # Convert set of indices to list of edges
    result = [list(edges[i]) for i in best_overall_S]
    return result

# To ensure we don't crash if something goes wrong, wrap in try-except?
# The contract says "invalid result, exception... scores 0".
# So we must ensure valid output.

if __name__ == "__main__":
    # Quick self-test
    edges = construct()
    print(f"Found {len(edges)} edges")
    # Validate
    d = 7
    n = 1 << d
    edge_set = set()
    for e in edges:
        u, v = e
        if u > v: u, v = v, u
        if (u, v) in edge_set:
            print("Duplicate edge")
        edge_set.add((u, v))
        # Check bit diff
        diff = u ^ v
        if diff & (diff - 1) != 0:
            print("Not a hypercube edge")
    
    # Check C4
    # For each face, check if all 4 edges are present
    # This is O(448 * 4) roughly
    for u in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(u, v1), max(u, v1))
                e2 = (min(u, v2), max(u, v2))
                e3 = (min(v1, v3), max(v1, v3))
                e4 = (min(v2, v3), max(v2, v3))
                
                if e1 in edge_set and e2 in edge_set and e3 in edge_set and e4 in edge_set:
                    print("C4 found!")
                    sys.exit(1)
    print("Valid")
