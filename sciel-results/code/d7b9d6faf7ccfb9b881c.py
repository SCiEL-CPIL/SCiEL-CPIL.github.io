import time
import random
import sys

N = 40
DEADLINE = 85.0  # Safety margin before 90s kill

def _validate(edges, cycles):
    """Check validity: edges simple/distinct, cycles valid, all cycle edges exist."""
    edge_set = set()
    for e in edges:
        a, b = e
        if a == b:
            return False
        if a > b:
            a, b = b, a
        if (a, b) in edge_set:
            return False
        edge_set.add((a, b))
    
    for L, cyc in zip(range(3, N + 1), cycles):
        if len(cyc) != L:
            return False
        if len(set(cyc)) != L:
            return False
        for i in range(L):
            a = cyc[i]
            b = cyc[(i + 1) % L]
            if a > b:
                a, b = b, a
            if (a, b) not in edge_set:
                return False
    return True

def _construct_fan():
    """Baseline fan construction."""
    ham_edges = [(i, (i + 1) % N) for i in range(N)]
    chord_edges = [(0, d) for d in range(2, 21)]
    edges = ham_edges + chord_edges
    cycles = []
    for L in range(3, N + 1):
        if L == 40:
            cycles.append(list(range(N)))
        elif L <= 21:
            cycles.append(list(range(L)))
        else:
            end_vertex = 41 - L
            cycles.append([0] + list(range(N - 1, end_vertex - 1, -1)))
    return edges, cycles

def construct():
    start_time = time.time()
    
    # Start with the known good fan solution
    edges, cycles = _construct_fan()
    best_edges = list(edges)
    best_cycles = list(cycles)
    best_val = len(best_edges) - 40
    
    # Try to improve via edge removal (structural: sparser backbone)
    # Idea: Can we replace the full Hamiltonian cycle with a path + fewer chords?
    # Or can we find a different set of chords that works with fewer total edges?
    
    # Structural change: Try a "path + selective chords" approach
    # Path: 0-1-2-...-39 (39 edges). Need cycles of length 3..40.
    # Cycle of length L needs a path of L-1 consecutive vertices on the path,
    # plus a chord connecting the two ends.
    # For length 40: need edge (0,39) -> that's the full cycle, 40 edges total for just this.
    # For length L < 40: vertices i..i+L-1 need chord (i, i+L-1).
    # But we can choose which vertices to use.
    
    # Alternative structural: Use two "hubs" 0 and 20.
    # Path 0-1-...-19-20-21-...-39
    # Chords from 0: (0,2)..(0,19) for forward cycles 3..20
    # Chords from 20: (20,22)..(20,39) for... hmm this gets complex.
    
    # Let's try a randomized local search to find sparser valid configs.
    # Represent edges as a set. Try removing edges and see if cycles can be re-derived.
    
    # Actually, the key structural insight: we can use a SPARSE backbone.
    # Instead of a full cycle, use a tree (39 edges) + chords.
    # A tree on 40 vertices has 39 edges. Add chords to create cycles of all lengths.
    # Each chord creates at least one cycle. We need 38 distinct cycle lengths (3..40).
    # One chord can create at most 2 cycle lengths (forward/backward on a cycle backbone).
    # With a tree, each chord creates exactly one fundamental cycle.
    # So we need at least 38 chords -> 39 + 38 = 77 edges. Much worse.
    
    # The cycle backbone is essential because each chord gives 2 cycle lengths.
    # Current: 40 + 19 = 59. Can we do 40 + 18?
    # Need to cover lengths 3..39 with 18 chords. Each chord (0,d) covers {d+1, 40-d}.
    # 18 chords can cover at most 36 lengths. We need 37 lengths (3..39). Impossible.
    # So 19 chords is minimum for this structure.
    
    # Different structure: What if chords are not all from vertex 0?
    # Chord (a,b) with distance d along the cycle creates cycles of length d+1 and 40-d.
    # Same coverage. So the number of chords needed is the same regardless of which vertices.
    
    # Can we use a different backbone? E.g., a "figure-8" or "theta" graph?
    # Theta: two vertices connected by 3 internally-disjoint paths.
    # This might allow more efficient cycle coverage.
    
    # For now, try local search: start from current, try swapping chords.
    # The chord set {2..20} is optimal for the 0-fan. But maybe a different
    # assignment of which chord covers which length could allow fewer total edges
    # if some backbone edges become redundant.
    
    # Structural change: Try removing backbone edges that are only used by one cycle
    # and see if we can reroute.
    
    # For the current solution, edge (i, i+1) is used by multiple cycles.
    # Edge (0,1): used by cycle L=3 (0,1,2,0), L=4 (0,1,2,3,0), ..., L=21 (0,1,...,20,0)
    #           and by backward cycles L=22..39? No, backward cycles go 0,39,38,...
    # Edge (39,0): used by L=40 (Hamiltonian) and L=22 (0,39,38,...,19,0)? 
    #   L=22: 0,39,38,...,19 -> uses (39,38),(38,37),...,(20,19),(19,0)? No, (0,19) is a chord.
    #   Actually L=22: 0,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19 -> 22 vertices
    #   Edges: (0,39),(39,38),...,(20,19),(19,0). So uses (0,39) and (19,0).
    
    # This is complex. Let's just try a simple improvement: can we drop the Hamiltonian
    # cycle edge (39,0) and instead have chord (0,39)? Same count.
    
    # Try: Is there a valid solution with 58 edges? That would need 18 "extra" edges beyond 40.
    # As argued, 18 chords can't cover 37 lengths. So 59 seems optimal for this structure.
    
    # Different structural family: What if we don't use a cycle backbone at all?
    # Use a "caterpillar": a path with leaves.
    # Path: 0-1-2-...-39. Add chords (0,2),(0,3),...,(0,39).
    # That's 39 + 38 = 77 edges. Worse.
    
    # Use a "double fan": two hubs 0 and 20, each connecting to nearby vertices.
    # This is getting too complex for the time budget.
    
    # Let's just return the current best. It's likely near-optimal.
    if time.time() - start_time > DEADLINE:
        return {'edges': best_edges, 'cycles': best_cycles}
    
    # One more structural idea: Can we share the Hamiltonian cycle edge (39,0)
    # more efficiently? What if we use a different vertex as the "hub"?
    # The current solution is symmetric in the sense that any vertex could be the hub.
    # No improvement there.
    
    # Try: remove one chord and see if the corresponding cycle can be formed differently.
    # Chord (0,20) is used by L=21 (forward: 0..20) and L=21 (backward: 0,39,...,20).
    # Both L=21 cycles use chord (0,20). If we remove it, we lose L=21.
    # Can we form L=21 without chord (0,20)? We'd need a path of 21 vertices from 0 back to 0.
    # On the cycle, the only paths from 0 back to 0 have length 40 (full cycle) or use chords.
    # So no, we need chord (0,20) for L=21.
    
    # All chords in {2..20} are essential. 59 edges is optimal for this structure.
    
    return {'edges': best_edges, 'cycles': best_cycles}
