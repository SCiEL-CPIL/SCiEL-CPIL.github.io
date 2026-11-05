import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    def is_2_colorable(n, edges):
        """
        Check if hypergraph is 2-colorable.
        Returns True if colorable, False if not.
        For n <= 12, brute force is fast enough for final checks, 
        but we rely on the construction logic for the main search.
        """
        if not edges:
            return True
        
        # Brute force check for small n
        if n > 20:
            # Fallback to a simpler check or assume valid if constructed correctly
            # But we should be safe. Let's do a quick backtracking if n is large.
            # However, our construction uses n <= 12.
            pass
            
        all_colorings = [mask for mask in range(1 << n)]
        # Optimization: Skip masks that are complements? No, need to check all.
        # Actually, we can just check if ANY coloring works.
        
        # Convert edges to masks for fast intersection
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
            
        for mask in all_colorings:
            valid = True
            for em in edge_masks:
                # If edge is monochromatic in this coloring:
                # 1. All 0: (mask & em) == 0
                # 2. All 1: (mask & em) == em
                if (mask & em) == 0 or (mask & em) == em:
                    valid = False
                    break
            if valid:
                return True
        return False

    best_edges = None
    best_n = 0
    best_count = float('inf')

    # Strategy: Greedy Set Cover on the Dual
    # For a fixed n, enumerate all 2^n colorings.
    # An edge "kills" a coloring if the edge is monochromatic in that coloring.
    # We want to select a set of edges such that EVERY coloring is killed by at least one edge.
    # This is equivalent to finding a hitting set for the collection of "valid colorings".
    # Wait, the condition is: NO coloring is valid.
    # A coloring is valid if NO edge is monochromatic.
    # So we need to select edges such that for every coloring C, there exists an edge E in our set such that E is monochromatic in C.
    
    # This is exactly a Set Cover problem:
    # Universe U = set of all 2^n colorings.
    # For each possible edge E, let S_E = { C in U | E is monochromatic in C }.
    # We want to find a minimum subset of edges {E_1, ..., E_k} such that Union(S_Ei) = U.
    
    # Greedy Set Cover:
    # 1. Start with U_current = U.
    # 2. At each step, pick the edge E that maximizes |S_E \ U_current| (i.e., kills the most remaining valid colorings).
    # 3. Remove those colorings from U_current.
    # 4. Repeat until U_current is empty.
    
    # Optimization:
    # Represent U as a bitset? 2^n bits. For n=12, 4096 bits = 512 bytes. Very fast.
    # For n=13, 8192 bits = 1024 bytes. Still fast.
    # For n=14, 16384 bits = 2048 bytes.
    # Python integers can handle arbitrary length, so we can use a single int as a bitset for the universe.
    
    # Precomputation for a given n:
    # 1. Generate all possible edges (5-subsets).
    # 2. For each edge, precompute the bitmask of colorings it kills.
    #    Coloring c is killed by edge E if E is mono in c.
    #    E is mono in c if (c & E_mask) == 0 or (c & E_mask) == E_mask.
    
    for n in range(9, 15):
        if time.time() > deadline - 10:
            break
            
        # Total colorings
        total_colorings = 1 << n
        
        # Generate all possible 5-edges
        all_edges = list(combinations(range(n), 5))
        
        # Precompute kill masks for each edge
        # kill_mask[i] is an integer where bit j is 1 if coloring j is killed by edge i
        # This might be memory heavy if we store all, but n=12 -> 792 edges, each mask 4096 bits.
        # 792 * 512 bytes ~ 400KB. Fine.
        
        kill_masks = []
        for e in all_edges:
            e_mask = 0
            for v in e:
                e_mask |= (1 << v)
                
            km = 0
            # Iterate all colorings
            # Optimization: Instead of iterating all 2^n, we can construct the mask directly?
            # Condition: (c & e_mask) == 0 OR (c & e_mask) == e_mask.
            # This defines a subset of the hypercube.
            # For speed, let's just loop if n is small.
            for c in range(total_colorings):
                if (c & e_mask) == 0 or (c & e_mask) == e_mask:
                    km |= (1 << c)
            kill_masks.append(km)
            
        # Greedy Set Cover
        # remaining is a bitset of colorings not yet killed
        remaining = (1 << total_colorings) - 1
        selected_edges = []
        selected_indices = set()
        
        while remaining:
            if time.time() > deadline - 5:
                break
                
            best_idx = -1
            best_count_killed = 0
            best_new_remaining = remaining
            
            # Try each edge
            for i, km in enumerate(kill_masks):
                if i in selected_indices:
                    continue
                    
                # How many new colorings does this edge kill?
                # new_killed = remaining & km
                # count = popcount(new_killed)
                new_killed = remaining & km
                if new_killed:
                    # Count bits
                    # Python 3.10+ has int.bit_count()
                    cnt = new_killed.bit_count()
                    if cnt > best_count_killed:
                        best_count_killed = cnt
                        best_idx = i
                        best_new_remaining = remaining & ~km
                        
            if best_idx == -1:
                # No edge can kill remaining colorings?
                # This means the hypergraph of ALL edges is not enough?
                # That's impossible, the complete hypergraph kills all colorings?
                # Wait. If we include ALL edges, does it kill all colorings?
                # Yes, because for any coloring, there is some edge that is monochromatic?
                # Not necessarily. But if we have all edges, we have all 5-sets.
                # If a coloring is valid, no 5-set is monochromatic.
                # Is it possible that no 5-set is monochromatic?
                # Yes, e.g., balanced coloring.
                # So the complete hypergraph IS 2-colorable?
                # Let's check n=9.
                # If n=9, can we color it such that no 5-set is mono?
                # Yes, 4-1 split? No, 4-1 has a 5-set of the 4s? No, 4 < 5.
                # 4-1: 4 zeros, 1 one.
                # Any 5-set must contain at least 1 one (since only 4 zeros).
                # So no 5-set is all zeros.
                # Any 5-set must contain at least 4 zeros?
                # Max ones is 1. So any 5-set has at least 4 zeros.
                # So no 5-set is all ones.
                # So the complete hypergraph on 9 vertices is 2-colorable!
                # This contradicts my earlier thought.
                
                # Let's re-verify: m(5) is the minimum number of edges in a non-2-colorable hypergraph.
                # If complete hypergraph on 9 is 2-colorable, then we need more vertices or specific edges.
                # Actually, if complete hypergraph on n is 2-colorable, then no subset of edges can be non-2-colorable?
                # No. A subset can be non-2-colorable even if the superset is?
                # No, if H is 2-colorable, any subgraph H' is 2-colorable (same coloring works).
                # So if Complete(n) is 2-colorable, then NO hypergraph on n vertices is non-2-colorable.
                # Therefore, n must be such that Complete(n) is NOT 2-colorable.
                
                # When is Complete(n) not 2-colorable?
                # It is not 2-colorable if there is NO coloring such that no 5-set is monochromatic.
                # This is equivalent to: For every coloring, there exists a 5-set that is monochromatic.
                # This is a Ramsey-type number.
                # R(5,5) is the smallest N such that any 2-coloring of K_N contains a monochromatic K_5.
                # But here we are coloring vertices, not edges.
                # This is the "Property B" for hypergraphs.
                # The complete 5-uniform hypergraph on n vertices is non-2-colorable iff n >= R_5?
                # Actually, it's simpler.
                # If we color vertices, a 5-set is monochromatic if all 5 have same color.
                # We want to avoid this.
                # Max size of a 2-colorable complete 5-uniform hypergraph?
                # If we have n vertices, and we split them into two sets of sizes a and b.
                # To avoid a monochromatic 5-set, we need a < 5 and b < 5.
                # So a <= 4 and b <= 4.
                # So n = a + b <= 8.
                # Thus, for n=8, Complete(8) is 2-colorable (4-4 split).
                # For n=9, Complete(9) is NOT 2-colorable?
                # If n=9, any split a+b=9 has a>=5 or b>=5.
                # So any 2-coloring has a monochromatic 5-set.
                # So Complete(9) is non-2-colorable.
                
                # So my previous check for n=9 complete graph was wrong?
                # Let's re-evaluate "4-1 split" on n=9.
                # Vertices: 4 zeros, 5 ones.
                # Is there a 5-set of ones? Yes, the 5 ones form a 5-set.
                # So the edge consisting of the 5 ones is monochromatic.
                # So the coloring is invalid.
                # So Complete(9) is indeed non-2-colorable.
                
                # So why did the greedy algorithm fail to find a solution?
                # It should not fail if Complete(n) is non-2-colorable.
                # If best_idx == -1, it means no single edge kills any remaining coloring.
                # This implies the remaining colorings are NOT killed by ANY edge in the full set.
                # Which means the remaining colorings are valid for the complete hypergraph.
                # Which means Complete(n) is 2-colorable.
                # So this branch should only be taken if n <= 8.
                # Since we start at n=9, this shouldn't happen.
                
                break
                
            selected_edges.append(all_edges[best_idx])
            selected_indices.add(best_idx)
            remaining = best_new_remaining
            
        if remaining == 0:
            # We have a valid non-2-colorable hypergraph
            if len(selected_edges) < best_count:
                best_count = len(selected_edges)
                best_edges = selected_edges
                best_n = n
                
            # Try to improve with local search / deletion
            # Greedy set cover is often close to optimal, but might have redundant edges.
            # Check if any edge can be removed.
            current_edges = selected_edges[:]
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                for i in range(len(current_edges)):
                    # Try removing current_edges[i]
                    test_edges = current_edges[:i] + current_edges[i+1:]
                    if not is_2_colorable(n, test_edges):
                        # Still non-2-colorable without this edge
                        current_edges = test_edges
                        changed = True
                        break # Restart loop with new size
                
            if len(current_edges) < best_count:
                best_count = len(current_edges)
                best_edges = current_edges
                best_n = n
                
        # If we found a very small one, we can stop
        if best_count <= 51:
            # We can't do better than known bounds? 
            # Actually, we want to minimize. 51 is the known best.
            # Can we go lower?
            # If we find < 51, that's a breakthrough.
            # Let's continue searching other n or restart with different randomization?
            # Greedy is deterministic. 
            # We can randomize the tie-breaking in greedy to find different solutions.
            pass

    if best_edges is None:
        # Fallback: Complete graph on 9
        n = 9
        best_edges = list(combinations(range(n), 5))
        best_n = n
    else:
        # Final verification
        if is_2_colorable(best_n, best_edges):
            # Should not happen, but fallback just in case
            n = 9
            best_edges = list(combinations(range(n), 5))
            best_n = n

    # Format output
    final_edges = [list(e) for e in best_edges]
    
    return {'n': best_n, 'edges': final_edges}

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, edges={len(res['edges'])}")
