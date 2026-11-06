import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def check_non_2colorable_bitmask(n, edges):
        """Check if hypergraph is NOT 2-colorable using bitmask enumeration.
        Returns True if non-2-colorable (no valid coloring exists)."""
        # Convert edges to masks
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        # Check all colorings (exclude all-0 and all-1 which are always mono for any edge)
        # Actually, we need to check if ANY coloring works.
        # A coloring is invalid if ANY edge is monochromatic.
        # So we look for a coloring where NO edge is monochromatic.
        
        # Optimization: Iterate colorings, check edges
        # For n=12, 4094 colorings. 51 edges. 200k ops. Fast.
        # For n=14, 16382 colorings. 51 edges. 800k ops. Fast.
        # For n=16, 65534 colorings. 51 edges. 3.3M ops. ~0.1s.
        
        for c in range(1, (1 << n) - 1):
            valid = True
            for em in edge_masks:
                # If mono, invalid
                if (c & em) == 0 or (c & em) == em:
                    valid = False
                    break
            if valid:
                return False  # Found a valid coloring, so it IS 2-colorable
        
        return True  # No valid coloring found, so it is NOT 2-colorable

    def greedy_minimize(n, target_edges, deadline):
        """
        Try to find a non-2-colorable hypergraph on n vertices with <= target_edges.
        Uses random sampling + local search / greedy addition.
        """
        all_edges = list(combinations(range(n), 5))
        m_total = len(all_edges)
        
        if m_total < target_edges:
            target_edges = m_total
            
        # Pre-compute masks for all edges
        all_masks = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            all_masks.append(mask)
        
        rng = random.Random(42 + n)
        
        # Strategy 1: Random subset of size target_edges, check.
        # If found, try to minimize by removing edges.
        # Strategy 2: Greedy addition starting from empty, adding edge that kills most colorings.
        
        # Let's try Greedy Addition first as it's more systematic.
        # But checking "kills most colorings" requires enumerating colorings.
        # For n=12, 4096 colorings.
        # We maintain a set of "alive" colorings.
        # Initially all are alive.
        # When we add an edge, we remove colorings that are mono for it.
        # We want to reach 0 alive colorings.
        
        # This is the approach in the current program. Let's optimize it.
        
        # Alive colorings as a set of ints
        alive = set(range(1, (1 << n) - 1))
        
        selected_indices = []
        selected_masks = []
        
        # To speed up "find edge that kills most", we can sample edges.
        # Or iterate all.
        # For n=12, 792 edges. 792 * 4096 = 3.2M checks per step.
        # If we need 50 steps, 160M checks.
        # In Python, 160M simple ops might take ~10-20s.
        # We have ~100s budget.
        
        # Let's try n=10, 11, 12.
        
        while alive:
            if time.time() > deadline - 2:
                return None
            
            # Find best edge to add
            best_idx = -1
            best_kills = 0
            
            # Sample if n is large
            if n > 12:
                sample_size = min(50, m_total - len(selected_indices))
                candidates = rng.sample(range(m_total), sample_size)
            else:
                candidates = range(m_total)
            
            # Filter out already selected
            selected_set = set(selected_indices)
            
            for i in candidates:
                if i in selected_set:
                    continue
                
                em = all_masks[i]
                kills = 0
                # Count how many alive colorings are killed by this edge
                # i.e., are mono
                for c in alive:
                    if (c & em) == 0 or (c & em) == em:
                        kills += 1
                
                if kills > best_kills:
                    best_kills = kills
                    best_idx = i
            
            if best_idx == -1 or best_kills == 0:
                return None
            
            selected_indices.append(best_idx)
            selected_masks.append(all_masks[best_idx])
            
            # Update alive
            em = all_masks[best_idx]
            new_alive = set()
            for c in alive:
                if not ((c & em) == 0 or (c & em) == em):
                    new_alive.add(c)
            alive = new_alive
            
            if len(selected_indices) > target_edges:
                return None
        
        # Found a non-2-colorable set
        # Try to minimize it by removing edges
        # We have selected_indices.
        # Try removing each edge, if still non-2-colorable, remove it.
        
        current_edges = list(selected_indices)
        
        # Optimization: Only try to remove if we have more than some minimum
        # We want to get as close to 51 as possible, or lower.
        
        # Shuffle removal order
        rng.shuffle(current_edges)
        
        # We need to re-check non-2-colorability after each removal.
        # This is expensive.
        # Let's just return the greedy result if it's good enough.
        # Or do a few removals.
        
        # For now, return the greedy set.
        # If len <= 51, it's great.
        
        return [all_edges[i] for i in current_edges]

    best_edges = None
    best_m = float('inf')
    
    # Try n from 10 to 13
    # n=10: 252 edges.
    # n=11: 462 edges.
    # n=12: 792 edges.
    # n=13: 1287 edges.
    
    for n in [10, 11, 12, 13]:
        if time.time() > deadline - 10:
            break
            
        edges = greedy_minimize(n, 100, deadline)
        
        if edges is not None:
            m = len(edges)
            if m < best_m:
                best_m = m
                best_edges = edges
                
            # If we found <= 51, we can likely stop or try to improve
            if m <= 51:
                # Try to minimize further?
                # Greedy already tries to be efficient.
                # Let's break if we find 51.
                break

    if best_edges is None:
        # Fallback: Random search
        for n in [10, 11, 12]:
            if time.time() > deadline - 5:
                break
            all_edges = list(combinations(range(n), 5))
            rng = random.Random(n)
            for trial in range(100):
                if time.time() > deadline - 2:
                    break
                # Try random subset of size 51
                if len(all_edges) < 51:
                    continue
                indices = rng.sample(range(len(all_edges)), 51)
                subset = [all_edges[i] for i in indices]
                if check_non_2colorable_bitmask(n, subset):
                    best_edges = subset
                    best_m = 51
                    break
            if best_edges:
                break

    if best_edges is None:
        # Absolute fallback: Complete hypergraph on 9 vertices
        # 126 edges.
        n = 9
        edges = [list(e) for e in combinations(range(n), 5)]
        best_edges = edges
        best_m = len(edges)

    return {'n': n, 'edges': best_edges}
