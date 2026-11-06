import math
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit

    n = 256
    max_sum = 2 * n  # 512

    # Precompute squares up to max_sum
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)

    # Filter out numbers where 2*a is a square (diagonal constraint)
    # If 2*a is a square, a cannot be in the set
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    num_valid = len(valid_numbers)
    
    if num_valid == 0:
        return []

    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}

    # Build compatibility graph (edges where sum is NOT square)
    # adj[i] is a bitmask of vertices j such that valid_numbers[i] + valid_numbers[j] is NOT a square
    # Note: We must ensure i != j for the edge to be valid in the clique context?
    # Actually, in a clique, we select distinct vertices. The "independent set" condition for a<=b includes a=b.
    # We already handled a=b by filtering valid_numbers.
    # So for i != j, we check if valid_numbers[i] + valid_numbers[j] is a square.
    # If it is NOT a square, they are compatible.
    
    adj = [0] * num_valid
    for i in range(num_valid):
        a = valid_numbers[i]
        mask = 0
        for j in range(num_valid):
            if i == j:
                continue
            b = valid_numbers[j]
            if (a + b) not in squares:
                mask |= (1 << j)
        adj[i] = mask

    all_mask = (1 << num_valid) - 1

    # Tomita Algorithm for Maximum Clique
    # We maintain a list of candidate vertices P (as a bitmask) and the current clique C (as a bitmask or count)
    # To get good bounds, we order the candidates in P by a greedy coloring heuristic.
    
    best_clique_size = [0]
    best_clique_set = [0]

    def greedy_color(P):
        """
        Greedily color the subgraph induced by P to provide an upper bound on clique size.
        Returns a list of (color, vertex_index) sorted by color, and the max color.
        The number of colors is an upper bound on the clique size in P.
        """
        P_list = []
        # Convert bitmask to list of indices
        v = P
        while v:
            lowbit = v & (-v)
            idx = lowbit.bit_length() - 1
            P_list.append(idx)
            v ^= lowbit
        
        if not P_list:
            return [], 0

        colored = [False] * len(P_list)
        colors = [0] * len(P_list)
        max_color = 0
        # Map original index to position in P_list for quick access? 
        # Or just use a set of uncolored indices.
        
        # Standard greedy coloring:
        # Iterate through colors 1, 2, ...
        # For each color, pick vertices that are not adjacent to each other.
        # This is essentially finding independent sets sequentially.
        
        uncolored = set(P_list)
        color = 0
        while uncolored:
            color += 1
            # Find a maximal independent set in the subgraph induced by uncolored
            # Greedy: pick a vertex, remove it and its neighbors
            # To maximize speed, we can just pick vertices one by one if they are not adjacent to already picked in this color.
            
            picked = set()
            remaining_to_check = list(uncolored)
            for v_idx in remaining_to_check:
                if v_idx not in uncolored:
                    continue
                # Check if v_idx is compatible with all already picked in this color
                compatible_with_picked = True
                for p_idx in picked:
                    # If they are connected in the original graph (compatible), they CANNOT be in the same independent set of the compatibility graph?
                    # Wait.
                    # We are finding a Clique in the Compatibility Graph.
                    # The coloring bound is for the Clique.
                    # A clique in G corresponds to an independent set in the complement? No.
                    # Tomita algorithm works on the graph G (Compatibility).
                    # The bound is derived by coloring G.
                    # A clique in G is a set of vertices all pairwise adjacent.
                    # In a coloring of G, no two adjacent vertices have the same color.
                    # Therefore, a clique must have vertices with distinct colors.
                    # So the number of colors is an upper bound on the clique size.
                    
                    # So, in the coloring step, we assign colors such that adjacent vertices have different colors.
                    # This means for a given color, we pick an independent set in G.
                    # So v_idx must NOT be adjacent to any picked vertex in the current color.
                    if adj[v_idx] & (1 << p_idx):
                        compatible_with_picked = False
                        break
                
                if compatible_with_picked:
                    picked.add(v_idx)
                    uncolored.remove(v_idx)
                    colors[v_idx] = color
            
            if picked:
                max_color = color
        
        # Sort vertices by color
        # We want to process vertices in order of increasing color to prune early?
        # Actually, we return the vertices sorted by color so that the recursive step can prune
        # if current_size + color(v) <= best_size.
        
        # Create list of (color, vertex)
        colored_vertices = []
        for v_idx in P_list:
            colored_vertices.append((colors[v_idx], v_idx))
        
        # Sort by color
        colored_vertices.sort(key=lambda x: x[0])
        
        return colored_vertices, max_color

    def expand(C, P):
        """
        Recursive expansion for Maximum Clique.
        C: bitmask of vertices in the current clique
        P: bitmask of candidate vertices
        """
        if time.time() > deadline:
            return
        
        # If P is empty, update best
        if P == 0:
            if C.bit_count() > best_clique_size[0]:
                best_clique_size[0] = C.bit_count()
                best_clique_set[0] = C
            return
        
        # Get coloring order
        colored_P, max_color = greedy_color(P)
        
        # If current size + max color <= best size, prune
        if C.bit_count() + max_color <= best_clique_size[0]:
            return
            
        # Iterate through vertices in descending order of color (reverse of sorted)
        # This allows for tighter pruning: if we process high color first, we know the remaining
        # vertices have lower or equal color.
        # Wait, standard Tomita processes in order of increasing color for the list, 
        # but the recursion usually pops from the end of the list (highest color) or start?
        
        # Let's look at the logic:
        # If we have vertices v1, v2, ... vk with colors c1 <= c2 <= ... <= ck.
        # If we pick vk, the new candidate set P' = P \ {vk} \ N(vk).
        # The max color in P' is at most ck-1? Not necessarily, but it's bounded by the color of vk.
        # Actually, the property is: if we order vertices v1..vk by color c1..ck,
        # then when we consider vk, the upper bound for any clique in the remaining candidates
        # (excluding vk and its neighbors) is at most ck - 1 + (some small factor)?
        # The standard prune condition is: if |C| + c(v) <= best, then no vertex u with c(u) <= c(v) can improve.
        # So we iterate from the vertex with the HIGHEST color down to the lowest.
        
        # colored_P is sorted by increasing color.
        # So we iterate from end to start.
        
        for i in range(len(colored_P) - 1, -1, -1):
            color, v_idx = colored_P[i]
            
            # Pruning check
            if C.bit_count() + color <= best_clique_size[0]:
                # Since colors are non-decreasing, all subsequent vertices (0 to i-1) have color <= color.
                # So none of them can lead to a better solution.
                break
            
            # v_idx must be in P
            if not (P & (1 << v_idx)):
                continue
                
            # Recurse
            new_C = C | (1 << v_idx)
            # New P is vertices in P that are compatible with v_idx, excluding v_idx itself
            new_P = P & adj[v_idx] & ~(1 << v_idx)
            
            expand(new_C, new_P)

    # Initial call
    # We can order the initial P by degree to improve performance
    # Tomita usually benefits from a good initial ordering.
    # Let's just pass all_mask.
    
    # To help greedy_color be fast and effective, we might want to reorder vertices initially.
    # However, the coloring inside the recursion handles the local ordering.
    
    expand(0, all_mask)

    # Extract result
    result = []
    v = best_clique_set[0]
    while v:
        lowbit = v & (-v)
        idx = lowbit.bit_length() - 1
        result.append(valid_numbers[idx])
        v ^= lowbit
        
    result.sort()
    
    # Fallback if empty (should not happen as num_valid > 0 and single vertex is always a clique)
    if not result:
         # Find any valid number
         for a in valid_numbers:
             result.append(a)
             break

    # Verify validity
    result_set = set(result)
    valid = True
    for a in result:
        if 2 * a in squares:
            valid = False
            break
        for b in result:
            if a <= b and (a + b) in squares:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # This should not happen, but just in case, return a single valid element
        for a in valid_numbers:
            return [a]
        return [1]

    return result

# To test locally
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(f"Set: {res}")
