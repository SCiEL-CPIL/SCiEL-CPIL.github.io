import time
import random

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    def build_multiway(vertices, next_color, randomize=False):
        """Build coloring using non-binary hierarchical clustering.
        Each internal node can have multiple children.
        One color is used for all edges between different children.
        """
        n = len(vertices)
        if n <= 1:
            return next_color
        if n == 2:
            u, v = vertices
            colors[u][v] = next_color
            colors[v][u] = next_color
            return next_color + 1
        if n == 3:
            u, v, w = vertices
            colors[u][v] = next_color
            colors[v][u] = next_color
            colors[v][w] = next_color + 1
            colors[w][v] = next_color + 1
            colors[u][w] = next_color + 2
            colors[w][u] = next_color + 2
            return next_color + 3
        
        # For larger sets, try multi-way splits
        # A k-way split uses 1 color for all cross-edges between the k groups
        # Then recursively color within each group
        
        # Try different numbers of groups
        best_next_color = next_color
        best_partition = None
        
        # Minimum group size is 2 (single vertices would waste colors)
        # Maximum groups: n // 2
        
        min_groups = 2
        max_groups = min(n // 2, 6)  # Limit to reasonable number
        
        for num_groups in range(min_groups, max_groups + 1):
            if time.time() > deadline:
                break
            
            # Generate random partition into num_groups of size >= 2
            # (except possibly one group of size 1 if n % num_groups != 0)
            
            # For simplicity, try to make groups as equal as possible
            base_size = n // num_groups
            remainder = n % num_groups
            
            # Try several random partitions
            for attempt in range(5 if randomize else 1):
                # Create list of sizes
                sizes = [base_size] * num_groups
                for i in range(remainder):
                    sizes[i] += 1
                
                # Shuffle vertices and assign to groups
                shuffled = list(vertices)
                random.shuffle(shuffled)
                
                groups = []
                idx = 0
                for size in sizes:
                    groups.append(shuffled[idx:idx+size])
                    idx += size
                
                # Build this partition
                next_color_test = next_color
                cross_color = next_color_test
                next_color_test += 1
                
                # Assign cross colors
                for i in range(num_groups):
                    for j in range(i + 1, num_groups):
                        for u in groups[i]:
                            for v in groups[j]:
                                colors[u][v] = cross_color
                                colors[v][u] = cross_color
                
                # Recursively build within each group
                valid = True
                for group in groups:
                    if len(group) > 1:
                        next_color_test = build_multiway(group, next_color_test, randomize)
                        if time.time() > deadline:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and next_color_test > best_next_color:
                    best_next_color = next_color_test
                    best_partition = groups
        
        if best_partition is None:
            # Fallback to binary split
            if n == 4: k = 1
            elif n == 5: k = 2
            elif n == 6: k = 3
            elif n == 7: k = 3
            elif n == 8: k = 4
            elif n == 9: k = 3
            elif n == 10: k = 4
            elif n == 11: k = 5
            elif n == 12: k = 6
            elif n == 13: k = 6
            elif n == 14: k = 7
            elif n == 15: k = 6
            elif n == 16: k = 7
            else: k = n // 2
            
            v1 = vertices[:k]
            v2 = vertices[k:]
            
            cross_color = next_color
            for u in v1:
                for v in v2:
                    colors[u][v] = cross_color
                    colors[v][u] = cross_color
            
            next_color += 1
            next_color = build_multiway(v1, next_color, randomize)
            if time.time() > deadline:
                return next_color
            next_color = build_multiway(v2, next_color, randomize)
            return next_color
        else:
            return best_next_color

    def is_valid_recolor(u, v, new_color):
        # Check all C4s containing edge (u,v)
        for w in range(N):
            if w == u or w == v:
                continue
            for x in range(N):
                if x == u or x == v or x == w:
                    continue
                if x >= w:
                    continue
                c1 = colors[u][w]
                c2 = colors[w][x]
                c3 = colors[x][v]
                if new_color != c1 and new_color != c2 and new_color != c3:
                    return False
        return True

    def local_search(colors, deadline):
        edge_list = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        used_colors = set(colors[i][j] for i in range(N) for j in range(i+1, N))
        all_colors = set(range(120))
        unused_colors = list(all_colors - used_colors)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(edge_list)
            for (u, v) in edge_list:
                if time.time() > deadline:
                    break
                if not unused_colors:
                    break
                
                old_color = colors[u][v]
                
                random.shuffle(unused_colors)
                
                for new_color in unused_colors:
                    if is_valid_recolor(u, v, new_color):
                        colors[u][v] = new_color
                        colors[v][u] = new_color
                        unused_colors.remove(new_color)
                        used_colors.add(new_color)
                        improved = True
                        break
        
        return len(used_colors)

    best_score = 0
    best_result = None
    
    # Run multiple attempts with different randomizations
    attempts = 15
    for i in range(attempts):
        if time.time() > deadline:
            break
            
        colors = [[0] * N for _ in range(N)]
        
        rand_flag = (i > 0)
        
        build_multiway(list(range(N)), 0, rand_flag)
        
        current_score = local_search(colors, deadline)
        
        if current_score > best_score:
            best_score = current_score
            best_result = []
            for r in range(N):
                for c in range(r+1, N):
                    best_result.append(colors[r][c])
    
    if best_result is None:
        # Fallback
        colors = [[0] * N for _ in range(N)]
        build_multiway(list(range(N)), 0, False)
        local_search(colors, deadline)
        best_result = []
        for r in range(N):
            for c in range(r+1, N):
                best_result.append(colors[r][c])
                
    return best_result
