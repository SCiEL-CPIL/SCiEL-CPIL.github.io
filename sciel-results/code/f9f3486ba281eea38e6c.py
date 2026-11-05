            # Heuristic: Try used colors first
            used_colors = set()
            for k4_idx in edge_to_k4s[e_idx]:
                for c in range(max_colors):
                    if k4_counts[k4_idx][c] > 0:
                        used_colors.add(c)
            
            # Create a list of colors to try
            # Used colors first, then unused
            candidates = list(used_colors)
            for c in range(max_colors):
                if c not in used_colors:
                    candidates.append(c)
            
            for c in candidates:
                # ... rest of the loop
