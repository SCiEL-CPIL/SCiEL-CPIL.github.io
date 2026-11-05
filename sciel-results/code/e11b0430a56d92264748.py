<<<<<<< SEARCH
        # Run DFS from a few promising starts
        # To avoid redundant work, we can try starts that are in the best_path or low degree
        # Let's try the first 10 low-degree vertices in the component
        for s in start_verts[:10]:
            if time.time() > deadline or node_count > NODE_LIMIT:
                break
            # If s is already in best_path, we might still want to explore different paths from it
            dfs(s, 1 << s, [s])
=======
        # Run DFS from a few promising starts
        # To avoid redundant work, we can try starts that are in the best_path or low degree
        # Let's try the first 5 low-degree vertices in the component
        # Also try the endpoints of the best_path if they are in the component
        dfs_starts = set(start_verts[:5])
        if best_path:
            for v in [best_path[0], best_path[-1]]:
                if main_comp & (1 << v):
                    dfs_starts.add(v)
        
        for s in dfs_starts:
            if time.time() > deadline or node_count > NODE_LIMIT:
                break
            # If s is already in best_path, we might still want to explore different paths from it
            dfs(s, 1 << s, [s])
>>>>>>> REPLACE
