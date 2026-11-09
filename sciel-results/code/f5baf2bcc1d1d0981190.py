    def check_local_validity(colors, edge_idx, new_color):
        """Check if changing edge_idx to new_color keeps all cycles valid."""
        # Only need to check cycles involving edge_idx
        for ci in edge_to_cycles_arr[edge_idx]:
            cyc = cycles_np[ci]
            # Get colors of all 4 edges in this cycle
            # One of them is edge_idx, which we're changing
            col_list = []
            for ei in cyc:
                if ei == edge_idx:
                    col_list.append(new_color)
                else:
                    col_list.append(colors[ei])
            # Check if all 4 are distinct
            if len(set(col_list)) == 4:
                return False
        return True
    
    # Hill climbing
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for ei in range(NUM_EDGES):
            if time.time() > deadline - 10:
                break
            # Try changing edge ei to a color not currently used, or a random color
            # First try to add a new color
            current_color = current[ei]
            # Try some random new colors
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                new_color = random.randint(0, 119)
                if new_color == current_color:
                    continue
                # Check validity
                if check_local_validity(current, ei, new_color):
                    # Tentatively apply
                    old_color = current[ei]
                    current[ei] = new_color
                    new_score = count_colors(current)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_colors = current[:]
                    else:
                        current[ei] = old_color
