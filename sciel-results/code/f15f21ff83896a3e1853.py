            # Calculate delta cost
            delta = 0
            
            for x in range(n):
                if x == u or x == v:
                    continue
                
                # Color old_c: v removed from N_old(u), u removed from N_old(v)
                if color[v][x] == old_c:
                    k = K[old_c][u][x]
                    if k == 2: # Becomes 1, so violation removed
                        delta -= 1
                if color[u][x] == old_c:
                    k = K[old_c][v][x]
                    if k == 2:
                        delta -= 1

                # Color new_c: v added to N_new(u), u added to N_new(v)
                if color[v][x] == new_c:
                    k = K[new_c][u][x]
                    if k == 1: # Becomes 2, so violation added
                        delta += 1
                if color[u][x] == new_c:
                    k = K[new_c][v][x]
                    if k == 1:
                        delta += 1

            # Acceptance
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                # Apply
                color[u][v] = new_c
                color[v][u] = new_c
                cost += delta
                
                # Update K and viol
                for x in range(n):
                    if x == u or x == v:
                        continue
                    
                    if color[v][x] == old_c:
                        K[old_c][u][x] -= 1
                        K[old_c][x][u] -= 1
                        viol[old_c][u][x] = 1 if K[old_c][u][x] >= 2 else 0
                        viol[old_c][x][u] = viol[old_c][u][x]
                    if color[u][x] == old_c:
                        K[old_c][v][x] -= 1
                        K[old_c][x][v] -= 1
                        viol[old_c][v][x] = 1 if K[old_c][v][x] >= 2 else 0
                        viol[old_c][x][v] = viol[old_c][v][x]

                    if color[v][x] == new_c:
                        K[new_c][u][x] += 1
                        K[new_c][x][u] += 1
                        viol[new_c][u][x] = 1 if K[new_c][u][x] >= 2 else 0
                        viol[new_c][x][u] = viol[new_c][u][x]
                    if color[u][x] == new_c:
                        K[new_c][v][x] += 1
                        K[new_c][x][v] += 1
                        viol[new_c][v][x] = 1 if K[new_c][v][x] >= 2 else 0
                        viol[new_c][x][v] = viol[new_c][v][x]

                if cost < best_cost:
                    best_cost = cost
                    best_color = [row[:] for row in color]
                    if best_cost == 0:
                        res_colors = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                res_colors.append(color[i][j])
                        return res_colors
