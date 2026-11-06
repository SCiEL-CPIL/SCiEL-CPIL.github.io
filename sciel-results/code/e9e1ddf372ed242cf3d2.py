                # Find first violation
                violation_idxs = None
                violation_type = None
                
                for idxs in red_cliques:
                    if colors[idxs[0]] == 0 and colors[idxs[1]] == 0 and colors[idxs[2]] == 0 and colors[idxs[3]] == 0:
                        violation_idxs = idxs
                        violation_type = 'red'
                        break
                        
                if not violation_idxs:
                    for idxs in blue_cliques:
                        if colors[idxs[0]] == 1 and colors[idxs[1]] == 1 and colors[idxs[2]] == 1 and colors[idxs[3]] == 1 and colors[idxs[4]] == 1 and colors[idxs[5]] == 1 and colors[idxs[6]] == 1 and colors[idxs[7]] == 1 and colors[idxs[8]] == 1 and colors[idxs[9]] == 1:
                            violation_idxs = idxs
                            violation_type = 'blue'
                            break

                if not violation_idxs:
                    return n, colors
