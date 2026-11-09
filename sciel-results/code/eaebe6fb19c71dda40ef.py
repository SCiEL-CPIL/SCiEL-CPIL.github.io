            active_triples_list = list(triples)
            # ... inside loop ...
            if not active_triples_list:
                break
            
            # Pick random index
            idx = rng.randint(0, len(active_triples_list) - 1)
            t = active_triples_list[idx]
            i, j, k = t
            
            # Find victim
            max_deg = max(degree[i], degree[j], degree[k])
            candidates = [c for c in (i, j, k) if degree[c] == max_deg]
            victim = rng.choice(candidates)
            
            selected[victim] = False
            
            # Remove all triples containing victim from active_triples_list
            # And update degrees
            new_active = []
            for tri in active_triples_list:
                if victim in tri:
                    # Update degrees
                    a, b, c = tri
                    for o in (a, b, c):
                        if o != victim:
                            degree[o] -= 1
                else:
                    new_active.append(tri)
            active_triples_list = new_active
