                        # Find d
                        # Intersect coverers of all u in U_total
                        # U_total = U1 + U2
                        if not U1 and not U2:
                            pass # handled
                        else:
                            # Start with coverers of the first u
                            # Sort U by len(coverers[u]) to minimize intersection size
                            U_total = U1 + U2
                            U_total.sort(key=lambda x: len(coverers[x]))
                            
                            candidates = set(coverers[U_total[0]])
                            for u in U_total[1:]:
                                candidates &= set(coverers[u])
                                if not candidates:
                                    break
                            
                            # Remove words already in code
                            candidates -= code_set
                            
                            if candidates:
                                found_d = next(iter(candidates))
                                # ... perform move
