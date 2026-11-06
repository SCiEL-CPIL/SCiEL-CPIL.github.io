def _try_local_improvement(S, evens, triples_by_elem, deadline):
    S_list = list(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to find an odd o to remove
        odds_in_S = [x for x in S if x % 2 == 1]
        random.shuffle(odds_in_S)
        
        for o in odds_in_S:
            if time.time() > deadline:
                break
            
            # Find evens that are NOT in S but would be valid if o was removed
            # i.e., evens x such that the only conflict with S is involving o
            candidates_to_add = []
            for x in evens:
                if x in S:
                    continue
                # Check if x is valid in S \ {o}
                # This is equivalent to: _is_valid_addition(S - {o}, x)
                # But we don't want to create new conflicts with other elements.
                # So we check if x forms a triple with any y in S\{o}.
                
                is_valid_without_o = True
                for t in triples_by_elem[x]:
                    # Check other two elements
                    others = [z for z in t if z != x]
                    # If both others are in S, and at least one is o, then removing o breaks this conflict.
                    # If both others are in S, and neither is o, then conflict remains.
                    # If one other is in S and one is not, no conflict.
                    # If both others not in S, no conflict.
                    
                    # Simpler: Just check _is_valid_addition on a temporary set? Too slow.
                    # Let's check conflicts explicitly.
                    
                    conflict_found = False
                    for z in others:
                        if z in S:
                            # If z is o, this specific triple is resolved by removing o.
                            # If z is not o, this triple is NOT resolved.
                            if z != o:
                                # We need to check if the *other* element of this triple is also in S.
                                # t is (a,b,c). x is one of them.
                                # The triple is formed if x and the other two are in S.
                                # We are checking if x can be added to S\{o}.
                                # So we need to know if the other two are in S\{o}.
                                pass 
                    # This logic is getting messy. Let's use a helper.
                
                # Helper: Check if x is valid in S given that o is removed.
                # This means: No triple (x, y, z) with y,z in S\{o}.
                
                valid = True
                for t in triples_by_elem[x]:
                    others = [z for z in t if z != x]
                    y, z = others[0], others[1]
                    # Are y and z both in S\{o}?
                    y_in = y in S and y != o
                    z_in = z in S and z != o
                    if y_in and z_in:
                        valid = False
                        break
                
                if valid:
                    candidates_to_add.append(x)
            
            if len(candidates_to_add) >= 2:
                # Try to add the first two
                # We must ensure that adding both doesn't create a conflict between them and the rest
                # Since they were valid individually w.r.t S\{o}, we need to check:
                # 1. Do they conflict with each other? (Triple with o? No o is removed. Triple with each other and some y in S\{o}?)
                #    If x1, x2, y form a triple, and y is in S\{o}, then we can't add both.
                #    But we checked x1 valid in S\{o} and x2 valid in S\{o}.
                #    This check ensures no triple (x1, y, z) with y,z in S\{o}.
                #    It does NOT check triple (x1, x2, y).
                
                # So we must check if x1 and x2 conflict with each other given S\{o}.
                
                x1, x2 = candidates_to_add[0], candidates_to_add[1]
                
                # Check if x1, x2 form a triple with any y in S\{o}
                conflict_between = False
                for t in triples_by_elem[x1]:
                    if x2 in t:
                        # t is (x1, x2, y)
                        y = [z for z in t if z != x1 and z != x2][0]
                        if y in S and y != o:
                            conflict_between = True
                            break
                        # If y == o, it's removed, so no conflict.
                        # If y not in S, no conflict.
                
                if not conflict_between:
                    # Perform the swap
                    S.remove(o)
                    S.add(x1)
                    S.add(x2)
                    improved = True
                    break # Restart loop with new S

    return S
