            # Check validity
            valid = True
            for idx in four_ap_members[x]:
                a, b, c, e = four_aps[idx]
                if x == a:
                    if in_S[b] and in_S[c] and in_S[e]:
                        valid = False
                        break
                elif x == b:
                    if in_S[a] and in_S[c] and in_S[e]:
                        valid = False
                        break
                elif x == c:
                    if in_S[a] and in_S[b] and in_S[e]:
                        valid = False
                        break
                else: # x == e
                    if in_S[a] and in_S[b] and in_S[c]:
                        valid = False
                        break
