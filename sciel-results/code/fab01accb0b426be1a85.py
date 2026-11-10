            # Also, x itself needs a res_counts entry?
            # res_counts[x] should contain residues of S \ {x}.
            # Currently S contains x. S \ {x} is the old S.
            # We need to populate res_counts[x] with residues of all other elements in S.
            if x not in res_counts:
                res_counts[x] = {}
                for y in S:
                    if y == x:
                        continue
                    r = y % x
                    res_counts[x][r] = res_counts[x].get(r, 0) + 1
