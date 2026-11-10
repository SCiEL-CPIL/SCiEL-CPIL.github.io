                    S.add(x)
                    counts[2*x] += 1
                    for y in S:
                        if y != x:
                            counts[x+y] += 1
