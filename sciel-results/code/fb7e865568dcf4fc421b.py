                if u < v:
                    row[2*idx] = -1.0
                    row[2*idx+1] = 1.0
                else:
                    row[2*idx+1] = -1.0
                    row[2*idx] = 1.0
