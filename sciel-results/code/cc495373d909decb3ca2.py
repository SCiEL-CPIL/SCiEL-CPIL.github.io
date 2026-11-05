    def row_to_bits(row):
        bits = 0
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                bits |= (1 << d)
        return bits
    