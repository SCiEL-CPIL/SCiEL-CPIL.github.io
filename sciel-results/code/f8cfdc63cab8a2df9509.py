    def _tournament_select(self, rank, cds):
        i = np.random.randint(0, len(rank))
        j = np.random.randint(0, len(rank))
        if rank[i] < rank[j]:
            return i
        elif rank[j] < rank[i]:
            return j
        else:
            if cds[i] > cds[j]:
                return i
            else:
                return j
