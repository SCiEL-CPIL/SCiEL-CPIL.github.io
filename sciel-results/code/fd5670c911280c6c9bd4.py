    def is_valid(code_set, cover_count):
        for i in range(total_words):
            if cover_count[i] == 0:
                return False
        return True
