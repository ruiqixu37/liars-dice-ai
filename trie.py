class Trie:
    """Flat dict-backed store replacing the original character-by-character trie.

    Preserves the same API: insert(), search(), all_end_state().
    search() returns ``{'#': [regret, strategy_p, counter]}`` (or *None*)
    so existing call-sites like ``node['#'][0]`` keep working.
    """

    def __init__(self):
        self.data = {}  # key -> [regret, strategy_p, action_counter]

    def insert(self, word: str, strategy_p=0) -> None:
        self.data[word] = [0.0, strategy_p, 0.0]

    def search(self, word: str):
        val = self.data.get(word)
        if val is None:
            return None
        return {'#': val}

    def all_end_state(self) -> list:
        return [(k, {'#': v}) for k, v in self.data.items()]
