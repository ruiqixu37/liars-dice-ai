from collections import defaultdict


class Trie:

    def __init__(self):
        self.trie = defaultdict(dict)

    def insert(self, word: str, strategy_p=0) -> None:
        """
        Inserts the string word into the trie.
        Store the values in the leaf node as "#" to indicate the end of the word.

        :param word: string, the word to be inserted
        :param strategy_p: float, the probability of the strategy for the word
        """
        curr = self.trie
        for w in word:
            if w not in curr:
                curr[w] = {}
            curr = curr[w]
        curr["#"] = [0.0, strategy_p, 0.0]

    def _dfs_search(self, word: str, trie) -> any:
        if not word:
            return None if '#' not in trie else trie
        if word[0] not in trie:
            return None
        return self._dfs_search(word[1:], trie[word[0]])

    def search(self, word: str) -> any:
        """
        Returns the node in the trie if the string word is in the trie (i.e., was inserted before),
        and None otherwise.
        """
        return self._dfs_search(word, self.trie)

    def startsWith(self, prefix: str) -> bool:
        """
        Returns true if there is a previously inserted string word
        that has the prefix prefix, and false otherwise.
        """
        self.curr = self.trie
        for w in prefix:
            if w not in self.curr:
                return False
            self.curr = self.curr[w]
        return True

    def _dfs_all_end(self, trie, path: str) -> list:
        if not trie:
            return []
        if "#" in trie:
            curr = [[path, trie]]
            for key in trie:
                if key != '#':
                    curr += self._dfs_all_end(trie[key], path + key)
            return curr
        res = []
        for key in trie:
            res += self._dfs_all_end(trie[key], path + key)
        return res

    def all_end_state(self) -> list:
        """
        Returns all the end states in the trie as well as the path to the end state.
        """
        return self._dfs_all_end(self.trie, '')