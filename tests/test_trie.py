from trie import Trie


def test_trie_insert_and_search():
    """Insert a key and verify [regret, strategy_p, counter] values."""
    trie = Trie()
    trie.insert("abc", strategy_p=0.5)
    node = trie.search("abc")
    assert node is not None
    assert node['#'] == [0.0, 0.5, 0.0]


def test_trie_search_nonexistent():
    """Returns None for missing keys."""
    trie = Trie()
    trie.insert("abc")
    assert trie.search("abd") is None
    assert trie.search("abcd") is None
    assert trie.search("ab") is None
    assert trie.search("xyz") is None


def test_trie_mutation_persists():
    """Mutate a searched node, re-search, verify mutation is visible."""
    trie = Trie()
    trie.insert("abc", strategy_p=0.0)
    node = trie.search("abc")
    node['#'][0] = 5.0  # set regret
    node['#'][1] = 0.75  # set strategy
    node['#'][2] = 3.0  # set counter

    # Re-search and verify mutations persisted
    node2 = trie.search("abc")
    assert node2['#'][0] == 5.0
    assert node2['#'][1] == 0.75
    assert node2['#'][2] == 3.0


def test_trie_all_end_state_format():
    """Verify all_end_state returns [[path, node], ...] with '#' key."""
    trie = Trie()
    trie.insert("abc", strategy_p=0.1)
    trie.insert("abd", strategy_p=0.2)
    trie.insert("xyz", strategy_p=0.3)

    results = trie.all_end_state()
    assert len(results) == 3

    # Each result should be [path, node] where node has '#' key
    paths = set()
    for path, node in results:
        paths.add(path)
        assert '#' in node
        assert len(node['#']) == 3

    assert paths == {"abc", "abd", "xyz"}


def test_trie_all_end_state_mutation():
    """Verify that mutating nodes from all_end_state updates the trie."""
    trie = Trie()
    trie.insert("abc", strategy_p=0.5)

    for path, node in trie.all_end_state():
        node['#'][0] = 99.0

    # Verify mutation visible through search
    assert trie.search("abc")['#'][0] == 99.0


def test_trie_insert_overwrites():
    """Inserting the same key again overwrites the values."""
    trie = Trie()
    trie.insert("abc", strategy_p=0.5)
    trie.insert("abc", strategy_p=0.9)
    node = trie.search("abc")
    assert node['#'][1] == 0.9
    assert node['#'][0] == 0.0  # regret reset to 0


def test_trie_negative_chars_in_key():
    """Trie can handle '-' characters (used in challenge encoding (-1,-1))."""
    trie = Trie()
    key = "123451-1-1"
    trie.insert(key, strategy_p=0.5)
    node = trie.search(key)
    assert node is not None
    assert node['#'][1] == 0.5
