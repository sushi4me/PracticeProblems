class SingleLinkedListNode(object):

    def __init__(self, value=None, next_node=None):
        self.next = next_node
        self.value = value


class TreeNode(object):

    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


class BSTNode(TreeNode):

    def __init__(self, value=None, left=None, right=None):
        super.__init__(value, left, right)


class Trie(object):

    def __init__(self, words):
        self._trie = {}
        for word in words:
            self.insert(word)

    def insert(self, word):
        trie = self._trie

        for char in word:
            if char not in trie:
                trie[char] = {}
            trie = trie[char]
        trie['#'] = True

    def find(self, word):
        trie = self._trie

        for char in word:
            if char in trie:
                trie = trie[char]
            else:
                return None

        return trie
