class SingleLinkedListNode(object):

    def __init__(self, value=None, next_node=None):
        self.next = next_node
        self.value = value


class BSTNode(object):

    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
