from data_structs.common import SingleLinkedListNode
from exercises.exceptions import NotImplementedException
from exercises.exercise_problems import ExerciseProblem


class DailyCodingProblem(ExerciseProblem):

    def __init__(self, diff=None):
        self.difficulty = diff

    def test(self):
        self.execute()

    def execute(self):
        raise NotImplementedException


class Problem26(DailyCodingProblem):

    def __init__(self):
        self.difficulty = 'Medium'

    def test(self):
        s = SingleLinkedListNode(0)
        head = s
        s_next = SingleLinkedListNode(0)

        for i in range(1, 10):
            s_next.value = i
            s.next = s_next
            s = s.next

        self.execute(head_of_list=head, last=2)

    def execute(self, **kwargs):
        fast = slow = kwargs['head_of_list']
        n_from_last = kwargs['last']

        for i in range(n_from_last):
            fast = fast.next

        prev = None
        while fast is not None:
            prev = slow
            fast = fast.next
            slow = slow.next

        # Fast will eventually hit a None condition exiting the loop and slow will be on the Nth from last
        prev.next = slow.next


class Problem27(DailyCodingProblem):

    def __init__(self):
        self.brackets = {')': '(', ']': '[', '}': '{'}

    def test(self):
        bracket_string = '([{}{}])'

        print(self.execute(bracket_string=bracket_string))

    def execute(self, **kwargs):
        bracket_string = kwargs['bracket_string']
        stack = []

        for char in bracket_string:
            if char in self.brackets:
                if self.brackets[char] != stack.pop():
                    return False
            else:
                stack.append(char)

        return True


class Problem28(DailyCodingProblem):

    def __init__(self):
        pass

    def test(self):
        list_of_words = ['hello', 'there', 'bye']
        k = 10

        print("Each line should be less than or equal to {} characters.".format(k))
        print(self.execute(list_of_words=list_of_words, k=k))

    def execute(self, **kwargs):
        list_of_words = kwargs['list_of_words']
        k = kwargs['k']
        sorted_words = []
        current_list = []

        # Find the character count in each word
        for word in list_of_words:
            # length of all words + spaces + current word < k
            if len(''.join(current_list)) + len(current_list) - 1 + len(word) < k:
                current_list.append(word)
            # Maxed out, move to the next line
            else:
                sorted_words.append(current_list)
                current_list = [word]

        # Append the last one
        sorted_words.append(current_list)

        # Return a list which has the justified strings
        current_list = []
        for lst in sorted_words:
            current_list.append(' '.join(lst))

        return current_list


class Problem29(DailyCodingProblem):

    def __init__(self):
        pass

    def test(self):
        raw_string = 'AAABBBCCDA'

        encoded = self.execute(func='encode', raw_string=raw_string)
        decoded = self.execute(func='decode', encoded_string=encoded)

        print("{0} --ENCODING--> {1}".format(raw_string, encoded))
        print("{0} --DECODING--> {1}".format(encoded, decoded))

    def execute(self, **kwargs):

        def encode(keyword_args):
            raw_string = keyword_args['raw_string']
            encoded_string = ""
            current = (None, 0)

            for char in raw_string:
                if current[1] == 0:
                    current = (char, 1)
                elif current[0] == char:
                    current = (char, current[1] + 1)
                else:
                    encoded_string += str(current[1]) + current[0]
                    current = (char, 1)
            # Print the last element
            encoded_string += str(current[1]) + current[0]

            return encoded_string

        def decode(keyword_args):
            encoded_string = keyword_args['encoded_string']
            decoded_string = ""
            current_count = 0

            for char in encoded_string:
                if char.isdigit():
                    current_count = current_count * 10 + int(char)
                else:
                    decoded_string += char * current_count
                    current_count = 0

            return decoded_string

        funcs = {'encode': encode, 'decode': decode}
        return funcs[kwargs['func']](kwargs)


class Problem30(DailyCodingProblem):
    """
    Notes: The original algorithm was missing cases in which your wall does not hold any water -- all the water runs off
    the edges.  See example 1.
    """

    def __init__(self):
        self.difficulty = 'Medium'

    def test(self):
        print("[1, 2, 3, 4, 3, 2, 1]={}".format(self.execute(wall=[1, 2, 3, 4, 3, 2, 1])))
        print("[1, 1, 1, 1, 1, 1, 1]={}".format(self.execute(wall=[1, 1, 1, 1, 1, 1, 1])))
        print("[1, 0, 1, 0, 1, 0, 1]={}".format(self.execute(wall=[1, 0, 1, 0, 1, 0, 1])))
        print("[3, 0, 4, 3, 5, 2, 1]={}".format(self.execute(wall=[3, 0, 4, 3, 5, 2, 1])))

    def execute(self, **kwargs):
        wall = kwargs['wall']
        total = 0

        edge = wall[0]
        # This will give you the max's index
        max_index = wall.index(max(wall))

        for value in wall[1:max_index]:
            if edge > value:
                total += edge - value
            edge = max(edge, value)

        edge = wall[-1]
        for value in wall[-2:max_index:-1]:
            if edge > value:
                total += edge - value
            edge = max(edge, value)

        return total


class Problem31(DailyCodingProblem):

    def __init__(self):
        self.difficulty = 'Easy'

    def test(self):
        print("cat, at={}".format(self.execute(s1='cat', s2='at')))

    def execute(self, **kwargs):
        string1 = kwargs['s1']
        string2 = kwargs['s2']

        x, y = len(string1) + 1, len(string2) + 1

        # Checking for NULL case in which we just need to insert N characters
        if len(string1) == 0 or len(string2) == 0:
            return max(len(string1), len(string2))

        edit = [[-1 for _ in range(x)] for _ in range(y)]

        # To get from NULL to N characters in another string we insert N times
        for i in range(x):
            edit[0][i] = i
        for j in range(y):
            edit[j][0] = j

        # Traverses the rows
        for j in range(1, y):
            # Traverses the columns
            for i in range(1, x):
                # If the characters match then we only need to worry about the string edit up to this point
                if string1[i - 1] == string2[j - 1]:
                    edit[j][i] = edit[j - 1][i - 1]
                # If the characters do not match then we need to find the minimum number of edits up to here plus 1
                else:
                    print(edit)
                    edit[j][i] = min(edit[j - 1][i - 1] + 1,
                                     edit[j - 1][i] + 1,
                                     edit[j][i - 1] + 1)
        return edit[-1][-1]


class Problem32(DailyCodingProblem):

    def __init__(self):
        pass

    def test(self):
        pass

    def execute(self):
        pass
