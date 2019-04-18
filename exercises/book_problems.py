from data_structs.common import *
from exercises.exercise_problems import ExerciseProblem


# n-Ways to Climb a Staircase
class Book_13_1(ExerciseProblem):

    """
    Notes:
        > Dynamic Programming
    """
    def __init__(self):
        pass

    def test(self):
        pass

    def execute(self, **kwargs):
        pass


# n-Ways to Decode a String
class Book_13_2(ExerciseProblem):

    """
    Notes:
        > Dynamic Programming
    """
    def __init__(self):
        pass

    def test(self):
        print(self.execute(encoded='123456'))

    def execute(self, **kwargs):
        from collections import defaultdict

        encoded_string = kwargs['encoded']
        cache = defaultdict(int)
        cache[len(encoded_string)] = 1

        for i in reversed(range(len(encoded_string))):
            # This means that it is part of a double digit encoding
            if encoded_string[i].startswith('0'):
                cache[i] = 0
            # The end of the encoded string should give at least one encoding if not 0
            elif i == len(encoded_string) - 1:
                cache[i] = 1
            else:
                # Add possible decodings up to now
                cache[i] += cache[i + 1]
                # There is an extra possible encoding if we find that the digit is possibly double, therefore we add the
                # possible decodings up to that point as well
                if int(encoded_string[i: i + 2]) <= 26:
                    cache[i] += cache[i + 2]

        return cache[0]


#
class Book_20_1(ExerciseProblem):

    def __init__(self):
        pass

    def test(self):
        words = ['bear', 'dog', 'cat', 'calf']
        print("Result: {}".format(self.execute(words=words)))

    def execute(self, **kwargs):

        def is_winning(trie, prefix):
            root = trie.find(prefix)

            if '#' in root:
                return False
            else:
                next_moves = [prefix + letter for letter in root]
                if any(is_winning(trie, move) for move in next_moves):
                    print(next_moves)
                    return False
                else:
                    #print(next_moves)
                    return True

        def optimal_starting_letters(words):
            trie = Trie(words)
            winners = []
            start = trie._trie.keys()

            for letter in start:
                if is_winning(trie, letter):
                    winners.append(letter)

            return winners

        words = kwargs['words']
        return optimal_starting_letters(words)