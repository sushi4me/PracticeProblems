from exercises.exceptions import NotImplementedException
from exercises.exercise_problems import ExerciseProblem


class DailyCodingProblem(ExerciseProblem):

    def __init__(self):
        pass

    def test(self):
        self.execute()

    def execute(self):
        raise NotImplementedException


class Problem(DailyCodingProblem):

    def __init__(self):
        pass

    def test(self, **kwargs):
        self.execute()

    def execute(self, **kwargs):
        pass
