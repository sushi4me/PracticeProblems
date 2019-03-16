from exercises.daily_coding_problems import DailyCodingProblem
from exercises.exercise_problems import ExerciseProblems

if __name__ == "__main__":
    e = ExerciseProblems()
    dcp = DailyCodingProblem()

    print(dcp.max_subarray([9, 7, 10, 2, 3, 5, 6], 3))
