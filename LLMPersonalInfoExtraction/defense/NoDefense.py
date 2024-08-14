from .Defense import Defense

class NoDefense(Defense):
    def __init__(self, defense):
        super().__init__(defense)

    def apply(self, profile, ground_truth_label):
        return profile
