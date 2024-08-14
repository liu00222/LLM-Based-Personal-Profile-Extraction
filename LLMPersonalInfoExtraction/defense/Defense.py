class Defense:
    def __init__(self, defense):
        self.defense = defense

    def apply(self, profile, ground_truth_label):
        raise NotImplementedError("ERROR: Interface doesn't have the implementation for apply")
