from .Defense import Defense

class SymbolReplacementDefense(Defense):
    def __init__(self, defense, type):
        super().__init__(defense)
        assert (type in ('at', 'dot', 'at_dot'))
        self.type = type

    def apply(self, profile, ground_truth_label):
        if type(profile) == str:
            found = 0
            if ground_truth_label['email'] in profile:
                found = 1
                if 'at' in self.type:
                    profile = profile.replace('@', ' AT ')
                if 'dot' in self.type:
                    profile = profile.replace('.', ' DOT ')
            if found != 1 or ground_truth_label['email'] == 'none':
                raise ValueError("Defense is not applicable to this profile")
        return profile
