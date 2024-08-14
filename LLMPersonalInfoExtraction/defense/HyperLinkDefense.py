from .Defense import Defense

class HyperLinkDefense(Defense):
    def __init__(self, defense):
        super().__init__(defense)

    def apply(self, profile, ground_truth_label):
        if type(profile) == list:
            found = 0
            for j in range(len(profile)):
                if ground_truth_label["email"] in profile[j]:
                    found = 1
                    profile[j] = f'<li><a href="mailto:{ground_truth_label["email"]}">Contact me</a></li>'
            if found != 1 or ground_truth_label['email'] == 'none':
                raise ValueError("Defense is not applicable to this profile")
        return profile