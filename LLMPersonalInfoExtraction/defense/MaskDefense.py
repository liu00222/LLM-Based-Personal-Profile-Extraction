from .Defense import Defense

class MaskDefense(Defense):
    def __init__(self, defense):
        super().__init__(defense)

    def apply(self, profile, ground_truth_label):
        if type(profile) == list:
            name = ground_truth_label['name']
            name_split = name.split(' ')

            if len(name_split) != 2:
                raise ValueError("Defense is not applicable to this profile")
            
            first_name = name_split[0].lower()
            last_name = name_split[1].lower()
            found = 0

            for j in range(len(profile)):
                if ground_truth_label["email"] in profile[j] and (first_name in profile[j] and last_name in profile[j]):
                    profile[j] = profile[j].replace(first_name, ' { first name } ').replace(last_name, ' { last name } ')
                    found = 1

            if found != 1 or ground_truth_label['email'] == 'none':
                raise ValueError("Defense is not applicable to this profile")
        return profile