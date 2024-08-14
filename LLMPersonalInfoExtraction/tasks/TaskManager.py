from os import walk

from ..utils import open_json, open_txt


class TaskManager:
    
    def __init__(self, config):
        # Initialize the meta data of the task
        self.task = config["task_info"]["task"]
        self.task_type = config["task_info"]["type"]
        self.dataset = config["dataset_info"]["dataset"]

        self.root = config["dataset_info"]["path"]
        self.label_path = config["dataset_info"]["label_path"]

        self.__prepare_data()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Each element in current_raw_profile_in_list is a line in the raw HTML source code of a 
        # personal profile, e.g., can be obtained by pressing F12 in any browser.
        current_raw_profile_in_list = open_txt(f'{self.root}/{self.filenames[idx]}')

        # Obtain the GT label of current profile
        name = self.filenames[idx].replace('.html', '')
        current_label = self.labels[name]

        return current_raw_profile_in_list, current_label
    
    def __prepare_data(self):
        """
        Prepare the HTML profiles and the labels
        """
        # Only load the filenames and will load the actual profile at run time.
        self.filenames = sorted(next(walk(self.root), (None, None, []))[2])
        self.filenames = [f for f in self.filenames if '.html' in f]
        self.labels = open_json(self.label_path)