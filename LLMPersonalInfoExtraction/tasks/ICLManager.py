from os import walk

from ..utils import open_json, open_txt, get_parser, parsed_data_to_string


class ICLManager:
    
    def __init__(self, config):
        # Initialize the meta data of the task
        self.task = config["task_info"]["task"]
        self.task_type = config["task_info"]["type"]
        self.dataset = config["dataset_info"]["dataset"]
        self.icl_root = config["dataset_info"]["icl_path"]
        self.icl_label_path = config["dataset_info"]["icl_label_path"]
        self.__prepare_icl_eamples()

    def __len__(self):
        return len(self.icl_names)

    def __getitem__(self, idx):
        return self.icl_data[self.icl_names[idx]], self.icl_labels[self.icl_names[idx]]
    
    def __prepare_icl_eamples(self):
        """
        Prepare the HTML profiles and the labels for the ICL data
        """
        # Post-process
        icl_filenames = sorted(next(walk(self.icl_root), (None, None, []))[2])
        icl_filenames = [f for f in icl_filenames if '.html' in f]
        self.icl_labels = open_json(self.icl_label_path)
        self.icl_data = {}
        self.icl_names = []
        for i, l in enumerate(icl_filenames):
            name = icl_filenames[i].replace('.html', '')
            self.icl_names.append(name)

            raw_list = open_txt(f'{self.icl_root}/{l}')
            raw = '\n'.join(raw_list)

            parser = get_parser(self.dataset, include_link=False)
            parser.feed(raw)
            parsed_data = parsed_data_to_string(self.dataset, parser.data)
            self.icl_data[name] = parsed_data.replace('href\n#\n', '')