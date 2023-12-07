import yaml

class DataReader:

    def __init__(self, input_file_name: str):
        self.input_file_name = input_file_name
        self.dft_dict = {}
        self.exploration_dict = {}
        self.manager_dict = {}
        self.pacemaker_dict = {}
        self.dft_dict, self.exploration_dict, self.manager_dict, self.pacemaker_dict = self.read_input(self.input_file_name)
        self.directory_dict = {}
        self.dft_connection = None
        self.train_connection = None
        self.exploration_connection = None


    def read_input(self, input_file_name: str):
        '''
        Reads the master input file and returns the sections in seperate dictonaries
        '''

        with open (input_file_name) as f:
            input_data = yaml.safe_load(f)

            manager_dict = input_data['manager']
            pacemaker_dict = input_data['pacemaker']
            exploration_dict = input_data['exploration']
            dft_dict = input_data['dft']

        return dft_dict, exploration_dict, manager_dict, pacemaker_dict

    def change_data(self, dict: str, key: str, value: any):
        '''
        Change/add the value of a key/value pair in one of the dictonaries
        '''

        if dict == 'dft_dict':
            self.dft_dict[key] = value
        elif dict == 'exploration_dict':
            self.exploration_dict[key] = value
        elif dict == 'manager_dict':
            self.manager_dict[key] = value
        elif dict == 'pacemaker_dict':
            self.pacemaker_dict[key] = value

        return dict
    
    def restore_input(self, dict: str, key: str):
        '''
        Restore an input parameter in one of the dictonaries 
        from the input file by provinding its key 
        '''

        with open (self.input_file_name) as f:
            input_data = yaml.safe_load(f)

        if dict == 'dft_dict':
            self.dft_dict[key] = input_data['dft'][key]
        elif dict == 'exploration_dict':
            self.dft_dict[key] = input_data['exploration'][key]
        elif dict == 'manager_dict':
            self.dft_dict[key] = input_data['manager'][key]
        elif dict == 'pacemaker_dict':
            self.dft_dict[key] = input_data['pacemaker'][key]