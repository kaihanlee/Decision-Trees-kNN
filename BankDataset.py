
# coding: utf-8

''' F20ML 2020-2021 - Utility class for Coursework 1

BankDataset loads the dataset and performs preprocessing such as feature scaling and 1-of-k mappings.

'''
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler
# get_ipython().run_line_magic('matplotlib', 'inline')


class BankDataset:
    def __init__(self, **kwargs):
        super(BankDataset, self).__init__(**kwargs)
        # This will be initialised by the load method with all the dataset features
        self.X = None
        # This will be initialised by the load method with all the dataset classes
        self.y = None       
        self.feature_names = ["age","job","marital","education","default","balance","housing",
                              "loan","contact","day","month","duration","campaign","pdays","previous","poutcome"]
        self.target_names = ["no", "yes"]

    def preprocess(self, type_, filter=[], apply_scaling=False):
        # filter is for any filtered variables that you dont want
        if type_ == "numerical":
            self.feature_encoders = [
                None,  
                # age
                preprocessing.LabelEncoder().fit(
                    ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", 
                     "self-employed", "services", "student", "technician", "unemployed", "unknown" ]),  
                # job
                preprocessing.LabelEncoder().fit(["divorced", "married",  "single"]),  
                # marital
                preprocessing.LabelEncoder().fit(["primary", "secondary", "tertiary", "unknown"]),
                # education
                preprocessing.LabelEncoder().fit(["no", "yes"]),
                # default
                None,
                # balance
                preprocessing.LabelEncoder().fit(["no", "yes"]),  
                # housing
                preprocessing.LabelEncoder().fit(["no", "yes"]),  
                # loan
                preprocessing.LabelEncoder().fit(["cellular",  "telephone", "unknown"]),  
                # contact
                None,
                # day
                preprocessing.LabelEncoder().fit([ "jan", "feb", "mar", "apr", "may", "jun", "jul","aug", "sep", "oct", "nov", "dec"]),
                # month
                None, 
                # duration
                None,
                # campaign
                None,
                # pdays
                None, 
                # previous
                preprocessing.LabelEncoder().fit([ "failure", "other", "success", "unknown"]),
                # poutcome
            ]
        elif type_ == "one-hot":
            self.feature_encoders = [
                None,  # age
                preprocessing.OneHotEncoder(sparse=False), # job
                preprocessing.OneHotEncoder(sparse=False), # marital
                preprocessing.OneHotEncoder(sparse=False), # education
                preprocessing.OneHotEncoder(sparse=False), # default
                None, # balance
                preprocessing.OneHotEncoder(sparse=False), # housing
                preprocessing.OneHotEncoder(sparse=False), # loan
                preprocessing.OneHotEncoder(sparse=False), # contact
                preprocessing.OneHotEncoder(categories='auto', sparse=False), # day
                preprocessing.OneHotEncoder(sparse=False), # month
                None, # duration
                None, # campaign
                None, # pdays
                None, # previous
                preprocessing.OneHotEncoder(sparse=False) # poutcome
            ]
        else:
            raise ValueError("Unable to load feature encoders for type {}".format(type_))

        self.class_encoder = preprocessing.LabelBinarizer().fit(["no", "yes"])

        num_features = self.X.shape[1]
        print("Number of features is {}".format(num_features))
        num_instances = self.X.shape[0]
        print("Number of instances is {}".format(num_instances))
        one_hot_applied = False
        new_features = []

        for f_id in [x for x in range(num_features) if not x in filter]:
            # convert them to integers
            if self.feature_encoders[f_id] is None:
                if type_ == "one-hot":
                    new_features.append(np.expand_dims(self.X[:, f_id].astype(np.float32), -1))
                else:
                    new_features.append(self.X[:, f_id].astype(np.float32))
            else:
                # apply the OneHotEncoder
                if isinstance(self.feature_encoders[f_id], preprocessing.OneHotEncoder):
                    one_hot_applied = True
                    new_features.append(self.feature_encoders[f_id].fit_transform(np.expand_dims(self.X[:, f_id], -1)))
                    # new_features.append(self.feature_encoders[f_id][1].fit_transform(np.expand_dims(temp, -1)))
                # # apply in sequence the preprocessors
                # if isinstance(self.feature_encoders[f_id], (list, tuple)):
                #     one_hot_applied = True
                #     temp = self.feature_encoders[f_id][0].transform(np.expand_dims(self.X[:, f_id], -1))
                #     new_features.append(self.feature_encoders[f_id][1].fit_transform(np.expand_dims(temp, -1)))
                else:
                    temp = self.feature_encoders[f_id].transform(np.expand_dims(self.X[:, f_id], -1))
                    new_features.append(np.expand_dims(temp, -1))

        if one_hot_applied or type_ == "one-hot":
            self.X = np.concatenate(new_features, -1)
            print("one-hot selected")
        else:
            self.X = np.array(self.X)
        # apply max abs scaling (useful for 1-hot representations)
        if apply_scaling:
            self.scaler = MaxAbsScaler().fit(self.X)
            self.X = self.scaler.transform(self.X)
        # print(type(self.y))
        self.y = np.array(self.class_encoder.transform(self.y))
        self.y = self.y.squeeze(-1)
        # print(type(self.y))
        
    def load(self, filename):
        """
        Loads the data from the specified file 
        """
        print("Loading bank dataset from file {}".format(filename))
        # we open the file in read mode
        with open(filename) as in_file:
            self.X = []
            self.y = []
            
            for line in in_file:
                # Reminder: each line is composed of values seperated by commas
                # e.g., 36,technician,married,tertiary,no,4596,yes,no,cellular,8,oct,234,2,175,2,success,yes
                values = line.strip().split(",")
                
                # we just make sure that we read a valid line
                if values and values[0] != '' and "?" not in values:
                    curr_X = values[:-1]
                    # we extract the class value for the current example
                    curr_y = values[-1]

                    # we store the current values by appending them to X and Y
                    self.X.append(curr_X)
                    self.y.append(curr_y)
            
            print("Dataset correctly loaded")
            self.X = np.array(self.X)
            self.y = np.array(self.y)


''' Loads dataset
 Inputs:
 filename: str corresponding to a dataset file (e.g., bank_train)
 preprocess_onehot: bool (Optional) instructs the preprocess(self) to convert categoricals to one-hot
 apply_scaling: bool (Optional) instructs the preprocess(self) to apply scaling

 Returns: dataset: BankDataset, an instance of the BankDataset class
'''
def load_dataset(filename, preprocess_onehot=False, apply_scaling=False):
    dataset = BankDataset()
    dataset.load(filename)
    if preprocess_onehot:
        dataset.preprocess("one-hot", apply_scaling=apply_scaling)
    return dataset

train_dataset = load_dataset('bank_train', preprocess_onehot=True)

