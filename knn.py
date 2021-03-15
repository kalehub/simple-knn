import numpy as np
import csv

class KNN():
    def __init__(self,dataset,data_uji,k):
        self.ddir = dataset
        self.uji = self.read_dataset(data_uji)
        self.__k__ = k
        self.data = self.read_dataset(self.ddir)
        self.data = self.cleaning_dataset(self.data)


    def read_dataset(self,ddir):
        with open(ddir,'r') as fi:
            reader = csv.reader(fi)
            data = [item for item in reader]
            data = np.asarray(data)
            return data

    def cleaning_dataset(self, data):
        label = (data[:,-1] == 'good').astype(int)
        data[:,-1] = label
        return data.astype(float)
        
            

        
            
            




