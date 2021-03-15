import numpy as np
import csv

class KNN():
    def __init__(self,dataset,data_uji,k):
        ''' class constructor '''
        self.ddir = dataset
        self.uji = self.read_dataset(data_uji)
        self.uji = self.cleaning_dataset(self.uji)
        self.__k__ = k
        self.data = self.read_dataset(self.ddir)
        self.data = self.cleaning_dataset(self.data)
        self.all_euclidean = self.get_euclidean(self.data, self.uji, self.__k__)


    def read_dataset(self,ddir):
        ''' methods to read data to numpy array '''
        with open(ddir,'r') as fi:
            reader = csv.reader(fi)
            data = [item for item in reader]
            data = np.asarray(data)
            return data

    def cleaning_dataset(self, data):
        ''' methods to clean data to numpy array '''
        label = (data[:,-1] == 'good').astype(int)
        data[:,-1] = label
        return data.astype(float)

    def get_euclidean(self,data,uji,k):
        ''' methods to calculate euclidean distance'''
        uji_atr = uji[0,:-1]
        data_without_label = data[:,:-1]
        print(data_without_label)
        uji_repeat = np.tile(uji_atr, data_without_label.shape[0]).reshape((data_without_label.shape[0],data_without_label.shape[1]))
        print(uji_repeat)
        _euclidean = np.sqrt(np.sum(np.square(abs(data_without_label-uji_repeat)), axis=1))
        print(_euclidean)
        

        
            

        
            
            




