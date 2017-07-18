import sys
sys.path.insert(0, './../')

import unittest
from Util_data import *



class Test_Util_data(unittest.TestCase):

    def setUp(self):
        self.data=[({'contains(one)': True, 'contains(of)': True},4),\
          ({'contains(two)': True, 'contains(you)': True},4),\
          ({'contains(three)': True, 'contains(can)': True},4),\
          ({ 'contains(four)': True, 'contains(?)': True},4),\
          ({ 'contains(five)': True, 'contains(?)': True},4),\
          ({ 'contains(six)': True, 'contains(?)': True},0),\
          ({ 'contains(seven)': True, 'contains(?)': True},0),\
          ({ 'contains(eight)': True, 'contains(?)': True},0),\
          ({ 'contains(nine)': True, 'contains(?)': True},0),\
          ({ 'contains(ten)': True, 'contains(?)': True},0)]

        self.dataLen=len(self.data)

    def test_split(self):
        train, test,y_test,X_test = split(self.data, train_size=0.8)#, random_state=42)

        self.assertEqual(len(train),int(0.8*self.dataLen))
        self.assertEqual(len(test),int(0.2*self.dataLen))
        self.assertEqual(len(y_test),int(0.2*self.dataLen))
        self.assertEqual(len(X_test),int(0.2*self.dataLen))
      

    def test_subSample(self):
        subSize=0.8
        tweets_list=subSample(self.data,subSize=subSize)

        pos_tweets=filter(lambda x: x[1] == 4, tweets_list)
        neg_tweets=filter(lambda x: x[1] == 0, tweets_list)

        labelSize=self.dataLen*subSize*0.5
        
        self.assertEqual(len(pos_tweets),labelSize)
        self.assertEqual(len(neg_tweets),labelSize)
        

if __name__ == '__main__':
    unittest.main()
