import sys
sys.path.insert(0, './../')

import unittest
from Feat_extractor import *

class Test_Feat_extractor(unittest.TestCase):

    def setUp(self):
        self.tweets_list=[('@switchfoot http://twitpic.com/2y1zl - I am good ;D',0), \
                   ('today is so good. but it started raining',4), \
                   ('@Kenichan today sucks!',4), \
                   ('final exam is good:D',0)]


    def test_process_many_uni(self):

        expected=[(['good'], 0), (['today', 'good', 'but', 'started', 'raining'], 4), (['today', 'sucks'], 4), (['final', 'exam', 'good'], 0)]
        actual=Feat_extractor(ngramType=1).process_many(self.tweets_list)           
        self.assertEqual(actual,expected)
        
    def test_process_many_bi(self):

        expected=[(['good', 'I am', 'am good', 'good D'], 0), (['today', 'good', 'but', 'started', 'raining', 'today is', 'is so', 'so good', 'good but', 'but it', 'it started', 'started raining'], 4), (['today', 'sucks', 'today sucks'], 4), (['final', 'exam', 'good', 'final exam', 'exam is', 'is good', 'good D'], 0)]
        actual=Feat_extractor(ngramType=2).process_many(self.tweets_list)
        self.assertEqual(actual,expected)

    def test_get_WordsSet(self):
        f=Feat_extractor(ngramType=2)
        tweets_words=f.process_many(self.tweets_list)
        actual= f.get_WordsSet(tweets_words)
        expected=['raining', 'so good', 'final exam', 'good D', 'today sucks', 'it started', 'exam is', 'started raining', 'is so', 'final', 'today', 'good', 'exam', 'but it', 'started', 'am good', 'but', 'is good', 'I am', 'sucks', 'today is', 'good but']
        self.assertEqual(actual,expected)
        
    def test_extractApplyFeat_many(self):
                
        f=Feat_extractor(ngramType=1)

        featFileName='wordFeat_uni_tcsv'
        featDir=""#'./../model/'+featFileName+'.pickle'

        train=f.extractApplyFeatTRAIN_many(self.tweets_list,featDir=featDir)

##        print "training_set:\n"
##        print '\n'.join(str(p) for p in train)

        expected=[({'contains(started)': False, 'contains(good)': True, 'contains(but)': False, 'contains(raining)': False, 'contains(exam)': False, 'contains(today)': False, 'contains(sucks)': False, 'contains(final)': False}, 0),               ({'contains(started)': True, 'contains(good)': True, 'contains(but)': True, 'contains(raining)': True, 'contains(exam)': False, 'contains(today)': True, 'contains(sucks)': False, 'contains(final)': False}, 4),              ({'contains(started)': False, 'contains(good)': False, 'contains(but)': False, 'contains(raining)': False, 'contains(exam)': False, 'contains(today)': True, 'contains(sucks)': True, 'contains(final)': False}, 4),                ({'contains(started)': False, 'contains(good)': True, 'contains(but)': False, 'contains(raining)': False, 'contains(exam)': True, 'contains(today)': False, 'contains(sucks)': False, 'contains(final)': True}, 0)]

##        print "expected:\n"
##        print '\n'.join(str(p) for p in expected)

        pairs = zip(train, expected)

##        for i in range(len(train)):
##            print train[i]==expected[i]

        # all dicts in the dict list are equal
        self.assertFalse(any(x != y for x, y in pairs)) 
        

 
if __name__ == '__main__':
    unittest.main()
