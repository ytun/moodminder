import unittest

import TweetProcess
import TweetClassify
import TweetRecommend
import TweetView

class TestCases(unittest.TestCase):
	# TweetProcess tests
	def test_TweetProcess(self):
			view = TweetView.PrintView()
			tp = TweetProcess.TweetProcess('realDonaldTrump', TweetClassify.NaiveBayesClassifier(), view.updateView)
			tp.tweet_process(0)
			view_out = view.getView()
			self.assertEqual(view_out[0]['name'], 'realDonaldTrump')

	# TweetClassify tests

	# TweetRecommend tests
	def test_tweet_filter(self):
		tweet_NN = TweetRecommend.tweet_filter('I like food')
		self.assertEqual(len(tweet_NN), 1)
		self.assertEqual(tweet_NN[0], 'food')

	def test_get_category(self):
		sim_value, c = TweetRecommend.get_category(['food'], ['food'])
		self.assertEqual(c, 'food')

	# TweetView tests
	def test_updateView(self):
		view = TweetView.PrintView()
		view.updateView('test')
		self.assertEqual(view.view_out[0], 'test')

	def test_getView(self):
		view = TweetView.PrintView()
		view.updateView('test')
		view_out = view.getView()
		self.assertEqual(view_out[0], 'test')

if __name__ == '__main__':
		unittest.main()