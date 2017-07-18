# views.py 

from django.shortcuts import render, HttpResponse, render_to_response
import requests
import sys, os, time
import TweetView
import tweepy, json, TweetProcess, TweetClassify, TweetView
import scipy
import threading

import pickle

##diry = os.path.dirname(__file__)
##diry = os.path.join(diry, '/code/model/')
##print "DIR: ",diry
##sys.path.insert(0, diry)

#cur_dir = os.getcwd()
#sys.path.append(cur_dir+"/app/code")

# Create your views here.

# Global data structure
view_past = TweetView.PrintView()
view_stream = TweetView.PrintView()

def index(request):
    return HttpResponse('Hello World!')

def test(request):
    return HttpResponse('My second view!')

def profile_old(request):
	import tweepy, json
	text = " * "
	parsedData = []
	if request.method == 'POST':
		handle = request.POST.get('user')
		access_token = "716775869046513664-5A4VLHy6O2AlxgPYnnDqK68hUgovsea"
		access_token_secret = "9imA5sFl6koFoweWeDbMzti0EwQs3b2gb5JszsAic4DSB"
		consumer_key = "XTb6wIyh9M2ZXaeNgIQcZ52JX"
		consumer_secret = "6XLYcm4YT2aAqxEpd9Ym4AZaW12rBRNsoXHNRwvLW3g1hG0L5B"

		auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
		auth.set_access_token(access_token, access_token_secret)

		api = tweepy.API(auth)
		tweets = api.user_timeline(screen_name = handle, count = 1, include_rts = True)
		
		userData = {}
		userData['name'] = handle
		userData['tweets'] = tweets[0].text
		userData['sentiment'] = ":)"
		parsedData.append(userData)
		print parsedData
	return render(request, 'app/profile.html', {'data': parsedData})
	
def wait_page(request, twitter_handle):
	#print 'HERE>>>'
	tp = TweetProcess.TweetProcess(twitter_handle, get_classifier(), view_stream.updateView)
#	tp = TweetProcess.TweetProcess(twitter_handle, TweetClassify.TweetClassify(ngramType=1), view_stream.updateView)
	tp.tweet_process(1) # 0=past tweet, 1=stream
	return True

# view for past tweets
def profile_stream(request):
	if request.method == 'POST':
		twitter_handle = request.POST.get('user')
		t = threading.Thread(target=wait_page, args=(request, twitter_handle,))
		t.start()
		return render(request, 'app/profile_stream.html', {'data': view_stream.getView()})
	elif request.method == 'GET':
		return render(request, 'app/profile_stream.html', {'data': view_stream.getView()})	
	else:
		return render(request, 'app/profile_stream.html', {'data': view_stream.getView()})

# view for streaming tweets		
def profile_past(request):
	if request.method == 'POST':
		twitter_handle = request.POST.get('user')
		tp = TweetProcess.TweetProcess(twitter_handle, get_classifier(), view_past.updateView)
		#tp = TweetProcess.TweetProcess(twitter_handle, TweetClassify.TweetClassify(ngramType=1), view_past.updateView)
		tp.tweet_process(0) # 0=past tweet, 1=stream
		return render(request, 'app/profile_past.html', {'data': view_past.getView()})
	elif request.method == 'GET':
		return render(request, 'app/profile_past.html', {'data': view_past.getView()})	
	else:
		return render(request, 'app/profile_past.html', {'data': view_past.getView()})	
		
def get_classifier():
        
        diry='app/code/model/'
        featFileName='bi_feature_old.pickle'
        classifierFileName='bi_NB_classifier_old.pickle'


        train_feat= pickle.load(open(diry+featFileName, 'rb'))
        classifier= pickle.load(open(diry+classifierFileName, 'rb'))
##        train_feat= pickle.load(open(featFileName, 'rb'))
##        classifier= pickle.load(open(classifierFileName, 'rb'))
        
        #print "classifier: ", classifier
        cl=TweetClassify.TweetClassify(classifier,ngramType=2,train_feat=train_feat)

        return cl
