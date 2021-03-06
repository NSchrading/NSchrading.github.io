---
layout: page
title: Research Data
permalink: data/
---

<style media="screen" type="text/css" media="only screen and (min-width: 900px)">

.nav > li > a:hover, .nav > li > a:focus {
    background-color: #ECF0F1;
}

.navbar-nav > li > a {
    line-height: 21px;
    padding-top: 19.5px;
    padding-bottom: 19.5px;
}

.nav > li > a {
    position: relative;
    display: block;
    padding: 10px 15px;
}

.navbar-nav {
    margin: 0px;
}

.nav > li {
    position: relative;
    display: block;
    float: left;
}

.navbar-nav.navbar-right:last-child {
    margin-right: -15px;
}

.navbar-right {
    float: right !important;
}

a:focus {
    outline: thin dotted;
    outline-offset: -2px;
}

a {
    background: transparent none repeat scroll 0% 0%;
    color: #2A7EAD;
    text-decoration: none;
}

.container {
    width: 900px;
    padding-left: 0px;
    padding-right: 0px;
    padding-bottom: 40px;
    margin-right: auto;
    margin-left: auto;
}

</style>

<style media="screen" type="text/css" media="only screen and (max-width: 899px)">

.nav > li > a:hover, .nav > li > a:focus {
    background-color: #ECF0F1;
}

.navbar-nav > li > a {
    line-height: 21px;
    padding-top: 19.5px;
    padding-bottom: 19.5px;
}

.nav > li > a {
    position: relative;
    display: block;
    padding: 10px 5px;
}

.navbar-nav {
    margin: 0px;
}

.nav > li {
    position: relative;
    display: block;
    float: left;
}

a:focus {
    outline: thin dotted;
    outline-offset: -2px;
}

a {
    background: transparent none repeat scroll 0% 0%;
    color: #2A7EAD;
    text-decoration: none;
}

.container {
    width: 100%;
    padding-left: 0px;
    padding-right: 0px;
    padding-bottom: 120px;
    margin-right: auto;
    margin-left: auto;
}

</style>

<div class="container">
    <!-- Collect the nav links, forms, and other content for toggling -->
    <div>
        <ul class="nav navbar-nav navbar-right">
            <li>
                <a class="scroll" data-speed="500" href="#twitter">#WhyIStayed / #WhyILeft</a>
            </li>
            <li>
                <a class="scroll" data-speed="500" href="#reddit">Reddit Domestic Abuse</a>
            </li>
            <li>
                <a class="scroll" data-speed="500" href="#terms">Terms of Use</a>
            </li>
        </ul>
    </div>
</div>

This page contains links to data I have collected for my thesis on [Analyzing Domestic Abuse using Natural Language Processing on Social Media Data]({{ site.baseurl }}/project/thesis/). You are welcome to use any of this data for your research, but please cite the relevant paper and follow the [terms of use](#terms) if you do so. 

Please read the papers first before contacting me with questions.

<section id="twitter"></section>

# \#WhyIStayed / \#WhyILeft Research Data

Twitter users unequivocally reacted to the Ray Rice assault scandal by unleashing personal stories of domestic abuse via the hashtags \#WhyIStayed or \#WhyILeft. In [Schrading et al. (2015a)]({{ site.baseurl }}/project/WhyIStayed-WhyILeft/) we explored at a macro-level firsthand accounts of domestic abuse from a corpus of tweeted instances designated with these tags to seek insights into the reasons victims give for staying in vs. leaving abusive relationships.

### Download

Download the extended dataset [here](https://www.dropbox.com/s/n4fp4sq572422ex/stayedLeftData.json.7z?dl=0){:target="_blank"}.

### Format
Twitter restricts public sharing of tweets to only tweet ids. Accordingly, the format of the data is as follows:

* An individual instance is on a new line of the file.
* Each instance is encoded as JSON.
* Some tweets were split automatically with regexes if they contained both hashtags - one part for the \#WhyIStayed reason and one part for the \#WhyILeft reason. Only a single tweet in the dataset needed to be split into 4s, but logic handled this case as well.
* Each instance has the properties ***id***, ***label***, ***startIdx***, and ***endIdx***
    * ***id***: The unique identifier of the tweet.
    * ***label***: The gold standard label this instance was given, either \#WhyIStayed or \#WhyILeft, based off of the hashtag used.
    * ***startIdx***: If the tweet was split into multiple parts, this is the starting index into the tweet text for the current label. If the tweet was not split this is null.
    * ***endIdx***: If the tweet was split into multiple parts, this is the index into the tweet text at which the instance ends. If the tweet was not split this is null.

### Getting the tweet text and additional information using Twitter's API

In order to gather the tweet text and additional information about the tweet, like user information and retweet count, you must use Twitter's API. I would recommend using the [GET statuses/lookup](https://dev.twitter.com/rest/reference/get/statuses/lookup){:target="_blank"} API call because it allows you to look up 100 tweet ids at a time in bulk. Provided below is Python code to do this, using [Twython](https://github.com/ryanmcgrath/twython){:target="_blank"} as a wrapper for Twitter's API.

#### Important
To use this code, you ***must*** have Twython installed. Installation instructions, and how to set up a Twitter API account are on [Twython's githup page](https://github.com/ryanmcgrath/twython){:target="_blank"}.

If you use this code, the text provided (Tweet.text) will still have hashtags, and is uncleaned and unprocessed.

{% highlight python %}

# collects data from the publicly released data file
import json
from twython import Twython

# enter your APP_KEY and ACCESS_TOKEN from your Twitter API account here
twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)

class Tweet():
    # A container class for tweet information
    def __init__(self, json, text, label, startIdx, endIdx, idStr):
        self.json = json
        self.text = text
        self.label = label
        self.id = idStr
        self.startIdx = startIdx
        self.endIdx = endIdx

    def __str__(self):
        return "id: " + self.id + " " + self.label + ": " + self.text

def collectTwitterData(twitter):
    tweetDict = {}
    # open the shared file and extract its data for all tweet instances
    with open("stayedLeftData.json") as f:
        for line in f:
            data = json.loads(line)
            label = data['label']
            startIdx = data['startIdx']
            endIdx = data['endIdx']
            idStr = data['id']
            tweet = Tweet(None, None, label, startIdx, endIdx, idStr)
            tweetDict[idStr] = tweet

    # download the tweets JSON to get the text and additional info
    i = 0
    chunk = []
    for tweetId in tweetDict:
        # gather up 100 ids and then call Twitter's API
        chunk.append(tweetId)
        i += 1
        if i >= 100:
            print("dumping 100...")
            # Make the API call
            results = twitter.lookup_status(id=chunk)
            for tweetJSON in results:
                idStr = tweetJSON['id_str']
                tweet = tweetDict[idStr]
                tweet.json = tweetJSON
                # If this tweet was split, get the right part of the text
                if tweet.startIdx is not None:
                    tweet.text = tweetJSON['text'][tweet.startIdx : tweet.endIdx]
                # Otherwise get all the text
                else:
                    tweet.text = tweetJSON['text']
            i = 0
            chunk = []
    # get the rest (< 100 tweets)
    print("dumping rest...")
    results = twitter.lookup_status(id=chunk)
    for tweetJSON in results:
        idStr = tweetJSON['id_str']
        tweet = tweetDict[idStr]
        tweet.json = tweetJSON
        if tweet.startIdx is not None:
            tweet.text = tweetJSON['text'][tweet.startIdx : tweet.endIdx]
        else:
            tweet.text = tweetJSON['text']

    # return the Tweet objects in a list
    return list(tweetDict.values())

{% endhighlight %}

### Data Information

Not every instance actually contains a reason for staying or leaving. Some may be sympathizing with those sharing, or reflecting on the trend itself. Others may be ads or jokes. The following chart shows the distribution of such classes in a random sample of 1000 instances. More information is in the paper.

![alt text]({{ site.baseurl }}/assets/images/tweet_data.png "Class distribution in Tweet Data")

> A = ads , J = jokes, L = reasons for leaving, M = meta commentary, O = other , S = reasons for staying, annotated by annotators A1-A4, with #L and #S being the assigned gold standard labels.

<section id="reddit"></section>

# Reddit Domestic Abuse Research Data

Some of this dataset was used in Schrading et al. (2015b) to study the dynamics of domestic abuse. Submissions and comments from the following subreddits were collected, and assigned a binary reference label (*abuse* or *non-abuse*) based on the subreddit title:

|                     | abuseinterrupted | domesticviolence | survivorsofabuse | casualconversation | advice    | anxiety   | anger     |
|-------------------- | ---------------- | ---------------- | ---------------- | ------------------ | --------- | --------- | --------- |
| gold standard label | abuse            | abuse            | abuse            | non abuse          | non abuse | non abuse | non abuse |
| num submissions collected | 1653             | 749              | 512              | 7286               | 5913      | 4183      | 837       |

Additional subreddit data were also collected and used to examine classification in unused subreddits:

|                           | relationships | relationship_advice |
| ------------------------- | ------------- | ------------------- |
| num submissions collected | 8201          | 5874                |

### Download

Download the entire reddit database [here](https://www.dropbox.com/s/1iqf9tx2s5rxdcr/new_reddit.db.7z?dl=0){:target="_blank"}.

Download the "shelved" sets of reddit data [here](https://www.dropbox.com/s/zsncys6m2hhqpi9/reddit_data_shelves.7z?dl=0){:target="_blank"}.

Download the "shelved" abuse classifier trained on the uneven set of data [here](https://www.dropbox.com/s/ehut622z4371yyb/abuseClassifier.7z?dl=0){:target="_blank"}.

### Format

There are two formats for the Reddit data. The most flexible is the entire database used to store all the data I collected. You will have to use sqlite to access the database ([Python has an API](https://docs.python.org/2/library/sqlite3.html){:target="_blank"} for interacting with sqlite databases). For those who do not wish to interact with the database but want to access the provided datasets used in my experiments, I have provided [Python shelved](https://docs.python.org/2/library/shelve.html){:target="_blank"} data files for use in Python.

The sqlite database named "new_reddit.db" has 3 tables within: ***submissions***, ***comments***, and ***submission_srls***. Their columns are as follows:

#### Submissions table

* ***subreddit*** - The name of the subreddit that the submission was posted to.
* ***author*** - The username of the author of the submission.
* ***title*** - The title of the submission.
* ***selftext*** - The selftext of the submission. Can be None or blank.
* ***id*** - The unique identifier of the submission. Can be accessed online by going to http://www.reddit.com/{id}, where {id} is the id of the submission.
* ***external_link*** - A link to a different website. Can be None or blank.
* ***score*** - The score of the submission as determined by Reddit.
* ***num_comments*** - The number of comments within the submission.
* ***created_utc*** - The UTC time at which the submission was created.
* ***link_flair*** - Any 'flair' that the user or moderators applied to this submission. Can be None or blank.
* ***sentiment*** - The overall sentiment of the submission as calculated by [VADER](https://github.com/cjhutto/vaderSentiment){:target="_blank"}.

#### Comments table

* ***subreddit*** - The name of the subreddit that the comment exists in.
* ***author*** - The username of the author of the comment.
* ***submission_id*** - The unique identifier of the submission that the comment is within. Can be accessed online by going to http://www.reddit.com/{id}, where {id} is the id of the submission.
* ***comment_id*** - The unique identifier of the comment.
* ***parent_id*** - The unique identifier of the comment that is a parent to the current comment. If the current comment is a top-level comment, this is the unique identifier of the submission.
* ***body*** - The text of the comment.
* ***depth*** - The depth of the comment in a comment chain. A top-level comment has a depth of 0. A comment in reply to a depth 0 comment has a depth of 1, etc.
* ***num_gilded*** - The number of times this comment was given "Reddit gold".
* ***score*** - The score of the comment as determined by Reddit.
* ***created_utc*** - The UTC time at which the submission was created.
* ***sim_score*** - The cosine similarity of the comment's text to the submission's text.
* ***sentiment*** - The overall sentiment of the comment as calculated by [VADER](https://github.com/cjhutto/vaderSentiment){:target="_blank"}.

#### Submission_srls table

* ***submission_id*** - The unique identifier of the submission that the semantic role exists within. Can be accessed online by going to http://www.reddit.com/{id}, where {id} is the id of the submission.
* ***role*** - The label of the semantic role as provided by [PropBank](http://verbs.colorado.edu/~mpalmer/projects/ace.html){:target="_blank"}.
* ***predicate*** - The predicate that the semantic role is associated with.
* ***text_slice*** - The text that the role is associated with.
* ***start*** - The start index of the text slice.
* ***end*** - The end index of the text slice.
* ***predicate_sense_num*** - The sense number of the predicate by the [Illinois Curator](http://cogcomp.cs.illinois.edu/page/software_view/Curator){:target="_blank"}.

The shelved files provided are as follows (note that lists are aligned e.g. submissionId lists align with the submissions in data lists. Also note that a submission is the initial post, and comments are linked to it by the associated submission ID):

* ***redditAbuseSubmissions*** This data is an even set of 552 *abuse* submissions and 552 *non-abuse* submissions. Each submission has been parsed by the Illinois Curator for Semantic Role Labels. It has the variables:
    * data: A list of submission titles and text concatenated, 1 entry per submission.
    * labels: A list of labels (abuse or non_abuse), 1 entry per submission.
    * subIds: A list of reddit submission ids, 1 entry per submission.
    * roles: A list of lists. Each inner list has the semantic role labels in a submission. 1 list per submission.
    * predicates: A list of lists. Each inner list is a tuple of (predicates, sense number) in a submission. 1 list per submission.
* ***redditAbuseComments*** This data contains all the comments within the submissions in the small even set of submissions. It has the variables:
    * commData: A dictionary, where the key is a reddit submission id and the value is a list of comments in that submission.
    * commLabels: A dictionary, where the key is a reddit submission id and the value is a list of labels given to the comments (abuse or non_abuse).
* ***redditAbuseOnlyNgrams*** This data contains a larger set of even data (1336 submissions per class), with no semantic roles or predicates. It has the variables:
    * XTrain: A list of submission title and text concatenated together, 90% training size (1202 per class).
    * XTest: A list of submission title and text concatenated together, 10% testing size (134 per class).
    * labelsTrain: A list of labels (abuse or non_abuse), 1 entry per submission.
    * labelsTest: A list of labels (abuse or non_abuse), 1 entry per submission.
    * subIdsTrain: A list of reddit submission ids, 1 entry per submission.
    * subIdsTest: A list of reddit submission ids, 1 entry per submission.
* ***redditAbuseUneven*** This data is an uneven set of data with 1336 *abuse* submissions and 17020 *non-abuse* submissions. It has the variables:
    * XTrain: A list of submission title, text, and comment data concatenated together, 85% training size.
    * XTest: A list of submission title, text, and comment data concatenated together, 15% testing size.
    * labelsTrain: A list of labels (abuse or non_abuse), 1 entry per submission.
    * labelsTest: A list of labels (abuse or non_abuse), 1 entry per submission.
    * subIdsTrain: A list of reddit submission ids, 1 entry per submission.
    * subIdsTest: A list of reddit submission ids, 1 entry per submission.
* ***redditRelationshipsData*** This data contains all relationship and relationship_advice submissions with at least 1 comment.
    * data: A list of submission title, text, and comment data concatenated together.
    * subIds: A list of reddit submission ids, 1 entry per submission.

The shelved classifier is a scikit-learn [Pipeline](http://scikit-learn.org/stable/modules/pipeline.html){:target="_blank"} consisting of [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html){:target="_blank"} with a custom tokenizer, followed by a [LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html){:target="_blank"}. Its key in the shelf is "classifier". To predict whether text is *abuse* or *non-abuse*, call the predict() function given a list of text. For example:

{% highlight python %}

shelf = shelve.open("abuseClassifier")
classifier = shelf['classifier']
shelf.close()
print(classifier.predict(["I was abused and hit and I was sad :(", "I am happy and stuff. Love you!"]))

{% endhighlight %}

> ['abuse', 'non_abuse']

Note that to use these shelved objects, you may need to use Python 3, not 2.

<section id="terms"></section>

# Terms of Use

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

Twitter paper, Schrading et al. (2015a):
    
[\#WhyIStayed, \#WhyILeft: Microblogging to Make Sense of Domestic Abuse](http://anthology.aclweb.org/N/N15/N15-1139.pdf){:target="_blank"}

    @InProceedings{schrading-EtAl:2015:NAACL-HLT,
      author    = {Schrading, Nicolas  and  Ovesdotter Alm, Cecilia  and  Ptucha, Raymond  and  Homan, Christopher},
      title     = {\#WhyIStayed, \#WhyILeft: Microblogging to Make Sense of Domestic Abuse},
      booktitle = {Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
      month     = {May--June},
      year      = {2015},
      address   = {Denver, Colorado},
      publisher = {Association for Computational Linguistics},
      pages     = {1281--1286},
      url       = {http://www.aclweb.org/anthology/N15-1139}
    }

Reddit paper, Schrading et al. (2015b):

    To appear at EMNLP 2015.