---
layout: page
title: Research Data
permalink: data/
---

<style media="screen" type="text/css">

.nav > li > a:hover, .nav > li > a:focus {
    text-decoration: none;
    background-color: #ECF0F1;
}

.navbar-nav > li > a {
    padding-top: 19.5px;
    padding-bottom: 19.5px;
}

.navbar-nav > li > a {
    padding-top: 10px;
    padding-bottom: 10px;
    line-height: 21px;
}

.nav > li > a {
    position: relative;
    display: block;
    padding: 10px 15px;
}

.navbar-nav {
    float: left;
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


.nav {
    list-style: outside none none;
}

a:focus {
    outline: thin dotted;
    outline-offset: -2px;
}

a {
    background: transparent none repeat scroll 0% 0%;
}

a {
    color: #2A7EAD;
    text-decoration: none;
}

* {
    box-sizing: border-box;
}

.container {
    width: 900px;
    padding-left: 0px;
    padding-right: 0px;
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
                <a class="scroll" data-speed="500" href="#reddit">Reddit Abuse</a>
            </li>
        </ul>
    </div>
</div>

<br>

This page contains links to data I have collected for my thesis. You are welcome to use any of this data for your research, but please cite the relevant paper if you do so :)

Twitter paper:
    
[\#WhyIStayed, \#WhyILeft: Microblogging to Make Sense of Domestic Abuse](anthology.aclweb.org/N/N15/N15-1139.pdf)

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


<section id="twitter"></section>

# \#WhyIStayed / \#WhyILeft Research Data

In September 2014, Twitter users unequivocally reacted to the Ray Rice assault scandal by unleashing personal stories of domestic abuse via the hashtags \#WhyIStayed or \#WhyILeft. This study explored at a macro-level firsthand accounts of domestic abuse from a corpus of tweeted instances designated with these tags to seek insights into the reasons victims give for staying in vs. leaving abusive relationships. A classifier that discriminates between the two hashtags was created, achieving 82% accuracy (50% is the binary baseline using an even dataset).

### Download

Download this data from my public dropbox [here](https://www.dropbox.com/s/n4fp4sq572422ex/stayedLeftData.json.7z?dl=0).

### Format
Twitter restricts public sharing of tweets to only tweet ids. Therefore the format of the data is as follows:

* An individual sample is on a new line of the file.
* Each sample is encoded as JSON.
* Some tweets were split into multiple parts if they contained both hashtags - one part for the \#WhyIStayed reason and one part for the \#WhyILeft reason. Only a single tweet in the dataset needed to be split into 4s, but logic handled this case as well.
* Each sample has the properties ***id***, ***label***, ***startIdx***, and ***endIdx***
    * ***id***: The unique identifier of the tweet.
    * ***label***: The ground truth label this sample was given, either \#WhyIStayed or \#WhyILeft, based off of the hashtag used.
    * ***startIdx***: If the tweet was split into multiple parts, this is the starting index into the tweet text. If the tweet was not split this is null.
    * ***endIdx***: If the tweet was split into multiple parts, this is the index into the tweet text at which the sample ends. If the tweet was not split this is null.

### Getting the tweet text and additional information using Twitter's API

In order to gather the tweet text and additional information about the tweet, like user information and retweet count, you must use Twitter's API. I would recommend using the [GET statuses/lookup](https://dev.twitter.com/rest/reference/get/statuses/lookup) API call because it allows you to look up 100 tweet ids at a time in bulk. Provided below is Python code to do this, using [Twython](https://github.com/ryanmcgrath/twython) as a wrapper for Twitter's API.

#### Important
To use this code, you ***must*** have Twython installed. Installation instructions, and how to set up a Twitter API account are on [Twython's githup page](https://github.com/ryanmcgrath/twython).

If you use this code, the text provided (Tweet.text) will still have hashtags, and is uncleaned. You will want to remove hashtags, and probably do standard cleaning procedures like lowercasing, lemmatizing, and stoplisting.

{% highlight python %}

# collects data from the publicly released data file
import json

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
    # open the shared file and extract its data for all tweet samples
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

### Data Cleanliness

Not every sample actually contains a reason for staying or leaving. Some may be sympathizing with those sharing, or reflecting on the trend itself. Others may be ads or jokes. An annotation study on 1000 random samples from the data was conducted with 4 annotators. The following chart shows the resulting distribution of classes in this data:

![alt text]({{ site.baseurl }}/assets/images/tweet_data.png "Class distribution in Tweet Data")

> According to the annotations in this random sample, on average 36% of the instances are reasons for staying (S), 44% are reasons for leaving (L), 12% are meta comments (M), 2% are jokes (J), 2% are ads (A), and 4%  do not match prior categories (O).

<section id="reddit"></section>

# Reddit Abuse Research Data

This data was used to create a general classifier to find text describing abuse. Analysis of this data can reveal the general dynamics of abuse, and can be used to study how online users discuss abuse. Discourse studies between submitters and commenters could be a very interesting experiment to conduct. Other studies on relationships and advice seeking behavior on Reddit could also be conducted. The following subreddits were collected from:

|                     | abuseinterrupted | domesticviolence | survivorsofabuse | casualconversation | advice    | anxiety   | anger     |
|-------------------- | ---------------- | ---------------- | ---------------- | ------------------ | --------- | --------- | --------- |
| gold standard label | abuse            | abuse            | abuse            | non abuse          | non abuse | non abuse | non abuse |
| num submissions collected | 1653             | 749              | 512              | 7286               | 5913      | 4183      | 837       |

Additionally, the following subreddits were collected from and used as a held out set. Annotators annotated to determine the precision of the abuse classifier.

|                           | relationships | relationship_advice |
| ------------------------- | ------------- | ------------------- |
| gold standard label       | unclassified  | unclassified        |
| num submissions collected | 8201          | 5874                |

### Download

Download this data from my public dropbox [here](https://www.dropbox.com/s/1iqf9tx2s5rxdcr/new_reddit.db.7z?dl=0).

### Format

All data in this study is contained in a sqlite database named "new_reddit.db". There are 3 tables within this database: ***submissions***, ***comments***, and ***submission_srls***. Their columns are as follows:

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
* ***link_flair*** - Any special flair that the user or moderators applied to this submission. Can be None or blank.
* ***sentiment*** - The overall sentiment of the submission as calculated by [VADER](https://github.com/cjhutto/vaderSentiment).

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
* ***label*** - Some comments may have been labeled as abuse or non_abuse in experiments. If they were, this is their label as assigned by a classifier. May be incorrect, and should not be considered ground truth.
* ***sentiment*** - The overall sentiment of the comment as calculated by [VADER](https://github.com/cjhutto/vaderSentiment).

#### Submission_srls table

* ***submission_id*** - The unique identifier of the submission that the semantic role exists within. Can be accessed online by going to http://www.reddit.com/{id}, where {id} is the id of the submission.
* ***role*** - The label of the semantic role as provided by [PropBank](http://verbs.colorado.edu/~mpalmer/projects/ace.html).
* ***predicate*** - The predicate that the semantic role is associated with.
* ***text_slice*** - The text that the role is associated with.
* ***start*** - The start index of the text slice.
* ***end*** - The end index of the text slice.
* ***predicate_sense_num*** - The sense number of the predicate as determined by the [Illinois Curator](http://cogcomp.cs.illinois.edu/page/software_view/Curator).

