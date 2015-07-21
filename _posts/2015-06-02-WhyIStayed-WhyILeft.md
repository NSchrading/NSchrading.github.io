---
layout: post
title: "#WhyIStayed, #WhyILeft: Microblogging to Make Sense of Domestic Abuse"
quote: "Analyzing a Twitter trend to determine the reasons victims of domestic abuse give for staying in and leaving their abusive relationships."
image:
    url: /media/2015-06-02-WhyIStayed-WhyILeft/naacl.jpg
    source:
video: false
comments: false
---

<style media="screen" type="text/css">

#post-info {
    background-color: rgba(33, 40, 42, 0.74);
    box-sizing: border-box;
    padding: 15px;
}

</style>

# Overview

As part of my research for my thesis and as a member of the computational linguistics and speech processing lab ([CLASP](https://www.rit.edu/clasp/){:target="_blank"}) at RIT, I began research into the #WhyIStayed trend on Twitter shortly after it began. Eventually, I published my findings at [NAACL-HLT 2015](http://naacl.org/naacl-hlt-2015/){:target="_blank"}, presenting a short paper poster during NAACL's June 2015 conference.

#### Published paper:

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

# Abstract

In September 2014, Twitter users unequivocally reacted to the Ray Rice assault scandal by unleashing personal stories of domestic abuse via the hashtags #WhyIStayed or #WhyILeft. We explore at a macro-level firsthand accounts of domestic abuse from a substantial, balanced corpus of tweeted instances designated with these tags. To seek insights into the reasons victims give for staying in vs. leaving abusive relationships, we analyze the corpus using linguistically motivated methods. We also report on an annotation study for corpus assessment. We perform classification, contributing a classifier that discriminates between the two hashtags exceptionally well at 82% accuracy with a substantial error reduction over its baseline.

# Technologies

I used primarily Python, Scikit-learn, NLTK, and [TurboParser](https://www.cs.cmu.edu/~ark/TurboParser/){:target="_blank"}. I also utilized some tools in MATLAB for experimentation with dimensionality reduction. Later research integrated [spaCy](https://honnibal.github.io/spaCy/){:target="_blank"} rather than NLTK and TurboParser for faster, more robust natural language processing, but this was after the NAACL paper was already published.

# Presentations

I have presented this work at 3 different venues:

#### University of Rochester Medical Center's Office of Mental Health Promotion: Community Counts Lunch and Discussion

Rochester, NY  
July 31, 2015 

#### Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies

Denver, Colorado  
June 02, 2015

#### Rochester Institute of Technologies Graduate Research Symposium

Rochester, NY  
February 27, 2015

# Data

Get the data used in this study [here]({{ site.baseurl }}/data/#twitter){:target="_blank"}.

