---
layout: post
title: "The Danger Ranger"
quote: "An Autonomous fixed-wing aircraft for identifying injured humans on the ground."
image:
    url: /media/2015-05-21-danger-ranger/danger_ranger.jpg
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

I, along with 4 teammates, worked on an unmanned fixed-wing aircraft for aerial surveillance for our 2-semester Senior Design project.

# Motivation

Every year across the globe, there are countless floods, tornados, hurricanes, ice storms, and other phenomena that turn large areas into disaster zones. People often go missing within these zones, and aerial search and rescue is one of the most effective methods of finding these missing persons. A manned aerial search and rescue operation can cost between $100,000 and $200,000 depending on the scale of the target area and number of volunteers. A rural community may not have a large enough budget for these operations. A system is required that can reduce the cost of search and rescue operations in large rural areas, as well as improve the speed at which missing persons are found.

The Danger Ranger is the first step towards creating a cheap and effective aerial search and rescue platform. It is capable of visually identifying “interesting” targets it has been trained to detect. Upon identification, the drone transmits its GPS coordinates and a snapshot of the identified target to a human operator. The operator can determine if it is a real target or a false-positive and can command the drone to surveil the target or continue searching the area.

# My Role

I designed and developed the image processing unit onboard the aircraft. This involved research into imaging devices that would work well at our desired altitude, research into small, cost-effective general purpose computing platforms, and development of the image capture, processing, and object-detection software.

{% include image.html url="/media/2015-05-21-danger-ranger/targets_found.jpg" description="Example of several objects (cardboard pieces) detected from the air during flight." %}

# Technologies Used

* ***Onboard processing platform***: [Beagleboard-xM](http://beagleboard.org/beagleboard-xm){:target="_blank"} running Ubuntu
* ***Camera***: [Leopard Imaging LI-USB30-M021](https://www.leopardimaging.com/LI-USB30-M021.html){:target="_blank"}.
* ***Software***: 
    * Custom video4linux (v4l) C++ code: To capture images with the camera on Linux.
    * Custom C++ code: To perform object detection and send any object snippets to the ground station via the radio.
    * [OpenCV](http://opencv.org/){:target="_blank"}: To perform [Viola-Jones](http://docs.opencv.org/modules/objdetect/doc/cascade_classification.html){:target="_blank"} object detection.
    * MATLAB: To [train a Viola-Jones object detector](https://www.mathworks.com/help/vision/ug/train-a-cascade-object-detector.html){:target="_blank"}.

# Awards

We showcased our project at the [Imagine RIT Innovation and Creativity Festival](https://www.rit.edu/imagine/){:target="_blank"} and entered into the ARM Student Design Contest. The Danger Ranger won the [Paychex Sponsor Award](https://www.rit.edu/imagine/sponsor_awards_2015.php){:target="_blank"} and the ARM 
Design Contest [Popular Vote](https://www.rit.edu/imagine/planyourday15/exhibit.php?id=601){:target="_blank"}.

{% include image.html url="/media/2015-05-21-danger-ranger/danger_ranger_win.jpeg" description="Receiving the Paychex Sponsor Award at the Imagine RIT Innovation and Creativity Festival." %}