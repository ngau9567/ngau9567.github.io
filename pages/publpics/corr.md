---
layout: frontpage
date: 12/14/2020
title: Analysis on Prediction of Trending YouTube Videos Applying Machine Learning Techniques
---

## Predicting the Number of likes for Trending YouTube Videos

YouTube is an influential and popular online video-sharing tool that is rated
as one of the largest search engines owned by Google. Because of its convenient
feature of uploading and sharing videos, it has reached 1.9 billion users worldwide
by the end of 2019. Each day, more than 1 million videos are being viewed in the
U.S., and almost 5 million videos are being viewed globally.

Hence, our goals for this project are: To identify key features that predict
trending videos are being liked the most in the U.S. Use Machine Learning to train
model(s) on prediction and then evaluate and improve model performance. We
targeted a broad range of audience, basically anyone who is interested in the topic.
Ideally, we wanted to show the process of using big data and build the model to
explain and predict the question of interest.

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)]({{ BASE_PATH }}/assets/SEAS6401/HW5/ML 14 - Capstone Project (1).html) 
[![Paper PDF](https://img.shields.io/badge/Project Report-red?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAABv1BMVEWpk5KxrKuyAAC0tbW2AAC5ubm9vb2/v7/AtbXEyMjJysrMnZzQ0NDRd3XUwcHU1NTYAADY2NjZjY3a2trbXFfbjYvb29vchoTchoXclJPcpKLc3NzdgoDdjo3dmJbdzs7d3d3eLSTekY/elpPe3t7e4eHfenffgn/f39/gBADgaWXgycngysng4ODg5eXhU07hxcTh4eHiycjiysni4uLjoqHjtbTj4uLj4+Pj5OTj5eXj6enklpTkx8bk2tnk5OTk6enlg4Ll4eHl5OTl5eXm3t3m5eXm5ubm6Ojm6+zn5ubn5+fn6enn6+vokY7o5OTo5+fo6Ojo6enp5eXp6Ojp6enp7O3q7ezq7u7rqafrvLrr2trr6urr6+vr9vbsiIbsubjs1NPs4+Ps6+vtpqPt1NPt1dTt7e3t7+/u6+vvko7vu7vvyMbv7+/v+fnwy8rw7e3x7+/x8vLx8vPy5eXy8vLy8/Py9/fy+vrz2dnz8fHz9vbz+fn0hID04N/08vL08/P1h4T19PT1/Pz2z873+fn4+Pj4+fn4+vr58/P5+fn6+vr7+/v8/Pz8/f38///98fH9///+/v7//v7////9cRVWAAAA2klEQVQYVwXBu00DQRRA0Ttv3s7sencsQ4IgICdBogVEWQSQ0IBjEuqgAwogQUR8JEJj4bXn8zjHrd3TV8i5zu02NQEhv8VyEPO7+600EIjTtJrSqYTHjTTEWVgO42KU8vt5t5EmtDCl5Tj5Qao9IAo+VieH1dXgPr73UaFL2qrtFjmfzKCgCyy/nzeJTg2FGHCHaVljMwWF2PvmL39c+usHEOhSDNuaQjcED4qJj3bmre6DeFAIWnwozroqAgpHY3QFMyvdsaG4l3luFCrweo3SXzwLAPh8E/kHIjxRqCpB+v4AAAAASUVORK5CYII=&style=plastic)]({{ BASE_PATH }}/assets/FinalPaper.pdf)
[![Slides PDF](https://img.shields.io/badge/Project Slides-orange?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAABv1BMVEWpk5KxrKuyAAC0tbW2AAC5ubm9vb2/v7/AtbXEyMjJysrMnZzQ0NDRd3XUwcHU1NTYAADY2NjZjY3a2trbXFfbjYvb29vchoTchoXclJPcpKLc3NzdgoDdjo3dmJbdzs7d3d3eLSTekY/elpPe3t7e4eHfenffgn/f39/gBADgaWXgycngysng4ODg5eXhU07hxcTh4eHiycjiysni4uLjoqHjtbTj4uLj4+Pj5OTj5eXj6enklpTkx8bk2tnk5OTk6enlg4Ll4eHl5OTl5eXm3t3m5eXm5ubm6Ojm6+zn5ubn5+fn6enn6+vokY7o5OTo5+fo6Ojo6enp5eXp6Ojp6enp7O3q7ezq7u7rqafrvLrr2trr6urr6+vr9vbsiIbsubjs1NPs4+Ps6+vtpqPt1NPt1dTt7e3t7+/u6+vvko7vu7vvyMbv7+/v+fnwy8rw7e3x7+/x8vLx8vPy5eXy8vLy8/Py9/fy+vrz2dnz8fHz9vbz+fn0hID04N/08vL08/P1h4T19PT1/Pz2z873+fn4+Pj4+fn4+vr58/P5+fn6+vr7+/v8/Pz8/f38///98fH9///+/v7//v7////9cRVWAAAA2klEQVQYVwXBu00DQRRA0Ttv3s7sencsQ4IgICdBogVEWQSQ0IBjEuqgAwogQUR8JEJj4bXn8zjHrd3TV8i5zu02NQEhv8VyEPO7+600EIjTtJrSqYTHjTTEWVgO42KU8vt5t5EmtDCl5Tj5Qao9IAo+VieH1dXgPr73UaFL2qrtFjmfzKCgCyy/nzeJTg2FGHCHaVljMwWF2PvmL39c+usHEOhSDNuaQjcED4qJj3bmre6DeFAIWnwozroqAgpHY3QFMyvdsaG4l3luFCrweo3SXzwLAPh8E/kHIjxRqCpB+v4AAAAASUVORK5CYII=&style=plastic)]({{ BASE_PATH }}/assets/Trending on YouTube Video.pdf)

[![Final Project](/assets/publpics/corr.PNG)]({{ BASE_PATH }}/pages/SEAS6401.html#seas6401-final-project)

<div class="navbar">
  <div class="navbar-inner">
      <ul class="nav">
          <li><a href="seas6401_hw3_capstone.html">Previous</a></li>
          <li><a href="emse6574_hw9_timeseries.html">Next</a></li>
      </ul>
  </div>
</div>
