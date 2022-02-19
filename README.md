# ga_learn
Rev UC 2022 Hack-a-thon. Visualizing GA to optimize the use of NN to classify.

## Goals
- Explore P5 JS for visualization
  - Basic understanding beforehand
- Explore Tensorflow in JS
  - Used in Python for simple FF classifier
- Visualize GA
  - Dissertation uses EC, but I haven't explore visualization past simple text description
- Visualize learning through a NN

## Data Set
The data set used in this work can be found at https://www.kaggle.com/ronitf/heart-disease-uci

## Major decisions
- Data set?
  - Check kaggle for simple data set.
  - Bottle neck of training is huge.
- GPU to train networks? 
  - Possibly train in python, output results, then visualize with P5 JS.
  - WHAT is needed for GPU training, and what can be an output?
- Use GA for learning, or use GA to refine connections and size of network?
  - Will define set up of GA
  - Plan to allow both, if data set is small enough, the second option is fine. Training of NN will be bottleneck

## Steps
- Learn tensorflow API. DONE.
- Run simple NN using data set. DONE
- Visualize NN from tensorflow in P5 JS.
- Build GA, and plot GA stats.
- Publish beta for hack-a-thon.

## Completed Before Hand
- Repo created. DONE.
- Issues defined in GitHub
- Libraries installed DONE.
- JS wed server prepared
  - In Python on a local network, viewable in browser
- Data set selected