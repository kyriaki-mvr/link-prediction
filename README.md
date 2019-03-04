# Link-Prediction
Scala


Technologies for Big Data Analytics 
Project: Link Prediction in Citation Networks
1. Introduction
In this data challenge you will work in teams of at most 2 persons. The problem you have to solve
is related to predicting links in a citation network. In our case, the citation network is a graph G(V,
E), where nodes represent scientific papers and a link between nodes u and v denotes that one of the
papers cites the other. Each node in the graph contains some properties such as: the paper abstract,
the year of publication and the author names. From the original citation network some randomly
selected links have been deleted. Given a set of possible links, your job is to determine the ones that
appeared in the original network. You will work in Spark using the Scala programming language.
2. File Description
There are five datasets provided. A short description of these files follows:
training_set.txt - 615,512 labeled node pairs (1 if there is an edge between the two nodes, 0 else).
One pair and label per row, as: source node ID, target node ID, and 1 or 0. The IDs match the papers
in the node_information.csv file .

testing_set.txt - 32,648 node pairs. The file contains one node pair per row, as: source node ID,
target node ID. Evidently, the label is not available (your job is to find the label for every pair).


node_information.csv - for each paper out of 27,770, contains the following information (1) unique
ID, (2) publication year (between 1993 and 2003), (3) title, (4) authors, (5) name of journal (not
available for all papers), and (6) abstract. Abstracts are already in lowercase, common English
stopwords have been removed, and punctuation marks have been removed except for intra-word
dashes

Cit-HepTh.txt – this is the complete ground truth network. You should use this file only to evaluate
your solutions with respect to the accuracy. You should not take it into account to create your
solutions.

3. Evaluation Metric
For each node pair in the testing set, your model should predict whether there is an edge between the
two nodes (1) or not (0). The testing set contains 50% of true edges (the ones that have been removed
from the original network) and 50% of synthetic, wrong edges (pairs of randomly selected nodes
between which there was no edge).
The evaluation metric for this competition is Mean F1-Score. The F1 score measures accuracy using
precision and recall. Precision is the ratio of true positives (tp) to all predicted positives (tp + fp).
Recall is the ratio of true positives to all actual positives (tp + fn). 

This metric weights recall and precision equally.

4. Problems to Solve
You have to provide solutions for the following problems:
P1. Given the network and a list of possible links, provide predictions if the links exist or not. You
ma attack the problem by different perspectives. For example, you may solve the problem as a
classification problem using a supervised approach or an unsupervised one.

P2. You are given only the papers and you are asked to discover all possible links that may appear in
the network. In this version, you do not have an initial insight about which papers are linked. This
problem is in general more difficult compared to P1, because the number of possible links is
quadratic in the number of papers.
Note that both problems must be solved using Spark, which means that you need to provide
distributed algorithms to solve them.
