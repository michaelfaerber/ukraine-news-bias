# A Multidimensional Dataset Based on Crowdsourcing for Analyzing and Detecting News Bias

## Abstract
The automatic detection of bias in news articles can have a high impact on society because undiscovered news bias may influence the political opinions, social views, and emotional feelings of readers. While various analyses and approaches to news bias detection have been proposed, large data sets with rich bias annotations on a fine-grained level are still missing. In this paper, we firstly aggregate the aspects of news bias in related works by proposing a new annotation schema for labeling news bias. This schema covers the overall bias, as well as the bias dimensions (1) hidden assumptions, (2) subjectivity, and (3) representation tendencies. Secondly, we propose a methodology based on crowdsourcing for obtaining a large data set for news bias analysis and identification. We then use our methodology to create a dataset consisting of more than 2,000 sentences annotated with 43,000 bias and bias dimension labels. Thirdly, we perform an in-depth analysis of the collected data. We show that the annotation task is difficult with respect to bias and specific bias dimensions. While crowdworkers' labels of representation tendencies correlate with experts' bias labels for articles, subjectivity and hidden assumptions do not correlate with experts' bias labels and, thus, seem to be less relevant when creating data sets with crowdworkers. The experts' article labels better match the inferred crowdworkers' article labels than the crowdworkers' sentence labels. The crowdworkers' countries of origin seem to affect their judgements. In our study, non-Western crowdworkers tend to annotate more bias either directly or in the form of bias dimensions (e.g., subjectivity) than Western crowdworkers do.

## Summary

On [Zenodo](https://doi.org/10.5281/zenodo.3885351) and in the folder [all-data-as-json](all-data-as-json), we provide a large data set consisting of **2,057 sentences** from 90 news articles and annotations of crowdworkers with respect to **bias** itself and the following **bias dimensions**:

* hidden assumptions
* subjectivity
* representation tendencies

Our data set contains **44,547 labels in total** (43,197 sentence labels and 1,350 article labels).

The news articles deal with the **Ukraine crisis**. They were published in 33 countries in total and were selected based on the data set of Cremisini et al. (Cremisini, A., Aguilar, D., & Finlayson, M. A. (2019). A Challenging Dataset for Bias Detection: The Case of the Crisis in the Ukraine. In International Conference on Social Computing, Behavioral-Cultural Modeling and Prediction and Behavior Representation in Modeling and Simulation (pp. 173-183). Springer, Cham.).

Each sentence was annotated by 5 crowdworkers. In total, we spent $ 3,335 for the crowdworkers annotations.

## Reference
Please cite our data set as follows:
```
@unpublished{Faerber2020Bias,
 author = {Michael F{\"{a}}rber and Victoria Burkard and Adam Jatowt and Sora Lim},
 title  = {{A Multidimensional Dataset Based on Crowdsourcing for Analyzing and Detecting News Bias}},
 booktitle = {{Proceedings of the 29th ACM International Conference on Information and Knowledge Management}},
 series = {{CIKM'20}},
 location = {{Virtual Event}},
 year   = {2020}
}
```
Our paper describing the data set is available [here](https://www.aifb.kit.edu/images/d/de/NewsBias-CIKM2020.pdf).
