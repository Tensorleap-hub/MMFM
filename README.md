
# MMFM model with IconQA dataset

<div style="text-align: center;">
    <img src="images/open.png" alt="Untitled" width="700"/>
</div>

<br>

The [IconQA](https://github.com/lupantech/IconQA/tree/main) (icon question answering) dataset aims to highlight the 
importance of abstract diagram understanding and comprehensive cognitive reasoning in real-world diagram word problems. 
It consists of three sub-tasks: multi-image-choice, multi-text-choice, and filling-in-the-blank. 
IconQA requires not only perception skills like object recognition and text understanding, but also diverse cognitive 
reasoning skills, such as geometric reasoning, commonsense reasoning, and arithmetic reasoning.

In this project we used the multi-text-choice data with a pre-trained Multimodal Foundation Model.

Using **Tensorleap** we can explore the latent space, easily detect unlabeled clusters, and handle those with high loss. 
This quick start guide will walk you through the steps to get started with this example repository project.

### Population Exploration

Below is a population exploration plot. It represents a samples similarity map based on the model's latent space,
built using the extracted features of the trained model.

It shows a visualization of the latent space where each dot represents a sample. The color and the size affected from the loss value.
In the figure below the latent space is colored by the questions type.

<div style="text-align: center;">
    <img src="images/PE1.png" alt="Untitled" width="600"/>
</div>

<br>

### *Detecting High Loss Clusters*

#### *- Using Tensorleap Insight*

When filtering the latent space by the higher loss samples the first "low performance" insight correlated to the
question word "how" and other metadata, indicates that the model fails to predict how many marbles are in the image.

<div style="display: flex; justify-content: center;">
  <div style="flex: 1; text-align: center;">
    <img src="images/insight.png" alt="Image 1" width="400" style="margin-right: 10px;">
  </div>
  <div style="display: flex; flex-direction: row;">
    <img src="images/marbles2.png" alt="Image 2" width="400" style="margin-right: 10px;">
    <img src="images/marbles3.png" alt="Image 3" width="400" style="margin-left: 10px;">
  </div>
</div>

#### *- Using PE*

When examining the population exploration (PE), we notice a group in the upper part of the latent space (marked with a 
yellow circle) that contains images associated with the same question: “On which color is the spinner less likely to land?“

<div style="text-align: center;">
    <img src="images/kmeans-1.png" alt="Untitled" width="600"/>
</div>

<br>

Further investigation revealed two distinct groups: one with higher loss and one with lower loss. We found that the model
consistently chooses 'white' as the answer, regardless of the actual conditions in the images.

<div style="display: flex; justify-content: center;">
  <img src="images/white1.png" alt="Image 1" width="350" style="margin-right: 10px;">
  <img src="images/white2.png" alt="Image 2" width="350" style="margin-left: 10px; margin-right: 10px;">
  <img src="images/white3.png" alt="Image 3" width="350" style="margin-left: 10px; margin-right: 10px;">
  <img src="images/white4.png" alt="Image 4" width="350" style="margin-left: 10px;">
</div>

#### *- Using Dashboards*

In tensorleap platform we can create and use dashboards in an easy way.
Each sample contains required skills (one or more) to answer the question correctly.
Using the dashboard, we found that tasks containing the 'fraction' skill tend to have a higher loss value.

![dashboard](images/dashboard.png)

### *Detecting Unlabeled Clusters in the Latent Space*
Now, let’s look for additional clusters in our data using an unsupervised clustering algorithm on the model’s latent
space.

Upon examination of these clusters, we can see that clusters 6, 13 and 18, located close to each other, 
contain different question and images, but they are all related to time and clock. 
The proximity of these clusters in the latent space suggests that the model has recognized a higher-level relationship 
among these concepts, grouping them together due to their shared relevance to the theme of time and clocks.

<div style="text-align: center;">
    <img src="images/bise.png" alt="Untitled" width="600"/>
</div>

<br>

### *Fetching similar samples*

Another approach to finding clusters using the model’s latent space is fetching similar samples to a selected sample.
It enables you to identify a cluster with an intrinsic property you want to investigate. 
By detecting this cluster, you can gain insights into how the model interprets this sample and, in general, retrieve 
clusters with more abstract patterns.

The figure below shows a cluster of images with the question: ״What has been done to this letter?"

<div style="text-align: center;">
    <img src="images/fetch_similar.png" alt="Untitled" width="600"/>
</div>

<br>

Upon analysis, we have noticed that the model consistently fails when the ground truth answer is 'flip'. 
This difficulty suggests a specific challenge for the model in accurately identifying changes related to flipping letters.

<div style="display: flex; justify-content: center;">
  <img src="images/flip1.png" alt="Image 1" width="350" style="margin-right: 10px;">
  <img src="images/flip2.png" alt="Image 2" width="350" style="margin-left: 10px; margin-right: 10px;">
</div>

#### *Sample Loss Analysis*

In this section, we can see the results of a gradient-based explanatory algorithm to interpret what drives the model to 
make specific predictions. It enables us to analyze which of the informative features contributes most 
to the loss function. We then generate a heatmap with these features that shows the relevant information.

Let’s analyze the following sample containing the question: “Are there enough carrot s for every rabbit?”. The correct 
predicted answer is: “no”. We see that the token that had the most impact on the model’s prediction is:
‘enough′.

<div style="display: flex; justify-content: center;">
  <img src="images/SA.png" alt="Image 1" width="350" style="margin-right: 10px;">
  <img src="images/SA2.png" alt="Image 2" width="350" style="margin-left: 10px; margin-right: 10px;">
</div>


# Getting Started with Tensorleap Project

This quick start guide will walk you through the steps to get started with this example repository project.

## Prerequisites

Before you begin, ensure that you have the following prerequisites installed:

- **[Python](https://www.python.org/)** (version 3.7 or higher).
- **[Poetry](https://python-poetry.org/)**.
- **[Tensorleap](https://tensorleap.ai/)** platform access. To request a free trial click [here](https://meetings.hubspot.com/esmus/free-trial).
- **[Tensorleap CLI](https://github.com/tensorleap/leap-cli)**.


## Tensorleap **CLI Installation**

with `curl`:

```
curl -s https://raw.githubusercontent.com/tensorleap/leap-cli/master/install.sh | bash
```

## Tensorleap CLI Usage

### Tensorleap **Login**

To login to Tensorleap:

```
tensorleap auth login [api key] [api url].
```

- API Key is your Tensorleap token (see how to generate a CLI token in the section below).
- API URL is your Tensorleap environment URL: https://api.CLIENT_NAME.tensorleap.ai/api/v2

<br>

**How To Generate CLI Token from the UI**

1. Login to the platform in 'CLIENT_NAME.tensorleap.ai'
2. Scroll down to the bottom of the **Resources Management** page, then click `GENERATE CLI TOKEN` in the bottom-left corner.
3. Once a CLI token is generated, just copy the whole text and paste it into your shell.


## Tensorleap **Project Deployment**

To deploy your local changes:

```
leap projects push
```

### **Tensorleap files**

Tensorleap files in the repository include `leap_binder.py` and `leap.yaml`. The files consist of the  required configurations to make the code integrate with the Tensorleap engine:

**leap.yaml**

leap.yaml file is configured to a dataset in your Tensorleap environment and is synced to the dataset saved in the environment.

For any additional file being used, we add its path under the `include` parameter:

```
include:
    - leap_binder.py
    - mmfm/configs.py
    - [...]
```

**leap_binder.py file**

`leap_binder.py` configures all binding functions used to bind to Tensorleap engine. These are the functions used to evaluate and train the model, visualize the variables, and enrich the analysis with external metadata variables

## Testing

To test the system we can run `leap_binder.check()` function using poetry:

```
poetry run test
```

This file will execute several tests on leap_binder.py script to assert that the implemented binding functions: preprocess, encoders,  metadata, etc,  run smoothly.

*For further explanation please refer to the [docs](https://docs.tensorleap.ai/)*




