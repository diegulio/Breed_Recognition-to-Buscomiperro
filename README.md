# Description

**This project aims to classify the images into dog's breed or cats.** This is part of a large project where the end goal is to recommend found dog / cat posters to people who have posted their own lost dogs / cats and viceversa. So one feature to identify similarities on these posts will be the images or breed of their pets. 

Labes from the data collected are the following:

- 111 dog's breed *
- cat
- No detectado (No detected)

* In the beginning, there was 120 breed, but in some steps decisions were made to delete some of them. (More info in the "Project life diary")

---

# Model aproach

On this project it was experimented with some pre-trained models (The selected pre-trained model was EfficientNet) and the metrics achieved were:

| Modelo  | Train Accuracy | Test Accuracy | Train Loss | Test Loss |
|---------|----------------|---------------|------------|-----------|
| ModelV3 | 98.9%          | 97.8%         | 0.0481     | 0.0819    |

---

# Project Life Diary

In this repository you can find a file called `[ProjectLifeDiary.md]` . There I put the steps that I did to carry out the project. You should get in mind that I am a student and Deeplearning enthusiast, so I could have had mistakes in the process. The entire process help me a lot to understand this incredible field, I expect you to enjoy it!

The principal steps in the process were:

1. Data Collection
2. Exploratory Analysis
3. Modelling
4. Validation and Error capturing
5. Error-based Decisions 
6. Update data and Model
7. App deployment

---

# App deployment demo

Note: I couldn't deploy it in heroku because the model was too heavy. But anyway there is a repository called [BMP-breed-classifier-deployment]([https://github.com/diegulio/BMP-breed-classifier-deployment](https://github.com/diegulio/BMP-breed-classifier-deployment)). There you can find all the files required to deploy the app.

![Breedog%20Classification%20ac06c7af099a468b865889b19a1c76e8/bmp.gif](Breedog%20Classification%20ac06c7af099a468b865889b19a1c76e8/bmp.gif)
