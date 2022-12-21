
![NLP Essentials Project Cover Page](https://user-images.githubusercontent.com/89140773/207842683-4443f5b6-b20e-487a-9eee-6e7af3fa2638.png)


![image](https://user-images.githubusercontent.com/89140773/208248644-3efdad3a-9ce7-4375-87fb-4065ca44c00e.png)

## Introduction
In this project, I was tasked to build a Neural Network that was capable of predicting whether a movie review is of a positive sentiment or a negative one, and it is categorised as a binary classification problem(0 or 1). 0 stands for negative sentiment, 1 stands for positive sentiment. 

## Given Scenario
In a recent market research by IBM, almost one-third of IT professionals surveyed globally say their business is using Artificial Intelligence (AI). The survey reported almost half of businesses today are now using applications powered by natural language processing (NLP).NLP, specifically 'Sentiment Analysis', can provide a key business advantage by abstracting from unstructured text data the negative or positive attitude of the writer/author. This crucuial insight can help turn web data into market intelligence for the business.A request has been received from the web development department to add 'Sentiment Analysis' feature to a movie reviews page.
The insights from the sentiment analysis will be used to promote more popular movies.In this project, you are tasked to create a prototype NLP project that is capable of 'Sentiment Analysis' from movie reviews.

*Note that Task 1 was provided as a guide*

## Some of the steps I did to achieve the final results:

```
- Data Cleansing
- Randomly Combine Review Texts From Positive & Negative Reviews Data Sets
- Perform Vectorisation and TF-IDF
- Perform AI Natural Language Inference
- Label Encoding
- Create model using Neural Network
- Test for unseen data
- Apply Cosine Similarity To Find Similar Texts
```

## To install dependencies:
```
pip install -r reqs\requirements.txt
```


## Difficulties encountered and learning points:
Friend of mine, Dr Chang, gave advices on how to build the tfidfvectorizer. He corrected my mistake specifically on how I used the fit_transform method. I had to be careful when building the vectorizer. Should only fit_transform on the X_train then use transform on X_test. The reason for not using fit_transform on X_test is because fit_transform chooses the best words you provide. So even though you may have equal amount of vocabs in both sets, using fit_transform may result in the mis-alignment in the arrays(because the vocabs are different). It would render your validation set useless because your model is validating against nonsense. He also used an analogy of describing the TF-IDF function like a mother function, it gave birth to the vectorizer,  then you can use it subsequently.


