
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
- Train Test Split
- Perform Vectorisation and TF-IDF
- Label Encoding
- Create model using Neural Network
- Test for unseen data
- Apply Cosine Similarity To Find Similar Texts
```

## Data Cleansing
Because the raw texts are dirty, I had to do some pre-processing to clean the text.

**Some of these processes are:**
| Tokenization | Remove Punctuation | Remove Stop Words | Lemmatization | 
| --- | --- | --- | --- |
| Sentence Tokenize | string.punctuation | stop = nltk.corpus.stopwords.words('english') | WordNetLemmatizer |
| Word Tokenize | ! | 's |  |
|  | " | http |  |
|  | # | `` |  |
|  | $ |  |  |
|  | % |  |  |
|  | & |  |  |
|  | ' |  |  |
|  | ( |  |  |
|  | ) |  |  |
|  | * |  |  |
|  | + |  |  |
|  | , |  |  |
|  | - |  |  |
|  | . |  |  |
|  | / |  |  |
|  | : |  |  |
|  | ; |  |  |
|  | < |  |  |
|  | = |  |  |
|  | > |  |  |
|  | ? |  |  |
|  | @ |  |  |
|  |[ |  |  |
|  | \ |  |  |
|  | ] |  |  |
|  | ^ |  |  |
|  | _ |  |  |
|  | ` |  |  |
|  | { |  |  |
|  | | |  |  |
|  | } |  |  |
|  | ~ |  |  |

## Randomly Combine Review Texts From Positive & Negative Reviews Data Sets
I had to combine both the positive and negative review texts together with a corresponding label.

**Size of texts and their labels after combining:**
| Size of texts | Size of labels |
| --- | --- | 
| 2000 | 2000 |

## Train Test Split
Splitted the texts into 80% for training, 20% for testing. I performed Train Test Split using [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

**After Train Test Split**
| Train Data | Train Targets | Test Data | Test Targets |
| --- | --- | --- | --- |
| 1600 | 1600 | 400 | 400 |

## Perform Vectorisation and TF-IDF
Because the model doesn't receive text as inputs, I had to vectorize each word into vectors. Then on these vectors, I performed Term Frequency and Inverse Document Frequency(TF-IDF) on them, scoring the uniqueness from a range of 0 - 1, using [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). The more often the word appears in the document, the less unique it is, and the lower the TF-IDF score. This allows me to know what are the important and less important words. Friend of mine, Dr Chang, gave advices on how to build the tfidfvectorizer. He corrected my mistake specifically on how I used the fit_transform method. I had to be careful when building the vectorizer. Should only fit_transform on the X_train then use transform on X_test. The reason for not using fit_transform on X_test is because fit_transform chooses the best words you provide. So even though you may have equal amount of vocabs in both sets, using fit_transform may result in the mis-alignment in the arrays(because the vocabs are different). It would render your validation set useless because your model is validating against nonsense. He also used an analogy of describing the TF-IDF function like a mother function, it gave birth to the vectorizer,  then you can use it subsequently.

## Label Encoding
I had to do categorical label encoding, reshaping them into a different shape of array. Both for y_train and y_test. 

**Shape of y_train after label encoding**
| Rows | Columns | dtype |
| --- | --- | --- |
| 1600 | 2 | int8 |

**Shape of y_test after label encoding**
| Rows | Columns | dtype |
| --- | --- | --- |
| 400 | 2 | int8 |

## Create model using Neural Network
**Model Architecture**

I created a simple neural network because this is only a binary classification problem. Didn't have to use a neural network to be honest but requirements said so. My Neural Network is of a [sequential class](https://keras.io/guides/sequential_model/), the input layer consists of 64 units, an activation of sigmoid and input dimensions of 500 because it is the maximum number of vocab my model is going to learn based on the vectorization performed earlier. The output layer will consists of 2 units( for class 0 & 1), an activation of softmax. 

![image](https://user-images.githubusercontent.com/89140773/209499086-315dcb72-2c07-4247-ae23-5d6c3c996412.png)

**Compiling the model**

To compile the model, I used the [Adam optimizer](https://keras.io/api/optimizers/adam/), loss of [binary crossentropy](https://keras.io/api/losses/probabilistic_losses/#binary_crossentropy-function) and metrics of [accuracy](https://keras.io/api/metrics/accuracy_metrics/). I could've also used a loss of [categorical crossentropy](https://keras.io/api/losses/probabilistic_losses/#categorical_crossentropy-function) to compile my model.

**Fitting the model**

Here I began by passing in the training data(x_train & y_train respectively),  a batch size of 20, 100 epochs, my validation data(x_test & y_test respectively) and also my callbacks such as [EarlyStopping](https://keras.io/api/callbacks/early_stopping/), [ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/) and [learningRateScheduler](https://keras.io/api/callbacks/learning_rate_scheduler/).


## Test for unseen data 
I used the model to make predictions on totally unseen texts,
the following two texts are:

**Text with negative sentiment:** Just when you think you’ve seen the worst movie ever made, along comes this pile of toxic waste.

**Text with positive sentiment:** This is going to go down as one of 2022’s most entertaining motion pictures.



## Apply Cosine Similarity To Find Similar Texts
Based on the given unseen positive and negative text previously, find the top most similar sentences from the training dataset.
In the mathematical analogy of cosine similarity, it is calculate based on the degree of angle point *A* & *B*. The closer the angles are to each other, the more similar the vectors are and the higher the cosine similarity. This degree of measure is computed within a score range of 0 to 1, using [sklearn.metrics.pairwise.cosinesimilarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html). In the case, we are trying to compare the similarity between the given negative and positive texts against the features in our training data and retrieve words with similar meaning. The reason I chose to compute the cosine similarity word to word was because there were semantic meanings in the text that made feature engineering difficult, it is easy for us to look at the sentence as a whole, but not likewise for the model. So to counter this issue, I break paragraphs to sentences, sentences to words, words to vectors and vectors to TF-IDF. Once the TF-IDF values were obtained, I computed the cosine similarity for each word in the unseen texts against each word(vocab/terms) in the training dataset. Then I sort the values from the largest to smallest and display the relevant word for the top cosine similarity values.




## To install dependencies:
```
pip install -r reqs\requirements.txt
```





