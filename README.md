# **Diskoveror-ML-Trainer**

This is the Trainer for Diskoveror Text Analytics Engine.

#### **Topic Extraction**

   **The Basic Idea**

We use these Word2Vec vectors to form clusters by feeding the desired topics to be learnt as seeds to the model, we  follow a unique approach in performing the clustering, that ultimately leads to the formation of clusters that       capture the semantics of any topic fed into it.

USP's of our model,
   *  It is capable of predicting topics based on the context in which a word/phrase of the text occurs.
      *   Example,

          1. **Text 1**: The number of by-products derived from cow’s milk is just unbelievable.
          2. **Text 2**: All the products sold by Flipkart are of high quality.
      *    Our model is capable of determining that Text 1 is about FOOD Products and Text 2 is about any GENERAL                 product based on the context of the sentence.
      *    It is equipped to learn any number of topics and the best part is you could specify what topics it needs to            learn and categorize any given document into.
      *    Example,

           1.  You could specify that you want to learn 2 topics, say, Sports and Technology and the model will                       train itself for these two categories/topics and later the trained model could be used to predict                      to which topic any wild document could come under.
           2.  A unique scoring mechanism has been developed, thus enabling us to tell how probable is each of the                    topics/categories given any document, we even rank the topics in descending order given any document (we                don’t give out the scores as part of our results, maybe in future we would include that feature) before                returning to the customer.
           3.  The model has also been crafted in a way that, it does not learn all the noisy topics from the                         text,meticulous evaluations are done before we tag the text under any topic (The algorithms that                       we use to achieve this is our secret recipe), thus ensuring the accuracy of the model.
           4.  We do analysis at both coarse and finer levels of topics, thus enabling our model to be bang on                        when it does a prediction.
                     *     We currently have a set of 22 coarse topics that in turn have                                                          750+ finer topics contained in them.                     
                           Eg. Music is a coarse topic and Jazz, Pop, Melody, etc are finer                                                       topics.

#### **The Working**

   <center>![The Working](https://raw.githubusercontent.com/tpsatish95/diskoveror-ml-trainer/master/Assets/topic-model-work-flow.jpg)</center>

#### **Results Snippet**
As our model is semi supervised we don’t have train or test data to evaluate our model, but we did randomly sample     a set of 5 tweets from twitter and have presented the results below (we got 4 out of 5 right !).

<center>![Results Snippet](https://raw.githubusercontent.com/tpsatish95/diskoveror-ml-trainer/master/Assets/topics-results.jpg)</center>



## Sentiment Analysis

Sentiment Analysis has always been an excellent source of information that expresses user opinion towards any particular product or service, topic, etc. It is also capable of providing numerous insights that can be used to formulate marketing strategies, improve campaign success, improve customer service etc of any company. In short, if perfect sentiment analysis is done it will improve any company's bottom line for sure.

Sentiment Analysis has always been a great research problem to solve, most of the solutions that companies use today are more rule based and are not capable of adapting to the dynamic world, we data scientists at Serendio are proposing a Machine Learning approach to this problem and we have developed a model that is capable of learning sentiments from nearly 36 Domains and also dynamically adapts to the new trends, unlike the rule based approaches. We also support 5 different types of text sources that include blogs, microblogs, news articles, reviews, comments and general text.

#### Key Features

*  **Social Text Processor (SocialFilters):**
   Of all the text data available to us, 60% to 70% are completely unstructured and are not suitable for most of the      NLP tools available today for processing them. We at serendio are working to process all of this unstructured data     into text suitable for Text Analytics applications with Processing Filters for:
   1. Acronyms [LOL, OMG, etc.]
   2. Emoticons [ :) , :( , etc.]
   3. Spell Check
   4. Contractions
   5. Hashtags
   6. Sentiment Scorer for Emoticons, Acronyms, Hashtags.
      We have discovered that by processing the hashtags, emoticons, acronyms, etc the sentiment engine becomes more         wiser in predicting the sentiment score.

*  **Multi Domain Support:**
   The main reason why Sentiment Analysis has always been a challenging problem is because of the wide range of domains    available and each domain has its own set of sentiment rich words or phrases that conflict with those in the other     domains. To overcome this we have built Domain specific models that are capable of capturing the sentiment semantics    of each domain and thus able to handle text from any domain with great precision. Especially for reviews we support    a set of:
   1. **11 Top Domains** Products, Electronics and Technology, Movies, Services, Books, Food, Hotels and Bars, Music,           Places, Restaurants, Travel and Tour.
   2. **36 Finer Sub Domains**,  that comes under each of the top domains.
       *  Example: The top domain “Products” has 16 sub domains ranging from apparels to grocery.

*  **Multiple Text Sources Support:**
   We at serendio have done extensive research on the best mechanisms to predict the sentiment of various text types      that include text from sources like: Blogs, Microblogs, News Articles, Reviews, Comments and General text. Each of     these text sources has its own sentiment rich spots that we need to concentrate on to get the precise sentiment        scores. Some Sentiment Rich Spots,
   *  Title of the text in the case of News Articles, reviews, etc.
   *  First and Last paragraphs in the case of Blogs.
   *  Emoticons, Acronyms and Abbreviations in the case of Comments and Microblogs.
      We have intelligent mechanism to dynamically vary the sentiment scoring of any text based on its source and            sentiment hotspots.
 We have intelligent mechanism to dynamically vary the sentiment scoring of any text based on its source and sentiment  hotspots.

*  **Scored Results and Power to the Customer:**
   Most often in the field of sentiment analysis the end results are either positive, negative or neutral given any       text, but we at serendio were really curious to know about the degree of positivity or negativity, that is, given      any text we wanted to determine its sentiment polarity. So instead of just giving class labels, we give integer        values ranging from -5 to +5 (Most Negative to Most Positive) depending on the polarity of the text as results. Thus    making our models more fine and precise in determining the sentiments with more granularity. We give the Customers     the power to determine what score ranges they feel is positive, negative or neutral, thus making the sentiment         models more adaptive to different kinds of businesses, rather than enforcing something predetermined on them.

#### **Modelling**

<center>![Modelling](https://raw.githubusercontent.com/tpsatish95/diskoveror-ml-trainer/master/Assets/sentiment-analysis-workflow.jpg)</center>

#### **Prediction and Scoring**

<center>![Prediction and Scoring](https://raw.githubusercontent.com/tpsatish95/diskoveror-ml-trainer/master/Assets/sentiments-working.jpg)</center>

#### **Results Snippets**
   On an average each of our models have exceptional recall values ranging from 85% to 90%. (we are still working on      ways to improve the accuracy). Find below the result snippets,

  <center>![Results Snippets](https://raw.githubusercontent.com/tpsatish95/diskoveror-ml-trainer/master/Assets/sentiments-results.jpg)</center>
