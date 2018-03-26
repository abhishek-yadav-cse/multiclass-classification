<h1>Medical Text Classification</h1>
<br><br>
<b>Description:</b><br>
The objectives of this task are the following:
 Choose appropriate techniques for modeling text. <br>
 Experiment with various classification models.<br>
 Think about dealing with imbalanced data.<br>
 F1 Scoring Metric<br>


<br>
<b>Detailed Description:</b><br>
Develop predictive models that can determine, given a medical abstract, which of 5 classes it falls in.
Medical abstracts describe the current conditions of a patient. Doctors routinely scan dozens or hundreds of abstracts each day as they do their rounds in a hospital and must quickly pick up on the salient information pointing to the patient’s malady. You are trying to design assistive technology that can identify, with high precision, the class of problems described in the abstract. In the given dataset, abstracts from 5 different conditions have been included: digestive system diseases, cardiovascular diseases, neoplasms, nervous system diseases, and general pathological conditions.
The goal of this competition is to allow you to develop predictive models that can determine, given a particular medical abstract, which one of 5 classes it belongs to. As such, the goal would be to develop the best classification model.
As we have learned in class, there are many ways to represent text as sparse vectors. Feel free to use any of the code in activities or write your own for the text processing step.
Since the dataset is imbalanced, the scoring function will be the F1-score instead of Accuracy.
<br><br>      
<b>Caveats:</b><br>
+ The dataset has an imbalanced distribution i.e., there are different numbers of samples in each class. No information is provided for the test set regarding the distribution.<br>
+ Use the data mining knowledge you have gained until now, wisely, to optimize your results.<br>
<br><br>
<b>Data Description:</b><br>
The training dataset consists of 14442 records and the test dataset consists of 14438 records. We provide you with the training class labels and the test labels are held out. The data are provided as text in train.dat and test.dat, which should be processed appropriately.
<br>
train.dat: Training set (class label, followed by a tab separating character and the text of the medical abstract).<br>
test.dat: Testing set (text of medical abstracts in lines, no class label provided). format.dat: A sample submission with 14438 entries randomly chosen to be 1 to 5.<br>
<br><br>
<b>Files Needed: </b><br>
 Training Data: train.dat <br>
 Test Data: test.dat<br>
 Format File: format.dat<br>
<br>
<hr>
<br>					
<b>Approach:</b><br>
Since the given file is a series of paragraph consisting of sentences, I need to convert it into numerical values in order to run the machine learning algorithm. Initially, my approach was to use bag of words consisting of train data in which the frequency for an unique word can be determined. However, on choosing that the Medical Abstract that are long and consist of more words will get more weightage than the Abstracts that are small. To avoid this problem,concept of Term-Frequency(TF) was used. Further, to avoid giving more weightage to common words, Term Frequency times inverse document frequency (TF-IDF) is used to scale weight of each words.<br>

My approach involved implementation of various classifiers and then comparing performance of each of them.<br>

The classifier used are - 
Naive Bayes Classifier<br>
Logistic Regression<br>
LinearSVC<br>

To Build everything all together, I used Scikit-learn's pipeline class<br>

“Scikit-learn's pipeline class is a useful tool for encapsulating multiple different transformers alongside an estimator into one object, so that you only have to call your important methods once (fit(), predict(), etc)”[1]<br>

“Transformer in scikit-learn - some class that have fit and transform method, or fit_transform method.” [2]<br>
“Predictor - some class that has fit and predict methods, or fit_predict method.”[2]
For this assignment I have fixed the Transformer to be CountVectorizer() and TfidfTransformer()	
The predictor here are the different classifiers mentioned above. 
For my approach I changed the classifiers and fit the dataset from ‘train.dat’ file.
The predict() method is then used to supply the ‘test.dat’ and predict the output of the model.	
The classification result was then written on the output “prediction.dat” file.					
<br><br>
<b>Initial Attempt:</b><br>
Used CountVectorizer(without any parameters)and TF-IDF transformer with MultinomialNB classifier and got F1 score of 0.7887
					
<br><br>
<b>Final Attempt:</b><br>	
In my final attempt, I used the TF-IDF vectorizer and tweaked various parameters in the form of ngram range. The classifier used was LinearSVC to get the best result.<br>
				
Here is tabular conclusion of all my submissions, the best is highlighted.<br>		
<pre>	
<b>N gram-range      Stemming      Classifier Used     Features Extracted      F1 - Score</b>
NA                  NA            Naive Bayes              yes                0.7887
<i>(1,4)               NA             LinearSVC               yes                0.7935</i>
(1,4)               NA         Logistic Regression         yes                0.7686
(1,2)               NA             LinearSVC               yes                0.7312
</pre>
<br><br>
<b>References:</b>
[1] ,[2] - https://stackoverflow.com/questions/33091376/python-what-is-exactly-sklearn-pipeline-pipeline<br>
https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a

