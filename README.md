# NLP_Curriculum_Notes
## NLP Natural Language Processing Curriculum, Study Topics and Applications Notes

## Recommended Courses:
https://www.udemy.com/course/nlp-natural-language-processing-with-python

## Recommended Books:
https://www.amazon.com/Natural-Language-Processing-Action-Understanding/dp/B07X37578L

## Real World Uses/Cases/User-Problems (common tools in parenthesis):
	- topic modeling (Latent Dirichlet Allocation, Non-negative Matrix Factorization)
	- text classification (sklearn, spaCy, NLTK)
		-- e.g. classifying email as spam or legit
- spam filter (Multinomial Naive Bayes Classifier (vs. Gaussian Naive Bayes classification)) 
	-- example of classification
-- statquest https://www.youtube.com/watch?v=O2L2Uv9pdDA
	- feature extraction (sklearn/spaCy?)
	- flagging offensive / inappropriate content (hate-speech and abuse detection)
	- NER = Named Entity Recognition
- POS = part of speech tagging 
- entity extraction (spaCy function)
- recommendation (various)
- fuzzy search (various)
- sentiment analysis (vector-embeddings: NLTK, spaCy, basilica, R)
	-- movie reviews
	-- customer feedback (restaurant, online shop)
	-- e,g, https://www.linkedin.com/learning/integrating-tableau-and-r-for-data-science/where-r-rules 
- chatbots / assistants / agents
- text generation (LSTM, GRU, NN)
- Summarizing Long Text Documents (RNN, CNN, Transformers)
- Understanding Commands ("computer, do X!")
	-- Understanding Text Commands
	?
	voice to text
	text to voice
	- fake-news detection

## Recommended Courses:
- https://www.udemy.com/course/nlp-natural-language-processing-with-python

## Tools:
	- python string functions
	- regex https://drive.google.com/file/d/1JVmK-pW4IKHIEj0UUvqLXelcYbbUPhuE/
- sklearn
- Multinomial Naive Bayes Classifier
- NLTK (designed to be adjustable, not for speed)
- spaCy (designed for speed using one-size fits all models)
- basilica
- 
- Neural Networks:
- RNN (Recurrent Neural Network)
- CNN (Convolutional Neural Networks)
- Transformers (NN)
- LSTM (Long Short Term Memory, (not forgetting) vs. RNN)
- GRU (Gated Recurrent Unit, a variation on LSTM system)
- Generative Auto-Encoder NN
https://www.udemy.com/course/nlp-natural-language-processing-with-python/learn/lecture/13195724#overview

## General Process:
- tokenization
- stop words
(my blog on this)
https://medium.com/wooden-information/what-are-stopwords-precisely-960590b21548
- Stemmatization (see note below)
- phrase-matching
- lemmatization (see notes below)
- semantics
- vectorizing/embeddings (example: 
On how vector-embeddings work:
https://colab.research.google.com/drive/1n0QHVKLmjHhb1J0PVumoxq58-1OevP5b
calculate distance with e.g. "Cosine Distance" https://reference.wolfram.com/language/ref/CosineDistance.html

## Methods, Approaches:
- Bag Of Words (e.g. with high bias, low variance, Multinomial Naive Bayes Classifier)
e.g. Spam Filter
- Vector Space / vectorizing / embeddings
- Lexicon Based Analysis
- Sequence-to-sequence



## Issues, Limitations, Challenges:
	- Apple's notorious problem of making (accurate) calendar appointments based on text from emails and messages (super hard)
	- 'negatives' such as 'not' in sentences (very difficult)
	- understanding from context (very difficult)
	- legal and contract english (very difficult)
	- tools for doctors to process large amounts of research publications  (very difficult)

## Ethics and NLP
	- Exclusion/Discrimination/Bias
- Privacy 
- VSD value sensitive design (http://faculty.washington.edu/ebender/2017_575/#vsd)
https://aclweb.org/aclwiki/Ethics_in_NLP
http://faculty.washington.edu/ebender/2017_575/


## Books:
	Discussion of Issues
https://www.amazon.com/Rebooting-AI-Building-Artificial-Intelligence/dp/1524748250

	Teaching Model Use
https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646


## History & Approaches
#### Symbolic vs. Subsymbolic
- heuristic approaches and "knowledge bases"




.......................................................................................................................................................


# On Stemmatize vs Lemmatize

For NLP it is helpful to reduce forms of words to ones that are easily compared.

Two methods for this are 
1. stemmatizing (or reducing to clipped 'stem' form of the word)
2. lemmatizing (or using a more general and abstract "lemma" form of a word)
https://en.wikipedia.org/wiki/Lemmatisation 

In various languages one 'word' may have different forms that are spelled very differently: 
- wolf, wolves
- be, are, was, were, is,

In some languages (such as Japanese) there is a 'dictionary form' separate from the gazillion other word-forms based on that. But in languages such as English, there is no 'dictionary form,' so a new 'meta-word-form' was created for this purpose and this is called a "lemma."

## Stem vs. lemma

In some cases you could use a 'stem' (form of a word) as a proxy for a 'dictionary form' or 'lemma' form of a word. 
- e.g. pour poured pouring

The stem of these examples is all the same "pour," which you can get by removing the extra letters in the other forms (the 'ed' and the 'ing')

The problem with 'stems' is that often clipping the end off a word will not give you the same 'root-step'
- fishes -> fish (this is ok)
- wolves -> wolv (this is not ok. 'wolf' and 'wolv' are not the same "root" form)

This is a problem when the stems "wolf" and "wolv" would be treated by the NLP engine as two unrelated words (assumed to have different meanings and uses, which is an error in the model).

The 'lemma' is one standard/root/dictionary/generalized/abstract form which all other forms can be converted to so that different texts can be compared.  




.......................................................................................................................................................


## On NLTK vs spaCy
Overall, spaCy is optimized for fast and easy out-of-the-box use.
NLTK is easier to customize. 

#### If you want more personal customization -> use NLTK
#### If you want optimized as-is performance -> use spaCy

.......................................................................................................................................................


On

...

New Steps (GGA):
Alias Standardization

....
spaCy Notes

Doc Object:  contains "processed text"
Voc Object (vocabulary object)

(see with "nlp.pipeline")
1. Tagger
2. Parser
3. 'ner' Describer


dep = syntactic dependency

ner = named entity recognizer
