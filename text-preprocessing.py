#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenize
    sentences = sent_tokenize(text)

    # Tokenize into words and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for sentence in sentences for word in word_tokenize(sentence) if word.isalnum() and word.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    return lemmatized_words

text = """While bismuth compounds (Pepto-Bismol) decreased the number of bowel movements in those with travelers' diarrhea, they do not decrease the length of illness.[91] Anti-motility agents like loperamide are also effective at reducing the number of stools but not the duration of disease.[8] These agents should be used only if bloody diarrhea is not present.[92]

Diosmectite, a natural aluminomagnesium silicate clay, is effective in alleviating symptoms of acute diarrhea in children,[93] and also has some effects in chronic functional diarrhea, radiation-induced diarrhea, and chemotherapy-induced diarrhea.[45] Another absorbent agent used for the treatment of mild diarrhea is kaopectate.

Racecadotril an antisecretory medication may be used to treat diarrhea in children and adults.[86] It has better tolerability than loperamide, as it causes less constipation and flatulence.[94]"""
preprocessed_text = preprocess_text(text)
print(preprocessed_text)


# In[2]:


import spacy

nlp = spacy.load('en_core_web_sm')

def pos_and_ner(text):
    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]

    return pos_tags, named_entities

pos_tags, named_entities = pos_and_ner(text)
print("POS Tags:", pos_tags)
print("Named Entities:", named_entities)


# In[3]:


def dependency_parsing_and_srl(text):
    doc = nlp(text)
    dependency_tree = [(token.text, token.dep_, token.head.text) for token in doc]
    semantic_roles = [(token.text, token.dep_, token.head.text, [child.text for child in token.children]) for token in doc]

    return dependency_tree, semantic_roles

dependency_tree, semantic_roles = dependency_parsing_and_srl(text)
print("Dependency Tree:", dependency_tree)
print("Semantic Roles:", semantic_roles)


# In[19]:





# In[23]:


import stanfordnlp

# Set the Stanford NLP server URL
stanfordnlp.download('en')  # Download English models
nlp = stanfordnlp.Pipeline(server='http://localhost:9000')


# Load the English model
nlp_stanford = stanfordnlp.Pipeline(processors='tokenize,ner,coref')

def coreference_resolution(text):
    doc = nlp_stanford(text)
    coreferences = []

    for cluster in doc._.coref_clusters:
        coref_text = ' '.join(
            doc.sentences[cluster.main.sentNum - 1].words[cluster.main.start - 1:cluster.main.end - 1])
        coreferences.append((cluster.main.text, coref_text))

    return coreferences

coreferences = coreference_resolution(text)
print("Coreferences:", coreferences)


# In[ ]:


def generate_questions(text):
    # Define rules for question generation based on Bloom's Taxonomy
    # You can customize these rules based on your requirements

    # Rule for remembering level
    remember_rule = lambda entity: f"What is {entity}?"

    # Rule for understanding level
    understand_rule = lambda entity, relation: f"Explain the relationship between {entity} and {relation}."

    # Rule for applying level
    apply_rule = lambda entity, action: f"How would you apply {action} to {entity}?"

    # Apply rules based on the semantic roles
    generated_questions = []

    for _, _, _, children in semantic_roles:
        if len(children) >= 2:
            entity = children[0]
            relation = children[1]
            action = children[-1]
            generated_questions.extend([
                remember_rule(entity),
                understand_rule(entity, relation),
                apply_rule(entity, action)
            ])

    return generated_questions

questions = generate_questions(text)
print("Generated Questions:")
for i, question in enumerate(questions, start=1):
    print(f"{i}. {question}")


# In[ ]:


def categorize_questions(questions):
    # Categorize questions based on Bloom's Taxonomy levels
    remember_questions = [q for q in questions if "What is" in q]
    understand_questions = [q for q in questions if "Explain the relationship" in q]
    apply_questions = [q for q in questions if "How would you apply" in q]

    return remember_questions, understand_questions, apply_questions

remember_questions, understand_questions, apply_questions = categorize_questions(questions)

print("Remember Questions:", remember_questions)
print("Understand Questions:", understand_questions)
print("Apply Questions:", apply_questions)


# In[ ]:





# In[ ]:




